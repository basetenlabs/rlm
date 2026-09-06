"""Deferred child trajectories retain the history that each turn actually saw."""

import json

import pytest

import rlm.core.types as types_module
from rlm.core.types import (
    CodeBlock,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.parsing import format_iteration


def diagnostic_iteration():
    # Only nine list objects, but tree serialization expands their shared edges.
    shared = [0]
    for _ in range(8):
        shared = [shared, shared]
    result = REPLResult(
        stdout="full output" * 7000,
        stderr="full error",
        locals={"shared": shared, "alias": shared},
        execution_time=0.25,
        rlm_calls=[
            RLMChatCompletion(
                root_model="leaf",
                prompt="full request",
                response="full response" * 20000,
                usage_summary=UsageSummary({"leaf": ModelUsageSummary(1, 7, 3)}),
                execution_time=0.2,
                metadata={"helper_provenance": {"call_id": "original-id"}},
            )
        ],
        final_answer="genuine answer",
    )
    return RLMIteration(
        prompt=[{"role": "user", "content": "full prompt"}],
        response="full model response",
        code_blocks=[CodeBlock("print('full output')", result)],
        final_answer="genuine answer",
        root_usage={"total_input_tokens": 11, "total_output_tokens": 5},
        reasoning_content="full reasoning",
    )


@pytest.mark.parametrize("level", ["result", "block", "iteration", "logger"])
def test_opt_out_omits_locals_before_any_serialization_and_preserves_feedback(monkeypatch, level):
    iteration = diagnostic_iteration()
    result = iteration.code_blocks[0].result
    feedback = format_iteration(iteration)
    expected = iteration.to_dict()
    del expected["code_blocks"][0]["result"]["locals"]

    def forbidden_serialization(*args, **kwargs):
        pytest.fail("excluded diagnostic locals must never be traversed")

    monkeypatch.setattr(types_module, "_serialize_value", forbidden_serialization)
    if level == "result":
        actual = result.to_dict(include_locals=False)
        assert actual == expected["code_blocks"][0]["result"]
    elif level == "block":
        actual = iteration.code_blocks[0].to_dict(include_locals=False)
        assert actual == expected["code_blocks"][0]
    elif level == "iteration":
        assert iteration.to_dict(include_locals=False) == expected
    else:
        logger = RLMLogger(include_locals=False)
        logger.log(iteration)
        assert {key: logger._iterations[0][key] for key in expected} == expected
        iteration.prompt[0]["content"] = "later mutation"
        result.rlm_calls[0].metadata["helper_provenance"]["call_id"] = "later mutation"
        assert logger._iterations[0]["prompt"][0]["content"] == "full prompt"
        saved_call = logger._iterations[0]["code_blocks"][0]["result"]["rlm_calls"][0]
        assert saved_call["metadata"]["helper_provenance"]["call_id"] == "original-id"
    assert result.locals["shared"] is result.locals["alias"]
    assert result.locals["shared"][0] is result.locals["shared"][1]
    assert format_iteration(iteration) == feedback


def test_default_logger_still_serializes_locals_and_child_loggers_remain_memory_only():
    logger = RLMLogger()
    iteration = diagnostic_iteration()
    logger.log(iteration)
    assert (
        logger._iterations[0]["code_blocks"][0]["result"]["locals"]
        == (iteration.code_blocks[0].result.to_dict()["locals"])
    )
    child = logger.child_logger()
    assert child.include_locals is True
    assert child.log_file_path is None
    assert child.log_model_responses is False


def test_child_loggers_inherit_policy_and_durable_directory_at_every_depth(tmp_path, monkeypatch):
    import rlm.logger.rlm_logger as logger_module

    root = RLMLogger(
        log_dir=str(tmp_path),
        include_locals=False,
        child_log_dir=str(tmp_path / "children"),
        log_model_responses=True,
    )
    fsync_calls = []
    real_fsync = logger_module.os.fsync

    def record_fsync(fd):
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(logger_module.os, "fsync", record_fsync)
    child = root.child_logger(helper_call_id="child-call")
    grandchild = child.child_logger(helper_call_id="grandchild-call")
    for logger, call_id in [(child, "child-call"), (grandchild, "grandchild-call")]:
        assert logger.include_locals is False
        assert logger.log_model_responses is True
        assert logger.log_dir == str(tmp_path / "children")
        logger.log_metadata(RLMMetadata("model", 3, 2, "openai", {}, "local", {}))
        logger.log(diagnostic_iteration())
        with open(logger.log_file_path) as stream:
            metadata, saved = [json.loads(line) for line in stream]
        assert metadata["helper_call_id"] == call_id
        assert metadata["log_file_path"] == logger.log_file_path
        assert metadata["include_locals"] is False
        assert "locals" not in saved["code_blocks"][0]["result"]
        assert saved["code_blocks"][0]["result"]["stdout"] == "full output" * 7000
        assert saved["code_blocks"][0]["result"]["rlm_calls"][0]["response"] == (
            "full response" * 20000
        )
    assert len(fsync_calls) == 4  # metadata and iteration are durable before returning
    assert len(list(tmp_path.glob("*.jsonl"))) == 0
    assert len(list((tmp_path / "children").glob("*.jsonl"))) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_model_response_is_disk_only_and_durable_before_code_execution(tmp_path, enabled):
    from unittest.mock import Mock

    from rlm import RLM

    logger = RLMLogger(log_dir=str(tmp_path), log_model_responses=enabled)
    rlm = RLM(logger=logger)
    history = [{"role": "user", "content": "exact request"}]
    handler = Mock()
    handler.completion.return_value = "```repl\nraise RuntimeError('stuck code')\n```"
    handler.default_client.get_last_usage.return_value = ModelUsageSummary(1, 7, 3)
    handler.default_client.last_finish_reason = "stop"
    handler.default_client.last_reasoning_content = "full thinking"

    def execute(code):
        with open(logger.log_file_path) as stream:
            records = [json.loads(line) for line in stream]
        received = [row for row in records if row["type"] == "model_response"]
        assert len(received) == int(enabled)
        if enabled:
            assert received[0]["prompt"] == history
            assert received[0]["response"] == handler.completion.return_value
            assert received[0]["root_usage"] == {
                "total_input_tokens": 7,
                "total_output_tokens": 3,
            }
            assert received[0]["iteration"] == 1
            assert received[0]["finish_reason"] == "stop"
            assert received[0]["reasoning_content"] == "full thinking"
        assert logger.iteration_count == 0
        assert logger._iterations == []
        raise RuntimeError("stuck code")

    try:
        with pytest.raises(RuntimeError, match="stuck code"):
            rlm._completion_turn(history, handler, Mock(execute_code=execute))
    finally:
        rlm.close()


def test_logger_snapshots_nested_history_before_subsequent_turn_mutations():
    history = [{"role": "user", "content": [{"text": "first"}]}]
    usage = {"total_input_tokens": 7}
    logger = RLMLogger()
    logger.log(
        RLMIteration(prompt=history, response="first response", code_blocks=[], root_usage=usage)
    )
    history[0]["content"][0]["text"] = "changed"
    history.append({"role": "assistant", "content": "later"})
    usage["total_input_tokens"] = 99
    logger.log(RLMIteration(prompt=history, response="second response", code_blocks=[]))
    assert logger._iterations[0]["prompt"] == [{"role": "user", "content": [{"text": "first"}]}]
    assert logger._iterations[0]["root_usage"] == {"total_input_tokens": 7}
    assert len(logger._iterations[1]["prompt"]) == 2
