"""Offline contracts for opt-in helper records and durable dispatch evidence."""

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from threading import Barrier
from unittest.mock import Mock

import pytest

import rlm.environments.local_repl as local_repl_module
from rlm.core.comms_utils import DEFAULT_WAVE_TIMEOUT, WAVE_TIMEOUT_SLACK, LMResponse
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.environments.local_repl import LocalREPL


@pytest.fixture(autouse=True)
def fresh_streams(monkeypatch):
    monkeypatch.setattr(local_repl_module, "_STREAM_ROUTERS", None)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        yield


def completion(prompt="question", *, error=None):
    return RLMChatCompletion(
        root_model="actual-model",
        prompt=prompt,
        response="provider result" if error is None else "",
        usage_summary=UsageSummary(
            model_usage_summaries={
                "actual-model": ModelUsageSummary(
                    total_calls=1, total_input_tokens=7, total_output_tokens=2
                )
            }
        ),
        execution_time=0.25,
        metadata={"provider_evidence": "retained", "usage_status": "observed"},
        error=error,
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "helper", ["llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"]
)
@pytest.mark.parametrize("failure", ["exception", "returned"])
def test_all_helpers_record_unprinted_failures_and_exact_events(
    monkeypatch, tmp_path, helper, failure
):
    path = tmp_path / "events.jsonl"
    full_prompt = "question" + "x" * 200010
    failed = completion(full_prompt, error="provider failed")
    if helper.startswith("rlm"):
        failed.response = "Error: recursive failure"

    def dispatch(*args, **kwargs):
        started = events(path)
        assert len(started) == 1
        assert started[0]["event"] == "started"
        assert started[0]["prompt"] == full_prompt
        if failure == "exception":
            raise TimeoutError("timed out")
        if helper.startswith("rlm"):
            return failed
        response = LMResponse(error="provider failed", chat_completion=failed)
        return [response] if "batched" in helper else response

    monkeypatch.setattr(local_repl_module, "send_lm_request", dispatch)
    monkeypatch.setattr(local_repl_module, "send_lm_request_batched", dispatch)
    with LocalREPL(
        lm_handler_address=("localhost", 1),
        subcall_fn=dispatch,
        record_failed_calls=True,
        helper_event_log=str(path),
    ) as env:
        arg = f"[{full_prompt!r}]" if "batched" in helper else repr(full_prompt)
        result = env.execute_code(f"{helper}({arg}, model='requested-model')")
    assert result.stdout == result.stderr == ""
    assert len(result.rlm_calls) == 1
    recorded = result.rlm_calls[0]
    assert recorded.prompt == full_prompt
    assert recorded.error
    provenance = recorded.metadata["helper_provenance"]
    assert provenance["helper"] == helper
    assert provenance["requested_model"] == "requested-model"
    assert provenance["elapsed_seconds"] >= 0
    if failure == "returned":
        assert recorded.root_model == failed.root_model
        assert recorded.usage_summary == failed.usage_summary
        assert recorded.metadata["provider_evidence"] == "retained"
        assert provenance["usage_status"] == "observed"
        assert provenance["completion_response"] == failed.response
    else:
        assert provenance["usage_status"] == "unknown"
        assert recorded.metadata["usage_status"] == "unknown"
        assert recorded.usage_summary.model_usage_summaries == {}
    started, completed = events(path)
    assert started["call_id"] == completed["call_id"] == provenance["call_id"]
    assert completed["event"] == "completed"
    assert completed["outcome"] == recorded.response
    assert completed["completion"]["error"] == recorded.error


@pytest.mark.parametrize("enabled", [False, True])
def test_single_timeout_alignment_is_opt_in(monkeypatch, enabled):
    send = Mock(return_value=LMResponse(error="timed out"))
    monkeypatch.setattr(local_repl_module, "send_lm_request", send)
    with LocalREPL(lm_handler_address=("localhost", 1), record_failed_calls=enabled) as env:
        result = env.execute_code("llm_query('question')")
    assert len(result.rlm_calls) == int(enabled)
    assert send.call_args.kwargs.get("timeout", 300) == (
        int(DEFAULT_WAVE_TIMEOUT + WAVE_TIMEOUT_SLACK) if enabled else 300
    )


@pytest.mark.parametrize(
    "helper", ["llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"]
)
def test_missing_handler_is_recorded_without_duplicate_fallback_events(tmp_path, helper):
    path = tmp_path / "events.jsonl"
    with LocalREPL(record_failed_calls=True, helper_event_log=str(path)) as env:
        arg = "['question']" if "batched" in helper else "'question'"
        result = env.execute_code(f"out = {helper}({arg})")
    assert len(result.rlm_calls) == 1
    assert result.rlm_calls[0].response == "Error: No LM handler configured"
    assert [event["helper"] for event in events(path)] == [helper, helper]


def test_parallel_children_share_durable_log_and_preserve_prompt_order(tmp_path):
    path = tmp_path / "events.jsonl"
    barrier = Barrier(2)

    def child(prompt, model):
        assert any(e["event"] == "started" and e["prompt"] == prompt for e in events(path))
        barrier.wait(timeout=5)
        if prompt == "bad":
            raise ValueError("child failed")
        return completion(prompt)

    with LocalREPL(subcall_fn=child, record_failed_calls=True, helper_event_log=str(path)) as env:
        result = env.execute_code(
            "out = rlm_query_batched(['good', 'bad'], model='requested-model')"
        )
    assert [c.prompt for c in result.rlm_calls] == ["good", "bad"]
    assert result.rlm_calls[0].error is None
    assert result.rlm_calls[1].error == "child failed"
    saved = events(path)
    assert len(saved) == 4
    assert len({e["call_id"] for e in saved}) == 2
    assert [e["event"] for e in saved[:2]] == ["started", "started"]


def test_started_event_survives_interruption_without_claiming_completion(tmp_path):
    path = tmp_path / "events.jsonl"

    def interrupt(*args):
        assert events(path)[0]["event"] == "started"
        raise KeyboardInterrupt()

    with LocalREPL(
        subcall_fn=interrupt, record_failed_calls=True, helper_event_log=str(path)
    ) as env:
        with pytest.raises(KeyboardInterrupt):
            env._rlm_query("question")
    assert [event["event"] for event in events(path)] == ["started"]


def test_failed_event_persistence_prevents_dispatch(monkeypatch, tmp_path):
    dispatch = Mock()
    monkeypatch.setattr(local_repl_module.os, "fsync", Mock(side_effect=OSError("disk failed")))
    with LocalREPL(
        subcall_fn=dispatch,
        record_failed_calls=True,
        helper_event_log=str(tmp_path / "events.jsonl"),
    ) as env:
        with pytest.raises(OSError, match="disk failed"):
            env._rlm_query("question")
    dispatch.assert_not_called()


def test_full_success_outcome_is_durable_without_preview_trimming(tmp_path):
    path = tmp_path / "events.jsonl"
    reply = completion()
    reply.response = "answer" + "y" * 200010
    with LocalREPL(
        subcall_fn=lambda *args: reply,
        record_failed_calls=True,
        helper_event_log=str(path),
    ) as env:
        assert env._rlm_query("question") == reply.response
    assert events(path)[-1]["outcome"] == reply.response


def test_event_omits_duplicate_nested_trajectory_but_call_record_retains_it(tmp_path):
    path = tmp_path / "events.jsonl"
    reply = completion()
    reply.metadata["iterations"] = [{"prompt": "large nested history"}]
    with LocalREPL(
        subcall_fn=lambda *args: reply, record_failed_calls=True, helper_event_log=str(path)
    ) as env:
        result = env.execute_code("rlm_query('question')")
    assert result.rlm_calls[0].metadata["iterations"] == reply.metadata["iterations"]
    saved = events(path)[-1]
    assert saved["embedded_trajectory_omitted"] is True
    assert "iterations" not in saved["completion"]["metadata"]
    assert saved["completion"]["metadata"]["provider_evidence"] == "retained"


def test_nested_child_event_is_linked_to_parent_before_dispatch(tmp_path):
    path = tmp_path / "events.jsonl"

    def child(prompt, model):
        with LocalREPL(record_failed_calls=True, helper_event_log=str(path), depth=2) as nested:
            nested._llm_query("leaf question", "leaf-model")
        return completion(prompt)

    with LocalREPL(subcall_fn=child, record_failed_calls=True, helper_event_log=str(path)) as env:
        env._rlm_query("question", "root-model")
    outer_start, inner_start, inner_end, outer_end = events(path)
    assert outer_start["call_id"] == outer_end["call_id"]
    assert inner_start["call_id"] == inner_end["call_id"]
    assert inner_start["parent_call_id"] == outer_start["call_id"]
    assert inner_start["depth"] == 2


def test_preserve_failed_completion_identity_even_when_request_differs(monkeypatch):
    failed = completion("provider-recorded-prompt", error="failed")
    monkeypatch.setattr(
        local_repl_module,
        "send_lm_request",
        Mock(return_value=LMResponse(error="failed", chat_completion=failed)),
    )
    with LocalREPL(lm_handler_address=("localhost", 1), record_failed_calls=True) as env:
        result = env.execute_code("llm_query('actual-request', model='requested-model')")
    recorded = result.rlm_calls[0]
    assert recorded.prompt == "provider-recorded-prompt"
    assert recorded.metadata["helper_provenance"]["prompt"] == "actual-request"
    assert failed.response == ""
    assert "helper_provenance" not in failed.metadata
