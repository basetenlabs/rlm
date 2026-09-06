"""Unit tests for RLM._subcall() method.

Tests for the parameter propagation to child RLM instances:
1. max_timeout (remaining time) is passed to child
2. max_tokens is passed to child
3. max_errors is passed to child
4. model= parameter overrides child's backend model
5. finalization safety settings are preserved through recursive children
"""

import asyncio
import time
from threading import Event
from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.exceptions import BudgetExceededError
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, RLM_SYSTEM_PROMPT_SLOTS
from tests.mock_lm import MockLM


def create_mock_lm(responses: list[str], model_name: str = "mock-model") -> Mock:
    """Create a mock LM that returns responses in order."""
    mock = Mock()
    mock.model_name = model_name
    mock.completion.side_effect = list(responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            model_name: ModelUsageSummary(
                total_calls=1, total_input_tokens=100, total_output_tokens=50
            )
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


def final(content: str) -> str:
    """Render a model response that submits ``content`` as the final answer."""
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


def test_recursive_log_policy_and_unfinished_child_turns_are_durable(tmp_path):
    import json

    from rlm.environments.local_repl import LocalREPL, current_helper_call_id
    from rlm.logger import RLMLogger

    children_dir = tmp_path / "children"
    journal = tmp_path / "helpers.jsonl"
    logger = RLMLogger(
        log_dir=str(tmp_path),
        include_locals=False,
        child_log_dir=str(children_dir),
        log_model_responses=True,
    )
    calls = 0

    def model_response(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "```repl\nanswer['content'] = rlm_query('grandchild task')\nanswer['ready'] = True\n```"
        if calls == 2:
            return "```repl\nshared = [0]\nfor _ in range(8): shared = [shared, shared]\nprint('x' * 70000)\n```"
        assert calls == 3
        # The grandchild has not finished, but its prior turn is on disk.
        logs = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in children_dir.glob("*.jsonl")
        ]
        assert len(logs) == 2
        started = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(started) == 2
        assert all(event["event"] == "started" for event in started)
        assert {rows[0]["helper_call_id"] for rows in logs} == {
            event["call_id"] for event in started
        }
        iterations = [row for rows in logs for row in rows if row["type"] == "iteration"]
        assert len(iterations) == 1
        saved = iterations[0]["code_blocks"][0]["result"]
        assert "locals" not in saved
        assert saved["stdout"] == "x" * 70000 + "\n"
        assert iterations[0]["final_answer"] is None
        return final("genuine child answer")

    mock_lm = MockLM(response_fn=model_response)
    with patch.object(rlm_module, "get_client", return_value=mock_lm):
        parent = RLM(
            backend_kwargs={"model_name": "mock-model"},
            max_depth=3,
            max_iterations=2,
            logger=logger,
            environment_kwargs={"record_failed_calls": True, "helper_event_log": str(journal)},
            fabricate_final_answer=False,
            recover_stub=False,
        )
        try:
            with LocalREPL(
                subcall_fn=parent._subcall,
                record_failed_calls=True,
                helper_event_log=str(journal),
            ) as env:
                result = env.execute_code("print(rlm_query('child task'))")
        finally:
            parent.close()
    assert current_helper_call_id() is None
    assert result.stdout == "genuine child answer\n"
    assert result.stderr == ""
    assert len(list(tmp_path.glob("rlm_*.jsonl"))) == 1
    child = result.rlm_calls[0].metadata
    grandchild = child["iterations"][0]["code_blocks"][0]["result"]["rlm_calls"][0]["metadata"]
    for trajectory in (child, grandchild):
        assert trajectory["run_metadata"]["include_locals"] is False
        assert (
            trajectory["run_metadata"]["helper_call_id"]
            == (trajectory["helper_provenance"]["call_id"])
        )
        assert trajectory["run_metadata"]["log_file_path"].startswith(str(children_dir))
        assert trajectory["iterations"][0]["root_usage"]["total_input_tokens"] == 10


def fixed_routing_rlm(**kwargs):
    return RLM(
        backend_kwargs={"model_name": "glm-root"},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": "qwen-leaf"}],
        fixed_model_routing=True,
        fabricate_final_answer=False,
        recover_stub=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"backend_kwargs": {"model_name": "glm-root"}},
        {
            "backend_kwargs": {"model_name": "glm-root"},
            "other_backends": ["openai"],
            "other_backend_kwargs": [{}],
        },
    ],
)
def test_fixed_routing_rejects_missing_model_configuration_before_clients(kwargs):
    with patch.object(rlm_module, "get_client") as clients:
        with pytest.raises(ValueError, match="fixed_model_routing"):
            RLM(fixed_model_routing=True, **kwargs)
    clients.assert_not_called()


@pytest.mark.parametrize(
    "depth, model, permitted, role",
    [
        (0, "qwen-leaf", "glm-root", "recursive"),
        (0, "typo-glm", "glm-root", "recursive"),
        (1, "glm-root", "qwen-leaf", "leaf"),
        (1, "typo-qwen", "qwen-leaf", "leaf"),
    ],
)
def test_fixed_routing_rejects_recursive_and_terminal_overrides(depth, model, permitted, role):
    with patch.object(rlm_module, "get_client") as clients:
        parent = fixed_routing_rlm(depth=depth, max_depth=2)
        try:
            with pytest.raises(ValueError, match=f"{role}.*{permitted}"):
                parent._subcall("task", model=model)
        finally:
            parent.close()
    clients.assert_not_called()


@pytest.mark.parametrize("model", [None, "qwen-leaf"])
def test_fixed_terminal_subcall_uses_leaf_and_shared_gate(model):
    leaf = MockLM(model_name="qwen-leaf", responses=["leaf answer"])
    from unittest.mock import MagicMock

    gate = MagicMock()
    with (
        patch.object(rlm_module, "get_client", return_value=leaf) as clients,
        patch("rlm.utils.global_gate.get_gate", return_value=gate),
    ):
        parent = fixed_routing_rlm(depth=1, max_depth=2)
        try:
            result = parent._subcall("task", model=model)
        finally:
            parent.close()

    assert result.response == "leaf answer"
    assert result.root_model == "qwen-leaf"
    assert set(result.usage_summary.model_usage_summaries) == {"qwen-leaf"}
    assert clients.call_args.args[1]["model_name"] == "qwen-leaf"
    gate.slot.assert_not_called()
    gate.async_slot.assert_called_once_with()
    gate.async_slot.return_value.__aenter__.assert_awaited_once_with()
    gate.async_slot.return_value.__aexit__.assert_awaited_once()


def test_fixed_direct_completion_at_depth_cap_uses_leaf():
    leaf = MockLM(model_name="qwen-leaf", responses=["leaf answer"])
    with patch.object(rlm_module, "get_client", return_value=leaf) as clients:
        parent = fixed_routing_rlm(depth=2, max_depth=2)
        try:
            result = parent.completion("task")
        finally:
            parent.close()
    assert result == "leaf answer"
    assert clients.call_args.args[1]["model_name"] == "qwen-leaf"


@pytest.mark.parametrize("depth,max_depth", [(0, 1), (1, 2)])
@pytest.mark.parametrize(
    "helper", ["llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"]
)
@pytest.mark.parametrize("model", ["glm-root", "typo-qwen"])
def test_fixed_leaf_helpers_return_routing_error_without_wrong_model_calls(
    depth, max_depth, helper, model
):
    batched = helper.endswith("_batched")
    args = "['task']" if batched else "'task'"
    root = MockLM(
        model_name="glm-root",
        responses=[
            "```repl\n"
            f"result = {helper}({args}, model={model!r})\n"
            + ("result = result[0]\n" if batched else "")
            + "assert 'Error' in result and 'qwen-leaf' in result\n"
            "answer['content'] = 'rejected correctly'\nanswer['ready'] = True\n```",
        ],
    )
    leaf = MockLM(model_name="qwen-leaf")
    models = {"glm-root": root, "qwen-leaf": leaf}
    with patch.object(
        rlm_module, "get_client", side_effect=lambda backend, kwargs: models[kwargs["model_name"]]
    ):
        parent = fixed_routing_rlm(depth=depth, max_depth=max_depth, max_iterations=1)
        try:
            result = parent.completion("task")
        finally:
            parent.close()

    assert result.response == "rejected correctly"
    assert root._call_count == 1
    assert leaf._call_count == 0


@pytest.mark.parametrize("explicit_models", [False, True])
def test_fixed_routing_full_root_child_leaf_path_all_four_helpers(explicit_models):
    leaf_kwarg = ", model='qwen-leaf'" if explicit_models else ""
    root_kwarg = ", model='glm-root'" if explicit_models else ""
    root_responses = iter(
        [
            "```repl\nanswer['content'] = rlm_query('child task'" + root_kwarg + ")\n"
            "answer['ready'] = True\n```",
            "```repl\n"
            f"a = llm_query('a'{leaf_kwarg})\n"
            f"b = llm_query_batched(['b', 'c']{leaf_kwarg})\n"
            f"c = rlm_query('d'{leaf_kwarg})\n"
            f"d = rlm_query_batched(['e', 'f']{leaf_kwarg})\n"
            "assert a == c == 'leaf answer'\n"
            "assert b == d == ['leaf answer', 'leaf answer']\n"
            "answer['content'] = 'child analysis'\nanswer['ready'] = True\n```",
        ]
    )
    clients = []
    root_prompts = []

    def root_response(prompt):
        root_prompts.append(prompt)
        return next(root_responses)

    def new_client(backend, kwargs):
        model = kwargs["model_name"]
        client = MockLM(
            model_name=model,
            response_fn=root_response if model == "glm-root" else lambda prompt: "leaf answer",
        )
        clients.append(client)
        return client

    with patch.object(rlm_module, "get_client", side_effect=new_client):
        parent = fixed_routing_rlm(max_depth=2, max_iterations=1)
        try:
            result = parent.completion("root context")
        finally:
            parent.close()

    assert result.response == "child analysis"
    assert sum(client._call_count for client in clients if client.model_name == "glm-root") == 2
    assert sum(client._call_count for client in clients if client.model_name == "qwen-leaf") == 6
    usage = result.usage_summary.model_usage_summaries
    assert usage["glm-root"].total_calls == 2
    assert usage["qwen-leaf"].total_calls == 6
    assert usage["glm-root"].total_input_tokens == usage["glm-root"].total_output_tokens == 20
    assert usage["qwen-leaf"].total_input_tokens == usage["qwen-leaf"].total_output_tokens == 60
    assert "llm_query_batched automatically use the leaf model 'qwen-leaf'" in str(root_prompts[1])


def test_fixed_terminal_failure_preserves_reported_usage_once():
    def fail(prompt):
        raise RuntimeError("provider failed after reporting billable usage")

    leaf = MockLM(model_name="qwen-leaf", response_fn=fail)
    with patch.object(rlm_module, "get_client", return_value=leaf):
        parent = fixed_routing_rlm(depth=1, max_depth=2)
        try:
            result = parent._subcall("task")
            recorded = parent._recursive_usage.model_usage_summaries
        finally:
            parent.close()

    assert result.response.startswith("Error:")
    assert result.error
    assert result.metadata["usage_status"] == "observed_partial"
    assert recorded["qwen-leaf"].total_calls == 1
    assert result.usage_summary.model_usage_summaries["qwen-leaf"].total_calls == 1


@pytest.mark.parametrize(
    "fixed,environment", [(True, "local"), (False, "local"), (True, "ipython")]
)
def test_failed_call_recording_is_mandatory_only_for_fixed_local_environments(fixed, environment):
    original = {"record_failed_calls": False, "helper_event_log": "/tmp/per-room-events.jsonl"}
    parent = RLM(
        backend_kwargs={"model_name": "glm-root"},
        other_backends=["openai"],
        other_backend_kwargs=[{"model_name": "qwen-leaf"}],
        fixed_model_routing=fixed,
        environment=environment,
        environment_kwargs=original,
    )
    try:
        assert parent.environment_kwargs["record_failed_calls"] is (
            fixed and environment == "local"
        )
        assert original["record_failed_calls"] is False
    finally:
        parent.close()


@pytest.mark.parametrize("limit", ["budget", "timeout"])
@pytest.mark.parametrize("depth", [0, 1])
def test_fixed_exhausted_subcalls_are_structured_and_never_dispatch(limit, depth):
    parent = fixed_routing_rlm(depth=depth, max_depth=2, max_budget=1, max_timeout=1)
    if limit == "budget":
        parent._cumulative_cost = 1
    else:
        parent._completion_start_time = time.perf_counter() - 2
    with patch.object(rlm_module, "get_client") as client:
        try:
            result = parent._subcall("task")
        finally:
            parent.close()
    client.assert_not_called()
    assert result.error
    assert result.metadata["usage_status"] == "unknown"
    assert result.usage_summary.model_usage_summaries == {}


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("budget_error", [False, True])
def test_fixed_child_exceptions_preserve_partial_usage_once(observed, budget_error):
    error = BudgetExceededError(1, 0.5) if budget_error else RuntimeError("child failed")
    usage = UsageSummary(model_usage_summaries={})
    if observed:
        usage = UsageSummary(
            model_usage_summaries={
                "glm-root": ModelUsageSummary(
                    total_calls=1, total_input_tokens=9, total_output_tokens=4
                )
            }
        )
        error.usage_summary = usage
    parent = fixed_routing_rlm(max_depth=2)
    with patch.object(RLM, "completion", side_effect=error):
        try:
            result = parent._subcall("task")
            folded = parent._recursive_usage
        finally:
            parent.close()
    assert result.error
    assert result.metadata["usage_status"] == ("observed_partial" if observed else "unknown")
    assert result.usage_summary == usage
    assert folded == usage


def test_fixed_terminal_slow_leaf_is_cancelled_without_sync_dispatch(monkeypatch):
    cancelled = Event()
    leaf = MockLM(model_name="qwen-leaf")

    async def slow(prompt):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    monkeypatch.setattr(leaf, "acompletion", slow)
    monkeypatch.setattr(
        leaf, "completion", Mock(side_effect=AssertionError("sync dispatch forbidden"))
    )
    monkeypatch.setattr("rlm.core.lm_handler.DEFAULT_WAVE_TIMEOUT", 0.03)
    monkeypatch.setattr("rlm.utils.global_gate.get_gate", lambda: None)
    with patch.object(rlm_module, "get_client", return_value=leaf):
        parent = fixed_routing_rlm(depth=1, max_depth=2)
        try:
            result = parent._subcall("task")
        finally:
            parent.close()
    assert cancelled.is_set()
    leaf.completion.assert_not_called()
    assert result.error and "wave exceeded" in result.error
    assert result.metadata["usage_status"] == "unknown"


def test_fixed_terminal_socket_timeout_is_aligned_and_failed_completion_retained(monkeypatch):
    from rlm.core.comms_utils import DEFAULT_WAVE_TIMEOUT, WAVE_TIMEOUT_SLACK, LMResponse
    from rlm.core.types import RLMChatCompletion

    usage = UsageSummary(
        model_usage_summaries={
            "qwen-leaf": ModelUsageSummary(
                total_calls=1, total_input_tokens=9, total_output_tokens=4
            )
        }
    )
    failed = RLMChatCompletion(
        root_model="qwen-leaf",
        prompt="task",
        response="",
        error="provider failed",
        usage_summary=usage,
        execution_time=0.2,
        metadata={"usage_status": "observed_partial"},
    )
    send = Mock(return_value=LMResponse(error=failed.error, chat_completion=failed))
    monkeypatch.setattr(rlm_module, "send_lm_request", send, raising=False)
    with patch.object(rlm_module, "get_client", return_value=MockLM(model_name="qwen-leaf")):
        parent = fixed_routing_rlm(depth=1, max_depth=2)
        try:
            result = parent._subcall("task")
            folded = parent._recursive_usage
        finally:
            parent.close()
    assert send.call_args.kwargs["timeout"] == int(DEFAULT_WAVE_TIMEOUT + WAVE_TIMEOUT_SLACK)
    assert send.call_args.args[1].depth == 0
    assert send.call_args.args[1].model == "qwen-leaf"
    assert result.error == "provider failed"
    assert result.usage_summary == folded == usage


def test_fixed_terminal_usage_collection_failure_remains_structured_and_unknown(monkeypatch):
    leaf = MockLM(model_name="qwen-leaf")
    monkeypatch.setattr(
        leaf, "get_usage_summary", Mock(side_effect=RuntimeError("usage unavailable"))
    )
    with patch.object(rlm_module, "get_client", return_value=leaf):
        parent = fixed_routing_rlm(depth=1, max_depth=2)
        try:
            result = parent._subcall("task")
        finally:
            parent.close()
    assert result.error
    assert result.metadata["usage_status"] == "unknown"
    assert result.metadata["usage_collection_error"] == "usage unavailable"
    assert result.usage_summary.model_usage_summaries == {}


class TestSubcallFinalizationPropagation:
    @pytest.mark.parametrize("fabricate_final_answer", [False, True])
    def test_iteration_exhaustion_respects_parent_setting_recursively(self, fabricate_final_answer):
        mock_lm = create_mock_lm(
            [
                "```repl\nanswer['content'] = rlm_query(context)\nanswer['ready'] = True\n```",
                "```repl\nprint('Still working')\n```",
                "Synthesized fallback",
            ]
        )
        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            parent = RLM(
                backend_kwargs={"model_name": "mock-model"},
                max_depth=3,
                max_iterations=1,
                fabricate_final_answer=fabricate_final_answer,
                recover_stub=False,
            )
            try:
                result = parent._subcall("Analyze this input.")
            finally:
                parent.close()

        assert result.response == ("Synthesized fallback" if fabricate_final_answer else "")
        assert mock_lm.completion.call_count == (3 if fabricate_final_answer else 2)

    @pytest.mark.parametrize("recover_stub", [False, True])
    def test_stub_recovery_respects_parent_setting_recursively(self, recover_stub):
        report = "# Analysis\n" + "The generated finding is supported.\n" * 200
        mock_lm = create_mock_lm(
            [
                "```repl\nanswer['content'] = rlm_query(context)\nanswer['ready'] = True\n```",
                f"```repl\nreport = {report!r}\nanswer['content'] = 'Saved'\n"
                "answer['ready'] = True\n```",
            ]
        )
        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            parent = RLM(
                backend_kwargs={"model_name": "mock-model"},
                max_depth=3,
                max_iterations=1,
                fabricate_final_answer=False,
                recover_stub=recover_stub,
            )
            try:
                result = parent._subcall("Analyze this input.")
            finally:
                parent.close()

        assert result.response == (report if recover_stub else "Saved")
        assert mock_lm.completion.call_count == 2

    def test_disabled_recovery_does_not_replace_child_analysis_with_input(self):
        source = "# Input document\n" + "Material to analyze.\n" * 300
        mock_lm = create_mock_lm([final("The analysis is complete.")])
        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            parent = RLM(
                backend_kwargs={"model_name": "mock-model"},
                deliverable_slots=["report.md"],
                max_depth=2,
                max_iterations=1,
                fabricate_final_answer=False,
                recover_stub=False,
            )
            try:
                result = parent._subcall(source)
            finally:
                parent.close()

        assert result.response == "The analysis is complete."
        assert result.deliverables is None
        assert mock_lm.completion.call_count == 1


class TestSubcallAnswerProtocol:
    @pytest.mark.parametrize(
        "custom_system_prompt",
        [None, RLM_SYSTEM_PROMPT_SLOTS + "\nAdditional root instructions."],
        ids=["default-slot-prompt", "custom-slot-prompt"],
    )
    def test_slot_parent_child_prompt_matches_content_protocol(self, custom_system_prompt):
        mock_lm = create_mock_lm([final("Child analysis")])
        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            parent = RLM(
                backend_kwargs={"model_name": "mock-model"},
                deliverable_slots=["report.md"],
                custom_system_prompt=custom_system_prompt,
                max_depth=2,
                max_iterations=1,
                fabricate_final_answer=False,
                recover_stub=False,
            )
            try:
                result = parent._subcall("Analyze this input.")
            finally:
                parent.close()

        assert result.response == "Child analysis"
        assert result.deliverables is None
        system_message = mock_lm.completion.call_args.args[0][0]["content"]
        assert system_message.startswith(RLM_SYSTEM_PROMPT.format(custom_tools_section=""))

    def test_content_parent_preserves_custom_system_prompt(self):
        custom_system_prompt = RLM_SYSTEM_PROMPT + "\nAdditional content instructions."
        mock_lm = create_mock_lm([final("Child analysis")])
        with patch.object(rlm_module, "get_client", return_value=mock_lm):
            parent = RLM(
                backend_kwargs={"model_name": "mock-model"},
                custom_system_prompt=custom_system_prompt,
                max_depth=2,
                max_iterations=1,
                fabricate_final_answer=False,
                recover_stub=False,
            )
            try:
                result = parent._subcall("Analyze this input.")
            finally:
                parent.close()

        assert result.response == "Child analysis"
        assert result.deliverables is None
        system_message = mock_lm.completion.call_args.args[0][0]["content"]
        assert system_message.startswith(custom_system_prompt.format(custom_tools_section=""))


class TestSubcallTimeoutPropagation:
    """Tests for max_timeout propagation to child RLM."""

    def test_child_receives_remaining_timeout(self):
        """When parent has max_timeout=60 and 10s have elapsed, child should get max_timeout approx 50."""
        captured_child_params = {}

        # Create a fake child RLM class to capture initialization params
        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                # Capture the kwargs before calling parent
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            # Create parent RLM with max_timeout
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,  # Need depth > 1 to allow child spawning
                max_timeout=60.0,
            )

            # Simulate that 10 seconds have elapsed since completion started
            parent._completion_start_time = time.perf_counter() - 10.0

            # Patch RLM class to capture child creation
            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall which should spawn a child RLM
                parent._subcall("test prompt")

            # Verify child received remaining timeout (approximately 50 seconds)
            assert "max_timeout" in captured_child_params
            remaining = captured_child_params["max_timeout"]
            # Allow some tolerance for test execution time
            assert 45.0 < remaining < 55.0, f"Expected ~50s remaining, got {remaining}"

            parent.close()

    def test_child_receives_none_timeout_when_parent_has_none(self):
        """When parent has no max_timeout, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=None,  # No timeout
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_timeout") is None

            parent.close()

    def test_subcall_returns_error_when_timeout_exhausted(self):
        """When timeout is already exhausted, _subcall should return error message."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_timeout=10.0,
            )

            # Simulate that more time has elapsed than the timeout
            parent._completion_start_time = time.perf_counter() - 15.0

            result = parent._subcall("test prompt")

            assert "Error: Timeout exhausted" in result.response

            parent.close()


class TestSubcallTokensPropagation:
    """Tests for max_tokens propagation to child RLM."""

    def test_child_receives_max_tokens(self):
        """Child RLM should get same max_tokens as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=50000,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") == 50000

            parent.close()

    def test_child_receives_none_tokens_when_parent_has_none(self):
        """When parent has no max_tokens, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_tokens=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_tokens") is None

            parent.close()


class TestSubcallErrorsPropagation:
    """Tests for max_errors propagation to child RLM."""

    def test_child_receives_max_errors(self):
        """Child RLM should get same max_errors as parent."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=5,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") == 5

            parent.close()

    def test_child_receives_none_errors_when_parent_has_none(self):
        """When parent has no max_errors, child should also have None."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
                max_errors=None,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt")

            assert captured_child_params.get("max_errors") is None

            parent.close()


class TestSubcallModelOverride:
    """Tests for model= parameter override in _subcall."""

    def test_model_override_sets_child_backend_kwargs(self):
        """When llm_query(prompt, model='test-model') is called, child's backend_kwargs should have model_name='test-model'."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall with model override
                parent._subcall("test prompt", model="override-model")

            # Verify child received overridden model in backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            # Original kwargs should be preserved
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()

    def test_model_override_does_not_mutate_parent_kwargs(self):
        """Model override should not mutate parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            original_model = parent.backend_kwargs["model_name"]

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Parent's backend_kwargs should be unchanged
            assert parent.backend_kwargs["model_name"] == original_model

            parent.close()

    def test_no_model_override_uses_parent_kwargs(self):
        """When no model override is provided, child uses parent's backend_kwargs."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                max_depth=3,
            )

            with patch.object(rlm_module, "RLM", CapturingRLM):
                # Call _subcall without model override
                parent._subcall("test prompt")

            # Child should use parent's backend_kwargs
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "parent-model"

            parent.close()


class TestSubcallModelOverrideAtLeafDepth:
    """Tests for model override at max_depth (leaf LM completion)."""

    def test_model_override_at_leaf_depth_uses_overridden_model(self):
        """When at max_depth, the leaf LM completion should use the overridden model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm(["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall with model override - should trigger leaf LM completion
            result = parent._subcall("test prompt", model="leaf-override-model")

            # Verify get_client was called with overridden model in backend_kwargs
            # The call should be: get_client("openai", {"model_name": "leaf-override-model"})
            call_args = mock_get_client.call_args_list
            # Find the call that has the overridden model
            found_override_call = False
            for call in call_args:
                args, kwargs = call
                if len(args) >= 2:
                    backend_kwargs = args[1]
                    if (
                        isinstance(backend_kwargs, dict)
                        and backend_kwargs.get("model_name") == "leaf-override-model"
                    ):
                        found_override_call = True
                        break

            assert found_override_call, (
                f"Expected get_client to be called with model_name='leaf-override-model', got calls: {call_args}"
            )
            assert result.response == "leaf response"

            parent.close()

    def test_leaf_depth_without_model_override_uses_parent_model(self):
        """When at max_depth without model override, uses parent's model."""
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")] * 2 + ["leaf response"])
            mock_get_client.return_value = mock_lm

            # Parent at depth 1, max_depth 2 means next depth (2) will be at max_depth
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model"},
                depth=1,
                max_depth=2,
            )

            # Call _subcall without model override
            parent._subcall("test prompt")

            # Verify get_client was called with parent's model
            # The last call should use the parent's backend_kwargs
            call_args = mock_get_client.call_args_list
            # Check the most recent call (for leaf completion)
            last_call = call_args[-1]
            args, _ = last_call
            if len(args) >= 2:
                backend_kwargs = args[1]
                assert backend_kwargs.get("model_name") == "parent-model"

            parent.close()


class TestSubcallCombinedParameters:
    """Tests for combined parameter propagation."""

    def test_all_parameters_propagate_together(self):
        """All parameters (timeout, tokens, errors, model) should propagate correctly together."""
        captured_child_params = {}

        original_rlm_class = rlm_module.RLM

        class CapturingRLM(original_rlm_class):
            def __init__(self, *args, **kwargs):
                captured_child_params.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_lm = create_mock_lm([final("answer")])
            mock_get_client.return_value = mock_lm

            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "parent-model", "api_key": "test-key"},
                max_depth=3,
                max_timeout=120.0,
                max_tokens=100000,
                max_errors=10,
            )

            # Simulate 30 seconds elapsed
            parent._completion_start_time = time.perf_counter() - 30.0

            with patch.object(rlm_module, "RLM", CapturingRLM):
                parent._subcall("test prompt", model="override-model")

            # Verify all parameters
            assert captured_child_params.get("max_tokens") == 100000
            assert captured_child_params.get("max_errors") == 10

            # Remaining timeout should be around 90 seconds
            remaining_timeout = captured_child_params.get("max_timeout")
            assert 85.0 < remaining_timeout < 95.0

            # Model should be overridden
            child_backend_kwargs = captured_child_params.get("backend_kwargs", {})
            assert child_backend_kwargs.get("model_name") == "override-model"
            assert child_backend_kwargs.get("api_key") == "test-key"

            parent.close()
