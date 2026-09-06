"""Tests for LMHandler using MockLM (no real LM required)."""

import asyncio
from threading import Event

import pytest

import rlm.core.lm_handler as handler_module
from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.lm_handler import LMHandler, LMRequestHandler
from rlm.utils.global_gate import GlobalSubcallGate
from tests.mock_lm import MockLM


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("model", [None, "qwen-leaf"])
def test_fixed_routing_selects_leaf_at_every_environment_depth(depth, model):
    root = MockLM(model_name="glm-root")
    leaf = MockLM(model_name="qwen-leaf")
    handler = LMHandler(root, other_backend_client=leaf, fixed_model_routing=True)
    handler.register_client(leaf.model_name, leaf)

    assert handler.get_client(model, depth=depth) is leaf
    assert handler.get_client("glm-root", depth=0) is root


def test_fixed_routing_keeps_root_sampling_when_root_and_leaf_names_match():
    root = MockLM(model_name="same-model")
    leaf = MockLM(model_name="same-model")
    handler = LMHandler(root, other_backend_client=leaf, fixed_model_routing=True)
    handler.register_client(leaf.model_name, leaf)

    assert handler.get_client("same-model", depth=0) is root
    assert handler.get_client("same-model", depth=2) is leaf


def test_fixed_routing_requires_an_explicit_leaf_backend():
    with pytest.raises(ValueError, match="leaf backend"):
        LMHandler(MockLM(), fixed_model_routing=True)


@pytest.mark.parametrize("model", ["glm-root", "misspelled-qwen", ""])
@pytest.mark.parametrize("batched", [False, True])
def test_fixed_routing_rejects_invalid_flat_overrides_without_model_calls(model, batched):
    root = MockLM(model_name="glm-root")
    leaf = MockLM(model_name="qwen-leaf")
    with LMHandler(root, other_backend_client=leaf, fixed_model_routing=True) as handler:
        if batched:
            responses = send_lm_request_batched(handler.address, ["a", "b"], model=model, depth=2)
        else:
            responses = [
                send_lm_request(handler.address, LMRequest(prompt="a", model=model, depth=2))
            ]

    assert all(not response.success for response in responses)
    assert all("qwen-leaf" in response.error and "flat" in response.error for response in responses)
    assert root._call_count == leaf._call_count == 0


def test_legacy_depth_and_explicit_routing_remain_unchanged():
    root = MockLM(model_name="glm-root")
    leaf = MockLM(model_name="qwen-leaf")
    handler = LMHandler(root, other_backend_client=leaf)
    handler.register_client(leaf.model_name, leaf)

    assert handler.get_client(depth=1) is leaf
    assert handler.get_client(depth=2) is root
    assert handler.get_client("qwen-leaf", depth=2) is leaf
    assert handler.get_client("unknown", depth=2) is root


def test_lm_handler_single_request():
    """Single prompt request returns success and echo-style content."""
    mock = MockLM(responses=["hello back"])
    with LMHandler(client=mock) as handler:
        request = LMRequest(prompt="hello")
        response = send_lm_request(handler.address, request)
    assert response.success
    assert response.chat_completion is not None
    assert response.chat_completion.response == "hello back"


def test_lm_handler_batched_request():
    """Batched prompts return one response per prompt in order."""
    responses = [f"r{i}" for i in range(5)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = [f"prompt-{i}" for i in range(5)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == 5
    for i, resp in enumerate(result):
        assert resp.success, resp.error
        assert resp.chat_completion is not None
        assert resp.chat_completion.response == f"r{i}"


def test_lm_handler_batched_partial_failure():
    """One failing call returns an error for that slot; the rest still succeed."""

    def response_fn(prompt):
        if prompt == "prompt-1":
            raise RuntimeError("boom")
        return f"ok {prompt}"

    mock = MockLM(response_fn=response_fn)
    with LMHandler(client=mock, batch_max_concurrent=3) as handler:
        prompts = ["prompt-0", "prompt-1", "prompt-2"]
        result = send_lm_request_batched(handler.address, prompts)

    assert len(result) == 3
    assert result[0].success
    assert result[0].chat_completion.response == "ok prompt-0"
    assert not result[1].success
    assert "boom" in result[1].error
    assert result[2].success
    assert result[2].chat_completion.response == "ok prompt-2"


def test_lm_handler_batched_many_prompts_semaphore_cap():
    """Many prompts complete successfully with semaphore limiting concurrency."""
    # 50 prompts, max 4 concurrent: should still all complete
    count = 50
    responses = [f"resp-{i}" for i in range(count)]
    mock = MockLM(responses=responses)
    with LMHandler(client=mock, batch_max_concurrent=4) as handler:
        prompts = [f"p-{i}" for i in range(count)]
        result = send_lm_request_batched(handler.address, prompts)
    assert len(result) == count
    for i, resp in enumerate(result):
        assert resp.success, (i, resp.error)
        assert resp.chat_completion.response == f"resp-{i}"


def test_sequential_batches_reuse_one_asyncio_event_loop():
    """A shared AsyncOpenAI client must never cross short-lived event loops."""

    class LoopRecordingLM(MockLM):
        def __init__(self):
            super().__init__()
            self.loop_ids: set[int] = set()

        async def acompletion(self, prompt):
            self.loop_ids.add(id(asyncio.get_running_loop()))
            return self.completion(prompt)

    mock = LoopRecordingLM()
    with LMHandler(client=mock, batch_max_concurrent=2) as handler:
        first = send_lm_request_batched(handler.address, ["a", "b"])
        second = send_lm_request_batched(handler.address, ["c", "d"])

    assert all(response.success for response in first + second)
    assert len(mock.loop_ids) == 1


class CancellableLeaf(MockLM):
    def __init__(self):
        super().__init__(model_name="qwen-leaf")
        self.cancelled = Event()

    def completion(self, prompt):
        raise AssertionError("strict flat requests must use the asynchronous client")

    async def acompletion(self, prompt):
        if prompt == "hang":
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        await asyncio.sleep(0.02)
        return MockLM.completion(self, prompt)


def test_strict_slow_single_uses_cancellable_executor(monkeypatch):
    monkeypatch.setattr(handler_module, "DEFAULT_WAVE_TIMEOUT", 0.2)
    leaf = CancellableLeaf()
    with LMHandler(
        MockLM(model_name="glm-root"), other_backend_client=leaf, fixed_model_routing=True
    ) as handler:
        response = LMRequestHandler._handle_single(
            object.__new__(LMRequestHandler), LMRequest(prompt="slow", depth=2), handler
        )

    assert response.success
    assert response.chat_completion.response == "Mock response to: slow"
    assert response.chat_completion.root_model == "qwen-leaf"
    assert leaf._call_count == 1


def test_strict_hung_single_cancels_and_releases_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(handler_module, "DEFAULT_WAVE_TIMEOUT", 0.15)
    gate = GlobalSubcallGate(str(tmp_path / "gate"), 1)
    monkeypatch.setattr("rlm.utils.global_gate.get_gate", lambda: gate)
    leaf = CancellableLeaf()
    with LMHandler(
        MockLM(model_name="glm-root"), other_backend_client=leaf, fixed_model_routing=True
    ) as handler:
        request_handler = object.__new__(LMRequestHandler)
        failed = request_handler._handle_single(LMRequest(prompt="hang", depth=2), handler)
        recovered = request_handler._handle_single(LMRequest(prompt="next", depth=2), handler)

    assert not failed.success
    assert failed.error.startswith("llm() call failed - ")
    assert failed.chat_completion.error == failed.error
    assert failed.chat_completion.metadata["usage_status"] == "unknown"
    assert leaf.cancelled.is_set()
    assert recovered.success
    assert recovered.chat_completion.response == "Mock response to: next"
    assert leaf._call_count == 1


def test_strict_wave_timeout_keeps_completed_results_and_usage(monkeypatch):
    monkeypatch.setattr(handler_module, "DEFAULT_WAVE_TIMEOUT", 0.15)
    leaf = CancellableLeaf()
    with LMHandler(
        MockLM(model_name="glm-root"), other_backend_client=leaf, fixed_model_routing=True
    ) as handler:
        response = LMRequestHandler._handle_batched(
            object(), LMRequest(prompts=["fast", "hang"], depth=2), handler
        )
        actual_usage = handler.get_usage_summary().model_usage_summaries["qwen-leaf"]

    first, second = response.chat_completions
    assert first.error is None
    assert first.response == "Mock response to: fast"
    assert second.error is not None
    assert leaf.cancelled.is_set()
    recorded = [
        call.usage_summary.model_usage_summaries.get("qwen-leaf")
        for call in response.chat_completions
    ]
    assert sum(usage.total_calls for usage in recorded if usage) == actual_usage.total_calls == 1
    assert (
        sum(usage.total_input_tokens for usage in recorded if usage)
        == actual_usage.total_input_tokens
        == 10
    )
    assert (
        sum(usage.total_output_tokens for usage in recorded if usage)
        == actual_usage.total_output_tokens
        == 10
    )


def test_strict_all_failed_wave_retains_reported_usage(monkeypatch):
    class BillableFailure(MockLM):
        async def acompletion(self, prompt):
            MockLM.completion(self, prompt)
            raise RuntimeError("failure after recorded usage")

    leaf = BillableFailure(model_name="qwen-leaf")
    with LMHandler(
        MockLM(model_name="glm-root"), other_backend_client=leaf, fixed_model_routing=True
    ) as handler:
        response = LMRequestHandler._handle_batched(
            object(), LMRequest(prompts=["a", "b"], depth=2), handler
        )

    assert all(call.error for call in response.chat_completions)
    assert response.chat_completions[0].metadata["usage_status"] == "observed_partial"
    assert response.chat_completions[1].metadata["usage_status"] == "unknown"
    recorded = [
        call.usage_summary.model_usage_summaries.get("qwen-leaf")
        for call in response.chat_completions
    ]
    assert sum(usage.total_calls for usage in recorded if usage) == 2
    assert sum(usage.total_input_tokens for usage in recorded if usage) == 20
