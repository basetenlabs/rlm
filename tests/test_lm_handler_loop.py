"""Batched waves must share ONE event loop per LMHandler.

Regression for the 2026-09-05 finding: ``_handle_batched`` ran every wave under
``asyncio.run`` (a fresh loop each time) while ``client.acompletion`` reused a
shared ``AsyncOpenAI``/httpx pool whose connections belonged to the previous,
closed loop. ~70% of batched sub-calls then failed inside the SDK with
``RuntimeError`` and were silently retried (edge-side 499s, wasted backend
work); the residual hard failures surfaced as ``Error: llm() call failed -
Connection error`` in REPL output and deliverables.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from rlm.clients.base_lm import BaseLM
from rlm.core.comms_utils import send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from rlm.core.types import ModelUsageSummary, UsageSummary


class _LoopRecordingLM(BaseLM):
    """Fake client that records which event loop each acompletion ran on."""

    def __init__(self) -> None:
        super().__init__(model_name="fake")
        self.loops: list[int] = []
        self.calls = 0

    def completion(self, prompt, model=None):  # noqa: D401 - BaseLM contract
        self.calls += 1
        return f"sync:{prompt}"

    async def acompletion(self, prompt, model=None):
        self.calls += 1
        self.loops.append(id(asyncio.get_running_loop()))
        await asyncio.sleep(0)
        return f"async:{prompt}"

    def get_last_usage(self):
        return ModelUsageSummary(total_calls=1, total_input_tokens=1, total_output_tokens=1)

    def get_usage_summary(self):
        return UsageSummary(model_usage_summaries={})


def test_batched_waves_share_one_event_loop():
    client = _LoopRecordingLM()
    handler = LMHandler(client, batch_max_concurrent=4)
    addr = handler.start()
    try:
        first = send_lm_request_batched(addr, ["a", "b", "c"], depth=1)
        second = send_lm_request_batched(addr, ["d", "e"], depth=1)
    finally:
        handler.stop()

    assert [r.success for r in first + second] == [True] * 5
    assert [r.chat_completion.response for r in first] == ["async:a", "async:b", "async:c"]
    assert len(client.loops) == 5
    # The whole point: one loop across waves, not one loop per wave.
    assert len(set(client.loops)) == 1
    # And the loop is the handler's own, not a caller thread's.
    assert handler._loop is None  # stop() tore it down


def test_concurrent_waves_share_one_handler_wide_limit():
    class ConcurrencyRecordingLM(_LoopRecordingLM):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0

        async def acompletion(self, prompt, model=None):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0.05)
                return f"async:{prompt}"
            finally:
                self.active -= 1

    client = ConcurrencyRecordingLM()
    handler = LMHandler(client, batch_max_concurrent=2)
    addr = handler.start()
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    send_lm_request_batched,
                    addr,
                    [f"{wave}:{idx}" for idx in range(4)],
                    depth=1,
                )
                for wave in range(3)
            ]
            results = [future.result() for future in futures]
    finally:
        handler.stop()

    assert all(result.success for wave in results for result in wave)
    assert client.peak == 2
