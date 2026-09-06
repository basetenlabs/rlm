"""Cancellation-safe asynchronous acquisition of the shared process gate."""

import asyncio

import pytest

from rlm.utils.global_gate import GlobalSubcallGate


def test_cancel_waiting_gate_does_not_acquire_later(tmp_path, monkeypatch):
    gate = GlobalSubcallGate(str(tmp_path / "gate"), 1)

    def no_background_acquisition(*args, **kwargs):
        raise AssertionError("gate acquisition must not outlive its cancelled coroutine")

    monkeypatch.setattr(asyncio, "to_thread", no_background_acquisition)

    async def exercise():
        acquired = []

        async def waiter():
            async with gate.async_slot():
                acquired.append(True)

        with gate.slot():
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        await asyncio.sleep(0.02)
        assert acquired == []
        async with asyncio.timeout(0.2):
            async with gate.async_slot():
                pass

    asyncio.run(exercise())


def test_cancel_acquired_gate_releases_for_sync_and_async_callers(tmp_path):
    gate = GlobalSubcallGate(str(tmp_path / "gate"), 1)

    async def exercise():
        acquired = asyncio.Event()

        async def holder():
            async with gate.async_slot():
                acquired.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(acquired.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(0.2):
            async with gate.async_slot():
                pass
        with gate.slot():
            pass

    asyncio.run(exercise())


def test_async_gate_acquire_deadline(tmp_path, monkeypatch):
    gate = GlobalSubcallGate(str(tmp_path / "gate"), 1)
    monkeypatch.setattr("rlm.utils.global_gate._ACQUIRE_TIMEOUT_S", 0.01)

    async def exercise():
        with gate.slot():
            with pytest.raises(TimeoutError, match="global sub-call gate"):
                async with gate.async_slot():
                    raise AssertionError("occupied gate must not be acquired")

    asyncio.run(exercise())
