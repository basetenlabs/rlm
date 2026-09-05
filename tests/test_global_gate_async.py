"""The sub-call gate must be acquirable from coroutines WITHOUT threads.

Regression for the 2026-09-05 Loops wedge: ``_handle_batched`` acquired slots
with ``asyncio.to_thread(slot.__enter__)``. The loop's default executor
(``min(32, cpu+4)`` workers) filled with acquirers spinning on flock, which
starved ``loop.getaddrinfo`` — the DNS step of every new httpx connection —
so the slot HOLDERS could never open their sub-model connections. The gate
then only drained at the 300 s connect timeout, in bursts of exactly ``limit``
ConnectTimeouts, and every wave came back all-errors.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from rlm.utils.global_gate import GlobalSubcallGate


def _run(coro):
    return asyncio.run(coro)


def test_async_slot_respects_limit_without_threads(tmp_path):
    gate = GlobalSubcallGate(str(tmp_path), limit=3)
    peak = 0
    held = 0
    lock = asyncio.Lock()
    before = threading.active_count()

    async def worker():
        nonlocal peak, held
        async with gate.async_slot():
            async with lock:
                held += 1
                peak = max(peak, held)
            await asyncio.sleep(0.05)
            async with lock:
                held -= 1

    async def main():
        await asyncio.gather(*(worker() for _ in range(12)))
        # No executor thread may have been spawned by the gate itself.
        return threading.active_count()

    after = _run(main())
    assert peak == 3
    assert held == 0
    assert after <= before


def test_async_slot_does_not_block_the_loop(tmp_path):
    """While acquirers wait for a slot, unrelated coroutines keep running."""
    gate = GlobalSubcallGate(str(tmp_path), limit=1)
    ticks = 0

    async def ticker(stop: asyncio.Event):
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    async def holder(release: asyncio.Event):
        async with gate.async_slot():
            await release.wait()

    async def main():
        stop = asyncio.Event()
        release = asyncio.Event()
        t = asyncio.create_task(ticker(stop))
        h = asyncio.create_task(holder(release))
        await asyncio.sleep(0.02)  # holder owns the only slot
        waiter = asyncio.create_task(holder(release))
        await asyncio.sleep(0.2)  # waiter spins for the slot
        assert ticks > 5, "gate acquisition blocked the event loop"
        release.set()
        await asyncio.gather(h, waiter)
        stop.set()
        await t

    _run(main())


def test_async_slot_releases_on_exception(tmp_path):
    gate = GlobalSubcallGate(str(tmp_path), limit=1)

    async def main():
        with pytest.raises(RuntimeError):
            async with gate.async_slot():
                raise RuntimeError("boom")
        # Slot must be free again: acquire immediately.
        fd = gate._try_acquire()
        assert fd is not None
        gate._release(fd)

    _run(main())


def test_async_slot_waiters_do_not_spin_on_lock_files(tmp_path, monkeypatch):
    """Waiters beyond ``limit`` must not rescan the lock files (2026-09-05 wedge #2).

    300 acquirers rescanning 64 files every 50-300 ms saturated the handler
    loop; only ``limit`` coroutines may ever touch flock at once.
    """
    gate = GlobalSubcallGate(str(tmp_path), limit=4)
    calls = 0
    orig = gate._try_acquire

    def counting():
        nonlocal calls
        calls += 1
        return orig()

    monkeypatch.setattr(gate, "_try_acquire", counting)

    async def worker():
        async with gate.async_slot():
            await asyncio.sleep(0.02)

    async def main():
        await asyncio.gather(*(worker() for _ in range(200)))

    _run(main())
    # One successful scan per acquisition (all slots are local, so a coroutine
    # admitted by the semaphore finds a free lock immediately); allow a little
    # slack for the rare race where a release has not yet reached the fd.
    assert calls <= 200 + 40, calls
