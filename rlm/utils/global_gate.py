"""Deployment-wide sub-call concurrency gate, shared by every RLM harness.

Why this exists (2026-08-05, L25): sub-call waves are fanned out per worker
process (``LMHandler.batch_max_concurrent``, default 16) with NO coordination
across workers, so 8 RL env workers could put 19-32+ simultaneous requests on
the frozen-sub deployment in same-second bursts. Measured effect on a
6-replica endpoint: p99 inference ~955s (vs p50 ~55s), ~9% client
cancellations, and timeout-retries duplicating server work — a density
amplification loop. A GLOBAL cap keeps deployment pressure bounded while a
lone wave still gets full width — strictly better than shrinking per-worker
fan-out, which slows healthy waves to protect against rare pileups.

Mechanism: a directory of ``limit`` lock files; holding slot k = holding an
``flock`` on file k. flock is per-fd, works across processes on one box, and —
the reason it beats counters or a token server — is released by the kernel
when the holder dies, so crashed workers cannot leak slots.

Config (read at first use, enforced in ``rlm.core.lm_handler`` around every
sub-call HTTP attempt, SDK retries included since the slot wraps the call):

* ``RLM_SUBCALL_GATE_DIR``   — enable by pointing at a directory (created if
  missing). Unset = gate off (single-process eval behavior unchanged).
* ``RLM_SUBCALL_GATE_LIMIT`` — slot count, default 16.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import random
import time

_ACQUIRE_TIMEOUT_S = 1800.0  # a saturated-but-moving gate clears long before this


class GlobalSubcallGate:
    def __init__(self, dir_path: str, limit: int) -> None:
        self.dir = dir_path
        self.limit = max(1, int(limit))
        os.makedirs(dir_path, exist_ok=True)
        # Per-event-loop admission semaphore for ``async_slot`` (see there).
        self._local_sems: dict[int, asyncio.Semaphore] = {}

    def _local_sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._local_sems.get(id(loop))
        if sem is None:
            sem = asyncio.Semaphore(self.limit)
            self._local_sems[id(loop)] = sem
        return sem

    def _try_acquire(self) -> int | None:
        """One non-blocking pass over the slots; returns a held fd or None."""
        indices = list(range(self.limit))
        random.shuffle(indices)  # fairness + no thundering herd on slot 0
        for i in indices:
            fd = os.open(
                os.path.join(self.dir, f"slot_{i:03d}.lock"), os.O_CREAT | os.O_RDWR, 0o644
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            return fd
        return None

    @staticmethod
    def _release(fd: int) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _timeout(self) -> TimeoutError:
        return TimeoutError(
            f"global sub-call gate: no slot free within "
            f"{_ACQUIRE_TIMEOUT_S:.0f}s ({self.limit} slots at {self.dir})"
        )

    @contextlib.contextmanager
    def slot(self):
        """Hold one deployment-wide slot; blocks (jittered spin) until free.

        Sync form for plain-thread callers. NEVER run this inside an event
        loop's default executor (``asyncio.to_thread``): see ``async_slot``.
        """
        deadline = time.monotonic() + _ACQUIRE_TIMEOUT_S
        while True:
            fd = self._try_acquire()
            if fd is not None:
                try:
                    yield
                finally:
                    self._release(fd)
                return
            if time.monotonic() > deadline:
                raise self._timeout()
            time.sleep(random.uniform(0.05, 0.30))  # doubles as launch jitter

    @contextlib.asynccontextmanager
    async def async_slot(self):
        """``slot`` for coroutines: thread-free, waits with ``asyncio.sleep``.

        Why not ``asyncio.to_thread(slot.__enter__)`` (the 2026-09-05 Loops
        wedge #1): the loop's default ``ThreadPoolExecutor`` has only
        ``min(32, cpu+4)`` workers (20 on a 16-CPU driver pod) and is ALSO what
        ``loop.getaddrinfo`` — every new httpx connection's DNS lookup — runs
        on. Once the gate saturated, 20 acquirers sat in the executor spinning
        on flock, the 64 slot holders could not resolve the sub-model's host
        to open their connections, nothing went on the wire, and each holder
        only failed at the 300 s connect timeout.

        Why the local semaphore (wedge #2, same day): a first thread-free
        version let EVERY waiting coroutine rescan all ``limit`` lock files
        each 50–300 ms. With 24 rollouts × 16-wide waves that was ~300
        acquirers × 64 open/flock/close syscalls per pass — the handler loop
        was 100% busy spinning (py-spy: every sample in ``_try_acquire``) and
        the holders' connects starved again. The semaphore admits at most
        ``limit`` coroutines per loop to the flock scan; the rest wait on a
        plain asyncio primitive that costs nothing until a holder releases.
        With a single gate-using process (the Loops driver) a coroutine that
        passes the semaphore finds a free flock on its first pass; the scan +
        sleep only matters when OTHER processes (PRIME env workers) hold slots.
        """
        async with self._local_sem():
            deadline = time.monotonic() + _ACQUIRE_TIMEOUT_S
            while True:
                fd = self._try_acquire()
                if fd is not None:
                    try:
                        yield
                    finally:
                        self._release(fd)
                    return
                if time.monotonic() > deadline:
                    raise self._timeout()
                await asyncio.sleep(random.uniform(0.1, 0.5))


_GATE: GlobalSubcallGate | None = None
_GATE_CHECKED = False


def get_gate() -> GlobalSubcallGate | None:
    """The process-wide gate, or None when RLM_SUBCALL_GATE_DIR is unset."""
    global _GATE, _GATE_CHECKED
    if not _GATE_CHECKED:
        _GATE_CHECKED = True
        d = os.environ.get("RLM_SUBCALL_GATE_DIR")
        if d:
            _GATE = GlobalSubcallGate(d, int(os.environ.get("RLM_SUBCALL_GATE_LIMIT", "16")))
    return _GATE
