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

    @contextlib.contextmanager
    def slot(self):
        """Hold one deployment-wide slot; blocks (jittered spin) until free."""
        deadline = time.monotonic() + _ACQUIRE_TIMEOUT_S
        indices = list(range(self.limit))
        while True:
            random.shuffle(indices)  # fairness + no thundering herd on slot 0
            for i in indices:
                fd = os.open(os.path.join(self.dir, f"slot_{i:03d}.lock"),
                             os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                try:
                    yield
                finally:
                    with contextlib.suppress(OSError):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"global sub-call gate: no slot free within "
                    f"{_ACQUIRE_TIMEOUT_S:.0f}s ({self.limit} slots at {self.dir})"
                )
            time.sleep(random.uniform(0.05, 0.30))  # doubles as launch jitter


_GATE: GlobalSubcallGate | None = None
_GATE_CHECKED = False


def get_gate() -> GlobalSubcallGate | None:
    """The process-wide gate, or None when RLM_SUBCALL_GATE_DIR is unset."""
    global _GATE, _GATE_CHECKED
    if not _GATE_CHECKED:
        _GATE_CHECKED = True
        d = os.environ.get("RLM_SUBCALL_GATE_DIR")
        if d:
            _GATE = GlobalSubcallGate(
                d, int(os.environ.get("RLM_SUBCALL_GATE_LIMIT", "16"))
            )
    return _GATE
