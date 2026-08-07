from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    final_answer: str | None = None
    execution_time: float = 0.0
    locals_keys: list[str] = field(default_factory=list)
    #: Completed sub-LLM calls made by THIS cell (len of the REPL's per-cell
    #: rlm_calls). Accumulated into state["rlm_sub_llm_calls"] by RLMTrainEnv
    #: so per-rollout sub-call volume is a first-class metric — the proxy
    #: counter never sees socket-path calls, so this is the only true count.
    n_subcalls: int = 0


class ReplBackend(ABC):
    @abstractmethod
    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None: ...

    @abstractmethod
    async def load_context(self, payload: Any, index: int | None = None) -> int: ...

    @abstractmethod
    async def execute(self, code: str) -> ExecResult: ...

    @abstractmethod
    async def stop(self) -> None: ...

    async def bootstrap(self, code: str) -> None:
        if not code:
            return
        await self.execute(code)
