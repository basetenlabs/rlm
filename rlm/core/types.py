from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "anthropic",
    "azure_openai",
    "gemini",
]
EnvironmentType = Literal["local", "ipython", "docker", "modal", "prime", "daytona", "e2b"]


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None = None  # Cost in USD, if available from provider

    def to_dict(self):
        result = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            total_cost=data.get("total_cost"),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    @property
    def total_cost(self) -> float | None:
        """Aggregate cost across all models. Returns None if no cost data available."""
        costs = [
            summary.total_cost
            for summary in self.model_usage_summaries.values()
            if summary.total_cost is not None
        ]
        return sum(costs) if costs else None

    @property
    def total_input_tokens(self) -> int:
        """Aggregate input tokens across all models."""
        return sum(summary.total_input_tokens for summary in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        """Aggregate output tokens across all models."""
        return sum(summary.total_output_tokens for summary in self.model_usage_summaries.values())

    def to_dict(self):
        result = {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################
DEFAULT_DELIVERABLE_SLOT = "answer"


def render_deliverables(deliverables: dict[str, str] | None) -> str:
    """Concatenate per-deliverable slots into a single string for trace/logging.

    Multi-slot output is rendered as ``===== <name> =====\\n<text>`` sections;
    a single slot renders as just its text (so the common single-deliverable
    case reads exactly like the old ``content``).
    """
    if not deliverables:
        return ""
    if len(deliverables) == 1:
        return next(iter(deliverables.values()))
    return "\n\n".join(f"===== {name} =====\n{text}" for name, text in deliverables.items())


@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within the environment."""

    root_model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float
    # Per-deliverable final output, keyed by exact deliverable filename. Set on
    # the completion that finalizes the RLM loop; ``response`` is a rendered
    # concatenation of these slots (trace/logging only).
    deliverables: dict[str, str] | None = None
    metadata: dict | None = (
        None  # Full trajectory (run_metadata + iterations) when logger captures it
    )
    error: str | None = (
        None  # Set when this single call failed (e.g. in a batch); response is empty.
    )

    def to_dict(self):
        out = {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }
        if self.deliverables is not None:
            out["deliverables"] = self.deliverables
        if self.metadata is not None:
            out["metadata"] = self.metadata
        if self.error is not None:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "RLMChatCompletion":
        return cls(
            root_model=data.get("root_model"),
            prompt=data.get("prompt"),
            response=data.get("response"),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary")),
            execution_time=data.get("execution_time"),
            deliverables=data.get("deliverables"),
            metadata=data.get("metadata"),
            error=data.get("error"),
        )


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    llm_calls: list["RLMChatCompletion"]
    # Final output captured when the model set ``answer["ready"] = True``.
    # Exactly one of these is set, keyed on the answer protocol in play:
    #   * content mode (default ``RLM``, no deliverable slots):
    #       ``final_answer`` holds the ``answer["content"]`` string.
    #   * slot mode (``MultiDeliverableRLM``, deliverable slots seeded):
    #       ``final_deliverables`` holds {exact filename: text}.
    # Both are None until the loop is finalized.
    final_answer: str | None = None
    final_deliverables: dict[str, str] | None = None

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float = None,
        rlm_calls: list["RLMChatCompletion"] = None,
        final_answer: str | None = None,
        final_deliverables: dict[str, str] | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.rlm_calls = rlm_calls or []
        self.final_answer = final_answer
        self.final_deliverables = final_deliverables

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time}, rlm_calls={len(self.rlm_calls)})"

    def to_dict(self):
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "rlm_calls": [call.to_dict() for call in self.rlm_calls],
            "final_answer": self.final_answer,
            "final_deliverables": self.final_deliverables,
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self):
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None
    # Per-iteration ROOT-model token usage ({"total_input_tokens", "total_output_tokens"}).
    # Sub-call usage lives in code_blocks[].result.rlm_calls[]; logging the root here makes the
    # run's total token count independently re-derivable from the trajectory.
    root_usage: dict[str, int] | None = None
    # The ROOT completion's finish_reason for this turn (e.g. "stop", "length"). "length"
    # means the turn was truncated at max_tokens BEFORE the model finished emitting content
    # (on reasoning turns this often means an empty response); surfaced so harnesses can count
    # truncated turns instead of the truncation being silent.
    finish_reason: str | None = None

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [code_block.to_dict() for code_block in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
            "root_usage": self.root_usage,
            "finish_reason": self.finish_reason,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
