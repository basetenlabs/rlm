import copy
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from rlm.core.comms_utils import (
    DEFAULT_WAVE_TIMEOUT,
    WAVE_TIMEOUT_SLACK,
    LMRequest,
    send_lm_request,
    send_lm_request_batched,
)
from rlm.core.types import DEFAULT_DELIVERABLE_SLOT, REPLResult, RLMChatCompletion
from rlm.environments.base_env import (
    RESERVED_TOOL_NAMES,
    NonIsolatedEnv,
    extract_tool_value,
    validate_custom_tools,
)


def normalize_slots(slots: list[str] | None) -> list[str]:
    """Slot names to pre-seed ``answer["deliverables"]`` with.

    None/empty falls back to a single generic ``"answer"`` slot so the engine
    works for non-b10 use. Order and duplicates are preserved-then-deduped.
    """
    if not slots:
        return [DEFAULT_DELIVERABLE_SLOT]
    seen: list[str] = []
    for s in slots:
        if s not in seen:
            seen.append(s)
    return seen or [DEFAULT_DELIVERABLE_SLOT]


class _AnswerDict(dict):
    """REPL-visible dict where ``answer["ready"] = True`` signals completion.

    Two protocols, gated on whether ``slots`` is passed:

    * Content mode (``slots is None`` — the default upstream ``RLM``): seeded as
      ``{"content": "", "ready": False}``. The model sets ``answer["content"]``,
      then ``answer["ready"] = True``. On ready, ``on_ready`` receives the
      ``content`` value (a string); the next ``execute_code`` surfaces it as
      ``REPLResult.final_answer``.
    * Slot mode (``slots`` is a list — ``MultiDeliverableRLM``): seeded as
      ``{"deliverables": {name: "" for name in slots}, "ready": False}``. The
      model fills each slot, then ``answer["ready"] = True``. On ready,
      ``on_ready`` receives the whole ``deliverables`` dict; the next
      ``execute_code`` surfaces it as ``REPLResult.final_deliverables``.
    """

    def __init__(self, on_ready=None, slots: list[str] | None = None):
        super().__init__()
        # slots is None -> content mode; a list -> slot mode. Distinguish here
        # (not via ``if slots``) so an explicit empty list still selects slots.
        self._slot_mode = slots is not None
        if self._slot_mode:
            super().__setitem__("deliverables", {name: "" for name in normalize_slots(slots)})
        else:
            super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                if self._slot_mode:
                    deliverables = self.get("deliverables") or {}
                    self._on_ready(
                        dict(deliverables) if isinstance(deliverables, dict) else {}
                    )
                else:
                    self._on_ready(self.get("content", ""))
            except Exception:
                pass


# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
class _ThreadRoutedStream(io.TextIOBase):
    """A process-wide stdout/stderr proxy that routes writes per-thread.

    Each REPL cell registers a buffer for ITS exec thread; writes from any
    unregistered thread pass through to the real stream. This is what makes
    concurrent LocalREPLs in one process safe — see ``_capture_output``.
    """

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._routes: dict[int, Any] = {}
        self._routes_lock = threading.Lock()

    def register(self, buf: Any) -> None:
        with self._routes_lock:
            self._routes[threading.get_ident()] = buf

    def unregister(self) -> None:
        with self._routes_lock:
            self._routes.pop(threading.get_ident(), None)

    def _target(self) -> Any:
        return self._routes.get(threading.get_ident(), self._fallback)

    def write(self, s: str) -> int:
        # A target can be closed by its owner (pytest capture teardown, a
        # cleaned-up REPL). Degrade to the real stream, never raise into the
        # printing thread.
        try:
            return self._target().write(s)
        except ValueError:
            real = getattr(sys, "__stdout__", None)
            try:
                return real.write(s) if real else len(s)
            except (ValueError, OSError):
                return len(s)

    def flush(self) -> None:
        target = self._target()
        flush = getattr(target, "flush", None)
        if flush is not None:
            try:
                flush()
            except (ValueError, OSError):
                pass

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:  # some libraries introspect this
        return getattr(self._fallback, "encoding", "utf-8") or "utf-8"


_STREAM_ROUTERS: tuple[_ThreadRoutedStream, _ThreadRoutedStream] | None = None
_STREAM_ROUTERS_LOCK = threading.Lock()


def _install_stream_routers() -> tuple[_ThreadRoutedStream, _ThreadRoutedStream]:
    """Idempotently replace sys.stdout/sys.stderr with thread-routing proxies."""
    global _STREAM_ROUTERS
    with _STREAM_ROUTERS_LOCK:
        if _STREAM_ROUTERS is None:
            out = _ThreadRoutedStream(sys.stdout)
            err = _ThreadRoutedStream(sys.stderr)
            sys.stdout, sys.stderr = out, err
            _STREAM_ROUTERS = (out, err)
    return _STREAM_ROUTERS


_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: Callable[[str, str | None], RLMChatCompletion] | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        max_concurrent_subcalls: int = 4,
        deliverable_slots: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
            **kwargs,
        )

        # Answer protocol: ``deliverable_slots is None`` -> content mode
        # (upstream ``answer["content"]``); a list -> slot mode
        # (``answer["deliverables"]`` seeded with these names).
        self._slot_mode = deliverable_slots is not None
        self.deliverable_slots = (
            normalize_slots(deliverable_slots) if self._slot_mode else None
        )
        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn  # Callback for recursive RLM calls (depth > 1 support)
        # os.getcwd() raises FileNotFoundError when another concurrent LocalREPL's
        # cleanup() has rmtree'd the directory this process is standing in (chdir is
        # process-global; see _temp_cwd/cleanup). Fall back so init never wedges.
        try:
            self.original_cwd = os.getcwd()
        except FileNotFoundError:
            self.original_cwd = tempfile.gettempdir()
            os.chdir(self.original_cwd)
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.compaction = compaction

        # Custom tools: functions available in the REPL
        self.custom_tools = custom_tools or {}
        # Sub-tools: inherited from custom_tools if not specified
        self.custom_sub_tools = (
            custom_sub_tools if custom_sub_tools is not None else self.custom_tools
        )

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Setup globals, locals, and modules in environment.
        self.setup()

        if compaction:
            self._compaction_history: list[Any] = []
            self.locals["history"] = self._compaction_history

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []
        # Captured the first time the model sets ``answer["ready"] = True``.
        # Exactly one is populated depending on the answer protocol.
        self._last_final_answer: str | None = None
        self._last_final_deliverables: dict[str, str] | None = None

        # Add helper functions
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched

        # The model marks completion via ``answer["ready"] = True``; the
        # custom dict captures the deliverables dict as soon as that happens so
        # we don't have to probe the namespace after every cell.
        self.locals["answer"] = _AnswerDict(
            on_ready=self._capture_answer, slots=self.deliverable_slots
        )

        # Add custom tools to globals
        # Tools can be either plain values or (value, description) tuples
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                # For non-callable values (constants, data), add to locals
                self.locals[name] = value

    def _capture_answer(self, result) -> None:
        if self._slot_mode:
            self._last_final_deliverables = {
                str(k): str(v) for k, v in (result or {}).items()
            }
        else:
            self._last_final_answer = str(result) if result is not None else ""

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call via the handler, regardless of depth.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            self._pending_llm_calls.append(response.chat_completion)
            return response.chat_completion.response
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Query the LM with multiple prompts concurrently (no REPL, no recursion).

        This always makes direct LM calls via the handler, regardless of depth.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input prompts.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)
        try:
            responses = send_lm_request_batched(
                self.lm_handler_address, prompts, model=model, depth=self.depth,
                timeout=int(DEFAULT_WAVE_TIMEOUT + WAVE_TIMEOUT_SLACK),
            )

            results = []
            for response in responses:
                if not response.success:
                    results.append(f"Error: {response.error}")
                else:
                    self._pending_llm_calls.append(response.chat_completion)
                    results.append(response.chat_completion.response)

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        """Spawn a recursive RLM sub-call for deeper thinking on a subtask.

        When a subcall callback is available (max_depth > 1), this spawns a child
        RLM with its own REPL that can reason over the prompt iteratively.
        Falls back to a plain llm_query if no recursive capability is configured.

        Args:
            prompt: The prompt to send to the child RLM.
            model: Optional model name override for the child.
        """
        if self.subcall_fn is not None:
            try:
                completion = self.subcall_fn(prompt, model)
                self._pending_llm_calls.append(completion)
                return completion.response
            except Exception as e:
                return f"Error: RLM query failed - {e}"

        # Fall back to plain LM call if no recursive capability
        return self._llm_query(prompt, model)

    def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Spawn recursive RLM sub-calls for multiple prompts in parallel.

        Each prompt gets its own child RLM for deeper thinking. When multiple
        prompts are provided, subcalls run concurrently using a thread pool
        (bounded by max_concurrent_subcalls) since they are independent and
        I/O-bound. Results are returned in the same order as input prompts.

        Falls back to llm_query_batched if no recursive capability is configured.

        Args:
            prompts: List of prompts for child RLMs.
            model: Optional model name override for the children.

        Returns:
            List of responses in the same order as input prompts.
        """
        if self.subcall_fn is not None:
            # For 0 or 1 prompts, no need for thread pool overhead
            if len(prompts) <= 1:
                results = []
                for prompt in prompts:
                    try:
                        completion = self.subcall_fn(prompt, model)
                        self._pending_llm_calls.append(completion)
                        results.append(completion.response)
                    except Exception as e:
                        results.append(f"Error: RLM query failed - {e}")
                return results

            # Parallel execution for multiple prompts
            max_workers = min(self.max_concurrent_subcalls, len(prompts))
            # Pre-allocate result slots to preserve ordering
            results: list[str] = [""] * len(prompts)
            completions: list[tuple[int, RLMChatCompletion]] = []
            lock = threading.Lock()

            def _run_subcall(index: int, prompt: str) -> None:
                try:
                    completion = self.subcall_fn(prompt, model)
                    with lock:
                        completions.append((index, completion))
                    results[index] = completion.response
                except Exception as e:
                    results[index] = f"Error: RLM query failed - {e}"

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_subcall, i, prompt) for i, prompt in enumerate(prompts)
                ]
                # Wait for all futures to complete; exceptions are captured inside _run_subcall
                for future in as_completed(futures):
                    future.result()  # Re-raises unexpected executor errors

            # Append completions in original prompt order for deterministic metadata
            completions.sort(key=lambda x: x[0])
            for _, completion in completions:
                self._pending_llm_calls.append(completion)

            return results

        # Fall back to plain batched LM call if no recursive capability
        return self._llm_query_batched(prompts, model)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: list[dict[str, Any]] | dict[str, Any]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))

    @contextmanager
    def _capture_output(self):
        """Capture THIS thread's stdout/stderr for the duration of a cell.

        ``sys.stdout`` is process-global, and multiple LocalREPLs execute
        cells concurrently on different threads (RL env workers run 2+
        rollouts per process). The old implementation swapped ``sys.stdout``
        under a per-INSTANCE lock, so concurrent cells captured EACH OTHER'S
        prints — rollout A received rollout B's output as its own REPL
        feedback (cross-rollout leakage, root-caused 2026-08-05 after
        masquerading as sub-LLM response misrouting), one side's output was
        silently lost, and the unordered restores could leave a dead StringIO
        installed as the process's stdout. Fix: install a process-wide
        thread-routing proxy ONCE; each cell registers its buffers for its
        own exec thread only. Threads the model spawns inside a cell are not
        registered and fall through to the real stream — never to another
        rollout's buffer.
        """
        out_router, err_router = _install_stream_routers()
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        out_router.register(stdout_buf)
        err_router.register(stderr_buf)
        try:
            yield stdout_buf, stderr_buf
        finally:
            out_router.unregister()
            err_router.unregister()

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution.

        chdir is process-global and multiple LocalREPLs run concurrently in one
        process, so both the saved cwd and the restore target can be deleted by a
        sibling's cleanup() at any await point. Never let that propagate — restore
        to the safest still-existing directory instead.
        """
        try:
            old_cwd = os.getcwd()
        except FileNotFoundError:
            old_cwd = self.original_cwd
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            for target in (old_cwd, self.original_cwd, tempfile.gettempdir()):
                try:
                    os.chdir(target)
                    break
                except FileNotFoundError:
                    continue

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites (e.g. context = 'x') don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                # If the model rebound ``answer`` to a plain dict, the
                # _AnswerDict callback never fired; capture deliverables here if
                # ``ready=True``, then re-wrap so the next cell signals.
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(
                        on_ready=self._capture_answer, slots=self.deliverable_slots
                    )
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready"):
                            if self._slot_mode and self._last_final_deliverables is None:
                                deliverables = current.get("deliverables") or {}
                                if isinstance(deliverables, dict):
                                    self._capture_answer(deliverables)
                            elif not self._slot_mode and self._last_final_answer is None:
                                self._capture_answer(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "history" and "history_0" in self.locals and not self.compaction:
                self.locals["history"] = self.locals["history_0"]
            elif name == "history" and self.compaction:
                self.locals["history"] = self._compaction_history

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold so model overwrites (context = ..., llm_query = ...) don't persist
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        final_answer = self._last_final_answer
        final_deliverables = self._last_final_deliverables
        self._last_final_answer = None
        self._last_final_deliverables = None

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
            final_deliverables=final_deliverables,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            # If the process is currently standing in (or under) our temp_dir —
            # possible because a sibling REPL's _temp_cwd restore can land here —
            # step out before deleting it, or every later os.getcwd() in this
            # process raises FileNotFoundError and all new episodes wedge.
            try:
                cwd = os.getcwd()
            except FileNotFoundError:
                cwd = None
            if cwd is not None and (cwd == self.temp_dir or cwd.startswith(self.temp_dir + os.sep)):
                for target in (self.original_cwd, tempfile.gettempdir()):
                    try:
                        os.chdir(target)
                        break
                    except FileNotFoundError:
                        continue
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()
