"""Comprehensive tests for LocalREPL environment."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from threading import Barrier
from unittest.mock import Mock

import pytest

import rlm.environments.local_repl as local_repl_module
from rlm.core.rlm import RLM
from rlm.core.types import UsageSummary
from rlm.environments.local_repl import LocalREPL, _ThreadRoutedStream


class TestNestedCapture:
    @pytest.fixture(autouse=True)
    def fresh_stream_routers(self, monkeypatch):
        # Pytest replaces process streams between tests; give each test a fresh
        # production router and restore the process streams after it finishes.
        monkeypatch.setattr(local_repl_module, "_STREAM_ROUTERS", None)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield

    @pytest.mark.parametrize(
        "query, count",
        [
            ("rlm_query('question')", 1),
            ("rlm_query_batched(['question'])", 1),
            ("rlm_query_batched(['question', 'question'])", 2),
        ],
    )
    @pytest.mark.parametrize("child_error", [False, True])
    def test_child_restores_parent_stdout_and_stderr(self, monkeypatch, query, count, child_error):
        barrier = Barrier(count)
        client = Mock(model_name="mock-model")
        ending = (
            "raise ValueError('child failed')"
            if child_error
            else "answer['content'] = 'child'; answer['ready'] = True"
        )

        def completion(*args):
            barrier.wait(timeout=5)
            return (
                "```repl\nimport sys\nprint('child stdout')\n"
                "print('child stderr', file=sys.stderr)\n" + ending + "\n```"
            )

        client.completion.side_effect = completion
        client.get_usage_summary.return_value = UsageSummary(model_usage_summaries={})
        monkeypatch.setattr("rlm.core.rlm.get_client", lambda *args: client)
        parent = RLM(
            backend_kwargs={"model_name": "mock-model"},
            max_depth=2,
            max_iterations=1,
            fabricate_final_answer=False,
            recover_stub=False,
        )
        try:
            with LocalREPL(subcall_fn=parent._subcall) as env:
                result = env.execute_code(
                    "import sys\nprint('parent before')\n"
                    "print('parent stderr before', file=sys.stderr)\n"
                    f"x = {query}\nprint('parent after')\n"
                    "print('parent stderr after', file=sys.stderr)"
                )
                following = env.execute_code("print('next block')")
        finally:
            parent.close()
        assert client.completion.call_count == count
        assert result.stdout == "parent before\nparent after\n"
        assert result.stderr == "parent stderr before\nparent stderr after\n"
        assert following.stdout == "next block\n"

    def test_nested_capture_exception_restores_outer_and_fallback_streams(self):
        fallback_out, fallback_err = StringIO(), StringIO()
        with (
            redirect_stdout(fallback_out),
            redirect_stderr(fallback_err),
            LocalREPL() as outer,
            LocalREPL() as inner,
        ):
            with outer._capture_output() as (outer_out, outer_err):
                print("outer before")
                print("outer error before", file=sys.stderr)
                with pytest.raises(ValueError, match="nested"):
                    with inner._capture_output() as (inner_out, inner_err):
                        print("inner")
                        print("inner error", file=sys.stderr)
                        raise ValueError("nested")
                print("outer after")
                print("outer error after", file=sys.stderr)
            print("fallback")
            print("fallback error", file=sys.stderr)
        assert outer_out.getvalue() == "outer before\nouter after\n"
        assert outer_err.getvalue() == "outer error before\nouter error after\n"
        assert inner_out.getvalue() == "inner\n"
        assert inner_err.getvalue() == "inner error\n"
        assert fallback_out.getvalue() == "fallback\n"
        assert fallback_err.getvalue() == "fallback error\n"

    def test_parallel_nested_routes_restore_their_own_parent(self):
        fallback, parent = StringIO(), StringIO()
        stream = _ThreadRoutedStream(fallback)
        stream.register(parent)
        barrier = Barrier(2)

        def worker(label):
            outer, inner = StringIO(), StringIO()
            stream.register(outer)
            stream.write(f"{label} before\n")
            stream.register(inner)
            barrier.wait(timeout=5)
            stream.write(f"{label} child\n")
            stream.unregister()
            barrier.wait(timeout=5)
            stream.write(f"{label} after\n")
            stream.unregister()
            return outer.getvalue(), inner.getvalue()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(worker, ("a", "b")))
        stream.write("main\n")
        stream.unregister()
        stream.write("fallback\n")
        assert results == [
            ("a before\na after\n", "a child\n"),
            ("b before\nb after\n", "b child\n"),
        ]
        assert parent.getvalue() == "main\n"
        assert fallback.getvalue() == "fallback\n"


class TestLocalREPLBasic:
    """Basic functionality tests for LocalREPL."""

    def test_simple_execution(self):
        """Test basic code execution."""
        repl = LocalREPL()
        result = repl.execute_code("x = 1 + 2")
        assert result.stderr == ""
        assert repl.locals["x"] == 3
        repl.cleanup()

    def test_print_output(self):
        """Test that print statements are captured."""
        repl = LocalREPL()
        result = repl.execute_code("print('Hello, World!')")
        assert "Hello, World!" in result.stdout
        repl.cleanup()

    def test_error_handling(self):
        """Test that errors are captured in stderr."""
        repl = LocalREPL()
        result = repl.execute_code("1 / 0")
        assert "ZeroDivisionError" in result.stderr
        repl.cleanup()

    def test_syntax_error(self):
        """Test syntax error handling."""
        repl = LocalREPL()
        result = repl.execute_code("def broken(")
        assert "SyntaxError" in result.stderr
        repl.cleanup()


class TestLocalREPLPersistence:
    """Tests for state persistence across executions."""

    def test_variable_persistence(self):
        """Test that variables persist across multiple code executions."""
        repl = LocalREPL()

        result1 = repl.execute_code("x = 42")
        assert result1.stderr == ""
        assert repl.locals["x"] == 42

        result2 = repl.execute_code("y = x + 8")
        assert result2.stderr == ""
        assert repl.locals["y"] == 50

        result3 = repl.execute_code("print(y)")
        assert "50" in result3.stdout

        repl.cleanup()

    def test_function_persistence(self):
        """Test that defined functions persist."""
        repl = LocalREPL()

        repl.execute_code(
            """
def greet(name):
    return f"Hello, {name}!"
"""
        )

        result = repl.execute_code("print(greet('World'))")
        assert "Hello, World!" in result.stdout
        repl.cleanup()

    def test_list_comprehension(self):
        """Test that list comprehensions work."""
        repl = LocalREPL()

        repl.execute_code("squares = [x**2 for x in range(5)]")
        assert repl.locals["squares"] == [0, 1, 4, 9, 16]

        result = repl.execute_code("print(sum(squares))")
        assert "30" in result.stdout
        repl.cleanup()


class TestLocalREPLBuiltins:
    """Tests for safe builtins and blocked functions."""

    def test_safe_builtins_available(self):
        """Test that safe builtins are available."""
        repl = LocalREPL()

        # Test various safe builtins
        _ = repl.execute_code("x = len([1, 2, 3])")
        assert repl.locals["x"] == 3

        _ = repl.execute_code("y = sum([1, 2, 3, 4])")
        assert repl.locals["y"] == 10

        _ = repl.execute_code("z = sorted([3, 1, 2])")
        assert repl.locals["z"] == [1, 2, 3]

        repl.cleanup()

    def test_imports_work(self):
        """Test that imports work."""
        repl = LocalREPL()
        result = repl.execute_code("import math\nx = math.pi")
        assert result.stderr == ""
        assert abs(repl.locals["x"] - 3.14159) < 0.001
        repl.cleanup()


class TestLocalREPLContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self):
        """Test using LocalREPL as context manager."""
        with LocalREPL() as repl:
            _ = repl.execute_code("x = 100")
            assert repl.locals["x"] == 100


class TestLocalREPLHelpers:
    """Tests for helper functions and the answer-dict completion signal."""

    def test_answer_dict_defaults(self):
        """The default (content-mode) ``answer`` dict starts unready with empty content."""
        repl = LocalREPL()
        assert repl.locals["answer"]["content"] == ""
        assert "deliverables" not in repl.locals["answer"]
        assert repl.locals["answer"]["ready"] is False
        repl.cleanup()

    def test_answer_ready_surfaces_final_answer(self):
        """Setting ``answer['ready'] = True`` surfaces ``content`` on final_answer.

        Content mode also leaves ``final_deliverables`` unset (None).
        """
        repl = LocalREPL()
        result = repl.execute_code('answer["content"] = "the result"\nanswer["ready"] = True')
        assert result.final_answer == "the result"
        assert result.final_deliverables is None
        repl.cleanup()

    def test_answer_ready_false_does_not_surface(self):
        """Mutating ``content`` without flipping ``ready`` must not end the run."""
        repl = LocalREPL()
        result = repl.execute_code('answer["content"] = "still working"')
        assert result.final_answer is None
        repl.cleanup()

    def test_answer_rebind_to_plain_dict(self):
        """Rebinding ``answer`` to a plain dict with ready=True is still picked up."""
        repl = LocalREPL()
        result = repl.execute_code('answer = {"content": "rebound", "ready": True}')
        assert result.final_answer == "rebound"
        repl.cleanup()

    def test_slot_mode_defaults(self):
        """With deliverable_slots the ``answer`` dict seeds per-file slots (no content)."""
        repl = LocalREPL(deliverable_slots=["a.md", "b.md"])
        assert repl.locals["answer"]["deliverables"] == {"a.md": "", "b.md": ""}
        assert "content" not in repl.locals["answer"]
        assert repl.locals["answer"]["ready"] is False
        repl.cleanup()

    def test_multi_deliverable_slots(self):
        """Slots seeded from deliverable_slots surface per-file text on final_deliverables."""
        repl = LocalREPL(deliverable_slots=["a.md", "b.md"])
        result = repl.execute_code(
            'answer["deliverables"]["a.md"] = "AAA"\n'
            'answer["deliverables"]["b.md"] = "BBB"\n'
            'answer["ready"] = True'
        )
        assert result.final_deliverables == {"a.md": "AAA", "b.md": "BBB"}
        assert result.final_answer is None
        repl.cleanup()

    def test_slot_mode_single_slot(self):
        """A single slot surfaces on final_deliverables (content stays None)."""
        repl = LocalREPL(deliverable_slots=["answer"])
        result = repl.execute_code(
            'answer["deliverables"]["answer"] = "the result"\nanswer["ready"] = True'
        )
        assert result.final_deliverables == {"answer": "the result"}
        assert result.final_answer is None
        repl.cleanup()

    def test_llm_query_no_handler(self):
        """Test llm_query without handler configured."""
        repl = LocalREPL()
        _ = repl.execute_code("response = llm_query('test')")
        assert "Error" in repl.locals["response"]
        repl.cleanup()


class TestLocalREPLContext:
    """Tests for context loading."""

    def test_string_context(self):
        """Test loading string context."""
        repl = LocalREPL(context_payload="This is the context data.")
        assert "context" in repl.locals
        assert repl.locals["context"] == "This is the context data."
        repl.cleanup()

    def test_dict_context(self):
        """Test loading dict context."""
        repl = LocalREPL(context_payload={"key": "value", "number": 42})
        assert "context" in repl.locals
        assert repl.locals["context"]["key"] == "value"
        assert repl.locals["context"]["number"] == 42
        repl.cleanup()

    def test_list_context(self):
        """Test loading list context."""
        repl = LocalREPL(context_payload=[1, 2, 3, "four"])
        assert "context" in repl.locals
        assert repl.locals["context"] == [1, 2, 3, "four"]
        repl.cleanup()


class TestLocalREPLScaffoldRestoration:
    """Tests that overwriting scaffold names (context, llm_query, etc.) is reverted after each execution."""

    def test_context_restored_after_overwrite(self):
        """If the model does context = 'something', the next execution still sees the real context."""
        repl = LocalREPL(context_payload="original context content")
        assert repl.locals["context"] == "original context content"

        repl.execute_code('context = "hijacked"')
        assert repl.locals["context"] == "original context content"

        out = repl.execute_code("print(context)")
        assert "original context content" in out.stdout
        repl.cleanup()

    def test_llm_query_restored_after_overwrite(self):
        """If the model does llm_query = lambda x: 'hijacked', the next execution still has real llm_query."""
        repl = LocalREPL()
        repl.execute_code("llm_query = lambda x: 'hijacked'")

        repl.execute_code("r = llm_query('test')")
        assert "Error" in repl.locals["r"]
        repl.cleanup()

    def test_answer_rewrap_after_rebind(self):
        """If the model rebinds ``answer`` to a plain dict, the next cell still triggers on ready=True."""
        repl = LocalREPL()
        repl.execute_code('answer = {"content": "intermediate", "ready": False}')
        # After scaffold restore, the rebound dict has been wrapped back into the
        # tracking subclass; setting ready=True now must fire the capture callback.
        result = repl.execute_code('answer["content"] = "done"; answer["ready"] = True')
        assert result.final_answer == "done"
        repl.cleanup()

    def test_answer_rewrap_after_rebind_slot_mode(self):
        """Slot-mode rebind to a plain dict is still captured on the next cell."""
        repl = LocalREPL(deliverable_slots=["answer"])
        repl.execute_code('answer = {"deliverables": {"answer": "intermediate"}, "ready": False}')
        result = repl.execute_code(
            'answer["deliverables"]["answer"] = "done"; answer["ready"] = True'
        )
        assert result.final_deliverables == {"answer": "done"}
        repl.cleanup()


class TestLocalREPLCleanup:
    """Tests for cleanup behavior."""

    def test_cleanup_clears_state(self):
        """Test that cleanup clears the namespace."""
        repl = LocalREPL()
        repl.execute_code("x = 42")
        assert "x" in repl.locals
        repl.cleanup()
        assert len(repl.locals) == 0

    def test_temp_dir_created_and_cleaned(self):
        """Test that temp directory is created and cleaned up."""
        repl = LocalREPL()
        temp_dir = repl.temp_dir
        assert os.path.exists(temp_dir)
        repl.cleanup()
        assert not os.path.exists(temp_dir)


class TestLocalREPLSimulatingRLMNoPersistence:
    """
    Tests simulating RLM's non-persistent completion behavior.

    When RLM is configured without persistent=True (the default), each
    get_completion() call spawns a fresh environment and destroys it after.
    This test suite simulates that behavior to prove variables don't survive
    across RLM completions.

    Why this matters: This is NOT just testing that two Python objects don't
    share state (trivially true). This simulates the actual RLM workflow where
    environments are created and destroyed per completion.
    """

    def test_simulated_rlm_completions_reset_environment(self):
        """
        Simulates 2 RLM completions to show env resets between calls.

        Without persistent=True, RLM creates a fresh environment for each
        completion, so state doesn't carry over.
        """
        completion_1_env = LocalREPL()
        completion_1_env.execute_code("important_result = 42")
        assert completion_1_env.locals["important_result"] == 42
        completion_1_env.cleanup()

        completion_2_env = LocalREPL()
        result = completion_2_env.execute_code("print(important_result)")

        assert "NameError" in result.stderr
        assert "important_result" in result.stderr
        completion_2_env.cleanup()

    def test_simulated_rlm_completions_functions_not_preserved(self):
        """
        Simulates 2 RLM completions to show functions don't persist.
        """
        completion_1_env = LocalREPL()
        completion_1_env.execute_code("def my_helper(): return 'useful'")
        assert completion_1_env.execute_code("print(my_helper())").stdout.strip() == "useful"
        completion_1_env.cleanup()

        completion_2_env = LocalREPL()
        result = completion_2_env.execute_code("my_helper()")

        assert "NameError" in result.stderr
        assert "my_helper" in result.stderr
        completion_2_env.cleanup()
