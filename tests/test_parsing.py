"""Tests for parsing utilities."""

from rlm.core.types import CodeBlock, REPLResult, RLMIteration
from rlm.environments.local_repl import LocalREPL
from rlm.utils.parsing import (
    convert_context_for_repl,
    find_code_blocks,
    format_execution_result,
    format_iteration,
)


class TestFindCodeBlocks:
    """Tests for find_code_blocks function."""

    def test_single_code_block(self):
        text = """Here's some code:
```repl
x = 1 + 2
print(x)
```
Done."""
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert "x = 1 + 2" in blocks[0]
        assert "print(x)" in blocks[0]

    def test_multiple_code_blocks(self):
        text = """First block:
```repl
a = 1
```
Second block:
```repl
b = 2
```
End."""
        blocks = find_code_blocks(text)
        assert len(blocks) == 2
        assert "a = 1" in blocks[0]
        assert "b = 2" in blocks[1]

    def test_no_code_blocks(self):
        text = "Just plain text without any code blocks."
        blocks = find_code_blocks(text)
        assert blocks == []

    def test_non_repl_code_blocks_ignored(self):
        text = """Python block:
```python
x = 1
```
REPL block:
```repl
y = 2
```
"""
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert "y = 2" in blocks[0]

    def test_repl_fence_with_trailing_junk_accepted(self):
        # GLM-5.2 with thinking disabled systematically emits these variants
        # (598 of 715 fence emissions in the 2026-07-09 nothink probe); the
        # strict ```repl fence rejected them silently and runs perseverated.
        for fence in ("```repl Python", "```repl Python block", "```repl  "):
            text = f"{fence}\nz = 3\nprint(z)\n```"
            blocks = find_code_blocks(text)
            assert len(blocks) == 1, fence
            assert "z = 3" in blocks[0]

    def test_repl_prefixed_languages_still_ignored(self):
        # "repl" must be the whole tag word: ```replace etc. are not REPL blocks.
        text = "```replace\nnot code\n```\n```repl\ny = 2\n```"
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert "y = 2" in blocks[0]

    def test_multiline_code_block(self):
        text = """```repl
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)
```"""
        blocks = find_code_blocks(text)
        assert len(blocks) == 1
        assert "def factorial(n):" in blocks[0]
        assert "return n * factorial(n - 1)" in blocks[0]


class TestAnswerDictFinalAnswer:
    """Tests for the default (content-mode) ``answer`` dict completion signal."""

    def test_answer_dict_ready_true_sets_final_answer(self):
        """Setting ``answer['ready'] = True`` must populate REPLResult.final_answer."""
        env = LocalREPL()
        try:
            result = env.execute_code('answer["content"] = "the result"\nanswer["ready"] = True')
            assert result.final_answer == "the result"
            assert result.final_deliverables is None
        finally:
            env.cleanup()

    def test_answer_dict_unset_keeps_final_answer_none(self):
        """If ``ready`` stays False, the REPL must not surface a final answer."""
        env = LocalREPL()
        try:
            result = env.execute_code('answer["content"] = "wip"')
            assert result.final_answer is None
        finally:
            env.cleanup()

    def test_answer_dict_rebind_with_ready(self):
        """Plain-dict rebind with ``ready=True`` must still be captured."""
        env = LocalREPL()
        try:
            result = env.execute_code('answer = {"content": "rebound", "ready": True}')
            assert result.final_answer == "rebound"
        finally:
            env.cleanup()

    def test_answer_content_can_be_non_string(self):
        """Any ``str()``-able content (numbers, lists) is coerced to a string final answer."""
        env = LocalREPL()
        try:
            result = env.execute_code('answer["content"] = [1, 2, 3]\nanswer["ready"] = True')
            assert result.final_answer == "[1, 2, 3]"
        finally:
            env.cleanup()


class TestSlotModeFinalDeliverables:
    """Tests for the slot-mode ``answer`` dict (MultiDeliverableRLM env path)."""

    def test_ready_true_sets_final_deliverables(self):
        env = LocalREPL(deliverable_slots=["answer"])
        try:
            result = env.execute_code(
                'answer["deliverables"]["answer"] = "the result"\nanswer["ready"] = True'
            )
            assert result.final_deliverables == {"answer": "the result"}
            assert result.final_answer is None
        finally:
            env.cleanup()

    def test_slot_value_coerced_to_string(self):
        env = LocalREPL(deliverable_slots=["answer"])
        try:
            result = env.execute_code(
                'answer["deliverables"]["answer"] = [1, 2, 3]\nanswer["ready"] = True'
            )
            assert result.final_deliverables == {"answer": "[1, 2, 3]"}
        finally:
            env.cleanup()


class TestFormatExecutionResult:
    """Tests for format_execution_result function."""

    def test_stdout_only(self):
        result = REPLResult(stdout="Hello, World!", stderr="", locals={})
        formatted = format_execution_result(result)
        assert "Hello, World!" in formatted

    def test_stderr_only(self):
        result = REPLResult(stdout="", stderr="Error occurred", locals={})
        formatted = format_execution_result(result)
        assert "Error occurred" in formatted

    def test_with_locals(self):
        result = REPLResult(stdout="", stderr="", locals={"x": 42, "name": "test"})
        formatted = format_execution_result(result)
        assert "x" in formatted
        assert "name" in formatted

    def test_excludes_private_vars(self):
        result = REPLResult(stdout="", stderr="", locals={"_private": 1, "public": 2})
        formatted = format_execution_result(result)
        assert "public" in formatted
        # Private vars should be excluded
        assert "_private" not in formatted

    def test_empty_result(self):
        result = REPLResult(stdout="", stderr="", locals={})
        formatted = format_execution_result(result)
        assert formatted == "No output"


class TestFormatIteration:
    """Tests for format_iteration function."""

    def test_iteration_with_code_blocks(self):
        code_result = REPLResult(stdout="3", stderr="", locals={"x": 3})
        iteration = RLMIteration(
            prompt="Calculate 1+2",
            response="Let me calculate that.",
            code_blocks=[CodeBlock(code="x = 1 + 2\nprint(x)", result=code_result)],
        )
        messages = format_iteration(iteration)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert "REPL output:" in messages[1]["content"]
        assert "3" in messages[1]["content"]

    def test_iteration_without_code_blocks(self):
        iteration = RLMIteration(
            prompt="Just thinking",
            response="I'm considering the options.",
            code_blocks=[],
        )
        messages = format_iteration(iteration)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_no_code_feedback_appended_when_nothing_parsed(self):
        # A silent no-op turn gives a fenceless/malformed-fence model no signal
        # to correct itself (2026-07-09 nothink probe: perseveration root cause).
        iteration = RLMIteration(
            prompt="Just thinking",
            response="I need to examine the context structure first.",
            code_blocks=[],
        )
        messages = format_iteration(iteration, no_code_feedback="No ```repl block found.")
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert "No ```repl block found." == messages[1]["content"]

    def test_no_code_feedback_not_appended_when_code_ran(self):
        code_result = REPLResult(stdout="3", stderr="", locals={})
        iteration = RLMIteration(
            prompt="Calculate",
            response="Running.",
            code_blocks=[CodeBlock(code="print(3)", result=code_result)],
        )
        messages = format_iteration(iteration, no_code_feedback="No ```repl block found.")
        assert len(messages) == 2
        assert "REPL output:" in messages[1]["content"]

    def test_truncates_long_results(self):
        long_output = "x" * 30000
        code_result = REPLResult(stdout=long_output, stderr="", locals={})
        iteration = RLMIteration(
            prompt="Test",
            response="Running...",
            code_blocks=[CodeBlock(code="print('x' * 30000)", result=code_result)],
        )
        messages = format_iteration(iteration, max_character_length=100)
        # Result should be truncated
        assert len(messages[1]["content"]) < 30000


class TestConvertContextForRepl:
    """Tests for convert_context_for_repl function."""

    def test_string_context(self):
        context_data, context_str = convert_context_for_repl("Hello world")
        assert context_data is None
        assert context_str == "Hello world"

    def test_dict_context(self):
        context_data, context_str = convert_context_for_repl({"key": "value"})
        assert context_data == {"key": "value"}
        assert context_str is None

    def test_list_of_strings(self):
        context_data, context_str = convert_context_for_repl(["a", "b", "c"])
        assert context_data == ["a", "b", "c"]
        assert context_str is None

    def test_list_of_message_dicts(self):
        messages = [
            {"content": "Hello"},
            {"content": "World"},
        ]
        context_data, context_str = convert_context_for_repl(messages)
        assert context_data == ["Hello", "World"]
        assert context_str is None
