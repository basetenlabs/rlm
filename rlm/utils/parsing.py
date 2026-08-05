"""
Parsing utilities for RLM trjaectories.
"""

import re

from rlm.core.types import REPLResult, RLMIteration


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.

    The fence tag tolerates trailing junk after "repl" (e.g. ```repl Python) —
    GLM-5.2 with thinking disabled emits these variants systematically, and a
    strict fence silently drops the code (2026-07-09 nothink probe). "repl"
    must still be the whole first word, so ```replace is not a REPL block.

    ```python / ```py fences are accepted too: Qwen3.5-122B nothink t0.6
    locks into ```python against explicit ```repl instructions in ~13% of RL
    episodes (2026-08-05, L25 transcripts: 79/79 turns ```python, zero cells
    executed, whole episode wasted with no error signal). Aaron's data-gen
    pod already ran a lenient parser, so leniency also matches the setup that
    produced the reference numbers.
    """
    pattern = r"```(?:repl|python|py)(?:[ \t][^\n]*)?\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


# Sent as the user turn when a response contained no parseable ```repl block.
# Without it the model gets zero corrective signal after a malformed or missing
# fence and can pattern-lock on its own prior turn (thinking-off GLM-5.2
# perseverated this way for 80-turn runs; see harvey-labs docs/glm-nothink-probe.md).
NO_CODE_FEEDBACK = (
    "No ```repl code block was found in your last message, so nothing was "
    "executed. To run code, open a block with ```repl on its own line, put "
    "your Python inside, and close it with ```. Remember that you can only "
    "inspect the context and produce your final answer through the REPL."
)


def render_block_output(
    stdout: str | None,
    stderr: str | None,
    locals_keys: list[str] | None,
    output_cap: int,
) -> str:
    """THE per-code-block REPL-output rendering, shared by every RLM loop.

    Extracted 2026-08-05: the eval loop (``format_execution_result``) and the
    RL loop (``rlm_train.env._format_one``) carried byte-identical copies of
    this — except the RL copy hardcoded a 20,000-char cap while eval's is
    configurable (``repl_output_cap``, raised for direct-read roots), a live
    divergence waiting to matter.
    """
    parts: list[str] = []
    if stdout:
        parts.append(f"\n{stdout}")
    if stderr:
        parts.append(f"\n{stderr}")
    if locals_keys:
        parts.append(f"REPL variables: {list(locals_keys)}\n")
    body = "\n\n".join(parts) if parts else "No output"
    if len(body) > output_cap:
        body = body[:output_cap] + f"... + [{len(body) - output_cap} chars...]"
    return body


def render_turn_feedback(
    block_bodies: list[str],
    no_code_feedback: str | None = None,
) -> str | None:
    """THE combined user-reply text for one turn's executed blocks.

    Headers + joining shared by both loops. Empty ``block_bodies`` (the turn
    parsed no code) returns ``no_code_feedback`` — so a loop that renders its
    feedback through here gets the no-code corrective signal by construction
    rather than by remembering to add it (the RL loop forgot once; 79-turn
    silent churn). Returns None when there is nothing to say.
    """
    if not block_bodies:
        return no_code_feedback
    multi = len(block_bodies) > 1
    parts = [
        f"{'REPL output (block %d):' % (i + 1) if multi else 'REPL output:'}\n{body}"
        for i, body in enumerate(block_bodies)
    ]
    return "\n\n".join(parts)


def format_iteration(
    iteration: RLMIteration,
    max_character_length: int = 20000,
    no_code_feedback: str | None = None,
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Each iteration produces exactly two messages in history: one assistant
    turn containing the model's response (with any ```repl``` blocks
    embedded), followed by a single user message that concatenates the
    outputs of all executed code blocks in that turn. This keeps the
    per-turn shape assistant-then-user even when the model emits several
    blocks in one response, and avoids redundantly echoing the code
    (which is already in the assistant message) back in the user reply.
    Each block's output is still individually truncated at
    ``max_character_length``.

    Args:
        iteration: The iteration to format
        max_character_length: Per-block cap on the formatted execution
            result. Longer outputs are tail-trimmed.
        no_code_feedback: If set and the iteration ran no code, this text is
            appended as the user reply instead of leaving the turn silent
            (see NO_CODE_FEEDBACK). If None, no-code turns get no user reply
            (stock behavior).

    Returns:
        A list of messages to add to the next prompt — length 2 (assistant
        + one combined user reply) when code ran or no_code_feedback is set,
        else length 1 (just the assistant).
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    bodies = [
        render_block_output(
            code_block.result.stdout,
            code_block.result.stderr,
            _important_locals(code_block.result),
            max_character_length,
        )
        for code_block in iteration.code_blocks
    ]
    reply = render_turn_feedback(bodies, no_code_feedback)
    if reply is not None:
        messages.append({"role": "user", "content": reply})
    return messages


def _important_locals(result: REPLResult) -> list[str]:
    """User-meaningful REPL variable names (simple types, non-dunder)."""
    keys: list[str] = []
    for key, value in result.locals.items():
        if key.startswith("_") or key in ("__builtins__", "__name__", "__doc__"):
            continue
        if isinstance(value, (str, int, float, bool, list, dict, tuple)):
            keys.append(key)
    return keys


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """Format one execution result for display (uncapped).

    Thin wrapper over :func:`render_block_output` — the shared renderer both
    loops use — kept for existing callers.
    """
    import sys

    return render_block_output(
        result.stdout, result.stderr, _important_locals(result), sys.maxsize
    )


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
