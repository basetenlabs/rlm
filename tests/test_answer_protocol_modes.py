"""Two-mode answer-protocol contract at the RLM / completion level.

* Default ``RLM`` (no ``deliverable_slots``) uses the upstream
  ``answer["content"]`` protocol; the completion's ``deliverables`` is None.
* ``MultiDeliverableRLM`` uses per-deliverable slots; the completion's
  ``deliverables`` is the {filename: text} dict and ``response`` is the render.
"""

from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM, MultiDeliverableRLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.utils.prompts import RLM_SYSTEM_PROMPT, RLM_SYSTEM_PROMPT_SLOTS


def create_mock_lm(responses: list[str], model_name: str = "mock-model") -> Mock:
    mock = Mock()
    mock.model_name = model_name
    mock.completion.side_effect = list(responses)
    mock.get_usage_summary.return_value = UsageSummary(
        model_usage_summaries={
            model_name: ModelUsageSummary(
                total_calls=1, total_input_tokens=100, total_output_tokens=50
            )
        }
    )
    mock.get_last_usage.return_value = mock.get_usage_summary.return_value
    return mock


def _content_final(text: str) -> str:
    return f"```repl\nanswer['content'] = {text!r}\nanswer['ready'] = True\n```"


def _slot_final(slot_values: dict[str, str]) -> str:
    lines = [f"answer['deliverables'][{k!r}] = {v!r}" for k, v in slot_values.items()]
    lines.append("answer['ready'] = True")
    body = "\n".join(lines)
    return f"```repl\n{body}\n```"


class TestDefaultRLMContentMode:
    def test_content_protocol_and_deliverables_none(self):
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = create_mock_lm([_content_final("hello")])
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_depth=1,
            )
            # Default RLM does not force slots and uses the content prompt.
            assert rlm.deliverable_slots is None
            assert rlm._slot_mode is False
            assert rlm.system_prompt is RLM_SYSTEM_PROMPT

            result = rlm.completion("q?")
            assert result.response == "hello"
            assert result.deliverables is None


class TestMultiDeliverableRLMSlotMode:
    def test_slot_protocol_populates_deliverables(self):
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = create_mock_lm(
                [_slot_final({"a.md": "AAA", "b.md": "BBB"})]
            )
            rlm = MultiDeliverableRLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_depth=1,
                deliverable_slots=["a.md", "b.md"],
            )
            assert rlm.deliverable_slots == ["a.md", "b.md"]
            assert rlm._slot_mode is True
            assert rlm.system_prompt is RLM_SYSTEM_PROMPT_SLOTS

            result = rlm.completion("q?")
            assert result.deliverables == {"a.md": "AAA", "b.md": "BBB"}
            # Multi-slot response is the rendered concat.
            assert "a.md" in result.response and "b.md" in result.response

    def test_single_slot_response_equals_content(self):
        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = create_mock_lm(
                [_slot_final({"answer": "just this"})]
            )
            rlm = MultiDeliverableRLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_depth=1,
                deliverable_slots=["answer"],
            )
            result = rlm.completion("q?")
            assert result.deliverables == {"answer": "just this"}
            # Single slot renders exactly like the old content.
            assert result.response == "just this"

    def test_requires_slots(self):
        with pytest.raises(ValueError, match="requires a non-empty"):
            MultiDeliverableRLM(
                backend="openai",
                backend_kwargs={"model_name": "test-model"},
                max_depth=1,
            )
