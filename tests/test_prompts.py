"""Tests for system-prompt assembly (addendum variants)."""

from rlm.core.types import QueryMetadata
from rlm.utils.prompts import build_rlm_system_prompt, RLM_SYSTEM_PROMPT


def _system_text(**kwargs):
    messages = build_rlm_system_prompt(
        system_prompt=RLM_SYSTEM_PROMPT,
        query_metadata=QueryMetadata("hello world"),
        **kwargs,
    )
    return messages[0]["content"]


class TestAddendumVariant:
    def test_default_orchestrator_addendum(self):
        text = _system_text()
        assert "orchestrator, not a solver" in text
        assert "Delegate everything else" in text

    def test_direct_read_variant_drops_delegation_pressure(self):
        text = _system_text(addendum_variant="direct-read")
        # delegation pressure gone
        assert "orchestrator, not a solver" not in text
        assert "Delegate everything else" not in text
        assert "Push every long-context operation" not in text
        assert "rather than reading them by hand" not in text
        # turn discipline kept
        assert "pause and plan" in text
        assert "submit your best inference" in text
        # neutral cross-check mention kept
        assert "optional cross-checks" in text

    def test_orchestrator_false_drops_all_addenda(self):
        text = _system_text(orchestrator=False, addendum_variant="direct-read")
        assert "pause and plan" not in text
        assert "orchestrator, not a solver" not in text
