"""get_usage_summary must not lose usage when two clients share a model name.

A root backend and a depth-1 "other backend" may serve the SAME model with
different sampling args (e.g. root thinking-on temp 1.0, sub thinking-off
temp 0.6). Merging their summaries with dict.update() dropped one side —
the whole-run token totals silently undercounted.
"""

from unittest.mock import patch

from rlm.core.lm_handler import LMHandler
from rlm.core.rlm import RLM
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.utils.exceptions import TokenLimitExceededError

from .mock_lm import MockLM


def _calls(client: MockLM, n: int) -> None:
    for i in range(n):
        client.completion(f"prompt {i}")


def test_same_name_root_and_sub_usage_is_summed():
    root = MockLM(model_name="shared/model")
    sub = MockLM(model_name="shared/model")
    handler = LMHandler(root, other_backend_client=sub)
    # Mirror rlm.py's registration of the other backend by model name.
    handler.register_client(sub.model_name, sub)

    _calls(root, 3)
    _calls(sub, 5)

    merged = handler.get_usage_summary().model_usage_summaries
    assert merged["shared/model"].total_calls == 8
    assert merged["shared/model"].total_input_tokens == 80
    assert merged["shared/model"].total_output_tokens == 80


def test_distinct_names_unchanged():
    root = MockLM(model_name="root/model")
    sub = MockLM(model_name="sub/model")
    handler = LMHandler(root, other_backend_client=sub)
    handler.register_client(sub.model_name, sub)

    _calls(root, 2)
    _calls(sub, 4)

    merged = handler.get_usage_summary().model_usage_summaries
    assert merged["root/model"].total_calls == 2
    assert merged["sub/model"].total_calls == 4


def test_default_client_not_double_counted():
    # __init__ also registers the default client in `clients`; identity dedupe
    # must keep it counted exactly once.
    root = MockLM(model_name="only/model")
    handler = LMHandler(root)

    _calls(root, 3)

    merged = handler.get_usage_summary().model_usage_summaries
    assert merged["only/model"].total_calls == 3


def test_recursive_child_usage_is_merged_into_parent_total():
    root = MockLM(model_name="root/model")
    handler = LMHandler(root)
    _calls(root, 2)
    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root/model"},
        max_depth=2,
    )
    parent._record_recursive_usage(
        UsageSummary(
            model_usage_summaries={
                "child/model": ModelUsageSummary(3, 300, 30),
                "leaf/model": ModelUsageSummary(5, 500, 50),
            }
        )
    )

    merged = parent._combined_usage(handler).model_usage_summaries

    assert merged["root/model"].total_calls == 2
    assert merged["child/model"].total_input_tokens == 300
    assert merged["leaf/model"].total_output_tokens == 50
    parent.close()


def test_recursive_same_model_usage_adds_to_root_instead_of_overwriting():
    root = MockLM(model_name="shared/model")
    handler = LMHandler(root)
    _calls(root, 2)
    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "shared/model"},
        max_depth=2,
    )
    parent._record_recursive_usage(
        UsageSummary(
            model_usage_summaries={
                "shared/model": ModelUsageSummary(3, 300, 30),
            }
        )
    )

    merged = parent._combined_usage(handler).model_usage_summaries["shared/model"]

    assert merged.total_calls == 5
    assert merged.total_input_tokens == 320
    assert merged.total_output_tokens == 50
    parent.close()


def test_completed_subcall_records_child_root_and_leaf_usage_on_parent():
    child_usage = UsageSummary(
        model_usage_summaries={
            "child/model": ModelUsageSummary(2, 200, 20),
            "leaf/model": ModelUsageSummary(4, 400, 40),
        }
    )

    class FakeChild:
        def __init__(self, **_kwargs):
            pass

        def completion(self, prompt, root_prompt=None):
            return RLMChatCompletion(
                root_model="child/model",
                prompt=prompt,
                response="done",
                usage_summary=child_usage,
                execution_time=1.0,
            )

        def close(self):
            pass

    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root/model"},
        max_depth=2,
    )
    with patch("rlm.core.rlm.RLM", FakeChild):
        result = parent._subcall("work", model="child/model")

    assert result.response == "done"
    recorded = parent._recursive_usage.model_usage_summaries
    assert recorded["child/model"].total_input_tokens == 200
    assert recorded["leaf/model"].total_output_tokens == 40
    parent.close()


def test_limited_subcall_records_child_root_and_leaf_usage_on_parent():
    child_usage = UsageSummary(
        model_usage_summaries={
            "child/model": ModelUsageSummary(2, 200, 20),
            "leaf/model": ModelUsageSummary(4, 400, 40),
        }
    )

    class LimitedChild:
        def __init__(self, **_kwargs):
            pass

        def completion(self, _prompt, root_prompt=None):
            error = TokenLimitExceededError(660, 600, partial_answer="partial")
            error.usage_summary = child_usage
            raise error

        def close(self):
            pass

    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "root/model"},
        max_depth=2,
    )
    with patch("rlm.core.rlm.RLM", LimitedChild):
        result = parent._subcall("work", model="child/model")

    assert "Child RLM completion failed" in result.response
    assert result.usage_summary.total_input_tokens == 600
    assert result.usage_summary.total_output_tokens == 60
    recorded = parent._recursive_usage.model_usage_summaries
    assert recorded["child/model"].total_input_tokens == 200
    assert recorded["leaf/model"].total_output_tokens == 40
    parent.close()
