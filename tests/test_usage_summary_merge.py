"""get_usage_summary must not lose usage when two clients share a model name.

A root backend and a depth-1 "other backend" may serve the SAME model with
different sampling args (e.g. root thinking-on temp 1.0, sub thinking-off
temp 0.6). Merging their summaries with dict.update() dropped one side —
the whole-run token totals silently undercounted.
"""

from rlm.core.lm_handler import LMHandler

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
