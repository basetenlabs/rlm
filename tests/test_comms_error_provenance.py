"""Failed helper outcomes retain the same evidence as successful outcomes."""

from rlm.core import comms_utils
from rlm.core.comms_utils import LMResponse
from rlm.core.types import RLMChatCompletion, UsageSummary


def failed_completion():
    return RLMChatCompletion(
        root_model="leaf",
        prompt="exact input",
        response="",
        error="provider timeout",
        execution_time=1.25,
        usage_summary=UsageSummary(model_usage_summaries={}),
        metadata={"usage_status": "unknown"},
    )


def test_single_error_roundtrip_retains_completion():
    completion = failed_completion()
    response = LMResponse(error=completion.error, chat_completion=completion)
    restored = LMResponse.from_dict(response.to_dict())
    assert not restored.success
    assert restored.chat_completion.to_dict() == completion.to_dict()


def test_batched_error_retains_completion(monkeypatch):
    completion = failed_completion()
    monkeypatch.setattr(
        comms_utils,
        "socket_request",
        lambda *args: LMResponse.batched_success_response([completion]).to_dict(),
    )
    responses = comms_utils.send_lm_request_batched(("localhost", 1), ["exact input"])
    assert len(responses) == 1
    assert not responses[0].success
    assert responses[0].chat_completion.to_dict() == completion.to_dict()
