"""Request-local formats through the actual socket handler and mocked SDK boundary."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from threading import Barrier, Lock

import httpx
import pytest
from openai import BadRequestError
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.types.chat import ChatCompletion

from rlm.clients.openai import OpenAIClient
from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.lm_handler import LMHandler
from rlm.environments.local_repl import LocalREPL
from tests.mock_lm import MockLM


def schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "source_findings_fixture",
            "strict": True,
            "schema": {"type": "object", "properties": {"finding": {"type": "string"}}},
        },
    }


class ObservedGate:
    def __init__(self):
        self.lock = Lock()
        self.active = 0
        self.entries = 0

    @contextmanager
    def slot(self):
        with self.lock:
            self.active += 1
            self.entries += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def sdk(monkeypatch):
    gate = ObservedGate()
    monkeypatch.setattr("rlm.utils.global_gate.get_gate", lambda: gate)
    state = {"requests": [], "barrier": None, "mutate": False, "gate": gate}
    lock = Lock()

    def finish(kwargs, interface):
        with lock:
            state["requests"].append((interface, deepcopy(kwargs), gate.active))
        if state["mutate"] and "response_format" in kwargs:
            kwargs["response_format"]["json_schema"]["name"] = "sdk-mutated-copy"
        prompt = kwargs["messages"][0]["content"]
        if prompt == "reject-format":
            raise BadRequestError(
                "fixture unsupported schema",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "https://fixture.invalid")
                ),
                body={"error": "fixture unsupported schema"},
            )
        return ChatCompletion(
            id="fixture",
            created=0,
            model=kwargs["model"],
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "unchanged " + prompt},
                }
            ],
            usage={"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        )

    def sync_create(resource, **kwargs):
        if state["barrier"]:
            state["barrier"].wait(timeout=5)
        return finish(kwargs, "sync")

    async def async_create(resource, **kwargs):
        if state["barrier"]:
            await asyncio.to_thread(state["barrier"].wait, 5)
        return finish(kwargs, "async")

    monkeypatch.setattr(Completions, "create", sync_create)
    monkeypatch.setattr(AsyncCompletions, "create", async_create)
    client = OpenAIClient(
        api_key="fixture-key",
        model_name="qwen-fixture",
        base_url="https://fixture.invalid/v1",
        sampling_args={
            "temperature": 0.6,
            "max_tokens": 32768,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    state["client"] = client
    yield state
    client.client.close()
    if not client.async_client.is_closed():
        asyncio.run(client.async_client.close())


def test_request_format_round_trip_is_detached_and_default_wire_unchanged():
    original = schema()
    request = LMRequest(prompt="source", model="qwen-fixture", depth=1, response_format=original)
    expected = deepcopy(original)
    original["json_schema"]["name"] = "caller-mutated"
    wire = request.to_dict()
    assert wire["response_format"] == expected
    restored = LMRequest.from_dict(wire)
    wire["response_format"]["json_schema"]["name"] = "wire-mutated"
    assert restored.response_format == expected
    assert request.response_format == expected
    assert LMRequest(prompt="ordinary").to_dict() == {"prompt": "ordinary", "depth": 0}
    assert LMRequest.from_dict({"prompt": "old-peer"}).response_format is None


@pytest.mark.parametrize("invalid", ["json_object", [], 1, False])
def test_non_dict_format_refused(invalid):
    with pytest.raises((TypeError, ValueError), match="response_format"):
        LMRequest(prompt="source", response_format=invalid)


@pytest.mark.parametrize("batched", [False, True])
def test_local_repl_transport_retains_outputs_calls_usage_and_gate(sdk, batched):
    client = sdk["client"]
    prompts = ["first", "last"] if batched else ["single"]
    function = "llm_query_batched" if batched else "llm_query"
    argument = prompts if batched else prompts[0]
    with LMHandler(client) as handler, LocalREPL(lm_handler_address=handler.address) as environment:
        result = environment.execute_code(
            f"result = {function}({argument!r}, model='qwen-fixture', response_format={schema()!r})\n"
            "print(result)"
        )
    assert not result.stderr
    assert [call.response for call in result.rlm_calls] == ["unchanged " + p for p in prompts]
    assert [call.prompt for call in result.rlm_calls] == prompts
    assert all(call.root_model == "qwen-fixture" for call in result.rlm_calls)
    assert sum(
        usage.total_input_tokens
        for call in result.rlm_calls
        for usage in call.usage_summary.model_usage_summaries.values()
    ) == 13 * len(prompts)
    usage = client.get_usage_summary().model_usage_summaries["qwen-fixture"]
    assert (usage.total_calls, usage.total_input_tokens, usage.total_output_tokens) == (
        len(prompts),
        13 * len(prompts),
        7 * len(prompts),
    )
    assert sdk["gate"].entries == len(prompts)
    assert sdk["gate"].active == 0
    assert all(
        active > 0 and request["response_format"] == schema()
        for _, request, active in sdk["requests"]
    )


def test_concurrent_sync_async_ordinary_and_structured_calls_are_isolated(sdk):
    sdk["barrier"], sdk["mutate"] = Barrier(4), True
    client = sdk["client"]
    sampling_before, selected = deepcopy(client.sampling_args), schema()
    with LMHandler(client) as handler, ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(
                send_lm_request,
                handler.address,
                LMRequest(prompt="structured-sync", response_format=selected),
            ),
            pool.submit(send_lm_request, handler.address, LMRequest(prompt="ordinary-sync")),
            pool.submit(
                send_lm_request_batched,
                handler.address,
                ["structured-async"],
                response_format=selected,
            ),
            pool.submit(send_lm_request_batched, handler.address, ["ordinary-async"]),
        ]
        results = [job.result(timeout=10) for job in jobs]
    assert all((result[0] if isinstance(result, list) else result).success for result in results)
    assert len(sdk["requests"]) == sdk["gate"].entries == 4
    assert sdk["gate"].active == 0
    assert selected == schema() and client.sampling_args == sampling_before
    for interface, request, active in sdk["requests"]:
        prompt = request["messages"][0]["content"]
        assert active > 0
        assert interface == ("async" if prompt.endswith("async") else "sync")
        assert (request.get("response_format") == schema()) == prompt.startswith("structured")
        if prompt.startswith("ordinary"):
            assert "response_format" not in request
        assert request["temperature"] == 0.6
        assert request["max_completion_tokens"] == 32768
        assert "max_tokens" not in request
        assert request["extra_body"] == sampling_before["extra_body"]


@pytest.mark.parametrize("batched", [False, True])
def test_unsupported_client_rejects_without_unformatted_fallback(monkeypatch, batched):
    gate = ObservedGate()
    monkeypatch.setattr("rlm.utils.global_gate.get_gate", lambda: gate)
    client = MockLM()
    with LMHandler(client) as handler:
        if batched:
            rejected = send_lm_request_batched(
                handler.address, ["a", "b"], response_format=schema()
            )
        else:
            rejected = [
                send_lm_request(handler.address, LMRequest(prompt="a", response_format=schema()))
            ]
        ordinary = send_lm_request(handler.address, LMRequest(prompt="ordinary"))
    assert all(not result.success and "response_format" in result.error for result in rejected)
    assert ordinary.success
    assert client._call_count == 1
    assert gate.entries == len(rejected) + 1 and gate.active == 0


def test_provider_rejection_preserves_each_outcome_without_fallback(sdk):
    with LMHandler(sdk["client"]) as handler:
        single = send_lm_request(
            handler.address, LMRequest(prompt="reject-format", response_format=schema())
        )
        batch = send_lm_request_batched(
            handler.address, ["reject-format", "success"], response_format=schema()
        )
    assert not single.success and "fixture unsupported schema" in single.error
    assert not batch[0].success and "fixture unsupported schema" in batch[0].error
    assert batch[1].success and batch[1].chat_completion.response == "unchanged success"
    assert len(sdk["requests"]) == sdk["gate"].entries == 3
    assert all(request["response_format"] == schema() for _, request, _ in sdk["requests"])
    assert sdk["gate"].active == 0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_client_copies_override_without_altering_backend_default(sdk, asynchronous):
    client, selected = sdk["client"], schema()
    backend_default = {"type": "json_object"}
    client.sampling_args["response_format"] = deepcopy(backend_default)
    sdk["mutate"] = True
    if asynchronous:
        asyncio.run(client.acompletion("override", response_format=selected))
    else:
        client.completion("override", response_format=selected)
    assert selected == schema()
    assert client.sampling_args["response_format"] == backend_default
    sdk["mutate"] = False
    client.completion("default")
    assert sdk["requests"][0][1]["response_format"] == schema()
    assert sdk["requests"][1][1]["response_format"] == backend_default


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_client_rejects_non_dict_before_sdk_dispatch(sdk, asynchronous):
    with pytest.raises(TypeError, match="response_format"):
        if asynchronous:
            asyncio.run(sdk["client"].acompletion("source", response_format="json_object"))
        else:
            sdk["client"].completion("source", response_format="json_object")
    assert sdk["requests"] == []
