import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

import anthropic

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


def _maybe_dump_reasoning(thinking_text: str, content_text: str, model: str) -> None:
    """Opt-in reasoning capture (set RLM_REASONING_DUMP=<path>).

    Anthropic adaptive thinking returns summarized reasoning in `thinking`
    content blocks that never reach the trajectory (history keeps text only,
    matching the chat-template behavior of served reasoning models). Persist
    one JSONL row per call so trajectories can be joined (by call order) with
    the reasoning that produced each response. Same row shape as the OpenAI
    client's dump ({model, reasoning, content_head}).
    """
    path = os.environ.get("RLM_REASONING_DUMP")
    if not path or not thinking_text:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps({
                "model": model,
                "reasoning": thinking_text,
                "content_head": content_text[:80],
            }, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass  # never let debug capture break a rollout


class AnthropicClient(BaseLM):
    """
    LM Client for running models with the Anthropic API.

    Notes for RLM use (long append-only conversations, e.g. force-read):
    - Prompt caching: the RLM resends the full cumulative history every
      iteration. We mark the system prompt and the last message block with
      `cache_control` so each iteration re-reads the previous prefix at
      ~0.1x input price instead of re-paying it in full. Messages are
      copied before annotation — the caller's history dicts are never
      mutated, so breakpoints don't accumulate across iterations.
    - Thinking: `enable_thinking=True` (default) requests adaptive thinking
      with summarized display. On Opus 4.7+ sampling params (temperature/
      top_p/top_k) are rejected by the API — this client never sends them.
    - Streaming: responses are streamed and joined via get_final_message()
      so long generations don't hit the SDK's non-streaming timeout guard.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
        max_tokens: int = 32768,
        max_retries: int = 8,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = anthropic.Anthropic(
            api_key=api_key, timeout=self.timeout, max_retries=max_retries)
        self.async_client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=self.timeout, max_retries=max_retries)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

    def _request_kwargs(
        self, prompt: str | list[dict[str, Any]], model: str | None
    ) -> tuple[dict[str, Any], str]:
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            kwargs["system"] = [{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        if self.enable_thinking:
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return kwargs, model

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> tuple[str, str]:
        """Return (text, thinking_summary). With thinking enabled the first
        content block is a thinking block — never assume content[0] is text."""
        text_parts, thinking_parts = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking" and getattr(block, "thinking", None):
                thinking_parts.append(block.thinking)
        return "".join(text_parts), "\n".join(thinking_parts)

    # SDK max_retries covers only the initial request; a 5xx/429/529 arriving
    # MID-STREAM raises through get_final_message() unretried (observed live:
    # api_error 500, then overloaded_error 529 storms). 529 storms against the
    # ~200K-token force-read prefills last 10+ minutes even single-stream, so
    # ride them out: exponential 15s..240s capped at 300s, ~23 min total,
    # before giving up and burning an RLM iteration.
    _STREAM_RETRIES = 8
    _BACKOFF_BASE_S = 15
    _BACKOFF_CAP_S = 300

    @staticmethod
    def _stream_retryable(e: Exception) -> bool:
        status = getattr(e, "status_code", None)
        return isinstance(e, anthropic.APIStatusError) and (status is None or status >= 429)

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        kwargs, model = self._request_kwargs(prompt, model)
        for attempt in range(self._STREAM_RETRIES + 1):
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                break
            except Exception as e:
                if attempt >= self._STREAM_RETRIES or not self._stream_retryable(e):
                    raise
                wait = min(2 ** attempt * self._BACKOFF_BASE_S, self._BACKOFF_CAP_S)
                print(f"[anthropic-retry] attempt {attempt + 1}/{self._STREAM_RETRIES} "
                      f"status={getattr(e, 'status_code', '?')} sleeping {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        self._track_cost(response, model)
        text, thinking = self._extract_text(response)
        _maybe_dump_reasoning(thinking, text, model)
        return text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        kwargs, model = self._request_kwargs(prompt, model)
        for attempt in range(self._STREAM_RETRIES + 1):
            try:
                async with self.async_client.messages.stream(**kwargs) as stream:
                    response = await stream.get_final_message()
                break
            except Exception as e:
                if attempt >= self._STREAM_RETRIES or not self._stream_retryable(e):
                    raise
                wait = min(2 ** attempt * self._BACKOFF_BASE_S, self._BACKOFF_CAP_S)
                print(f"[anthropic-retry] attempt {attempt + 1}/{self._STREAM_RETRIES} "
                      f"status={getattr(e, 'status_code', '?')} sleeping {wait}s",
                      file=sys.stderr, flush=True)
                await asyncio.sleep(wait)
        self._track_cost(response, model)
        text, thinking = self._extract_text(response)
        _maybe_dump_reasoning(thinking, text, model)
        return text

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Prepare messages and extract system prompt for Anthropic API.

        Copies every message (never mutates the caller's history — the RLM
        reuses its message list across iterations) and puts a cache_control
        breakpoint on the last message's content so the next iteration's
        prefix is served from cache.
        """
        system = None

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Extract system message if present (Anthropic handles system separately)
            messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    system = msg.get("content")
                else:
                    messages.append({"role": msg.get("role"), "content": msg.get("content")})
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        if messages:
            last = messages[-1]
            content = last["content"]
            if isinstance(content, str):
                last["content"] = [{
                    "type": "text", "text": content,
                    "cache_control": {"type": "ephemeral"},
                }]

        return messages, system

    def _track_cost(self, response: anthropic.types.Message, model: str):
        usage = response.usage
        # usage.input_tokens excludes cached tokens; total prompt size is the
        # sum of fresh + cache-written + cache-read. Record the sum so RLM
        # token accounting reflects what the model actually consumed.
        input_total = (
            usage.input_tokens
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += input_total
        self.model_output_tokens[model] += usage.output_tokens
        self.model_total_tokens[model] += input_total + usage.output_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = input_total
        self.last_completion_tokens = usage.output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
