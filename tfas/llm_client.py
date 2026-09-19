"""Thin OpenRouter chat client with usage, cost, and latency accounting.

A single entry point, ``OpenRouterClient.chat(...)``, returns a ``ChatResult``
carrying the text, any tool calls, token usage, dollar cost, and wall-clock
latency for the call. Cost is taken from OpenRouter's own accounting when
available (``usage.include``) and otherwise computed from ``config.PRICING``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # parsed from JSON; {} if unparseable


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_s: float
    model: str
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None  # reasoning trace when the model exposes one
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def first_tool_call(self) -> Optional[ToolCall]:
        return self.tool_calls[0] if self.tool_calls else None


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    """Minimal synchronous client for the OpenRouter chat-completions endpoint."""

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or config.get_api_key()
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning: Optional[dict] = None,
        extra_body: Optional[dict] = None,
    ) -> ChatResult:
        """Make one chat-completions call and return a ChatResult.

        Retries on 429/5xx with exponential backoff (config.MAX_RETRIES).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # Ask OpenRouter to include its own cost/usage accounting.
            "usage": {"include": True},
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **config.OPENROUTER_HEADERS,
        }

        last_err: Optional[Exception] = None
        for attempt in range(config.MAX_RETRIES):
            start = time.perf_counter()
            try:
                resp = self.session.post(
                    config.OPENROUTER_CHAT_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=config.REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as exc:  # network error
                last_err = exc
                self._sleep(attempt)
                continue

            latency = time.perf_counter() - start

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise OpenRouterError(f"Non-JSON 200 response: {resp.text[:500]}") from exc
                # OpenRouter can return an error object inside a 200 body.
                if "error" in data and not data.get("choices"):
                    last_err = OpenRouterError(str(data["error"]))
                    self._sleep(attempt)
                    continue
                return self._parse(data, model=model, latency=latency)

            if resp.status_code in (408, 409, 429) or resp.status_code >= 500:
                last_err = OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                self._sleep(attempt)
                continue

            # Non-retryable client error.
            raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        raise OpenRouterError(f"Request failed after {config.MAX_RETRIES} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(config.RETRY_BACKOFF_S * (2 ** attempt))

    @staticmethod
    def _parse(data: dict, *, model: str, latency: float) -> ChatResult:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        reasoning = message.get("reasoning")

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if not isinstance(args, dict):
                    args = {}
            except (ValueError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))

        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        cost = usage.get("cost")
        if cost is None:
            cost = OpenRouterClient._estimate_cost(model, prompt_tokens, completion_tokens)
        cost = float(cost or 0.0)

        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_s=latency,
            model=model,
            finish_reason=finish_reason,
            reasoning=reasoning,
            raw=data,
        )

    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = config.PRICING.get(model)
        if not price:
            return 0.0
        return prompt_tokens * price["prompt"] + completion_tokens * price["completion"]
