"""System 1: the fast, intuitive model.

System 1 is a small non-reasoning model that answers via native tool calling.
Given a multiplication problem it either commits to an ``answer`` (when highly
confident) or, in hybrid mode, hands the problem off with ``escalate``. Parsing
is deliberately forgiving: model quirks fall back to scraping an integer from the
reply rather than raising.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .llm_client import ChatResult, OpenRouterClient
from .schemas import Problem

# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI / OpenRouter function-calling format)
# --------------------------------------------------------------------------- #
ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Give the final product when highly confident it is exactly correct.",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "integer",
                    "description": "The exact integer product.",
                },
            },
            "required": ["result"],
        },
    },
}

ESCALATE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "escalate",
        "description": "Hand the problem to the slower, more careful System 2.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the problem needs deliberate calculation.",
                },
            },
            "required": [],
        },
    },
}

_INT_RE = re.compile(r"-?\d+")


def _last_int(text: Optional[str]) -> Optional[int]:
    """Return the last integer appearing in ``text``, or None if there is none."""
    if not text:
        return None
    matches = _INT_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except (ValueError, TypeError):
        return None


@dataclass
class S1Outcome:
    """The result of one System-1 turn."""

    decision: str  # "answer" | "escalate"
    answer: Optional[int]
    chat: ChatResult


def run_system1(
    client: OpenRouterClient,
    problem: Problem,
    *,
    allow_escalate: bool,
    model: str = config.SYSTEM1_MODEL,
) -> S1Outcome:
    """Run System 1 on ``problem`` and parse its tool call into an S1Outcome.

    When ``allow_escalate`` is True the model may answer or escalate; otherwise it
    is forced to always answer (pure-System-1 baseline). Never raises on model
    quirks: unparseable output falls back to an integer scraped from the text.
    """
    system_prompt = config.S1_SYSTEM_PROMPT if allow_escalate else config.S1_FORCED_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"What is {problem.text}?"},
    ]
    tools = [ANSWER_TOOL, ESCALATE_TOOL] if allow_escalate else [ANSWER_TOOL]

    chat = client.chat(
        model,
        messages,
        tools=tools,
        tool_choice="required",
        temperature=config.S1_TEMPERATURE,
        max_tokens=config.S1_MAX_TOKENS,
    )

    tc = chat.first_tool_call
    if tc is not None:
        if tc.name == "answer":
            try:
                return S1Outcome("answer", int(tc.arguments["result"]), chat)
            except (KeyError, ValueError, TypeError):
                pass  # malformed args -> fall through to the text fallback
        elif tc.name == "escalate":
            return S1Outcome("escalate", None, chat)

    # Robust fallback: no usable tool call.
    scraped = _last_int(chat.text)
    if scraped is not None:
        return S1Outcome("answer", scraped, chat)
    if allow_escalate:
        return S1Outcome("escalate", None, chat)
    return S1Outcome("answer", None, chat)
