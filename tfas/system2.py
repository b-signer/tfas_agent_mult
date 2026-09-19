"""System 2: the slow, deliberate reasoning model.

System 2 reasons step by step and ends its reply with a line ``ANSWER: <int>``.
Parsing prefers the last such tagged line and, failing that, the last integer
anywhere in the text, so a well-behaved reply and a rambling one both yield an
answer when one is present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .llm_client import ChatResult, OpenRouterClient
from .schemas import Problem

_ANSWER_RE = re.compile(r"ANSWER:\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def _parse_answer(text: Optional[str]) -> Optional[int]:
    """Extract the final integer answer from a System-2 reply.

    Prefers the LAST ``ANSWER: <int>`` tag; falls back to the last integer
    anywhere in the text; returns None if neither is present.
    """
    if not text:
        return None
    tagged = _ANSWER_RE.findall(text)
    if tagged:
        try:
            return int(tagged[-1])
        except (ValueError, TypeError):
            pass
    ints = _INT_RE.findall(text)
    if ints:
        try:
            return int(ints[-1])
        except (ValueError, TypeError):
            pass
    return None


@dataclass
class S2Outcome:
    """The result of one System-2 turn."""

    answer: Optional[int]
    chat: ChatResult


def run_system2(
    client: OpenRouterClient,
    problem: Problem,
    *,
    model: str = config.SYSTEM2_MODEL,
) -> S2Outcome:
    """Run System 2 on ``problem`` and parse its final answer."""
    messages = [
        {"role": "system", "content": config.S2_SYSTEM_PROMPT},
        {"role": "user", "content": f"Compute {problem.text}."},
    ]
    chat = client.chat(
        model,
        messages,
        temperature=config.S2_TEMPERATURE,
        max_tokens=config.S2_MAX_TOKENS,
        extra_body={"provider": config.S2_PROVIDER},
    )
    return S2Outcome(_parse_answer(chat.text), chat)
