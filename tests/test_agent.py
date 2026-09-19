"""Tests for System 1/2 parsing and the TFaSAgent routing.

The OpenRouter client is fully mocked: ``FakeClient`` returns hand-built
ChatResult objects from a scripted queue, so no network calls are made.
"""
from __future__ import annotations

import pytest

from tfas import config
from tfas.agent import TFaSAgent
from tfas.config import ExperimentConfig
from tfas.llm_client import ChatResult, ToolCall
from tfas.memory import Memory
from tfas.schemas import Problem
from tfas.system1 import run_system1
from tfas.system2 import run_system2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_chat(
    *,
    text="",
    tool_calls=None,
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=None,
    cost_usd=0.0,
    latency_s=0.0,
    model="fake",
    reasoning=None,
) -> ChatResult:
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return ChatResult(
        text=text,
        tool_calls=tool_calls or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_s=latency_s,
        model=model,
        reasoning=reasoning,
    )


def answer_chat(result, **kw) -> ChatResult:
    kw.setdefault("prompt_tokens", 10)
    kw.setdefault("completion_tokens", 5)
    kw.setdefault("cost_usd", 0.001)
    kw.setdefault("latency_s", 0.1)
    return make_chat(tool_calls=[ToolCall(id="c1", name="answer", arguments={"result": result})], **kw)


def escalate_chat(**kw) -> ChatResult:
    kw.setdefault("prompt_tokens", 10)
    kw.setdefault("completion_tokens", 5)
    kw.setdefault("cost_usd", 0.001)
    kw.setdefault("latency_s", 0.1)
    return make_chat(
        tool_calls=[ToolCall(id="c1", name="escalate", arguments={"reason": "too big"})], **kw
    )


def s2_chat(result, **kw) -> ChatResult:
    kw.setdefault("prompt_tokens", 100)
    kw.setdefault("completion_tokens", 200)
    kw.setdefault("cost_usd", 0.05)
    kw.setdefault("latency_s", 2.0)
    return make_chat(text=f"Working it out step by step...\nANSWER: {result}", **kw)


def malformed_chat(text, **kw) -> ChatResult:
    """No tool call at all; only free text."""
    return make_chat(text=text, tool_calls=[], **kw)


class FakeClient:
    """Mock OpenRouterClient: pops scripted responses and records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeClient.chat called more times than scripted")
        return self.responses.pop(0)


class MemorySpy(Memory):
    """Memory that records every put() call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.put_calls = []

    def put(self, a, b, value):
        self.put_calls.append((a, b, value))
        super().put(a, b, value)


PROB = Problem(id=0, a=7, b=8, difficulty="easy")  # answer 56


def cfg(mode, use_memory=False, name="t"):
    return ExperimentConfig(name=name, mode=mode, use_memory=use_memory)


def fresh_memory():
    return MemorySpy(path="/tmp/tfas_test_mem_unused.json", enabled=True, load=False)


# --------------------------------------------------------------------------- #
# Route: memory hit
# --------------------------------------------------------------------------- #
def test_memory_hit_returns_without_model_calls():
    mem = fresh_memory()
    mem.put(7, 8, 56)
    client = FakeClient([])  # any call would raise
    agent = TFaSAgent(client, mem, cfg("hybrid", use_memory=True))

    res = agent.solve(PROB)

    assert res.route == "memory"
    assert res.memory_hit is True
    assert res.predicted == 56
    assert res.correct is True
    assert res.escalated is False
    assert res.s1_decision is None
    assert res.total_tokens == 0 and res.cost_usd == 0.0 and res.latency_s == 0.0
    assert client.calls == []  # no model was invoked


# --------------------------------------------------------------------------- #
# Route: hybrid, System 1 answers
# --------------------------------------------------------------------------- #
def test_hybrid_system1_answers():
    client = FakeClient([answer_chat(56)])
    agent = TFaSAgent(client, fresh_memory(), cfg("hybrid", use_memory=False))

    res = agent.solve(PROB)

    assert res.route == "system1"
    assert res.escalated is False
    assert res.s1_decision == "answer"
    assert res.predicted == 56
    assert res.correct is True
    assert res.s2_text is None
    assert len(client.calls) == 1
    # Confidence-framed prompt + both tools offered.
    call = client.calls[0]
    assert call["messages"][0]["content"] == config.S1_SYSTEM_PROMPT
    assert {t["function"]["name"] for t in call["kwargs"]["tools"]} == {"answer", "escalate"}
    assert call["kwargs"]["tool_choice"] == "required"


# --------------------------------------------------------------------------- #
# Route: hybrid, System 1 escalates -> System 2
# --------------------------------------------------------------------------- #
def test_hybrid_escalate_to_system2_sums_tokens():
    client = FakeClient([escalate_chat(), s2_chat(56)])
    agent = TFaSAgent(client, fresh_memory(), cfg("hybrid", use_memory=False))

    res = agent.solve(PROB)

    assert res.route == "system2"
    assert res.escalated is True
    assert res.s1_decision == "escalate"
    assert res.predicted == 56
    assert res.correct is True
    assert res.s2_text is not None and "ANSWER: 56" in res.s2_text
    assert len(client.calls) == 2

    # Tokens/cost/latency are the SUM of both calls (10+100, 5+200, 15+300...).
    assert res.prompt_tokens == 110
    assert res.completion_tokens == 205
    assert res.total_tokens == 315
    assert res.cost_usd == pytest.approx(0.051)
    assert res.latency_s == pytest.approx(2.1)


# --------------------------------------------------------------------------- #
# Route: s1_only
# --------------------------------------------------------------------------- #
def test_s1_only_forced_answer():
    client = FakeClient([answer_chat(56)])
    agent = TFaSAgent(client, fresh_memory(), cfg("s1_only"))

    res = agent.solve(PROB)

    assert res.route == "system1"
    assert res.escalated is False
    assert res.s1_decision == "answer"
    assert res.predicted == 56
    # Forced prompt, only the answer tool offered.
    call = client.calls[0]
    assert call["messages"][0]["content"] == config.S1_FORCED_SYSTEM_PROMPT
    assert [t["function"]["name"] for t in call["kwargs"]["tools"]] == ["answer"]


# --------------------------------------------------------------------------- #
# Route: s2_only
# --------------------------------------------------------------------------- #
def test_s2_only():
    client = FakeClient([s2_chat(56)])
    agent = TFaSAgent(client, fresh_memory(), cfg("s2_only"))

    res = agent.solve(PROB)

    assert res.route == "system2"
    assert res.predicted == 56
    assert res.correct is True
    assert res.s1_decision is None
    assert res.s1_text is None
    # System 2 gets the configured provider passed through extra_body.
    call = client.calls[0]
    assert call["messages"][0]["content"] == config.S2_SYSTEM_PROMPT
    assert call["kwargs"]["extra_body"] == {"provider": config.S2_PROVIDER}


# --------------------------------------------------------------------------- #
# Correctness flag
# --------------------------------------------------------------------------- #
def test_correct_flag_false_on_wrong_answer():
    client = FakeClient([answer_chat(55)])  # 7x8 = 56, not 55
    agent = TFaSAgent(client, fresh_memory(), cfg("s1_only"))

    res = agent.solve(PROB)

    assert res.predicted == 55
    assert res.correct is False


# --------------------------------------------------------------------------- #
# Memory write-back after a hybrid solve
# --------------------------------------------------------------------------- #
def test_memory_put_called_after_hybrid_solve():
    mem = fresh_memory()
    client = FakeClient([answer_chat(56)])
    agent = TFaSAgent(client, mem, cfg("hybrid", use_memory=True))

    res = agent.solve(PROB)

    assert res.route == "system1"
    assert res.memory_hit is False
    assert mem.put_calls == [(7, 8, 56)]
    assert mem.get(7, 8) == 56


def test_memory_not_written_when_use_memory_false():
    mem = fresh_memory()
    client = FakeClient([answer_chat(56)])
    agent = TFaSAgent(client, mem, cfg("hybrid", use_memory=False))

    agent.solve(PROB)

    assert mem.put_calls == []


# --------------------------------------------------------------------------- #
# run_system1 malformed-tool-call fallback
# --------------------------------------------------------------------------- #
def test_run_system1_fallback_scrapes_integer():
    client = FakeClient([malformed_chat("I think the product is 56.")])
    outcome = run_system1(client, PROB, allow_escalate=True)
    assert outcome.decision == "answer"
    assert outcome.answer == 56


def test_run_system1_fallback_escalates_when_no_integer():
    client = FakeClient([malformed_chat("Hmm, that is tricky.")])
    outcome = run_system1(client, PROB, allow_escalate=True)
    assert outcome.decision == "escalate"
    assert outcome.answer is None


def test_run_system1_fallback_forced_answer_none_when_no_integer():
    client = FakeClient([malformed_chat("no digits here")])
    outcome = run_system1(client, PROB, allow_escalate=False)
    assert outcome.decision == "answer"
    assert outcome.answer is None


def test_run_system1_answer_with_bad_args_falls_back_to_text():
    bad = make_chat(
        text="The answer is 56",
        tool_calls=[ToolCall(id="c1", name="answer", arguments={})],  # missing result
    )
    client = FakeClient([bad])
    outcome = run_system1(client, PROB, allow_escalate=True)
    assert outcome.decision == "answer"
    assert outcome.answer == 56


# --------------------------------------------------------------------------- #
# run_system2 answer parsing
# --------------------------------------------------------------------------- #
def test_run_system2_prefers_last_answer_tag():
    client = FakeClient([make_chat(text="ANSWER: 12\nOops, recompute.\nANSWER: 56")])
    outcome = run_system2(client, PROB)
    assert outcome.answer == 56


def test_run_system2_falls_back_to_last_integer():
    client = FakeClient([make_chat(text="The product works out to 56")])
    outcome = run_system2(client, PROB)
    assert outcome.answer == 56


# --------------------------------------------------------------------------- #
# solve() never raises: errors are captured
# --------------------------------------------------------------------------- #
def test_solve_captures_error_instead_of_raising():
    class BoomClient:
        def chat(self, *a, **k):
            raise RuntimeError("network down")

    agent = TFaSAgent(BoomClient(), fresh_memory(), cfg("s2_only"))
    res = agent.solve(PROB)

    assert res.error == "network down"
    assert res.predicted is None
    assert res.correct is False
    assert res.route == "system2"
