"""TFaSAgent: the memory -> System 1 -> System 2 orchestration.

One agent wraps a client, a memory, and an ExperimentConfig, and turns each
Problem into a SolveResult. The routing depends on ``config.mode``:

    s1_only  -- System 1 forced to answer.
    s2_only  -- System 2 only.
    hybrid   -- optional memory check -> System 1 (answer/escalate) ->
                System 2 on escalation, then write the answer back to memory.

Token, cost, and latency fields aggregate every LLM call actually made (a memory
hit contributes zero). ``solve`` never raises: failures are captured in the
result's ``error`` field.
"""
from __future__ import annotations

from . import config
from .llm_client import OpenRouterClient
from .memory import Memory
from .schemas import Problem, SolveResult
from .system1 import run_system1
from .system2 import run_system2


class TFaSAgent:
    """Solve multiplication problems under one ExperimentConfig."""

    def __init__(self, client: OpenRouterClient, memory: Memory, config: config.ExperimentConfig):
        self.client = client
        self.memory = memory
        self.config = config

    def solve(self, problem: Problem) -> SolveResult:
        """Route ``problem`` per the configured mode and return a SolveResult."""
        cfg = self.config
        chats = []  # every ChatResult produced, for token/cost/latency aggregation
        predicted = None
        route = "system1"  # best-effort default (updated as we progress)
        escalated = False
        memory_hit = False
        s1_decision = None
        s1_text = None
        s2_text = None
        error = None

        try:
            if cfg.mode == "s1_only":
                route = "system1"
                s1 = run_system1(self.client, problem, allow_escalate=False, model=cfg.system1_model)
                chats.append(s1.chat)
                s1_decision = "answer"
                s1_text = s1.chat.text or s1.chat.reasoning
                predicted = s1.answer

            elif cfg.mode == "s2_only":
                route = "system2"
                s2 = run_system2(self.client, problem, model=cfg.system2_model)
                chats.append(s2.chat)
                s2_text = s2.chat.text
                predicted = s2.answer

            elif cfg.mode == "hybrid":
                if cfg.use_memory:
                    cached = self.memory.get(problem.a, problem.b)
                    if cached is not None:
                        # Served from cache: zero model calls, negligible latency.
                        return SolveResult(
                            problem_id=problem.id,
                            a=problem.a,
                            b=problem.b,
                            difficulty=problem.difficulty,
                            ground_truth=problem.answer,
                            predicted=cached,
                            correct=(cached == problem.answer),
                            route="memory",
                            escalated=False,
                            memory_hit=True,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            cost_usd=0.0,
                            latency_s=0.0,
                            s1_decision=None,
                            s1_text=None,
                            s2_text=None,
                            error=None,
                        )

                route = "system1"
                s1 = run_system1(self.client, problem, allow_escalate=True, model=cfg.system1_model)
                chats.append(s1.chat)
                s1_decision = s1.decision
                s1_text = s1.chat.text or s1.chat.reasoning

                if s1.decision == "answer":
                    route = "system1"
                    escalated = False
                    predicted = s1.answer
                else:
                    route = "system2"
                    escalated = True
                    s2 = run_system2(self.client, problem, model=cfg.system2_model)
                    chats.append(s2.chat)
                    s2_text = s2.chat.text
                    predicted = s2.answer

                if cfg.use_memory and predicted is not None:
                    self.memory.put(problem.a, problem.b, predicted)

            else:
                raise ValueError(f"Unknown experiment mode: {cfg.mode!r}")

        except Exception as exc:  # never raise out of solve()
            error = str(exc)
            predicted = None

        prompt_tokens = sum(c.prompt_tokens for c in chats)
        completion_tokens = sum(c.completion_tokens for c in chats)
        total_tokens = sum(c.total_tokens for c in chats)
        cost_usd = sum(c.cost_usd for c in chats)
        latency_s = sum(c.latency_s for c in chats)

        return SolveResult(
            problem_id=problem.id,
            a=problem.a,
            b=problem.b,
            difficulty=problem.difficulty,
            ground_truth=problem.answer,
            predicted=predicted,
            correct=(predicted == problem.answer),
            route=route,
            escalated=escalated,
            memory_hit=memory_hit,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_s=latency_s,
            s1_decision=s1_decision,
            s1_text=s1_text,
            s2_text=s2_text,
            error=error,
        )
