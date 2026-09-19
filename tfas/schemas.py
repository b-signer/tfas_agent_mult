"""Shared data contract: Problem and SolveResult.

Every module (dataset, agent, runner, analysis) imports these so the shape of a
problem and of a per-problem result is defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

Difficulty = str  # "easy" | "hard"


@dataclass
class Problem:
    """A single multiplication problem.

    The ground-truth answer is derived (a * b) rather than stored, so it can
    never drift from the operands.
    """

    id: int
    a: int
    b: int
    difficulty: Difficulty  # "easy" (single-digit x single-digit) or "hard"

    @property
    def text(self) -> str:
        return f"{self.a} x {self.b}"

    @property
    def answer(self) -> int:
        return self.a * self.b

    @property
    def key(self) -> str:
        """Exact-match memory key for this problem as presented."""
        return f"{self.a}x{self.b}"

    def to_dict(self) -> dict:
        return {"id": self.id, "a": self.a, "b": self.b, "difficulty": self.difficulty}

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        return cls(id=int(d["id"]), a=int(d["a"]), b=int(d["b"]), difficulty=str(d["difficulty"]))


@dataclass
class SolveResult:
    """The outcome of solving one problem under one configuration.

    route is where the answer ultimately came from:
        "memory"   -- served from the deterministic cache (0 LLM calls)
        "system1"  -- answered by the small model
        "system2"  -- answered by the reasoning model (after escalation or in s2_only)
    Token/cost/latency fields aggregate every LLM call made for this problem
    (a memory hit contributes zero).
    """

    problem_id: int
    a: int
    b: int
    difficulty: Difficulty
    ground_truth: int
    predicted: Optional[int]
    correct: bool
    route: str
    escalated: bool = False
    memory_hit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    # System-1 escalation bookkeeping (present even when memory served the answer).
    s1_decision: Optional[str] = None  # "answer" | "escalate" | None
    s1_text: Optional[str] = None
    s2_text: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SolveResult":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})
