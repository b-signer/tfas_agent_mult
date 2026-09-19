"""Tests for tfas.analysis (no network, synthetic results only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfas.analysis import make_figures, save_summary, summarize
from tfas.schemas import SolveResult


def _r(pid, difficulty, route, correct, *, tokens=0, cost=0.0, latency=0.0,
       error=None, memory_hit=False, escalated=False):
    """Build a SolveResult; operands are cosmetic, difficulty/route drive metrics."""
    a, b = (3, 4) if difficulty == "easy" else (37, 48)
    gt = a * b
    return SolveResult(
        problem_id=pid, a=a, b=b, difficulty=difficulty, ground_truth=gt,
        predicted=gt if correct else gt + 1, correct=correct, route=route,
        escalated=escalated, memory_hit=memory_hit, total_tokens=tokens,
        cost_usd=cost, latency_s=latency, error=error,
    )


def _synthetic():
    """Hand-made results with known aggregates (see the assertions below)."""
    hybrid = [
        _r(0, "easy", "memory", True, tokens=0, cost=0.0, latency=0.0, memory_hit=True),
        _r(1, "easy", "system1", True, tokens=100, cost=0.0010, latency=0.5),
        _r(2, "easy", "system1", True, tokens=110, cost=0.0011, latency=0.6),
        _r(3, "easy", "system1", False, tokens=120, cost=0.0012, latency=0.7),  # wrong answer
        _r(4, "hard", "system2", True, tokens=2000, cost=0.0200, latency=10.0, escalated=True),
        _r(5, "hard", "system2", False, tokens=1800, cost=0.0180, latency=9.0,
           escalated=True, error="timeout"),  # wrong + error
    ]
    pure_system1 = [
        _r(0, "easy", "system1", True, tokens=50, cost=0.0005, latency=0.4),
        _r(1, "hard", "system1", False, tokens=60, cost=0.0006, latency=0.5),
    ]
    pure_system2 = [
        _r(0, "easy", "system2", True, tokens=1500, cost=0.015, latency=8.0),
        _r(1, "hard", "system2", True, tokens=1700, cost=0.017, latency=9.0),
    ]
    return {
        "pure_system1": pure_system1,
        "pure_system2": pure_system2,
        "hybrid": hybrid,
    }


def test_summarize_metrics():
    results = _synthetic()
    metadata = {"configs": {"hybrid": {"wall_clock_s": 42.5}}}
    df = summarize(results, metadata)

    # Index preserved in insertion order.
    assert list(df.index) == ["pure_system1", "pure_system2", "hybrid"]

    h = df.loc["hybrid"]
    # 4 of 6 correct overall; easy 3/4; hard 1/2.
    assert h["accuracy"] == pytest.approx(4 / 6)
    assert h["accuracy_easy"] == pytest.approx(0.75)
    assert h["accuracy_hard"] == pytest.approx(0.5)
    assert h["n"] == 6
    assert h["n_easy"] == 4
    assert h["n_hard"] == 2
    # 1 memory route of 6; 2 system2 routes of 6.
    assert h["memory_hit_rate"] == pytest.approx(1 / 6)
    assert h["escalation_rate"] == pytest.approx(2 / 6)
    assert h["n_system2_calls"] == 2
    assert h["n_errors"] == 1
    assert h["total_tokens"] == 4130
    assert h["total_cost_usd"] == pytest.approx(0.0413)
    assert h["total_latency_s"] == pytest.approx(20.8)
    assert h["mean_latency_s"] == pytest.approx(20.8 / 6)
    assert h["wall_clock_s"] == pytest.approx(42.5)

    # Pure baselines: no memory, escalation_rate == fraction of system2 routes.
    p1 = df.loc["pure_system1"]
    assert p1["memory_hit_rate"] == 0.0
    assert p1["escalation_rate"] == 0.0
    assert p1["n_system2_calls"] == 0
    assert p1["accuracy"] == pytest.approx(0.5)

    p2 = df.loc["pure_system2"]
    assert p2["memory_hit_rate"] == 0.0
    assert p2["escalation_rate"] == pytest.approx(1.0)  # every problem is system2
    assert p2["n_system2_calls"] == 2

    # wall_clock_s is NaN where metadata is missing.
    assert math.isnan(p1["wall_clock_s"])


def test_summarize_without_metadata():
    df = summarize(_synthetic())
    assert math.isnan(df.loc["hybrid"]["wall_clock_s"])


def test_save_summary_writes_files(tmp_path):
    df = summarize(_synthetic())
    paths = save_summary(df, results_dir=tmp_path)
    for kind in ("csv", "md", "tex"):
        p = paths[kind]
        assert p.exists() and p.stat().st_size > 0
    tex = paths["tex"].read_text(encoding="utf-8")
    assert r"\label{tab:summary}" in tex
    assert r"\begin{table}" in tex


def test_make_figures_writes_all_pngs(tmp_path):
    results = _synthetic()
    df = summarize(results)
    figures = make_figures(results, df, out_dir=tmp_path)

    expected = {
        "accuracy_by_config.png", "cost_by_config.png", "tokens_by_config.png",
        "latency_by_config.png", "route_composition.png", "accuracy_vs_cost.png",
        "cumulative_cost.png",
    }
    assert {p.name for p in figures} == expected
    for name in expected:
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 0
