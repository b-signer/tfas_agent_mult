"""Load raw experiment results, summarize them, and render paper-ready figures.

This module is the reporting layer: it never talks to the network and never runs
an experiment. It reads the per-config JSON files the runner wrote (see
``tfas.runner``), computes a per-config summary table, and draws a small set of
colorblind-safe figures for the white paper.

Everything is computed from the ``SolveResult`` records themselves; run metadata
is consulted only for ``wall_clock_s`` (a whole-config timing the per-problem
records cannot carry).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless: no display, must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .schemas import SolveResult

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- #
# Palette (validated colorblind-safe categorical slots from the dataviz skill)
# --------------------------------------------------------------------------- #
# A fixed, non-cycled hue order. One consistent color per config across every
# figure; routes get their own three-hue set (all clear the all-pairs gate).
_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

CONFIG_COLORS: dict[str, str] = {
    "pure_system1": "#2a78d6",       # blue
    "pure_system2": "#eb6834",       # orange
    "hybrid_no_memory": "#1baf7a",   # aqua
    "hybrid": "#eda100",             # yellow
}
ROUTE_COLORS: dict[str, str] = {
    "memory": "#1baf7a",   # aqua  -- served free from cache
    "system1": "#2a78d6",  # blue  -- fast, cheap
    "system2": "#eb6834",  # orange -- slow, expensive
}
CONFIG_LABELS: dict[str, str] = {
    "pure_system1": "Pure S1",
    "pure_system2": "Pure S2",
    "hybrid_no_memory": "Hybrid (no mem)",
    "hybrid": "Hybrid (mem)",
}

# Chart chrome / ink tokens (light surface).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

# Column order of the summary DataFrame.
SUMMARY_COLUMNS = [
    "accuracy", "accuracy_easy", "accuracy_hard",
    "n", "n_easy", "n_hard",
    "mean_latency_s", "total_latency_s", "wall_clock_s",
    "total_tokens", "total_cost_usd",
    "memory_hit_rate", "escalation_rate", "n_system2_calls", "n_errors",
]

# Per-column rounding for the rendered tables.
_ROUND = {
    "accuracy": 3, "accuracy_easy": 3, "accuracy_hard": 3,
    "mean_latency_s": 2, "total_latency_s": 2, "wall_clock_s": 2,
    "total_cost_usd": 4, "memory_hit_rate": 3, "escalation_rate": 3,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_all_results(results_dir: PathLike = config.RESULTS_DIR) -> dict[str, list[SolveResult]]:
    """Load per-config result files into ``{config_name: [SolveResult, ...]}``.

    Reads ``results/<name>.json`` for every config in ``config.EXPERIMENTS`` that
    has a file on disk (a partial run is fine). Records are rebuilt with
    ``SolveResult.from_dict`` and returned in file (stream) order.
    """
    results_dir = Path(results_dir)
    out: dict[str, list[SolveResult]] = {}
    for cfg in config.EXPERIMENTS:
        path = results_dir / f"{cfg.name}.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        out[cfg.name] = [SolveResult.from_dict(d) for d in raw]
    return out


def load_metadata(results_dir: PathLike = config.RESULTS_DIR) -> dict:
    """Return the parsed ``run_metadata.json`` if present, else an empty dict."""
    path = Path(results_dir) / "run_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def _accuracy(results: list[SolveResult], difficulty: Optional[str] = None) -> float:
    """Mean correctness over ``results`` (optionally restricted to a difficulty)."""
    subset = results if difficulty is None else [r for r in results if r.difficulty == difficulty]
    if not subset:
        return float("nan")
    return float(np.mean([1.0 if r.correct else 0.0 for r in subset]))


def summarize(
    results_by_config: dict[str, list[SolveResult]],
    metadata: Optional[dict] = None,
) -> pd.DataFrame:
    """Build a per-config summary table indexed by config name.

    Every metric is computed from the ``SolveResult`` records; ``metadata`` (the
    runner's ``run_metadata.json``) is used only to fill ``wall_clock_s``, which
    the per-problem records cannot carry. Rates are fractions of the whole stream:

    * ``memory_hit_rate`` -- share of problems with ``route == "memory"``.
    * ``escalation_rate`` -- share of ALL problems with ``route == "system2"``
      (so it is 1.0 for ``pure_system2`` and 0.0 for ``pure_system1``).

    Metrics that are not meaningful for a config fall out as ``0.0`` naturally
    (e.g. ``memory_hit_rate`` for the pure baselines has no ``memory`` routes).
    """
    metadata = metadata or {}
    meta_configs = metadata.get("configs", {})

    rows: dict[str, dict] = {}
    for name, results in results_by_config.items():
        n = len(results)
        n_easy = sum(1 for r in results if r.difficulty == "easy")
        n_hard = sum(1 for r in results if r.difficulty == "hard")
        n_mem = sum(1 for r in results if r.route == "memory")
        n_s2 = sum(1 for r in results if r.route == "system2")
        latencies = [r.latency_s for r in results]
        wall = meta_configs.get(name, {}).get("wall_clock_s", float("nan"))

        rows[name] = {
            "accuracy": _accuracy(results),
            "accuracy_easy": _accuracy(results, "easy"),
            "accuracy_hard": _accuracy(results, "hard"),
            "n": n,
            "n_easy": n_easy,
            "n_hard": n_hard,
            "mean_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
            "total_latency_s": float(np.sum(latencies)) if latencies else 0.0,
            "wall_clock_s": float(wall) if wall is not None else float("nan"),
            "total_tokens": int(sum(r.total_tokens for r in results)),
            "total_cost_usd": float(sum(r.cost_usd for r in results)),
            "memory_hit_rate": (n_mem / n) if n else 0.0,
            "escalation_rate": (n_s2 / n) if n else 0.0,
            "n_system2_calls": n_s2,
            "n_errors": sum(1 for r in results if r.error is not None),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.empty:  # keep the schema stable even with no results on disk
        df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = df.reindex(columns=SUMMARY_COLUMNS)
    df.index.name = "config"
    return df


# --------------------------------------------------------------------------- #
# Summary persistence (csv / markdown / latex)
# --------------------------------------------------------------------------- #
def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Round float columns to sensible precisions; leave integer columns intact."""
    decimals = {c: d for c, d in _ROUND.items() if c in df.columns}
    return df.round(decimals)


def _latex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _fmt_cell(col: str, value) -> str:
    """Format one summary value for the LaTeX/markdown table."""
    if pd.isna(value):
        return "--"
    if col in _ROUND:
        return f"{value:.{_ROUND[col]}f}"
    if col == "total_tokens":
        return f"{int(value):,}"
    return f"{int(value)}"


def _to_latex(df: pd.DataFrame) -> str:
    """Render a booktabs-style LaTeX table with a caption and ``tab:summary`` label."""
    cols = list(df.columns)
    colspec = "l" + "r" * len(cols)
    header = " & ".join([r"\textbf{config}"] + [rf"\textbf{{{_latex_escape(c)}}}" for c in cols]) + r" \\"
    body = []
    for name, row in df.iterrows():
        cells = [_latex_escape(str(name))] + [_fmt_cell(c, row[c]) for c in cols]
        body.append(" & ".join(cells) + r" \\")
    lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx} in the document preamble.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Per-configuration accuracy, latency, token and cost summary for the "
        r"Thinking-Fast-and-Slow multiplication agent.}",
        r"  \label{tab:summary}",
        # The full table has many columns; scale it to the text width so it never
        # overflows the page (readable on a landscape page in the appendix).
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
        *["    " + b for b in body],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# Human-readable labels for the transposed (portrait) appendix table.
_CONFIG_LABELS = {
    "pure_system1": "Pure S1",
    "pure_system2": "Pure S2",
    "hybrid_no_memory": "Hybrid (no mem)",
    "hybrid": "Hybrid (mem)",
}
# (column key, row label, format kind, group id) for the transposed table.
_METRIC_ROWS = [
    ("accuracy",        r"Accuracy (overall)", "f3",   0),
    ("accuracy_easy",   r"Accuracy (easy)",    "f3",   0),
    ("accuracy_hard",   r"Accuracy (hard)",    "f3",   0),
    ("n",               r"Problems ($n$)",     "int",  1),
    ("n_easy",          r"\quad easy",         "int",  1),
    ("n_hard",          r"\quad hard",         "int",  1),
    ("mean_latency_s",  r"Mean latency (s)",   "f2",   2),
    ("total_latency_s", r"Total latency (s)",  "f2",   2),
    ("wall_clock_s",    r"Wall-clock (s)",     "f2",   2),
    ("total_tokens",    r"Total tokens",       "intc", 3),
    ("total_cost_usd",  r"Total cost (USD)",   "f4",   3),
    ("memory_hit_rate", r"Memory hit rate",    "f3",   4),
    ("escalation_rate", r"Escalation rate",    "f3",   4),
    ("n_system2_calls", r"System~2 calls",     "int",  4),
    ("n_errors",        r"Errors",             "int",  4),
]


def _fmt_val(kind: str, value) -> str:
    """Format one value for the transposed table by declared kind."""
    if pd.isna(value):
        return "--"
    if kind == "f2":
        return f"{value:.2f}"
    if kind == "f3":
        return f"{value:.3f}"
    if kind == "f4":
        return f"{value:.4f}"
    if kind == "intc":
        return f"{int(round(value)):,}"
    return f"{int(round(value))}"


def _to_latex_transposed(df: pd.DataFrame) -> str:
    """Portrait, readable appendix table: metrics as rows, configs as columns.

    Unlike the wide auto-generated table, this fits a normal page without scaling.
    """
    order = [c for c in _CONFIG_LABELS if c in df.index]
    headers = [_CONFIG_LABELS[c] for c in order]
    ncol = len(order)
    lines = [
        "% Requires \\usepackage{booktabs} in the document preamble.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Full per-configuration results across all recorded metrics. "
        r"Configurations: Pure S1/S2 use only System~1 / System~2; Hybrid (no mem) "
        r"is routing without memory; Hybrid (mem) is the full TFaS agent.}",
        r"  \label{tab:summary}",
        rf"  \begin{{tabular}}{{l{'r' * ncol}}}",
        r"    \toprule",
        "    " + " & ".join([r"\textbf{Metric}"] + [rf"\textbf{{{h}}}" for h in headers]) + r" \\",
        r"    \midrule",
    ]
    prev_group = None
    for key, label, kind, group in _METRIC_ROWS:
        if key not in df.columns:
            continue
        if prev_group is not None and group != prev_group:
            lines.append(r"    \addlinespace")
        cells = [label] + [_fmt_val(kind, df.at[cfg, key]) for cfg in order]
        lines.append("    " + " & ".join(cells) + r" \\")
        prev_group = group
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def save_summary(df: pd.DataFrame, results_dir: PathLike = config.RESULTS_DIR) -> dict[str, Path]:
    """Write ``summary.csv``, ``summary.md`` and ``summary_table.tex``.

    Returns the three paths written. Numbers are rounded sensibly (accuracy and
    rates 3dp, latency 2dp, cost 4dp).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rounded = _rounded(df)

    csv_path = results_dir / "summary.csv"
    md_path = results_dir / "summary.md"
    tex_path = results_dir / "summary_table.tex"
    tex_t_path = results_dir / "summary_table_transposed.tex"

    rounded.to_csv(csv_path)
    try:
        md = rounded.to_markdown()
    except Exception:  # pragma: no cover - tabulate fallback
        from tabulate import tabulate

        md = tabulate(rounded, headers="keys", tablefmt="github")
    md_path.write_text(md + "\n", encoding="utf-8")
    tex_path.write_text(_to_latex(rounded), encoding="utf-8")
    # Portrait, readable version used in the paper's appendix (metrics as rows).
    tex_t_path.write_text(_to_latex_transposed(df), encoding="utf-8")

    return {"csv": csv_path, "md": md_path, "tex": tex_path, "tex_transposed": tex_t_path}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _label(name: str) -> str:
    return CONFIG_LABELS.get(name, name.replace("_", " "))


def _color(name: str, order: list[str]) -> str:
    """Stable color for a config: the fixed map, else the next categorical slot."""
    if name in CONFIG_COLORS:
        return CONFIG_COLORS[name]
    return _SLOTS[order.index(name) % len(_SLOTS)]


def _apply_style() -> None:
    """Publication defaults: clean sans, recessive grid, hairline axes."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _AXIS,
        "axes.linewidth": 1.0,
        "text.color": _INK,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "xtick.labelcolor": _INK_2,
        "ytick.labelcolor": _INK_2,
        "grid.color": _GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def _new_ax(figsize=(7.0, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _annotate_bars(ax, bars, values, fmt: str, fontsize: int = 8, rotation: int = 0) -> None:
    for bar, v in zip(bars, values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.annotate(
            fmt.format(v),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize, color=_INK_2, rotation=rotation,
        )


def _fig_accuracy(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    metrics = [("accuracy", "Overall"), ("accuracy_easy", "Easy"), ("accuracy_hard", "Hard")]
    fig, ax = _new_ax()
    x = np.arange(len(metrics))
    n = max(len(configs), 1)
    slot = 0.8 / n
    for i, name in enumerate(configs):
        vals = [df.loc[name, m] for m, _ in metrics]
        off = (i - (n - 1) / 2) * slot
        bars = ax.bar(x + off, vals, slot * 0.9, color=_color(name, configs),
                      label=_label(name), edgecolor=_SURFACE, linewidth=1.2)
        _annotate_bars(ax, bars, vals, "{:.2f}", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by configuration and difficulty")
    ax.grid(axis="y")
    ax.legend(ncol=min(n, 4), loc="lower center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, out_dir / "accuracy_by_config.png")


def _fig_simple_bar(df: pd.DataFrame, configs: list[str], column: str, *, title: str,
                    ylabel: str, fmt: str, out_name: str, out_dir: Path, log: bool = False) -> None:
    fig, ax = _new_ax(figsize=(6.4, 4.4))
    vals = [df.loc[name, column] for name in configs]
    colors = [_color(name, configs) for name in configs]
    bars = ax.bar(range(len(configs)), vals, width=0.6, color=colors,
                  edgecolor=_SURFACE, linewidth=1.2)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([_label(name) for name in configs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y")
    if log:
        ax.set_yscale("log")
    else:
        top = max([v for v in vals if v == v] or [0]) or 1.0
        ax.set_ylim(0, top * 1.18)
    _annotate_bars(ax, bars, vals, fmt, fontsize=9)
    _save(fig, out_dir / out_name)


def _fig_route_composition(results_by_config: dict[str, list[SolveResult]], out_dir: Path) -> None:
    hybrids = [c for c in ("hybrid_no_memory", "hybrid") if c in results_by_config]
    fig, ax = _new_ax(figsize=(6.0, 4.4))
    routes = ["memory", "system1", "system2"]
    if hybrids:
        x = np.arange(len(hybrids))
        fracs = {r: [] for r in routes}
        for name in hybrids:
            res = results_by_config[name]
            n = len(res) or 1
            for r in routes:
                fracs[r].append(sum(1 for rr in res if rr.route == r) / n)
        bottom = np.zeros(len(hybrids))
        for r in routes:
            vals = np.array(fracs[r])
            bars = ax.bar(x, vals, width=0.55, bottom=bottom, color=ROUTE_COLORS[r],
                          label=r, edgecolor=_SURFACE, linewidth=1.5)
            for bar, v, b in zip(bars, vals, bottom):
                if v >= 0.06:  # only label a segment tall enough to hold text
                    ax.annotate(f"{v:.0%}", (bar.get_x() + bar.get_width() / 2, b + v / 2),
                                ha="center", va="center", fontsize=8, color=_INK)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([_label(name) for name in hybrids])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Share of problem stream")
        ax.legend(title="route", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        ax.text(0.5, 0.5, "no hybrid configs in results", ha="center", va="center",
                color=_MUTED, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Route composition (memory / System 1 / System 2)")
    _save(fig, out_dir / "route_composition.png")


def _fig_accuracy_vs_cost(df: pd.DataFrame, configs: list[str], out_dir: Path) -> None:
    fig, ax = _new_ax(figsize=(6.4, 4.6))
    for name in configs:
        x = df.loc[name, "total_cost_usd"]
        y = df.loc[name, "accuracy"]
        ax.scatter([x], [y], s=120, color=_color(name, configs), edgecolor=_SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(_label(name), (x, y), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=_INK_2)
    ax.set_xlabel("Total cost (USD)")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Accuracy vs. cost trade-off")
    ax.grid(axis="both")
    ax.margins(0.18)
    _save(fig, out_dir / "accuracy_vs_cost.png")


def _fig_cumulative_cost(results_by_config: dict[str, list[SolveResult]], configs: list[str],
                         out_dir: Path) -> None:
    fig, ax = _new_ax(figsize=(7.2, 4.6))
    for name in configs:
        res = sorted(results_by_config[name], key=lambda r: r.problem_id)
        if not res:
            continue
        cum = np.cumsum([r.cost_usd for r in res])
        idx = np.arange(len(cum))
        color = _color(name, configs)
        ax.plot(idx, cum, color=color, linewidth=2, label=_label(name))
        ax.scatter([idx[-1]], [cum[-1]], s=30, color=color, edgecolor=_SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"${cum[-1]:.4f}", (idx[-1], cum[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=_INK_2)
    ax.set_xlabel("Problem position in stream")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("Cumulative cost over the problem stream")
    ax.grid(axis="both")
    ax.margins(x=0.08)
    ax.legend(loc="upper left")
    _save(fig, out_dir / "cumulative_cost.png")


def make_figures(
    results_by_config: dict[str, list[SolveResult]],
    df: pd.DataFrame,
    out_dir: PathLike = config.FIGURES_DIR,
) -> list[Path]:
    """Render all seven paper figures to ``out_dir`` (created if needed).

    Colors are consistent per config across every figure. Returns the list of
    PNG paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # Keep a stable config order shared by all figures (summary index order).
    configs = list(df.index)

    _fig_accuracy(df, configs, out_dir)
    _fig_simple_bar(df, configs, "total_cost_usd", title="Total cost by configuration",
                    ylabel="Total cost (USD)", fmt="${:.4f}", out_name="cost_by_config.png",
                    out_dir=out_dir, log=_should_log([df.loc[c, "total_cost_usd"] for c in configs]))
    _fig_simple_bar(df, configs, "total_tokens", title="Total tokens by configuration",
                    ylabel="Total tokens", fmt="{:,.0f}", out_name="tokens_by_config.png",
                    out_dir=out_dir)
    _fig_simple_bar(df, configs, "mean_latency_s", title="Mean per-problem latency by configuration",
                    ylabel="Mean latency (s)", fmt="{:.2f}", out_name="latency_by_config.png",
                    out_dir=out_dir)
    _fig_route_composition(results_by_config, out_dir)
    _fig_accuracy_vs_cost(df, configs, out_dir)
    _fig_cumulative_cost(results_by_config, configs, out_dir)

    return [
        out_dir / p for p in (
            "accuracy_by_config.png", "cost_by_config.png", "tokens_by_config.png",
            "latency_by_config.png", "route_composition.png", "accuracy_vs_cost.png",
            "cumulative_cost.png",
        )
    ]


def _should_log(values: list[float]) -> bool:
    """Use a log y-axis when the positive spread across configs exceeds ~50x."""
    pos = [v for v in values if v is not None and v == v and v > 0]
    return len(pos) >= 2 and (max(pos) / min(pos)) > 50
