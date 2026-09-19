#!/usr/bin/env python3
"""Summarize experiment results and render the paper figures.

Reads whatever the runner has written under ``results/`` (per-config JSON plus
optional ``run_metadata.json``), prints the markdown summary, and writes the
summary tables (csv/md/tex) and the seven figures.

Run from the repo root:

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when invoked as a bare script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfas import config
from tfas.analysis import (
    load_all_results,
    load_metadata,
    make_figures,
    save_summary,
    summarize,
)


def main() -> None:
    results_by_config = load_all_results()
    if not results_by_config:
        print(f"No per-config result files found in {config.RESULTS_DIR}. "
              "Run scripts/run_experiments.py first.")
        return

    metadata = load_metadata()
    df = summarize(results_by_config, metadata)

    print("\n=== Summary ===\n")
    try:
        print(df.round(4).to_markdown())
    except Exception:
        print(df.round(4).to_string())

    paths = save_summary(df)
    figures = make_figures(results_by_config, df)

    print("\nWrote summary tables:")
    for kind, p in paths.items():
        print(f"  {kind}: {p}")
    print(f"\nWrote {len(figures)} figures to {config.FIGURES_DIR}:")
    for p in figures:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
