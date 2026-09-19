#!/usr/bin/env python3
"""Run the TFaS experiment matrix over the multiplication dataset.

Examples:
    python scripts/run_experiments.py                 # full run, all configs
    python scripts/run_experiments.py --limit 6       # quick smoke on 6 problems
    python scripts/run_experiments.py --configs pure_system1,hybrid --workers 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfas import config
from tfas.dataset import load_dataset
from tfas.llm_client import OpenRouterClient
from tfas.runner import run_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TFaS experiment matrix.")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N problems (smoke test).")
    ap.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS, help="Concurrent workers for stateless configs.")
    ap.add_argument("--configs", type=str, default=None,
                    help="Comma-separated config names to run (default: all four).")
    ap.add_argument("--dataset", type=str, default=str(config.DATASET_PATH), help="Path to dataset.json.")
    args = ap.parse_args()

    problems = load_dataset(Path(args.dataset))
    if args.limit is not None:
        problems = problems[: args.limit]

    experiments = config.EXPERIMENTS
    if args.configs:
        wanted = {c.strip() for c in args.configs.split(",")}
        experiments = [c for c in config.EXPERIMENTS if c.name in wanted]
        missing = wanted - {c.name for c in experiments}
        if missing:
            ap.error(f"Unknown config name(s): {sorted(missing)}. "
                     f"Valid: {[c.name for c in config.EXPERIMENTS]}")

    print(f"Dataset: {len(problems)} problems | configs: {[c.name for c in experiments]} | workers: {args.workers}")
    client = OpenRouterClient()
    run_all(problems, client, experiments=experiments, workers=args.workers)
    print(f"\nAll done. Results in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
