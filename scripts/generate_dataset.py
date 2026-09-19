#!/usr/bin/env python3
"""Generate the two-tier multiplication dataset and write it to ``config.DATASET_PATH``.

Run from the repo root:

    python scripts/generate_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when invoked as a bare script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabulate import tabulate

from tfas import config
from tfas.dataset import dataset_stats, generate_dataset, save_dataset


def main() -> None:
    problems = generate_dataset()
    save_dataset(problems, config.DATASET_PATH)
    stats = dataset_stats(problems)

    print(f"Wrote {len(problems)} problems to {config.DATASET_PATH}")
    rows = [[k, f"{v:.3f}" if isinstance(v, float) else v] for k, v in stats.items()]
    print(tabulate(rows, headers=["metric", "value"], tablefmt="github"))


if __name__ == "__main__":
    main()
