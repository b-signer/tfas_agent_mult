"""Deterministic exact-match memory of solved problems.

The memory is the "System 0" of the agent: before any model is invoked, the
problem key is looked up here. A hit returns the stored answer instantly with
zero tokens and zero latency (beyond a dict lookup). Every problem solved by a
model is written back so that repeats become free.

Backed by a plain JSON object: {"7x8": 56, "12x13": 156, ...}.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from . import config


class Memory:
    def __init__(self, path: Union[str, Path, None] = None, *, enabled: bool = True, load: bool = True):
        self.path = Path(path) if path is not None else config.DEFAULT_MEMORY_PATH
        self.enabled = enabled
        self._store: dict[str, int] = {}
        # Live counters for the experiment metrics.
        self.lookups = 0
        self.hits = 0
        self.writes = 0
        if load and self.enabled and self.path.exists():
            self.load()

    # ------------------------------------------------------------------ #
    @staticmethod
    def make_key(a: int, b: int) -> str:
        """Exact-match key for the problem as presented (order preserved)."""
        return f"{a}x{b}"

    def get(self, a: int, b: int) -> Optional[int]:
        """Return the cached product, or None on a miss. Records a lookup."""
        if not self.enabled:
            return None
        self.lookups += 1
        val = self._store.get(self.make_key(a, b))
        if val is not None:
            self.hits += 1
        return val

    def put(self, a: int, b: int, value: int) -> None:
        """Store a solved answer. No-op when memory is disabled."""
        if not self.enabled:
            return
        self._store[self.make_key(a, b)] = int(value)
        self.writes += 1

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "entries": len(self._store),
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.lookups - self.hits,
            "writes": self.writes,
            "hit_rate": self.hit_rate,
        }

    def reset_counters(self) -> None:
        self.lookups = self.hits = self.writes = 0

    def clear(self) -> None:
        self._store.clear()
        self.reset_counters()

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            self._store = {str(k): int(v) for k, v in json.load(fh).items()}

    def save(self, path: Union[str, Path, None] = None) -> None:
        out = Path(path) if path is not None else self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self._store, fh, indent=2, sort_keys=True)
