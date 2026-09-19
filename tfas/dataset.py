"""Two-tier multiplication dataset with intentional repeats.

The agent is evaluated on a *stream* of problems that deliberately revisits a
smaller *unique pool*, so the memory cache has something to hit. The pool has two
tiers:

    easy -- single-digit x single-digit, both factors in 2..9 (0 and 1 are
            skipped as trivial).
    hard -- at least one factor with two or three digits, mixed evenly across
            2d x 1d, 2d x 2d, 3d x 1d and 3d x 2d.

The stream includes every pool problem at least once and then samples the pool
uniformly (with replacement) to reach ``n_stream``; each occurrence after the
first counts as a repeat. Generation is fully deterministic given ``seed`` and
never touches the global ``random`` state.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Union

from tfas import config
from tfas.schemas import Problem

# Factor ranges (inclusive) for each hard sub-category, in the order the
# ``n_unique_hard`` budget is spread across them.
_HARD_CATEGORIES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((10, 99), (2, 9)),      # 2-digit x 1-digit
    ((10, 99), (10, 99)),    # 2-digit x 2-digit
    ((100, 999), (2, 9)),    # 3-digit x 1-digit
    ((100, 999), (10, 99)),  # 3-digit x 2-digit
]

# The easy pool is single-digit x single-digit with factors in 2..9.
_EASY_LO, _EASY_HI = 2, 9
_MAX_EASY_PAIRS = (_EASY_HI - _EASY_LO + 1) ** 2  # 8 x 8 = 64 distinct ordered pairs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _difficulty(a: int, b: int) -> str:
    """Label a problem: easy iff both factors are single-digit."""
    return "easy" if a < 10 and b < 10 else "hard"


def _even_split(total: int, k: int) -> list[int]:
    """Split ``total`` into ``k`` counts as evenly as possible (earlier buckets larger)."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _sample_distinct_pairs(
    rng: random.Random, a_range: tuple[int, int], b_range: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    """Sample ``count`` distinct ordered (a, b) pairs from the given inclusive ranges.

    Uses rejection sampling with insertion-ordered bookkeeping so results depend
    only on ``rng`` (never on set iteration order).
    """
    (a_lo, a_hi), (b_lo, b_hi) = a_range, b_range
    space = (a_hi - a_lo + 1) * (b_hi - b_lo + 1)
    if count > space:
        raise ValueError(f"cannot draw {count} distinct pairs from a space of {space}")
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < count:
        pair = (rng.randint(a_lo, a_hi), rng.randint(b_lo, b_hi))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_dataset(
    n_stream: int = config.N_STREAM,
    n_unique_easy: int = config.N_UNIQUE_EASY,
    n_unique_hard: int = config.N_UNIQUE_HARD,
    repeat_fraction: float = config.REPEAT_FRACTION,
    seed: int = config.DATASET_SEED,
) -> list[Problem]:
    """Build a deterministic stream of multiplication problems with repeats.

    A unique pool of ``n_unique_easy`` easy and ``n_unique_hard`` hard problems is
    sampled first. The stream then contains every pool problem once, padded to
    ``n_stream`` by sampling the pool uniformly with replacement, and shuffled.
    Because every unique problem appears at least once, the resulting repeat rate
    is ``(n_stream - pool_size) / n_stream``; ``repeat_fraction`` is the design
    target the default config sizes are chosen to hit.

    ``Problem.id`` is the stream index (0..n_stream-1); repeats keep identical
    (a, b) and difficulty.
    """
    pool_size = n_unique_easy + n_unique_hard
    if n_stream < pool_size:
        raise ValueError(
            f"n_stream={n_stream} is smaller than the unique pool ({pool_size}); "
            "every unique problem must appear at least once."
        )
    if n_unique_easy > _MAX_EASY_PAIRS:
        raise ValueError(
            f"n_unique_easy={n_unique_easy} exceeds the {_MAX_EASY_PAIRS} distinct "
            "single-digit pairs available."
        )

    rng = random.Random(seed)

    # --- Unique pool ---------------------------------------------------- #
    easy_space = [(a, b) for a in range(_EASY_LO, _EASY_HI + 1) for b in range(_EASY_LO, _EASY_HI + 1)]
    easy_pairs = rng.sample(easy_space, n_unique_easy)

    hard_pairs: list[tuple[int, int]] = []
    for (a_range, b_range), count in zip(_HARD_CATEGORIES, _even_split(n_unique_hard, len(_HARD_CATEGORIES))):
        hard_pairs.extend(_sample_distinct_pairs(rng, a_range, b_range, count))

    pool: list[tuple[int, int, str]] = [(a, b, _difficulty(a, b)) for a, b in easy_pairs + hard_pairs]

    # --- Stream (each unique once, then fill with replacement) ---------- #
    stream = list(pool)
    for _ in range(n_stream - len(pool)):
        stream.append(rng.choice(pool))
    rng.shuffle(stream)

    return [Problem(id=i, a=a, b=b, difficulty=d) for i, (a, b, d) in enumerate(stream)]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_dataset(problems: list[Problem], path: Union[str, Path] = config.DATASET_PATH) -> None:
    """Write ``problems`` as a JSON list of ``Problem.to_dict()``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([p.to_dict() for p in problems], fh, indent=2)


def load_dataset(path: Union[str, Path] = config.DATASET_PATH) -> list[Problem]:
    """Read a dataset written by :func:`save_dataset` back into ``Problem`` objects."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        return [Problem.from_dict(d) for d in json.load(fh)]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def dataset_stats(problems: list[Problem]) -> dict:
    """Summarize a stream: occurrences (with repeats) and distinct-key counts."""
    total = len(problems)
    unique_keys = {(p.a, p.b) for p in problems}
    n_unique = len(unique_keys)
    n_repeats = total - n_unique
    easy = [p for p in problems if p.difficulty == "easy"]
    hard = [p for p in problems if p.difficulty == "hard"]
    return {
        "total": total,
        "n_unique": n_unique,
        "n_repeats": n_repeats,
        "repeat_rate": n_repeats / total if total else 0.0,
        "easy_occurrences": len(easy),
        "hard_occurrences": len(hard),
        "unique_easy": len({(p.a, p.b) for p in easy}),
        "unique_hard": len({(p.a, p.b) for p in hard}),
    }
