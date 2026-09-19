"""Tests for tfas.dataset (no network required)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfas import config
from tfas.dataset import dataset_stats, generate_dataset, load_dataset, save_dataset


def _keys(problems):
    return [(p.a, p.b) for p in problems]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_is_identical():
    a = generate_dataset(seed=0)
    b = generate_dataset(seed=0)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_different_seed_differs():
    a = generate_dataset(seed=0)
    b = generate_dataset(seed=1)
    assert [p.to_dict() for p in a] != [p.to_dict() for p in b]


# --------------------------------------------------------------------------- #
# Unique pool
# --------------------------------------------------------------------------- #
def test_unique_pool_sizes():
    problems = generate_dataset()
    stats = dataset_stats(problems)
    assert stats["unique_easy"] == config.N_UNIQUE_EASY
    assert stats["unique_hard"] == config.N_UNIQUE_HARD
    assert stats["n_unique"] == config.N_UNIQUE_EASY + config.N_UNIQUE_HARD


def test_easy_are_single_digit():
    problems = generate_dataset()
    easy = [p for p in problems if p.difficulty == "easy"]
    assert easy  # tier is present
    for p in easy:
        assert 2 <= p.a <= 9 and 2 <= p.b <= 9  # both single-digit, no 0/1


def test_hard_have_multidigit_factor():
    problems = generate_dataset()
    hard = [p for p in problems if p.difficulty == "hard"]
    assert hard  # tier is present
    for p in hard:
        assert max(p.a, p.b) >= 10  # at least one 2-3 digit factor


def test_hard_categories_are_mixed():
    problems = generate_dataset()
    hard_keys = set(_keys([p for p in problems if p.difficulty == "hard"]))
    has_2digit_a = any(10 <= a <= 99 for a, _ in hard_keys)
    has_3digit_a = any(100 <= a <= 999 for a, _ in hard_keys)
    has_1digit_b = any(b < 10 for _, b in hard_keys)
    has_2digit_b = any(b >= 10 for _, b in hard_keys)
    assert has_2digit_a and has_3digit_a and has_1digit_b and has_2digit_b


def test_difficulty_labels_correct():
    for p in generate_dataset():
        expected = "easy" if p.a < 10 and p.b < 10 else "hard"
        assert p.difficulty == expected


# --------------------------------------------------------------------------- #
# Stream shape
# --------------------------------------------------------------------------- #
def test_stream_length_and_ids():
    n = config.N_STREAM
    problems = generate_dataset(n_stream=n)
    assert len(problems) == n
    assert [p.id for p in problems] == list(range(n))


def test_every_unique_appears_at_least_once():
    problems = generate_dataset()
    stats = dataset_stats(problems)
    # Distinct keys in the stream equals the pool size -> pool fully covered.
    assert stats["n_unique"] == config.N_UNIQUE_EASY + config.N_UNIQUE_HARD


def test_repeat_rate_near_target():
    problems = generate_dataset()
    stats = dataset_stats(problems)
    assert abs(stats["repeat_rate"] - config.REPEAT_FRACTION) <= 0.1
    assert stats["n_repeats"] == stats["total"] - stats["n_unique"]


def test_repeats_keep_operands_and_difficulty():
    problems = generate_dataset()
    by_key: dict[tuple[int, int], str] = {}
    for p in problems:
        key = (p.a, p.b)
        if key in by_key:
            assert by_key[key] == p.difficulty  # difficulty stable across repeats
        else:
            by_key[key] = p.difficulty


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "dataset.json"
    problems = generate_dataset()
    save_dataset(problems, path)
    loaded = load_dataset(path)
    assert [p.to_dict() for p in problems] == [p.to_dict() for p in loaded]


# --------------------------------------------------------------------------- #
# Custom parameters
# --------------------------------------------------------------------------- #
def test_custom_parameters():
    problems = generate_dataset(n_stream=100, n_unique_easy=10, n_unique_hard=8, seed=3)
    stats = dataset_stats(problems)
    assert stats["total"] == 100
    assert stats["unique_easy"] == 10
    assert stats["unique_hard"] == 8
    assert stats["n_unique"] == 18
