"""Run the experiment configurations over a dataset and persist raw results.

Concurrency model
-----------------
Three of the four configurations (``pure_system1``, ``pure_system2``,
``hybrid_no_memory``) have no cross-problem state, so they are executed with a
thread pool. Each worker thread uses its own ``OpenRouterClient`` (a
``requests.Session`` is not guaranteed thread-safe), created lazily via
thread-local storage.

The ``hybrid`` configuration with memory enabled MUST run sequentially in stream
order: a problem solved at step *i* is written to the cache and a later repeat at
step *j > i* is expected to hit it. Running it concurrently would race on the
cache and understate the memory benefit.

Per-problem ``latency_s`` (recorded on each ``SolveResult``) is always the real
model latency for that problem; concurrency only affects the wall-clock time of a
whole configuration, which is recorded separately in the run metadata.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .agent import TFaSAgent
from .llm_client import OpenRouterClient
from .memory import Memory
from .schemas import Problem, SolveResult

_thread_local = threading.local()


def _thread_client(api_key: str) -> OpenRouterClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenRouterClient(api_key=api_key)
        _thread_local.client = client
    return client


def _progress(name: str, done: int, total: int, every: int = 10) -> None:
    if done == total or done % every == 0:
        print(f"  [{name}] {done}/{total}", flush=True)


def run_experiment(
    cfg: config.ExperimentConfig,
    problems: list[Problem],
    client: OpenRouterClient,
    *,
    workers: int = config.DEFAULT_WORKERS,
    progress: bool = True,
) -> list[SolveResult]:
    """Run one configuration over the full problem stream and return results in order."""
    total = len(problems)

    # Memory-enabled hybrid: sequential, one shared cache that fills as we go.
    if cfg.use_memory:
        mem = Memory(enabled=True, load=False)
        agent = TFaSAgent(client, mem, cfg)
        results: list[SolveResult] = []
        for i, p in enumerate(problems):
            results.append(agent.solve(p))
            if progress:
                _progress(cfg.name, i + 1, total)
        return results

    # Stateless configs: concurrent. Disabled memory is a no-op and safe to share.
    shared_disabled_mem = Memory(enabled=False, load=False)
    api_key = client.api_key
    results = [None] * total  # type: ignore[assignment]

    def work(idx: int, problem: Problem):
        c = _thread_client(api_key)
        agent = TFaSAgent(c, shared_disabled_mem, cfg)
        return idx, agent.solve(problem)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, p) for i, p in enumerate(problems)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            done += 1
            if progress:
                _progress(cfg.name, done, total)
    return results  # type: ignore[return-value]


def save_results(name: str, results: list[SolveResult], results_dir: Path = config.RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2)
    return path


def run_all(
    problems: list[Problem],
    client: Optional[OpenRouterClient] = None,
    *,
    experiments: Optional[list[config.ExperimentConfig]] = None,
    workers: int = config.DEFAULT_WORKERS,
    results_dir: Path = config.RESULTS_DIR,
    progress: bool = True,
) -> dict[str, list[SolveResult]]:
    """Run every configuration, persist per-config JSON + combined JSON + metadata."""
    client = client or OpenRouterClient()
    experiments = experiments or config.EXPERIMENTS

    all_results: dict[str, list[SolveResult]] = {}
    meta: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_problems": len(problems),
        "workers": workers,
        "system1_model": config.SYSTEM1_MODEL,
        "system2_model": config.SYSTEM2_MODEL,
        "configs": {},
    }

    for cfg in experiments:
        print(f"\n=== Running config: {cfg.name} (mode={cfg.mode}, memory={cfg.use_memory}) ===", flush=True)
        start = time.perf_counter()
        results = run_experiment(cfg, problems, client, workers=workers, progress=progress)
        wall = time.perf_counter() - start
        save_results(cfg.name, results, results_dir)
        all_results[cfg.name] = results
        n_correct = sum(1 for r in results if r.correct)
        n_err = sum(1 for r in results if r.error)
        meta["configs"][cfg.name] = {
            "mode": cfg.mode,
            "use_memory": cfg.use_memory,
            "wall_clock_s": wall,
            "n": len(results),
            "n_correct": n_correct,
            "n_errors": n_err,
            "total_cost_usd": sum(r.cost_usd for r in results),
            "total_tokens": sum(r.total_tokens for r in results),
        }
        print(
            f"  done in {wall:.1f}s wall | acc {n_correct}/{len(results)} "
            f"| tokens {meta['configs'][cfg.name]['total_tokens']} "
            f"| ${meta['configs'][cfg.name]['total_cost_usd']:.4f} | errors {n_err}",
            flush=True,
        )

    with open(results_dir / "all_results.json", "w", encoding="utf-8") as fh:
        json.dump({k: [r.to_dict() for r in v] for k, v in all_results.items()}, fh, indent=2)
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return all_results
