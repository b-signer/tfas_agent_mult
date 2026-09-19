# TFaS module interface contract

The shared foundation (`config`, `schemas`, `llm_client`, `memory`) is already
implemented and **must not be modified** by module authors. Build each remaining
module to the interfaces below so the pieces integrate without edits.

## Already implemented (import, do not change)

### `tfas.config`
- `SYSTEM1_MODEL`, `SYSTEM2_MODEL` — OpenRouter model IDs.
- `S1_TEMPERATURE`, `S1_MAX_TOKENS`, `S2_TEMPERATURE`, `S2_MAX_TOKENS`.
- `EXPERIMENTS: list[ExperimentConfig]` where
  `ExperimentConfig(name, mode, use_memory, system1_model, system2_model)`;
  `mode ∈ {"s1_only", "s2_only", "hybrid"}`.
- Paths: `DATA_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `PAPER_DIR`, `DATASET_PATH`, `DEFAULT_MEMORY_PATH`.
- `get_api_key() -> str`.

### `tfas.schemas`
- `Problem(id, a, b, difficulty)` with props `.text` (`"7 x 8"`), `.answer` (`a*b`),
  `.key` (`"7x8"`), and `.to_dict()/.from_dict()`.
- `SolveResult(problem_id, a, b, difficulty, ground_truth, predicted, correct, route,
  escalated, memory_hit, prompt_tokens, completion_tokens, total_tokens, cost_usd,
  latency_s, s1_decision, s1_text, s2_text, error)` with `.to_dict()/.from_dict()`.
  `route ∈ {"memory", "system1", "system2"}`.

### `tfas.llm_client`
- `OpenRouterClient().chat(model, messages, *, tools=None, tool_choice=None,
  temperature=None, max_tokens=None, reasoning=None) -> ChatResult`.
- `ChatResult(text, tool_calls: list[ToolCall], prompt_tokens, completion_tokens,
  total_tokens, cost_usd, latency_s, model, finish_reason, reasoning, raw)` with
  `.first_tool_call`.
- `ToolCall(id, name, arguments: dict)`.

### `tfas.memory`
- `Memory(path=None, *, enabled=True)`: `.get(a,b) -> int|None`, `.put(a,b,val)`,
  `.stats()`, `.hit_rate`, `.reset_counters()`, `.clear()`, `.save()`, `.load()`.

## To be implemented

### `tfas.dataset` (module author A)
- `Problem` reused from `schemas`.
- `generate_dataset(n_stream=400, n_unique_easy=?, n_unique_hard=?, repeat_fraction≈0.5, seed=0) -> list[Problem]`
  producing a **stream with intentional repeats** drawn from a smaller unique pool.
  Easy = single-digit × single-digit (factors 2–9). Hard = at least one 2–3 digit factor.
  `Problem.id` is the stream index (0..n-1). Repeats keep identical (a,b).
- `save_dataset(problems, path=config.DATASET_PATH)` / `load_dataset(path) -> list[Problem]`
  (JSON list of `Problem.to_dict()`).
- `dataset_stats(problems) -> dict` (counts, unique count, repeat count, easy/hard split).
- CLI: `scripts/generate_dataset.py` writes `data/dataset.json` and prints stats.

### `tfas.system1`, `tfas.system2`, `tfas.agent` (module author B)
- `system1.run_system1(client, problem, *, allow_escalate: bool) -> S1Outcome`
  where `S1Outcome` has `.decision ∈ {"answer","escalate"}`, `.answer: int|None`, `.chat: ChatResult`.
  Uses native tool calling with two tools: `answer(result:int)` and `escalate(reason?:str)`.
  When `allow_escalate=False`, expose only `answer` and instruct the model to always answer.
  Parse the first tool call; on malformed output, fall back to parsing an integer from text.
- `system2.run_system2(client, problem) -> S2Outcome` with `.answer: int|None`, `.chat: ChatResult`.
  Prompt the reasoning model to end with `ANSWER: <int>`; parse robustly.
- `agent.TFaSAgent(client, memory, config: ExperimentConfig)` with
  `.solve(problem) -> SolveResult`. Behaviour by `config.mode`:
  - `s1_only`: System 1 forced to answer; `route="system1"`.
  - `s2_only`: System 2 only; `route="system2"`.
  - `hybrid`: if `use_memory` and memory hit → `route="memory"` (0 tokens);
    else System 1 answer/escalate; escalate → System 2 (`route="system2"`),
    else `route="system1"`; then write the answer back to memory (if `use_memory`).
  Aggregate tokens/cost/latency across whatever calls were made; set `correct`
  against `problem.answer`; populate `s1_decision/s1_text/s2_text`.

### `tfas.runner` (author C, after A+B)
- `run_experiment(config, problems, client, *, progress=True) -> list[SolveResult]`
  (fresh `Memory` per hybrid run; process the stream in order).
- `run_all(problems, client) -> dict[str, list[SolveResult]]` over `config.EXPERIMENTS`.
- Persist per-config JSON to `results/<name>.json` and a combined `results/all_results.json`.
- CLI: `scripts/run_experiments.py`.

### `tfas.analysis` (author C)
- `summarize(results_by_config) -> pandas.DataFrame` with per-config accuracy
  (overall + by difficulty), total/mean latency, total tokens, total cost,
  escalation rate (hybrid), memory hit rate (hybrid).
- Render figures to `figures/` (PNG) and a markdown/LaTeX summary table.
- CLI: `scripts/make_figures.py`.
