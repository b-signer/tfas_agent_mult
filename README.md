# TFaS — Thinking, Fast and Slow

A dual-model agent inspired by the two-systems account of cognition in Daniel
Kahneman's *Thinking, Fast and Slow*, applied to elementary multiplication.

- **System 1** — a small (<4B), non-reasoning model (`mistralai/ministral-3b-2512`).
  Fast and cheap. It answers what it is confident about and *escalates* the rest
  via a tool call.
- **System 2** — a larger reasoning model (`deepseek/deepseek-r1-0528`). Slow and
  expensive. It handles the problems System 1 hands off.
- **Memory** — a deterministic exact-match cache checked *before* any model runs;
  a previously seen problem is answered instantly with zero tokens.

Both models are served through [OpenRouter](https://openrouter.ai), so latency and
token accounting are measured on the same footing.

## The experiment

The agent is run over a two-tier multiplication dataset (easy single-digit
problems and hard multi-digit problems) presented as a stream **with repeats**, in
four configurations:

| Config | Memory | Router |
|---|---|---|
| `pure_system1` | off | System 1 only |
| `pure_system2` | off | System 2 only |
| `hybrid_no_memory` | off | System 1 → System 2 |
| `hybrid` (full TFaS) | on | memory → System 1 → System 2 |

We report accuracy (overall and by difficulty), latency, token usage, dollar cost,
how often System 2 was invoked, and the memory hit rate. Results are written up as
a LaTeX white paper in `paper/`.

## Results (200-problem run)

| Config | Accuracy (easy / hard) | Mean latency | Tokens | Cost | → System 2 | Memory hits |
|---|---|---|---|---|---|---|
| `pure_system1` | 92.0% (100% / 81.0%) | 0.53 s | 26 K | $0.0026 | 0 | — |
| `pure_system2` | 99.0% (100% / 97.6%) | 24.1 s | 238 K | $0.5282 | 200 (100%) | — |
| `hybrid_no_memory` | 99.5% (100% / 98.8%) | 9.4 s | 153 K | $0.2272 | 31 (15.5%) | — |
| **`hybrid` (full TFaS)** | **98.5%** (100% / 96.4%) | **3.7 s** | **58 K** | **$0.0835** | 12 (6%) | 60% |

Takeaways:

- **Routing works.** With no memory, System 1 escalates only 15.5% of problems yet
  the hybrid matches pure-System-2 accuracy (99.5% vs 99.0%) at 43% of the cost.
- **Memory is the multiplier.** The full agent serves 60% of the stream from cache
  for free, escalates just 6%, and lands at **$0.0835 — 6.3× cheaper than
  pure-System-2, 24% of its tokens, and 6.5× lower mean latency** — for a
  0.5-point accuracy cost.
- **Memory is double-edged.** The full hybrid (98.5%) trails `hybrid_no_memory`
  (99.5%) because the cache locks in System 1's *first-occurrence* answer: a
  confidently-wrong first answer propagates to every repeat and forgoes a later
  chance to escalate.

See `paper/main.pdf` for the full write-up and `figures/` for the plots.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your OpenRouter key (OPENROUTER_KEY=...)

python scripts/generate_dataset.py     # -> data/dataset.json
python scripts/run_experiments.py      # -> results/*.json   (calls the API)
python scripts/make_figures.py         # -> figures/*.png, summary tables
```

## Layout

```
tfas/        core package (config, schemas, llm_client, memory, dataset,
             system1, system2, agent, runner, analysis)
scripts/     CLI entry points
tests/       unit tests (API mocked)
data/        dataset + memory
results/     raw experiment output
figures/     generated plots
paper/       LaTeX white paper
docs/        module interface contract
```

## Notes

- The code reads the API key from `OPENROUTER_KEY` (or `OPENROUTER_API_KEY`) in the
  environment or in `.env`. `.env` is gitignored.
- A full run costs roughly a few US dollars in OpenRouter credits.
