"""Central configuration: models, pricing, generation parameters, paths, API key.

Everything that another module might want to tune lives here so the rest of the
codebase never hard-codes a model name, price, or path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"

DATASET_PATH = DATA_DIR / "dataset.json"
DEFAULT_MEMORY_PATH = DATA_DIR / "memory.json"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# OpenRouter API
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

# Optional attribution headers (shown on the OpenRouter dashboard).
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/b-signer/tfas_agent_mult",
    "X-Title": "TFaS-Agent",
}


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (avoids importing python-dotenv at module import time).

    Only sets keys that are not already present in the environment.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_api_key() -> str:
    """Return the OpenRouter API key, loading .env if needed.

    Accepts OPENROUTER_KEY (preferred) or OPENROUTER_API_KEY.
    """
    _load_dotenv()
    key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenRouter API key found. Set OPENROUTER_KEY (or OPENROUTER_API_KEY) "
            "in your environment or in a .env file at the project root."
        )
    return key


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# System 1: small (<4B), non-reasoning, supports native tool calling.
SYSTEM1_MODEL = "mistralai/ministral-3b-2512"
# System 2: larger reasoning model.
SYSTEM2_MODEL = "deepseek/deepseek-r1-0528"

# Fallback price table (USD per token) used only if OpenRouter does not return a
# cost field. Values from the OpenRouter catalog (USD per 1M tokens / 1e6).
PRICING = {
    SYSTEM1_MODEL: {"prompt": 0.10e-6, "completion": 0.10e-6},
    SYSTEM2_MODEL: {"prompt": 0.50e-6, "completion": 2.15e-6},
}

# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
# System 1 answers directly; a small token budget is plenty.
S1_TEMPERATURE = 0.0
S1_MAX_TOKENS = 512

# System 2 is a reasoning model and needs headroom for chain-of-thought.
S2_TEMPERATURE = 0.0
S2_MAX_TOKENS = 6000
# Prefer the fastest available provider for the reasoning model. Calibration
# showed this cuts DeepSeek-R1 latency ~4x (126s -> ~13-31s per problem).
S2_PROVIDER = {"sort": "throughput"}

# HTTP behaviour
REQUEST_TIMEOUT_S = 180
MAX_RETRIES = 4
RETRY_BACKOFF_S = 2.0

# Concurrency for stateless experiment configs. The memory-enabled hybrid run is
# always executed sequentially (in stream order) so cache write-backs are visible
# to later repeats; the other three configs have no cross-problem state.
DEFAULT_WORKERS = 8

# --------------------------------------------------------------------------- #
# Dataset defaults (see tfas.dataset)
# --------------------------------------------------------------------------- #
DATASET_SEED = 0
N_STREAM = 200          # total problems presented (with repeats)
N_UNIQUE_EASY = 45      # distinct easy problems in the pool
N_UNIQUE_HARD = 35      # distinct hard problems in the pool
REPEAT_FRACTION = 0.6   # target fraction of the stream that are repeats

# --------------------------------------------------------------------------- #
# Prompts (finalized from calibration)
# --------------------------------------------------------------------------- #
# Confidence-framed self-assessment: System 1 answers what it is sure of and
# escalates the rest. Calibration: answers 6x7 and 12x8, escalates 23x47 and
# 384x279 -- the intended fast/slow split via genuine self-assessment.
S1_SYSTEM_PROMPT = (
    "You are System 1: fast, intuitive arithmetic - the instant, effortless kind of "
    "thinking. You are given a multiplication problem. Call answer(result) ONLY if you "
    "are highly confident the result is exactly correct from memory or quick intuition. "
    "If the problem would require careful, deliberate, multi-step calculation, call "
    "escalate() instead of guessing. It is much better to escalate than to give a wrong "
    "answer. Always respond with exactly one tool call."
)
# Forced-answer variant for the pure-System-1 baseline (no escalation tool).
S1_FORCED_SYSTEM_PROMPT = (
    "You are a fast mental-arithmetic assistant. Multiply the two numbers and call "
    "answer(result) with the exact integer product. Always respond with one tool call."
)
S2_SYSTEM_PROMPT = (
    "Solve the multiplication problem. Reason step by step as carefully as needed, then "
    "end your reply with a line in exactly this format:\nANSWER: <integer>"
)


@dataclass
class ExperimentConfig:
    """One column of the experiment matrix.

    mode:
        "s1_only"  -- always answer with System 1 (no escalation tool)
        "s2_only"  -- always answer with System 2
        "hybrid"   -- memory check -> System 1 (answer/escalate) -> System 2
    use_memory:
        whether the deterministic memory cache is consulted / written.
    """

    name: str
    mode: str
    use_memory: bool
    system1_model: str = SYSTEM1_MODEL
    system2_model: str = SYSTEM2_MODEL


# The four columns compared in the write-up.
EXPERIMENTS = [
    ExperimentConfig(name="pure_system1", mode="s1_only", use_memory=False),
    ExperimentConfig(name="pure_system2", mode="s2_only", use_memory=False),
    ExperimentConfig(name="hybrid_no_memory", mode="hybrid", use_memory=False),
    ExperimentConfig(name="hybrid", mode="hybrid", use_memory=True),
]
