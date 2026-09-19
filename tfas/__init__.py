"""TFaS: a Thinking, Fast and Slow dual-model agent for elementary multiplication.

Package layout:
    config      -- model IDs, pricing, generation params, paths, API key loading
    schemas     -- Problem and SolveResult dataclasses (shared data contract)
    llm_client  -- OpenRouterClient: one chat() call -> ChatResult(text, tool_calls, usage, cost, latency)
    memory      -- Memory: JSON-backed exact-match cache of solved problems
    dataset     -- two-tier multiplication dataset generation (built by dataset module)
    system1     -- small non-reasoning model: answer-or-escalate via tool calls
    system2     -- large reasoning model: solve hard problems
    agent       -- TFaSAgent: memory -> System 1 -> System 2 orchestration
    runner      -- run the experiment configurations over a dataset
    analysis    -- summarize results and render figures
"""

__version__ = "0.1.0"
