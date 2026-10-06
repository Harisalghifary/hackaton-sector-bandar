"""Runtime LLM configuration (§3, [LOCKED] — exact block, do not edit here)."""

RUNTIME_LLM = {
    "primary":  "gemini-flash-3.8",    # all tiers: intent, plan, synthesis
    "fallback": "claude-sonnet-4.5",   # config-only hot swap
    "enforce":  ["response_schema:plan", "response_schema:submit_brief",
                 "numeric_trace_validator", "plan_validator"],
    "temp": 0.2,
}

# §3 [LOCKED]: Max 3 LLM calls per run — intent → plan → synthesis.
MAX_LLM_CALLS_PER_RUN = 3
