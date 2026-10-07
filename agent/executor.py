"""BANDAR executor (§8, D5, FR10, FR12): intent -> plan -> execute -> synthesis.

§3 LOCKED: max 3 LLM calls per run, enforced by LLMCallCounter.
D5 LOCKED: synthesis output is the required boxed JSON (submit_brief), never
free-form; any failure path returns the §8 fallback {"type":"fallback","message"}.
Policy 4: budget breach -> abort step, narrate truncation honestly.
D6/D7: scored results upsert to memory; prior snapshot feeds delta context.

The LLM narrates only; every number it may quote comes from the tool-result
corpus, and numeric_trace_validator drops/flags anything else (FR10, §14).
"""

from __future__ import annotations

import json

from agent.llm import LLMError, make_llm
from agent.planner import LLMCallCounter, Planner, deterministic_plan
from agent.tools import ToolContext, run_tool
from agent.validators import (PlanValidationError, collect_allowed_numbers,
                              validate_brief_numbers)

# §8 synthesis schema — exact fields: submit_brief {symbol, extracted[],
# interpretation, action_plan[], risk_flags[]}
SUBMIT_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "extracted": {"type": "array", "items": {"type": "string"}},
        "interpretation": {"type": "string"},
        "action_plan": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["symbol", "extracted", "interpretation", "action_plan", "risk_flags"],
}

SYNTHESIS_SYSTEM = (
    "You are the synthesis voice of Bandar, a pre-market desk note for IDX swing "
    "traders. Write ONLY from the tool results provided — the engine computed every "
    "figure; you compute NOTHING.\n"
    "Voice rules (LOCKED §8): desk-note tone; at most 2 sentences per section; "
    "NO derived arithmetic; NO invented deltas — a delta may only compare figures "
    "present in the results/memory context.\n"
    "Human language: NEVER print raw field names (no snake_case like net_foreign, "
    "no_chase, extension_atr, f5_broker) — translate to trader language: "
    "net_foreign -> 'net foreign buying/selling'; f5_broker -> 'smart-money "
    "(broker/foreign) check'; f1_macd -> 'MACD trend'; f2_volume -> 'volume "
    "confirmation'; f3_ad -> 'accumulation/distribution'; f4_ma_stack -> "
    "'moving-average stack'; no_chase/extension_atr -> 'no-chase rule: price "
    "closed too far above yesterday's high'.\n"
    "Sign semantics (flow data): net_foreign_inflow > 0 = foreign money ENTERING "
    "the stock (accumulation pressure) -> say 'net foreign inflow of Rp X'; "
    "< 0 = money LEAVING (distribution/profit-taking) -> say 'net foreign outflow "
    "of Rp X' (quote the magnitude, carry the direction in words). Cumulative or "
    "window claims MUST quote the engine's flow summary fields (net_total, "
    "net_last_5d, net_last_20d, inflow_days, outflow_days, top_inflow_value, "
    "top_outflow_value, bias) — never sum or average rows yourself.\n"
    "Figures: quote values from the tool results; large rupiah amounts MAY be "
    "rescaled into human units ('Rp 152.1B', 'Rp 850M') — the machine checker "
    "knows these unit aliases. No other reformatting, no additional rounding, "
    "no new figures.\n"
    "Every number you output is machine-checked against the tool results; "
    "unverifiable figures are dropped, so do not add any.\n"
    "If data is missing, truncated, or a step failed (budget_abort, not_found), "
    "say so honestly in risk_flags. symbol = the primary ticker of the answer "
    "(top pick for watchlist-wide runs). Output ONLY the submit_brief JSON."
)

# Human names for engine factor keys (used by the deterministic fallback brief
# so it reads like a desk note, not a schema dump — same glossary as the prompt).
_FACTOR_NAMES = {
    "f1_macd": "MACD trend",
    "f2_volume": "Volume confirmation",
    "f3_ad": "Accumulation/distribution",
    "f4_ma_stack": "Moving-average stack",
    "f5_broker": "Smart-money (broker/foreign) check",
}


def _fallback(message: str, **extra) -> dict:
    """§8 fallback schema — graceful, honest, never a crash (FR12)."""
    return {"type": "fallback", "message": message, **extra}


def _valid_brief_shape(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    if not isinstance(raw.get("symbol"), str) or not isinstance(raw.get("interpretation"), str):
        return False
    for key in ("extracted", "action_plan", "risk_flags"):
        if not isinstance(raw.get(key), list) or not all(isinstance(x, str) for x in raw[key]):
            return False
    return True


def _deterministic_brief(scored: list[dict], truncated: bool) -> dict | None:
    """Engine-built submit_brief-shaped fallback (FR10-safe: every figure is copied
    verbatim from a tool result; nothing computed or invented). Used when LLM
    synthesis is unavailable so the Answer never shows a bare error."""
    if not scored:
        return None
    top = max(scored, key=lambda r: (r.get("score") or 0))
    dec = top.get("decision") or {}
    gates = top.get("gates") or {}
    tp = top.get("trade_plan") or {}
    factors = top.get("factors") or {}
    extracted = []
    for key, label in _FACTOR_NAMES.items():
        f = factors.get(key)
        extracted.append(f"{label}: no data (honest — not fabricated)" if f is None
                         else f"{label}: {'confirmed' if f.get('pass') else 'not confirmed'}")
    ez = tp.get("entry_zone") or []
    no_chase = bool((gates.get("no_chase") or {}).get("triggered"))
    interpretation = (f"{top.get('symbol')} scores {top.get('score')}/"
                      f"{top.get('denominator')} — decision {dec.get('label')}. "
                      + ("The no-chase rule triggered (price closed too far above "
                         "yesterday's high) — WAIT for a pullback instead of chasing."
                         if no_chase else "The no-chase rule did not trigger."))
    action_plan = []
    if len(ez) > 1:
        action_plan.append(f"entry zone {ez[0]}-{ez[1]}")
    elif ez:
        action_plan.append(f"entry {ez[0]}")
    if tp.get("stop_close") is not None:
        action_plan.append(f"close-based stop {tp['stop_close']}")
    if tp.get("lots") is not None:
        action_plan.append(f"size {tp['lots']} lots")
    risk_flags = []
    if no_chase:
        risk_flags.append("no-chase rule triggered — never chase an extended price")
    if (gates.get("ara") or {}).get("triggered"):
        risk_flags.append("ARA limit-up block")
    if truncated:
        risk_flags.append("run truncated by budget")
    risk_flags.append("LLM narration unavailable — deterministic engine summary")
    return {"symbol": top.get("symbol"), "extracted": extracted,
            "interpretation": interpretation, "action_plan": action_plan,
            "risk_flags": risk_flags}


def run_agent(question: str, ctx: ToolContext, llm=None, planner: Planner | None = None) -> dict:
    """One full run. Returns {"type": "brief", ...} or {"type": "fallback", ...}."""
    llm = llm or make_llm("primary")
    counter = LLMCallCounter()
    p = planner or Planner(llm, ctx, call_counter=counter)
    trace_start = len(ctx.client.trace_log)

    # ---- LLM call 1/3: intent ------------------------------------------------
    intent = p.classify_intent(question)
    if intent.get("intent") == "fallback":
        reason = intent.get("reason", "")
        # A "reason" means the intent call FAILED (transport/parse), not that the user
        # asked out-of-scope. Report it as a retry-able outage, not a scope refusal (FR12).
        if reason:
            if "quota" in reason.lower():
                msg = ("Bandar's LLM quota is exhausted (HTTP 429 quota/billing). "
                       "Scoring and refresh still work (deterministic), but free-text "
                       "answers need quota to reset, billing to be enabled, or a "
                       "different API key.")
            else:
                msg = ("Bandar's model is busy or unreachable right now (temporary "
                       "overload) — your question looked in-scope but the intent call "
                       "could not complete. Please try again in a moment.")
            return _fallback(
                msg,
                intent=intent, reason=reason, llm_calls=list(counter.used), outage=True,
            )
        return _fallback(
            "Outside Bandar's scope: I plan and analyze IDX watchlist setups, and I "
            "never execute trades. Ask about scores, smart money, valuation, deltas, "
            "or the universe screen.",
            intent=intent, reason="out-of-scope request", llm_calls=list(counter.used),
        )

    # ---- LLM call 2/3: plan (code-validated; no repair loop — §3 max 3 calls) -
    plan_degraded = ""
    try:
        plan = p.plan(question, intent=intent)
    except PlanValidationError as exc:
        return _fallback(f"Planner produced an invalid plan; rejected by validator: "
                         f"{exc.errors}", intent=intent, llm_calls=list(counter.used))
    except LLMError as exc:
        # K: deterministic plan fallback — the intent is known (often via the 0-LLM
        # fast-path), so a hardcoded validator-safe plan still fetches real data
        # during an outage; synthesis then degrades to the engine brief (E).
        plan = deterministic_plan(intent, ctx.memory.load_watchlist(), question)
        if plan is None:
            return _fallback(f"Planner unavailable: {exc}", intent=intent,
                             llm_calls=list(counter.used))
        plan_degraded = (f"plan LLM unavailable ({exc}); used the deterministic "
                         f"fallback plan for intent '{intent.get('intent')}'")

    # ---- deterministic execution (0 LLM) -------------------------------------
    step_results: list[dict] = []
    truncated = False
    for step in plan["steps"]:
        out = run_tool(step["tool"], step.get("args"), ctx)
        out["reason"] = step.get("reason", "")
        if not out["ok"] and out.get("error") == "budget_abort":
            truncated = True          # policy 4: narrate truncation honestly
        step_results.append(out)

    # ---- memory: delta context first (D6), then upsert scores (D7) -----------
    scored: list[dict] = []
    for out in step_results:
        if not out["ok"]:
            continue
        if out["tool"] == "score_ticker":
            scored.append(out["data"])
        elif out["tool"] == "rank_watchlist":
            scored.extend(out["data"]["ranked"])
    memory_context = []
    for r in scored:
        prev = ctx.memory.get_prev(r["symbol"], r["as_of"])
        if prev:
            memory_context.append({
                "symbol": r["symbol"], "prior_as_of": prev["as_of"],
                "prior_score": prev["score"], "prior_denominator": prev["denominator"],
                "prior_decision": prev["decision"],
            })
        ctx.memory.record_score(r)     # any run upserts by (ticker, as_of)

    # ---- LLM call 3/3: synthesis (boxed submit_brief, D5) ---------------------
    corpus = {
        "tool_results": step_results,
        "memory_context": memory_context,
        "trace": ctx.client.trace_log[trace_start:],
        "truncated": truncated,
    }
    user = (f"Question: {question}\nIntent: {json.dumps(intent)}\n"
            f"Plan executed ({len(step_results)} steps; truncated={truncated}).\n"
            f"Tool results and memory context (the ONLY quotable figures):\n"
            f"{json.dumps(corpus, default=str)}")
    try:
        counter.tick("synthesis")
        raw_brief = llm.complete(system=SYNTHESIS_SYSTEM, user=user,
                                 schema=SUBMIT_BRIEF_SCHEMA, purpose="synthesis")
    except LLMError as exc:
        det = _deterministic_brief(scored, truncated)
        if det is not None:
            return {"type": "brief", "brief": det, "deterministic_fallback": True,
                    "dropped_figures": [], "intent": intent, "plan": plan,
                    "plan_degraded": plan_degraded,
                    "tool_results": step_results, "truncated": truncated,
                    "llm_calls": list(counter.used),
                    "trace": ctx.client.trace_log[trace_start:],
                    "synthesis_error": str(exc)}
        return _fallback(f"Synthesis unavailable: {exc}", intent=intent, plan=plan,
                         plan_degraded=plan_degraded,
                         tool_results=step_results, truncated=truncated,
                         llm_calls=list(counter.used),
                         trace=ctx.client.trace_log[trace_start:])

    if not _valid_brief_shape(raw_brief):
        det = _deterministic_brief(scored, truncated)
        if det is not None:
            return {"type": "brief", "brief": det, "deterministic_fallback": True,
                    "dropped_figures": [], "intent": intent, "plan": plan,
                    "plan_degraded": plan_degraded,
                    "tool_results": step_results, "raw_brief": raw_brief,
                    "llm_calls": list(counter.used),
                    "trace": ctx.client.trace_log[trace_start:]}
        return _fallback("Synthesis returned a schema-invalid brief; nothing fabricated "
                         "in its place.", intent=intent, plan=plan,
                         plan_degraded=plan_degraded,
                         tool_results=step_results, raw_brief=raw_brief,
                         llm_calls=list(counter.used),
                         trace=ctx.client.trace_log[trace_start:])

    # ---- FR10: numeric trace validator (drop/flag) ----------------------------
    allowed = collect_allowed_numbers(corpus)
    brief, dropped = validate_brief_numbers(raw_brief, allowed)

    return {
        "type": "brief",
        "brief": brief,
        "dropped_figures": dropped,           # honest audit trail for the workings pane
        "intent": intent,
        "plan": plan,
        "plan_degraded": plan_degraded,       # K: non-empty when the plan LLM was down
        "tool_results": step_results,
        "memory_context": memory_context,
        "truncated": truncated,
        "llm_calls": list(counter.used),
        "trace": ctx.client.trace_log[trace_start:],
    }
