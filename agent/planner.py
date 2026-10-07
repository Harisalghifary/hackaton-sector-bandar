"""BANDAR planner (§8, D4/D5): policy rules live in the system prompt; the
executor (M4b) enforces budget. Max 3 LLM calls per run (§3 LOCKED):
intent → plan → synthesis — so the planner makes exactly ONE call and does NOT
repair-loop; an invalid plan becomes the §8 fallback (FR12), never a 4th call.

INTERPRETATION NOTE (FR12 "all 5 intents"): the spec never enumerates the five
intents; this module defines them from §8 planner policy (2) routing + §1 brief:
  daily_brief · score_ticker · smart_money · valuation · screen  (+ "fallback")
Flagged for user confirmation before the M7 video gate.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.config import MAX_LLM_CALLS_PER_RUN, RUNTIME_LLM
from agent.llm import LLMError
from agent.validators import validate_plan_or_raise
from client import _normalize_symbol

WIB = ZoneInfo("Asia/Jakarta")

INTENTS = ("daily_brief", "score_ticker", "smart_money", "valuation", "screen", "fallback")

# Deterministic smart-money fast-path (§8 hardening). If the question names a watchlist
# symbol AND uses smart-money vocabulary, route to smart_money WITHOUT an LLM call. This
# makes routing immune to transient model overload (503) and to LLM mis-classification,
# and saves an intent call. Non-matching questions still use the LLM intent router.
SMART_MONEY_RE = re.compile(
    r"accumulat|akumulasi|distribusi|broker|foreign|flow|smart\s*money|bandar", re.I)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "symbols": {"type": "array", "items": {"type": "string"}},
        "question": {"type": "string"},
    },
    "required": ["intent", "symbols", "question"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string",
                             "enum": ["score_ticker", "rank_watchlist", "score_history",
                                      "get_fundamentals", "get_foreign_flow", "screen",
                                      "watchlist_rw"]},
                    # Concrete optional arg properties: Gemini controlled generation
                    # fills "args": {} when the object has no declared properties
                    # (learned live 2026-10-06). §8 shape {tool,args,reason} is kept.
                    "args": {"type": "object", "properties": {
                        "symbol": {"type": "string"},
                        "limit": {"type": "integer"},
                        "where": {"type": "object"},
                        "q": {"type": "string"},
                        "op": {"type": "string", "enum": ["add", "remove", "replace"]},
                        "list": {"type": "array", "items": {"type": "string"}},
                    }},
                    "reason": {"type": "string"},
                },
                "required": ["tool", "args", "reason"],
            },
        },
    },
    "required": ["steps"],
}

# §8 planner policy — verbatim rules, injected into the system prompt (D4).
PLANNER_POLICIES = (
    "(1) active-position/risk items before new candidates; "
    "(2) route by sub-question: price/TA->engine, valuation->fundamentals, "
    "smart-money->broksum/flow, universe->screener; "
    "(3) adaptivity: score>=3 -> fundamentals, score<=1 -> skip context, "
    "tie -> broksum/flow tiebreak; "
    "(4) budget breach -> abort step, narrate truncation honestly; "
    "(5) read history before scoring."
)

TOOL_COST_TABLE = (
    "score_ticker(symbol) ~2cr (0 on warm cache) | rank_watchlist() 2xN cr | "
    "score_history(symbol) 0cr | get_fundamentals(symbol) 1cr | "
    "get_foreign_flow(symbol) 1cr | screen(where)=1cr screen(q)=3cr | "
    "watchlist_rw(op,list) 0cr"
)

# Explicit arg contract for the system prompt — the live model fills "args": {}
# when only tool names are given, and plan_validator (correctly) rejects that.
TOOL_ARGS_CONTRACT = (
    "Tool args (EXACT names; every per-symbol tool REQUIRES \"symbol\"):\n"
    "- score_ticker {\"symbol\": \"BBRI\"}\n"
    "- rank_watchlist {}\n"
    "- score_history {\"symbol\": \"BBRI\", \"limit\": 10}\n"
    "- get_fundamentals {\"symbol\": \"BBRI\"}\n"
    "- get_foreign_flow {\"symbol\": \"BBRI\"}\n"
    "- screen {\"where\": {\"market_cap\": {\">\": 100000000000000}}} (1cr) OR "
    "{\"q\": \"natural language query\"} (3cr) — exactly one of where/q\n"
    "- watchlist_rw {\"op\": \"add\"|\"remove\"|\"replace\", \"list\": [\"TLKM\"]}\n"
    "Example valid plan:\n"
    "{\"steps\": [{\"tool\": \"score_history\", \"args\": {\"symbol\": \"BBRI\"}, "
    "\"reason\": \"policy 5: read history before scoring\"}, "
    "{\"tool\": \"score_ticker\", \"args\": {\"symbol\": \"BBRI\"}, "
    "\"reason\": \"current confluence score and decision\"}]}"
)


class LLMCallCounter:
    """§3 LOCKED: max 3 LLM calls per run (intent -> plan -> synthesis)."""

    def __init__(self, limit: int = MAX_LLM_CALLS_PER_RUN):
        self.limit = limit
        self.used: list[str] = []

    def tick(self, purpose: str) -> None:
        if len(self.used) >= self.limit:
            raise LLMError(f"LLM call budget exhausted ({self.limit}/run): "
                           f"tried '{purpose}' after {self.used}")
        self.used.append(purpose)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - len(self.used))


class Planner:
    def __init__(self, llm, ctx, call_counter: LLMCallCounter | None = None):
        self.llm = llm
        self.ctx = ctx
        self.calls = call_counter or LLMCallCounter()

    # ---------------------------------------------------------------- prompts

    def system_prompt(self) -> str:
        ctx = self.ctx
        watchlist = ctx.memory.load_watchlist()
        budget = ctx.budget.status()
        today = datetime.now(WIB).date().isoformat()
        return (
            "You are the planning layer of Bandar, an autonomous pre-market analyst for "
            "IDX swing traders. You plan tool calls; a deterministic engine computes every "
            "number — you NEVER compute or invent figures yourself.\n"
            f"Today (WIB): {today}\n"
            f"Watchlist (the ONLY symbols per-ticker tools may use): {watchlist}\n"
            f"Credits: run {budget['run_spent']}/{budget['run_cap']} used, "
            f"{budget['remaining_total']} spendable today-of-grant.\n"
            f"Tools & costs: {TOOL_COST_TABLE}\n"
            f"{TOOL_ARGS_CONTRACT}\n"
            f"Policy — follow exactly: {PLANNER_POLICIES}\n"
            "Output: ONLY the JSON plan {steps:[{tool,args,reason}]}; keep it minimal; "
            "every step must have a trader-meaningful reason."
        )

    # ------------------------------------------------------------------ calls

    def _hint_smart_money(self, question: str) -> dict | None:
        """Deterministic smart-money fast-path (no LLM). Returns an intent dict or None."""
        try:
            watchlist = [str(s).upper() for s in
                         (self.ctx.memory.load_watchlist() if self.ctx else [])]
        except Exception:                                    # noqa: BLE001 - never block routing
            watchlist = []
        up = question.upper()
        syms = [s for s in watchlist if s and s in up]
        if syms and SMART_MONEY_RE.search(question):
            return {"intent": "smart_money", "symbols": syms, "question": question,
                    "hint": "deterministic-fast-path"}
        return None

    def classify_intent(self, question: str) -> dict:
        """LLM call 1/3: classify free text into one of the 5 intents (+fallback)."""
        hint = self._hint_smart_money(question)
        if hint is not None:
            return hint
        self.calls.tick("intent")
        system = (
            "You are the intent router for Bandar (IDX swing-trading analyst). "
            f"Classify the user's message into exactly one intent: {list(INTENTS)}.\n"
            "- daily_brief: rankings, top pick, watchlist overview, what changed since yesterday\n"
            "- score_ticker: price/TA/confluence/decision about specific watchlist symbols\n"
            "- smart_money: broker activity, foreign flow, accumulation/distribution\n"
            "- valuation: fundamentals, financials, valuation of a symbol\n"
            "- screen: questions about the wider IDX universe beyond the watchlist\n"
            "- fallback: anything else (greetings, out-of-scope, execution requests — "
            "Bandar NEVER executes trades)\n"
            "Also extract IDX symbols mentioned (uppercase, no .JK suffix) and echo the "
            "original question."
        )
        try:
            out = self.llm.complete(system=system, user=question,
                                    schema=INTENT_SCHEMA, purpose="intent")
        except LLMError as exc:
            return {"intent": "fallback", "symbols": [], "question": question,
                    "reason": f"intent transport failed: {exc}"}
        if not isinstance(out, dict) or out.get("intent") not in INTENTS:
            return {"intent": "fallback", "symbols": [], "question": question,
                    "reason": f"intent value invalid: {out!r:.200}"}
        out.setdefault("symbols", [])
        out.setdefault("question", question)
        return out

    def plan(self, question: str, intent: dict | None = None) -> dict:
        """LLM call 2/3: produce a schema-valid, code-validated plan.

        No repair loop (§3: max 3 calls/run). PlanValidationError propagates; the
        executor turns it into the §8 fallback {type:'fallback', message} (FR12).
        """
        self.calls.tick("plan")
        user = question if intent is None else (
            f"Classified intent: {intent.get('intent')} | symbols: {intent.get('symbols')}\n"
            f"User question: {question}"
        )
        raw = self.llm.complete(system=self.system_prompt(), user=user,
                                schema=PLAN_SCHEMA, purpose="plan")
        watchlist = self.ctx.memory.load_watchlist()
        return validate_plan_or_raise(raw, watchlist)


# ------------------------------------------------- deterministic plan fallback (K)

def deterministic_plan(intent: dict, watchlist: list[str], question: str = "") -> dict | None:
    """Hardcoded, validator-safe tool plan per intent — used ONLY when the plan
    LLM call fails (transport/quota outage). The intent is already known (often
    from the 0-LLM fast-path), so the run can still fetch real data and end in
    the deterministic engine brief instead of an error. Returns None when no
    sensible plan exists (e.g. per-symbol intent without a watchlist symbol)."""
    name = (intent or {}).get("intent")
    wl_norm = {_normalize_symbol(str(s)) for s in watchlist}
    syms = [str(s).upper() for s in ((intent or {}).get("symbols") or [])
            if _normalize_symbol(str(s)) in wl_norm][:2]      # budget-safe cap
    steps: list[dict] = []
    if name == "daily_brief":
        steps = [{"tool": "rank_watchlist", "args": {},
                  "reason": "deterministic fallback plan: rank the watchlist"}]
    elif name == "score_ticker" and syms:
        for s in syms:
            steps.append({"tool": "score_history", "args": {"symbol": s},
                          "reason": "policy 5: read history before scoring"})
            steps.append({"tool": "score_ticker", "args": {"symbol": s},
                          "reason": "deterministic fallback plan: current score/decision"})
    elif name == "smart_money" and syms:
        for s in syms:
            steps.append({"tool": "get_foreign_flow", "args": {"symbol": s},
                          "reason": "deterministic fallback plan: foreign flow (smart money)"})
            steps.append({"tool": "score_ticker", "args": {"symbol": s},
                          "reason": "deterministic fallback plan: broker factor via score"})
    elif name == "valuation" and syms:
        for s in syms:
            steps.append({"tool": "get_fundamentals", "args": {"symbol": s},
                          "reason": "deterministic fallback plan: fundamentals"})
    elif name == "screen":
        q = (question or "").strip() or "largest IDX stocks by market cap"
        steps = [{"tool": "screen", "args": {"q": q},
                  "reason": "deterministic fallback plan: universe screen"}]
    if not steps:
        return None
    return validate_plan_or_raise({"steps": steps, "deterministic": True}, watchlist)
