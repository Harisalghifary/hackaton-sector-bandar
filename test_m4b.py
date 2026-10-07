"""M4b tests (§8 executor + numeric_trace_validator + submit_brief synthesis).

FR10 (every figure traces), FR12 (graceful fallback), D5 (boxed JSON), D6/D7
(memory delta + upsert), policy 4 (budget truncation narration), §3 (max 3 LLM
calls). All offline: FakeLLM + offline/cached client — 0 credits, 0 LLM spend.

Run: python -m pytest test_m4b.py -v
"""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest

from agent.executor import SUBMIT_BRIEF_SCHEMA, SYNTHESIS_SYSTEM, run_agent
from agent.llm import LLMError
from agent.tools import ToolContext
from agent.validators import collect_allowed_numbers, validate_brief_numbers
from budget import CreditBudget
from client import SectorsClient
from memory import ScoreMemory

REPO = Path(__file__).resolve().parent
SEED_AS_OF = date(2026, 10, 6)
WATCHLIST = ["BBRI", "BBCA", "BMRI", "ANTM", "DSSA", "PTBA", "MDKA", "ICBP"]

INTENT_SMART = {"intent": "score_ticker", "symbols": ["BBRI"], "question": "how is BBRI?"}
PLAN_BBRI = {"steps": [
    {"tool": "score_history", "args": {"symbol": "BBRI"}, "reason": "policy 5: history first"},
    {"tool": "score_ticker", "args": {"symbol": "BBRI"}, "reason": "current confluence"},
]}
# Real cached BBRI figures (verified in M2 demo): 0/5, WAIT, close 3120, as_of 2026-10-05
CLEAN_BRIEF = {
    "symbol": "BBRI",
    "extracted": ["BBRI score 0/5, decision WAIT, deploy 0%",
                  "Close 3120 as of 2026-10-05", "234 bars scored"],
    "interpretation": "No confluence while distribution persists; standing aside is the disciplined play.",
    "action_plan": ["Stay in cash and re-check after the next session."],
    "risk_flags": ["score 0 of 5 with bearish bias"],
}


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M4b tests — must be 0 live calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture()
def ctx(tmp_path):
    budget = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    client = SectorsClient(api_key="TESTKEY", mode="offline", budget=budget,
                           cache_dir=REPO / "cache")
    memory = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    memory.save_watchlist(WATCHLIST)
    return ToolContext(client=client, memory=memory, budget=budget, as_of=SEED_AS_OF)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, schema=None, purpose=""):
        self.calls.append({"system": system, "user": user, "schema": schema, "purpose": purpose})
        if not self.responses:
            raise AssertionError("FakeLLM called more times than expected")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ------------------------------------------------- numeric trace validator units


def test_allowed_numbers_from_corpus():
    corpus = [{"data": {"score": 4, "close": 3120.0, "conc": 0.4567,
                        "decision": {"deploy_pct": "60-80%"},
                        "as_of": "2026-10-05"}},
              ["flow:/v2/foreign-flow/BBRI/ [live, 1cr]"]]
    allowed = collect_allowed_numbers(corpus)
    for tok in ("4", "3120", "3120.0", "0.4567", "60", "80", "2026", "10", "05", "5", "1"):
        assert tok in allowed, tok


def test_clean_brief_passes_unchanged():
    allowed = collect_allowed_numbers([CLEAN_BRIEF, {"score": 0, "denominator": 5,
                                                     "close": 3120.0, "bars": 234,
                                                     "as_of": "2026-10-05", "deploy_pct": "0%"}])
    cleaned, dropped = validate_brief_numbers(CLEAN_BRIEF, allowed)
    assert dropped == [] and cleaned["extracted"] == CLEAN_BRIEF["extracted"]


def test_fabricated_items_dropped_flagged():
    allowed = collect_allowed_numbers([{"score": 0, "close": 3120}])
    brief = dict(CLEAN_BRIEF,
                 extracted=["score 0", "target price 9999"],
                 action_plan=["buy at 3120", "average down 20% more"],
                 risk_flags=["clean", "insider buying 123456 lots"])
    cleaned, dropped = validate_brief_numbers(brief, allowed)
    assert cleaned["extracted"] == ["score 0"]                    # 9999 item dropped
    assert cleaned["action_plan"] == ["buy at 3120"]
    assert cleaned["risk_flags"] == ["clean"]
    assert {d["section"] for d in dropped} == {"extracted", "action_plan", "risk_flags"}
    assert any("9999" in d["figures"] for d in dropped)


def test_interpretation_tokens_removed_not_silent():
    allowed = collect_allowed_numbers([{"score": 0}])
    brief = dict(CLEAN_BRIEF, interpretation="score 0 now but could rally 42% to 9999 soon")
    cleaned, dropped = validate_brief_numbers(brief, allowed)
    assert "42" not in cleaned["interpretation"] and "9999" not in cleaned["interpretation"]
    assert cleaned["interpretation"].count("[removed]") == 2
    assert cleaned["interpretation_flag"] == ["42", "9999"]
    assert any(d["section"] == "interpretation" for d in dropped)


def test_comma_and_float_normalization():
    allowed = collect_allowed_numbers([{"value": 5200, "ratio": 1.50}])
    brief = dict(CLEAN_BRIEF, extracted=["turned over 5,200 lots", "ratio 1.5"],
                 risk_flags=["value 5200 flagged"])
    assert validate_brief_numbers(brief, allowed)[1] == []


# ------------------------------------------------------------------- executor


def test_happy_path_brief_zero_credits(ctx):
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, CLEAN_BRIEF])
    out = run_agent("how is BBRI today?", ctx, llm=fake)

    assert out["type"] == "brief"
    assert out["brief"]["symbol"] == "BBRI"
    assert out["dropped_figures"] == []                            # every figure traced (FR10)
    assert out["llm_calls"] == ["intent", "plan", "synthesis"]     # exactly 3 (§3)
    assert out["truncated"] is False
    assert all(t.endswith("[cache]") or t == "" for t in out["trace"])
    assert ctx.budget.status()["total_spent"] == 0                 # 0 credits
    assert fake.calls[2]["schema"] == SUBMIT_BRIEF_SCHEMA          # D5 boxed JSON
    assert fake.calls[2]["purpose"] == "synthesis"


def test_synthesis_prompt_carries_voice_rules_and_corpus(ctx):
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, CLEAN_BRIEF])
    run_agent("how is BBRI?", ctx, llm=fake)
    system, user = fake.calls[2]["system"], fake.calls[2]["user"]
    assert "2 sentences" in system and "EXACTLY" in system
    assert "NO derived arithmetic" in system and "NO invented deltas" in system
    assert SYNTHESIS_SYSTEM == system
    assert "f1_macd" in user and "[cache]" in user                 # corpus embedded


def test_fabricated_figures_dropped_end_to_end(ctx):
    dirty = dict(CLEAN_BRIEF,
                 extracted=CLEAN_BRIEF["extracted"] + ["fair value 9999 per share"],
                 interpretation="Cheap at 3120 with 123456% upside ahead.")
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, dirty])
    out = run_agent("how is BBRI?", ctx, llm=fake)
    assert out["type"] == "brief"
    assert all("9999" not in item for item in out["brief"]["extracted"])
    assert "123456" not in out["brief"]["interpretation"]
    assert out["brief"]["interpretation_flag"] == ["123456"]
    assert len(out["dropped_figures"]) >= 2                        # audit trail kept


def test_intent_fallback_short_circuits(ctx):
    fake = FakeLLM([{"intent": "fallback", "symbols": [], "question": "execute buy order"}])
    out = run_agent("execute buy order now", ctx, llm=fake)
    assert out["type"] == "fallback"
    assert "never execute trades" in out["message"]
    assert out["llm_calls"] == ["intent"]                          # no wasted calls
    assert ctx.budget.status()["total_spent"] == 0


def test_invalid_plan_becomes_fallback(ctx):
    bad_plan = {"steps": [{"tool": "score_ticker", "args": {"symbol": "GOTO"}, "reason": "r"}]}
    fake = FakeLLM([INTENT_SMART, bad_plan])
    out = run_agent("how is GOTO?", ctx, llm=fake)
    assert out["type"] == "fallback"
    assert "not on the watchlist" in out["message"]
    assert out["llm_calls"] == ["intent", "plan"]                  # NO synthesis, no repair


def test_synthesis_error_becomes_deterministic_fallback(ctx):
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, LLMError("gemini HTTP 500 (synthesis)")])
    out = run_agent("how is BBRI?", ctx, llm=fake)
    # synthesis down -> engine-built brief, never a bare error (section E)
    assert out["type"] == "brief" and out["deterministic_fallback"] is True
    assert out["brief"]["symbol"] == "BBRI"
    assert "gemini HTTP 500" in out["synthesis_error"]
    assert len(out["tool_results"]) == 2                           # honest: data still shown
    assert out["llm_calls"] == ["intent", "plan", "synthesis"]


def test_schema_invalid_brief_becomes_deterministic_fallback(ctx):
    broken = {"symbol": "BBRI", "extracted": [], "action_plan": [], "risk_flags": []}
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, broken])              # interpretation missing
    out = run_agent("how is BBRI?", ctx, llm=fake)
    assert out["type"] == "brief" and out["deterministic_fallback"] is True
    assert out["raw_brief"] == broken                              # never silently patched


def test_budget_abort_truncation_narrated(tmp_path):
    """Policy 4: budget breach -> step aborted, truncation narrated honestly."""
    budget = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    budget._state["run_spent"] = 60                                # hard cap already reached
    budget._save()
    client = SectorsClient(api_key="TESTKEY", mode="cache-first", budget=budget,
                           cache_dir=REPO / "cache")                # miss -> precheck aborts
    memory = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    memory.save_watchlist(WATCHLIST)
    ctx2 = ToolContext(client=client, memory=memory, budget=budget, as_of=SEED_AS_OF)

    plan = {"steps": [{"tool": "get_foreign_flow", "args": {"symbol": "BBRI"}, "reason": "flow"}]}
    clean = {"symbol": "BBRI", "extracted": [],
             "interpretation": "Data collection was cut short by the run budget.",
             "action_plan": [], "risk_flags": ["run budget exhausted; results truncated"]}
    fake = FakeLLM([INTENT_SMART, plan, clean])
    # no watchlist symbol in the question -> exercises the LLM intent path (not fast-path)
    out = run_agent("flow picture for the tape today?", ctx2, llm=fake)

    assert out["truncated"] is True
    assert out["tool_results"][0]["error"] == "budget_abort"
    assert '"truncated": true' in fake.calls[2]["user"]            # narrated to synthesis
    assert out["type"] == "brief"
    assert budget.status()["total_spent"] == 0                     # aborted BEFORE any HTTP


def test_memory_upsert_and_delta_context(ctx):
    """D6/D7: prior snapshot feeds delta context; any run upserts by (ticker, as_of)."""
    prior = {"symbol": "BBRI", "as_of": "2026-10-02", "score": 2, "denominator": 5,
             "status": "ok", "factors": None, "gates": None, "trade_plan": None,
             "decision": {"label": "SCALP", "deploy_pct": "15-20%", "bias": "NEUTRAL"}}
    ctx.memory.record_score(prior)

    brief_quoting_prior = dict(CLEAN_BRIEF,
                               extracted=CLEAN_BRIEF["extracted"] + ["was 2/5 SCALP on 2026-10-02"])
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, brief_quoting_prior])
    out = run_agent("how is BBRI?", ctx, llm=fake)

    assert out["memory_context"] == [{"symbol": "BBRI", "prior_as_of": "2026-10-02",
                                      "prior_score": 2, "prior_denominator": 5,
                                      "prior_decision": "SCALP"}]
    # prior figures are quotable (corpus includes memory context) -> not dropped
    assert any("was 2/5 SCALP" in item for item in out["brief"]["extracted"])
    hist = ctx.memory.get_history("BBRI")
    assert [h["as_of"] for h in hist] == ["2026-10-05", "2026-10-02"]   # upserted (D7)
    assert hist[0]["score"] == 0 and hist[0]["decision"] == "WAIT"


def test_max_three_llm_calls_never_exceeded(ctx):
    fake = FakeLLM([INTENT_SMART, PLAN_BBRI, CLEAN_BRIEF, CLEAN_BRIEF])
    out = run_agent("how is BBRI?", ctx, llm=fake)
    assert len(out["llm_calls"]) == 3
    assert len(fake.calls) == 3                                    # 4th response untouched
