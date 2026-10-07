"""M4a tests (§8 tools + planner + plan_validator; §10: 'tools + planner with
plan_validator tests'). All LLM calls faked, all data calls offline/cached or
monkeypatched — 0 Sectors credits, 0 LLM spend, socket-blocked.

Run: python -m pytest test_m4a.py -v
"""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest

from agent import llm as llm_mod
from agent.config import MAX_LLM_CALLS_PER_RUN, RUNTIME_LLM
from agent.llm import ClaudeLLM, GeminiLLM, LLMError, make_llm
from agent.planner import (INTENT_SCHEMA, PLAN_SCHEMA, PLANNER_POLICIES,
                           LLMCallCounter, Planner)
from agent.tools import TOOLS, TOOL_NAMES, ToolContext, run_tool
from agent.validators import PlanValidationError, validate_plan, validate_plan_or_raise
from budget import CreditBudget, RunCapExceeded
from client import SectorsClient, SectorsNotFoundError
from memory import ScoreMemory

REPO = Path(__file__).resolve().parent
SEED_AS_OF = date(2026, 10, 6)
WATCHLIST = ["BBRI", "BBCA", "BMRI", "ANTM", "DSSA", "PTBA", "MDKA", "ICBP"]


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M4a tests — must be 0 live calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture()
def ctx(tmp_path):
    budget = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    client = SectorsClient(api_key="TESTKEY", mode="offline", budget=budget,
                           cache_dir=REPO / "cache")
    memory = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    memory.save_watchlist(WATCHLIST)
    return ToolContext(client=client, memory=memory, budget=budget, as_of=SEED_AS_OF)


class FakeLLM:
    """Queued responses; records every call. Exception items are raised."""

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


VALID_PLAN = {"steps": [
    {"tool": "rank_watchlist", "args": {}, "reason": "rank the watchlist first"},
    {"tool": "score_history", "args": {"symbol": "BBRI"}, "reason": "policy 5: history before score"},
]}


# -------------------------------------------------------------------- the tools


def test_tool_registry_is_exactly_the_seven():
    assert tuple(TOOLS) == TOOL_NAMES
    assert set(TOOL_NAMES) == {"score_ticker", "rank_watchlist", "score_history",
                               "get_fundamentals", "get_foreign_flow", "screen",
                               "watchlist_rw"}


def test_score_ticker_tool_cached_zero_cr(ctx):
    out = run_tool("score_ticker", {"symbol": "BBRI"}, ctx)
    assert out["ok"] is True
    data = out["data"]
    assert data["symbol"] == "BBRI" and data["as_of"] == "2026-10-05"
    assert data["denominator"] in (4, 5) and 0 <= data["score"] <= data["denominator"]
    assert {"f1_macd", "f2_volume", "f3_ad", "f4_ma_stack", "f5_broker"} <= set(data["factors"])
    assert out["trace"] and all(t.endswith("[cache]") for t in out["trace"])
    assert ctx.budget.status()["total_spent"] == 0


def test_rank_watchlist_tool(ctx, monkeypatch):
    ctx.memory.save_watchlist(["BBRI", "BMRI"])
    out = run_tool("rank_watchlist", {}, ctx)
    assert out["ok"] is True
    ranked = out["data"]["ranked"]
    assert [r["symbol"] for r in ranked] == sorted(
        [r["symbol"] for r in ranked], key=lambda s: -next(
            x["score"] for x in ranked if x["symbol"] == s))
    assert len(ranked) == 2
    assert ctx.budget.status()["total_spent"] == 0


def test_score_history_tool_zero_cr(ctx):
    out = run_tool("score_history", {"symbol": "BBRI"}, ctx)
    assert out["ok"] is True and out["data"]["history"] == []      # nothing recorded yet

    r = run_tool("score_ticker", {"symbol": "BBRI"}, ctx)
    ctx.memory.record_score(r["data"])
    out2 = run_tool("score_history", {"symbol": "BBRI", "limit": 5}, ctx)
    assert len(out2["data"]["history"]) == 1
    assert out2["data"]["history"][0]["as_of"] == "2026-10-05"


def test_watchlist_rw_tool(ctx):
    out = run_tool("watchlist_rw", {"op": "add", "list": ["tlkm.jk"]}, ctx)
    assert out["ok"] is True and "TLKM" in out["data"]["watchlist"]
    out = run_tool("watchlist_rw", {"op": "remove", "list": ["TLKM"]}, ctx)
    assert "TLKM" not in out["data"]["watchlist"]
    out = run_tool("watchlist_rw", {"op": "replace", "list": ["BBRI"]}, ctx)
    assert out["data"]["watchlist"] == ["BBRI"]
    assert ctx.memory.load_watchlist() == ["BBRI"]                 # persisted


def test_screen_routes_structured_vs_nl(ctx, monkeypatch):
    calls = []
    monkeypatch.setattr(ctx.client, "screener_structured",
                        lambda where: calls.append(("where", where)) or {"results": []})
    monkeypatch.setattr(ctx.client, "screener_nl",
                        lambda q: calls.append(("q", q)) or {"results": []})

    out1 = run_tool("screen", {"where": {"market_cap": {">": 1e14}}}, ctx)
    out2 = run_tool("screen", {"q": "banks with foreign inflow"}, ctx)
    assert out1["ok"] and out1["data"]["mode"] == "structured"
    assert out2["ok"] and out2["data"]["mode"] == "nl"
    assert calls == [("where", {"market_cap": {">": 1e14}}), ("q", "banks with foreign inflow")]


def test_fundamentals_and_flow_passthrough_and_404(ctx, monkeypatch):
    monkeypatch.setattr(ctx.client, "fundamentals", lambda s: {"profile": {"name": s}})
    out = run_tool("get_fundamentals", {"symbol": "BBRI"}, ctx)
    assert out["ok"] is True and out["data"]["report"] == {"profile": {"name": "BBRI"}}

    def not_found(s):
        raise SectorsNotFoundError(f"404 on /v2/foreign-flow/{s}/")
    monkeypatch.setattr(ctx.client, "foreign_flow", not_found)
    out = run_tool("get_foreign_flow", {"symbol": "BBRI"}, ctx)
    assert out["ok"] is False and out["error"] == "not_found"     # honest-null (FR2)


def test_budget_breach_aborts_step_honestly(ctx, monkeypatch):
    """Planner policy 4: budget breach -> abort step, narrate — never crash the loop."""
    from agent import tools as tools_mod

    def boom(*a, **k):
        raise RunCapExceeded("run cap: run_spent=60 + cost=1 > 60")
    monkeypatch.setattr(tools_mod.engine, "score_ticker", boom)
    out = run_tool("score_ticker", {"symbol": "BBRI"}, ctx)
    assert out["ok"] is False and out["error"] == "budget_abort"
    assert "run cap" in out["message"]


def test_unknown_tool_rejected_at_runtime(ctx):
    out = run_tool("launch_missiles", {}, ctx)
    assert out["ok"] is False and out["error"] == "unknown_tool"


# --------------------------------------------------- plan_validator (§8 LOCKED)


def test_valid_plan_passes():
    assert validate_plan(VALID_PLAN, WATCHLIST) == []
    assert validate_plan_or_raise(VALID_PLAN, WATCHLIST) is VALID_PLAN


def test_validator_known_tools_only():
    plan = {"steps": [{"tool": "execute_trade", "args": {}, "reason": "yolo"}]}
    errors = validate_plan(plan, WATCHLIST)
    assert any("unknown tool" in e for e in errors)
    assert any("execute_trade" in e for e in errors)


def test_validator_requires_schema_fields():
    plan = {"steps": [{"tool": "rank_watchlist", "args": {}}]}       # missing reason
    assert any("missing 'reason'" in e for e in validate_plan(plan, WATCHLIST))
    plan = {"steps": [{"tool": "rank_watchlist", "reason": "x"}]}    # missing args
    assert any("missing 'args'" in e for e in validate_plan(plan, WATCHLIST))
    assert any("must be an array" in e for e in validate_plan({"steps": "nope"}, WATCHLIST))
    assert any("must be an object" in e for e in validate_plan(["steps"], WATCHLIST))
    assert any("empty" in e for e in validate_plan({"steps": []}, WATCHLIST))


def test_validator_watchlist_symbols_only():
    """§8: per-symbol data tools accept watchlist symbols ONLY (normalized)."""
    plan = {"steps": [{"tool": "score_ticker", "args": {"symbol": "GOTO"}, "reason": "r"}]}
    errors = validate_plan(plan, WATCHLIST)
    assert any("not on the watchlist" in e for e in errors)

    ok = {"steps": [{"tool": "score_ticker", "args": {"symbol": "bbri.jk"}, "reason": "r"}]}
    assert validate_plan(ok, WATCHLIST) == []                        # normalization

    for tool in ("score_history", "get_fundamentals", "get_foreign_flow"):
        p = {"steps": [{"tool": tool, "args": {"symbol": "XYZZ"}, "reason": "r"}]}
        assert any("not on the watchlist" in e for e in validate_plan(p, WATCHLIST)), tool


def test_validator_arg_types_and_enums():
    p = {"steps": [{"tool": "score_ticker", "args": {"symbol": 42}, "reason": "r"}]}
    assert any("must be str" in e for e in validate_plan(p, WATCHLIST))

    p = {"steps": [{"tool": "score_history", "args": {"symbol": "BBRI", "limit": "ten"},
                    "reason": "r"}]}
    assert any("'limit' must be int" in e for e in validate_plan(p, WATCHLIST))

    p = {"steps": [{"tool": "watchlist_rw", "args": {"op": "nuke", "list": []}, "reason": "r"}]}
    assert any("must be one of" in e for e in validate_plan(p, WATCHLIST))

    p = {"steps": [{"tool": "score_ticker", "args": {"symbol": "BBRI", "leveraged": True},
                    "reason": "r"}]}
    assert any("unexpected args" in e for e in validate_plan(p, WATCHLIST))


def test_validator_screen_exactly_one_of():
    both = {"steps": [{"tool": "screen", "args": {"where": {}, "q": "x"}, "reason": "r"}]}
    assert any("exactly one of" in e for e in validate_plan(both, WATCHLIST))
    neither = {"steps": [{"tool": "screen", "args": {}, "reason": "r"}]}
    assert any("exactly one of" in e for e in validate_plan(neither, WATCHLIST))
    ok = {"steps": [{"tool": "screen", "args": {"q": "coal miners"}, "reason": "r"}]}
    assert validate_plan(ok, WATCHLIST) == []


def test_validator_watchlist_rw_may_add_new_names():
    """watchlist_rw is the ONE tool allowed non-watchlist names (it manages the list)."""
    p = {"steps": [{"tool": "watchlist_rw", "args": {"op": "add", "list": ["TLKM", "PGAS"]},
                    "reason": "user asked to track TLKM"}]}
    assert validate_plan(p, WATCHLIST) == []


def test_validate_plan_or_raise_collects_errors():
    p = {"steps": [{"tool": "nope", "args": {}, "reason": ""},
                   {"tool": "score_ticker", "args": {"symbol": "GOTO"}}]}
    with pytest.raises(PlanValidationError) as excinfo:
        validate_plan_or_raise(p, WATCHLIST)
    assert len(excinfo.value.errors) >= 3        # unknown tool + empty reason + missing/symbol


# ---------------------------------------------------------------------- planner


def test_planner_system_prompt_carries_policies_and_context(ctx):
    p = Planner(FakeLLM([]), ctx)
    prompt = p.system_prompt()
    for marker in ("(1)", "(2)", "(3)", "(4)", "(5)"):            # all five §8 policies
        assert marker in prompt
    assert "read history before scoring" in prompt                # policy 5 verbatim-ish
    assert "BBRI" in prompt and "watchlist" in prompt.lower()
    assert "60" in prompt                                          # run cap surfaced
    assert "NEVER compute" in prompt                               # engine-computes rule (§1)
    assert '{"symbol": "BBRI"}' in prompt                          # explicit arg contract
    assert "Example valid plan" in prompt


def test_planner_plan_happy_path(ctx):
    fake = FakeLLM([VALID_PLAN])
    p = Planner(fake, ctx)
    plan = p.plan("what changed on BBRI?", intent={"intent": "score_ticker", "symbols": ["BBRI"]})
    assert plan == VALID_PLAN
    assert len(fake.calls) == 1                                    # exactly ONE LLM call
    call = fake.calls[0]
    assert call["schema"] == PLAN_SCHEMA                           # response_schema:plan (§3)
    assert call["purpose"] == "plan"
    assert "score_ticker" in call["user"] and "daily_brief" not in call["user"]


def test_planner_invalid_plan_raises_without_repair(ctx):
    """§3 LOCKED max-3-calls: no repair loop — invalid plan -> error -> §8 fallback."""
    bad = {"steps": [{"tool": "score_ticker", "args": {"symbol": "GOTO"}, "reason": "r"}]}
    fake = FakeLLM([bad, VALID_PLAN])                              # 2nd never consumed
    p = Planner(fake, ctx)
    with pytest.raises(PlanValidationError):
        p.plan("rank my watchlist")
    assert len(fake.calls) == 1                                    # NO second call


def test_intent_classification_and_graceful_fallback(ctx):
    fake = FakeLLM([{"intent": "smart_money", "symbols": ["BMRI"], "question": "q"}])
    p = Planner(fake, ctx)
    # deterministic fast-path: watchlist symbol + smart-money vocab -> 0 LLM calls
    out = p.classify_intent("who is accumulating BMRI?")
    assert out["intent"] == "smart_money" and out["symbols"] == ["BMRI"]
    assert out.get("hint") == "deterministic-fast-path"
    assert fake.calls == []                      # fast path must not spend an LLM call

    # non-fast-path question (no symbol) still routes through the LLM intent router
    out = p.classify_intent("who's accumulating in the market today?")
    assert out["intent"] == "smart_money" and out["symbols"] == ["BMRI"]
    assert fake.calls[0]["schema"] == INTENT_SCHEMA

    # transport failure -> graceful fallback (FR12), never a crash — with observable reason
    fake_err = FakeLLM([LLMError("gemini HTTP 500")])
    out = Planner(fake_err, ctx).classify_intent("hello?")
    assert out["intent"] == "fallback" and out["symbols"] == []
    assert "transport failed" in out["reason"]

    # garbage intent value -> fallback
    fake_bad = FakeLLM([{"intent": "moon", "symbols": [], "question": "x"}])
    out = Planner(fake_bad, ctx).classify_intent("x")
    assert out["intent"] == "fallback" and "invalid" in out["reason"]


def test_llm_call_counter_enforces_max_3(ctx):
    counter = LLMCallCounter()
    fake = FakeLLM([{"intent": "daily_brief", "symbols": [], "question": "q"}, VALID_PLAN])
    p = Planner(fake, ctx, call_counter=counter)
    p.classify_intent("q")                                          # 1 intent
    p.plan("q")                                                     # 2 plan
    assert counter.remaining == 1                                   # synthesis left
    counter.tick("synthesis")                                       # 3
    assert counter.remaining == 0
    with pytest.raises(LLMError, match="exhausted"):
        counter.tick("extra")                                       # 4th blocked
    assert MAX_LLM_CALLS_PER_RUN == 3


def test_runtime_llm_config_exact():
    """§3 [LOCKED] block — byte-for-byte."""
    assert RUNTIME_LLM == {
        "primary": "gemini-flash-3.8",
        "fallback": "claude-sonnet-4.5",
        "enforce": ["response_schema:plan", "response_schema:submit_brief",
                    "numeric_trace_validator", "plan_validator"],
        "temp": 0.2,
    }


# ----------------------------------------------------------------- LLM transport


def test_gemini_request_shape(monkeypatch):
    captured = {}
    payload = {"candidates": [{"content": {"parts": [{"text": json.dumps(VALID_PLAN)}]}}]}

    class R:
        status_code = 200
        text = "ok"
        @staticmethod
        def json():
            return payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    llm = GeminiLLM(api_key="GKEY")
    out = llm.complete(system="SYS", user="USER", schema=PLAN_SCHEMA, purpose="plan")

    assert out == VALID_PLAN
    assert "gemini-3.8-flash:generateContent" in captured["url"]   # spec name resolved to API id
    assert captured["headers"]["x-goog-api-key"] == "GKEY"
    gc = captured["body"]["generationConfig"]
    assert gc["temperature"] == 0.2                                # §3 temp
    assert gc["response_mime_type"] == "application/json"          # §3 controlled generation
    assert gc["response_schema"] == PLAN_SCHEMA
    assert captured["body"]["system_instruction"]["parts"][0]["text"] == "SYS"


def test_gemini_http_error_raises_llm_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "RETRY_WAITS", [0, 0])

    class R:
        status_code = 429
        text = "RATE_LIMIT"
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: R())
    with pytest.raises(LLMError, match="429"):
        GeminiLLM(api_key="GKEY").complete("s", "u")


def test_transient_503_retried_then_succeeds(monkeypatch):
    """A 503 transport attempt never completed a call — bounded retry is allowed
    within the §3 max-3-LLM-calls budget (the logical call still counts once)."""
    monkeypatch.setattr(llm_mod, "RETRY_WAITS", [0, 0, 0])
    attempts = []
    payload = {"candidates": [{"content": {"parts": [{"text": '{"steps": []}'}]}}]}

    class Bad:
        status_code = 503
        text = "UNAVAILABLE"

    class Good:
        status_code = 200
        text = "ok"
        @staticmethod
        def json():
            return payload

    def fake_post(url, headers=None, json=None, timeout=None):
        attempts.append(1)
        return Bad() if len(attempts) == 1 else Good()

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    out = GeminiLLM(api_key="GKEY").complete("s", "u")
    assert out == {"steps": []} and len(attempts) == 2


def test_claude_fallback_hotswap_shape(monkeypatch):
    captured = {}
    payload = {"content": [{"text": json.dumps(VALID_PLAN)}]}

    class R:
        status_code = 200
        text = "ok"
        @staticmethod
        def json():
            return payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    llm = make_llm("fallback", api_key="AKEY")
    assert isinstance(llm, ClaudeLLM) and llm.model == "claude-sonnet-4.5"   # D8 config-only swap
    out = llm.complete(system="SYS", user="USER", schema=PLAN_SCHEMA, purpose="plan")

    assert out == VALID_PLAN
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "AKEY"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-4-5"          # resolved API id
    assert captured["body"]["temperature"] == 0.2
    assert "JSON Schema" in captured["body"]["system"]              # schema folded into prompt


def test_make_llm_primary_is_gemini():
    llm = make_llm("primary", api_key="GKEY")
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-flash-3.8"


def test_model_id_resolution_keeps_locked_config():
    """RUNTIME_LLM keeps the LOCKED §3 spec names; transport resolves real API ids."""
    from agent.llm import resolve_model
    assert resolve_model("gemini-flash-3.8") == "gemini-3.8-flash"   # verified via ListModels
    assert resolve_model("claude-sonnet-4.5") == "claude-sonnet-4-5"
    assert resolve_model("some-future-model") == "some-future-model"  # passthrough
    assert RUNTIME_LLM["primary"] == "gemini-flash-3.8"              # config untouched


def test_network_timeout_becomes_graceful_fallback(ctx, monkeypatch):
    """A read/connect timeout must become LLMError -> graceful fallback, never a crash."""
    import requests as _requests
    from agent import llm as llm_mod

    def boom(*a, **k):
        raise _requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    p = Planner(llm_mod.GeminiLLM(api_key="TESTKEY"), ctx)
    out = p.classify_intent("top pick today?")
    assert out["intent"] == "fallback"
    assert "transport failed" in out["reason"] and "ReadTimeout" in out["reason"]


# ------------------------------------------- K: deterministic plan fallback


def test_deterministic_plan_builders_pass_validator():
    """Every intent gets a hardcoded, validator-safe plan (used when plan LLM is down)."""
    from agent.planner import deterministic_plan
    wl = ["BBRI", "DSSA"]
    cases = [
        ({"intent": "smart_money", "symbols": ["DSSA"]}, ["get_foreign_flow", "score_ticker"]),
        ({"intent": "score_ticker", "symbols": ["bbri"]}, ["score_history", "score_ticker"]),
        ({"intent": "daily_brief", "symbols": []}, ["rank_watchlist"]),
        ({"intent": "valuation", "symbols": ["BBRI"]}, ["get_fundamentals"]),
        ({"intent": "screen", "symbols": []}, ["screen"]),
    ]
    for intent, tools in cases:
        plan = deterministic_plan(intent, wl, "which energy stocks are in play?")
        assert plan is not None, intent
        assert [s["tool"] for s in plan["steps"]] == tools
        assert plan["deterministic"] is True
        assert validate_plan(plan, wl) == []


def test_deterministic_plan_refuses_when_unusable():
    """No watchlist symbol for a per-symbol intent (or unknown intent) -> None,
    so the executor keeps the honest 'Planner unavailable' fallback."""
    from agent.planner import deterministic_plan
    wl = ["BBRI"]
    assert deterministic_plan({"intent": "smart_money", "symbols": ["GOTO"]}, wl) is None
    assert deterministic_plan({"intent": "score_ticker", "symbols": []}, wl) is None
    assert deterministic_plan({"intent": "fallback", "symbols": []}, wl) is None
    assert deterministic_plan({}, wl) is None


def test_deterministic_plan_caps_symbols_for_budget():
    from agent.planner import deterministic_plan
    plan = deterministic_plan({"intent": "smart_money", "symbols": ["BBRI", "DSSA", "ANTM"]},
                              WATCHLIST)
    uniq = {s["args"]["symbol"] for s in plan["steps"] if "symbol" in s.get("args", {})}
    assert uniq == {"BBRI", "DSSA"}                            # budget-safe cap: 2 symbols


def test_foreign_flow_summary_engine_aggregates(ctx):
    """O1: engine computes signed window aggregates so synthesis can quote cumulative
    flow without deriving arithmetic itself (FR10). Signs preserved: <0 = outflow."""
    out = run_tool("get_foreign_flow", {"symbol": "PTBA"}, ctx)
    assert out["ok"] is True
    flow = out["data"]["flow"]
    rows = flow if isinstance(flow, list) else (flow.get("data") or flow.get("results") or [])
    s = out["data"]["summary"]
    dated = sorted((r["date"], r["net_foreign_inflow"]) for r in rows
                   if isinstance(r.get("net_foreign_inflow"), (int, float)))
    vals = [v for _, v in dated]
    assert s["net_total"] == sum(vals)
    assert s["net_last_5d"] == sum(vals[-5:])
    assert s["net_last_20d"] == sum(vals[-20:])
    assert s["n_days"] == len(vals)
    assert s["window_start"] == dated[0][0]
    assert s["window_end"] == dated[-1][0]
    assert s["inflow_days"] == sum(1 for v in vals if v > 0)
    assert s["outflow_days"] == sum(1 for v in vals if v < 0)
    assert s["bias"] == ("net inflow" if s["net_total"] > 0
                         else "net outflow" if s["net_total"] < 0 else "balanced")
    assert s["top_outflow_value"] < 0 < s["top_inflow_value"]   # signs are the info
