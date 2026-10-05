"""M1 data-layer tests (§10 prompt + §5 contract + FR7/FR8).

Proof obligations:
- offline cache miss raises, never spends (FR8)
- cache hit spends 0, guard never touched (FR8, §5)
- billing: 2xx -> cost · 404 -> exactly 1 · all other 4xx/5xx -> free (FR7)
- caps: run 60 hard, total 800 spendable (reserve 200), daily 200 soft warn
- persistence, rollover, logging, no-Bearer auth, deterministic cache keys

ALL HTTP is mocked and an autouse fixture blocks sockets → 0 Sectors credits burned.
Run: python -m pytest test_m1.py -v
"""

from __future__ import annotations

import json
import logging
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import requests

import client as client_mod
from budget import (
    CreditBudget,
    ReserveExceeded,
    RunCapExceeded,
    billable_amount,
)
from client import (
    OfflineCacheMissError,
    SectorsAPIError,
    SectorsConnectionError,
    SectorsClient,
    SectorsNotFoundError,
    _cache_key,
)

WIB = ZoneInfo("Asia/Jakarta")


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Hard guarantee: any socket use during M1 tests fails the suite (0 live calls)."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access attempted during M1 tests — must be 0 live calls")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)


@pytest.fixture()
def budget(tmp_path):
    return CreditBudget(
        state_path=tmp_path / "state" / "budget_state.json",
        log_path=tmp_path / "state" / "budget_log.jsonl",
    )


class SpyBudget(CreditBudget):
    """Counts guard touches — cache hits must never call precheck/charge (§5)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precheck_calls = 0
        self.charge_calls = 0

    def precheck(self, cost):
        self.precheck_calls += 1
        return super().precheck(cost)

    def charge(self, cost, status, endpoint):
        self.charge_calls += 1
        return super().charge(cost, status, endpoint)


@pytest.fixture()
def make_client(tmp_path, budget):
    def _make(mode="cache-first", api_key="TESTKEY", budget_override=None):
        return SectorsClient(
            api_key=api_key,
            mode=mode,
            budget=budget_override if budget_override is not None else budget,
            cache_dir=tmp_path / "cache",
        )
    return _make


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}

    def json(self):
        return self._payload


def install_fake_get(monkeypatch, response=None, exc=None):
    """Patch requests.get inside client.py; returns the list of captured calls."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if exc is not None:
            raise exc
        return response if response is not None else FakeResponse(200)

    monkeypatch.setattr(client_mod.requests, "get", fake_get)
    return calls


def read_log_lines(budget):
    if not budget.log_path.exists():
        return []
    return [json.loads(line) for line in budget.log_path.read_text().splitlines() if line]


# ------------------------------------------------------------------ FR8: offline


def test_offline_miss_raises_no_http_no_spend(make_client, budget, monkeypatch):
    calls = install_fake_get(monkeypatch, response=FakeResponse(200))
    c = make_client(mode="offline", api_key=None)  # offline needs no API key

    with pytest.raises(OfflineCacheMissError):
        c.broker_summary("BBRI")

    assert calls == []                     # never spends, never calls
    assert budget.status()["total_spent"] == 0
    assert read_log_lines(budget) == []    # no live call happened


@pytest.mark.parametrize("mode", ["cache-first", "offline"])
def test_cache_hit_spends_zero(mode, tmp_path, make_client, monkeypatch):
    """The prompt's core proof: a cache hit returns data with 0 credits and no guard touch."""
    spy = SpyBudget(
        state_path=tmp_path / "state" / "budget_state.json",
        log_path=tmp_path / "state" / "budget_log.jsonl",
    )
    payload = {"results": [{"date": "2026-09-21", "close": 5200}]}

    # Seed the cache through ONE live 2xx (bills 1) using cache-first mode —
    # offline mode never calls HTTP, so seeding must not go through it.
    install_fake_get(monkeypatch, response=FakeResponse(200, payload))
    seeder = make_client(mode="cache-first", budget_override=spy)
    assert seeder.ohlcv("BBRI", "2026-06-23", "2026-09-21") == payload
    assert spy.charge_calls == 1 and spy.precheck_calls == 1
    spent_after_seed = spy.status()["total_spent"]

    # Mode under test now reads the warm cache.
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, payload))
    c = make_client(mode=mode, api_key=None if mode == "offline" else "TESTKEY",
                    budget_override=spy)
    for _ in range(3):
        assert c.ohlcv("BBRI", "2026-06-23", "2026-09-21") == payload

    assert calls == []                                     # no HTTP on hits
    assert spy.precheck_calls == 1 and spy.charge_calls == 1  # guard NEVER touched by hits
    assert spy.status()["total_spent"] == spent_after_seed    # 0 credits spent on hits


# ------------------------------------------------------------- FR7: billing rules


def test_charge_bills_2xx_and_writes_cache(make_client, budget, monkeypatch):
    payload = {"results": [{"close": 5200}]}
    install_fake_get(monkeypatch, response=FakeResponse(200, payload))
    c = make_client()

    assert c.foreign_flow("BBRI") == payload

    st = budget.status()
    assert (st["total_spent"], st["daily_spent"], st["run_spent"]) == (1, 1, 1)
    # cache file written under the right namespace
    assert len(list((c.cache_dir / "foreign_flow").glob("*.json"))) == 1


def test_404_bills_exactly_one_and_not_cached(make_client, budget, monkeypatch):
    """404 on a cost-3 endpoint bills exactly 1 (LOCKED) and writes no cache."""
    install_fake_get(monkeypatch, response=FakeResponse(404))
    c = make_client()

    with pytest.raises(SectorsNotFoundError):
        c.screener_nl("banks with foreign inflow")

    st = budget.status()
    assert st["total_spent"] == 1          # not 3
    assert st["run_spent"] == 1
    assert not (c.cache_dir / "screener").exists()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 422, 500, 502, 503])
def test_free_statuses(status, make_client, budget, monkeypatch):
    """FR7: free on 4xx/5xx except 404 — including unlisted ones like 422."""
    install_fake_get(monkeypatch, response=FakeResponse(status))
    c = make_client()

    with pytest.raises(SectorsAPIError) as excinfo:
        c.broker_summary("BBRI")

    assert excinfo.value.status_code == status
    assert budget.status()["total_spent"] == 0
    lines = read_log_lines(budget)
    assert len(lines) == 1 and lines[0]["billed"] == 0   # free calls still logged


def test_billable_amount_table():
    """Pure billing rule: 2xx -> cost, 404 -> 1, everything else -> 0."""
    assert billable_amount(1, 200) == 1
    assert billable_amount(3, 201) == 3
    assert billable_amount(3, 404) == 1
    assert billable_amount(1, 404) == 1
    for free in (400, 401, 403, 422, 429, 500, 503):
        assert billable_amount(1, free) == 0


# ------------------------------------------------------------------ caps & state


def test_run_cap_boundary(budget):
    budget._state["run_spent"] = 59
    budget.precheck(1)                       # 59 + 1 == 60 is allowed (LOCKED boundary)
    with pytest.raises(RunCapExceeded):
        budget.precheck(2)                   # 59 + 2 > 60 aborts

    budget._state["run_spent"] = 60
    with pytest.raises(RunCapExceeded):
        budget.precheck(1)


def test_reserve_boundary(budget):
    budget._state["total_spent"] = 799
    budget.precheck(1)                       # landing exactly on 800 is allowed
    with pytest.raises(ReserveExceeded):
        budget.precheck(2)

    budget._state["total_spent"] = 800       # reserve 200 is untouchable
    with pytest.raises(ReserveExceeded):
        budget.precheck(1)


def test_daily_rollover_resets_daily_only(tmp_path):
    b = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    b.charge(1, 200, "/v2/daily/BBRI/")
    yesterday = str((datetime.now(WIB) - timedelta(days=1)).date())
    b._state["day"] = yesterday
    b._state["daily_spent"] = 150
    b._save()

    reloaded = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    st = reloaded.status()
    assert st["daily_spent"] == 0            # daily reset on WIB rollover
    assert st["total_spent"] == 1            # total untouched
    assert st["day"] == str(datetime.now(WIB).date())


def test_daily_soft_warn_logs_only(budget, caplog):
    budget._state["daily_spent"] = 199
    with caplog.at_level(logging.WARNING, logger="bandar.budget"):
        billed = budget.charge(1, 200, "/v2/daily/BBRI/")   # crosses 200

    assert billed == 1                                        # soft: never raises
    assert "daily soft warn" in caplog.text
    # warns once per day only
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bandar.budget"):
        budget.charge(1, 200, "/v2/daily/BBRI/")
    assert "daily soft warn" not in caplog.text


def test_reset_run_zeroes_run_only(budget):
    budget.charge(1, 200, "/a/")
    budget.charge(1, 200, "/b/")
    assert budget.status()["run_spent"] == 2

    budget.reset_run()
    st = budget.status()
    assert st["run_spent"] == 0
    assert st["total_spent"] == 2 and st["daily_spent"] == 2


def test_state_persists_across_instances(tmp_path):
    b1 = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    b1.charge(1, 200, "/v2/daily/BBRI/")
    b1.charge(3, 404, "/v2/companies/")

    b2 = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    st = b2.status()
    assert st["total_spent"] == 2            # 1 (2xx cost-1) + 1 (404 bills exactly 1, not 3)
    assert st["run_spent"] == 2 and st["daily_spent"] == 2


# ------------------------------------------------------- client contract details


def test_auth_header_no_bearer(make_client, monkeypatch):
    calls = install_fake_get(monkeypatch, response=FakeResponse(200))
    make_client(api_key="TESTKEY").foreign_flow("BBRI")

    headers = calls[0]["headers"]
    assert headers["Authorization"] == "TESTKEY"          # raw key, exactly
    assert "Bearer" not in json.dumps(headers)            # NO Bearer anywhere


def test_cache_key_deterministic_and_param_order_stable(make_client, budget, monkeypatch):
    # pure function: sorted params -> identical key regardless of insertion order
    k1 = _cache_key("/v2/daily/BBRI/", {"start": "2026-06-23", "end": "2026-09-21"})
    k2 = _cache_key("/v2/daily/BBRI/", {"end": "2026-09-21", "start": "2026-06-23"})
    k3 = _cache_key("/v2/daily/BBRI/", {"end": "2026-09-22", "start": "2026-06-23"})
    assert k1 == k2 and k1 != k3 and len(k1) == 40

    # integration: second call with reordered kwargs hits the same cache file
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"results": []}))
    c = make_client()
    c.ohlcv("BBRI", "2026-06-23", "2026-09-21")
    c.ohlcv("BBRI", "2026-06-23", "2026-09-21")
    assert len(calls) == 1                                # second served from cache
    assert len(list((c.cache_dir / "ohlcv").glob("*.json"))) == 1
    assert budget.status()["total_spent"] == 1


def test_log_records_every_live_call_only(make_client, budget, monkeypatch):
    install_fake_get(monkeypatch, response=FakeResponse(200, {"results": []}))
    c = make_client()
    c.ohlcv("BBRI", "2026-06-23", "2026-09-21")            # live 2xx

    install_fake_get(monkeypatch, response=FakeResponse(500))
    with pytest.raises(SectorsAPIError):
        c.foreign_flow("BBRI")                              # live 5xx (free)

    c.ohlcv("BBRI", "2026-06-23", "2026-09-21")            # cache hit — NOT a live call

    lines = read_log_lines(budget)
    assert len(lines) == 2                                  # one line per live call only
    assert lines[0]["billed"] == 1 and lines[0]["status"] == 200
    assert lines[1]["billed"] == 0 and lines[1]["status"] == 500
    assert all("endpoint" in ln and "ts" in ln for ln in lines)


def test_network_error_no_charge(make_client, budget, monkeypatch):
    install_fake_get(monkeypatch, exc=requests.Timeout("boom"))
    c = make_client()

    with pytest.raises(SectorsConnectionError):
        c.broker_summary("BBRI")

    assert budget.status()["total_spent"] == 0              # charge never called
    assert read_log_lines(budget) == []


def test_live_mode_bypasses_cache_read_and_rewrites(make_client, budget, monkeypatch):
    c = make_client(mode="cache-first")
    install_fake_get(monkeypatch, response=FakeResponse(200, {"v": "old"}))
    assert c.foreign_flow("BBRI") == {"v": "old"}

    c.mode = "live"
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"v": "new"}))
    assert c.foreign_flow("BBRI") == {"v": "new"}           # HTTP despite warm cache
    assert len(calls) == 1

    c.mode = "cache-first"
    assert c.foreign_flow("BBRI") == {"v": "new"}           # cache rewritten by live mode
    assert budget.status()["total_spent"] == 2


def test_ohlcv_over_90d_rejected_preflight(make_client, budget, monkeypatch):
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"results": []}))
    c = make_client()

    with pytest.raises(ValueError, match="exceeds 90d"):
        c.ohlcv("BBRI", "2026-01-01", "2026-05-01")         # 120 days

    assert calls == []                                      # pre-flight: no HTTP
    assert budget.status()["total_spent"] == 0              # and 0 cr

    c.ohlcv("BBRI", "2026-06-23", "2026-09-21")             # exactly 90d is fine
    assert len(calls) == 1


def test_symbol_normalization_in_url(make_client, monkeypatch):
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"results": []}))
    c = make_client()

    c.ohlcv("bbri.jk", "2026-06-23", "2026-09-21")
    assert calls[0]["url"] == "https://api.sectors.app/v2/daily/BBRI/"
    assert calls[0]["params"] == {"start": "2026-06-23", "end": "2026-09-21"}

    c.foreign_flow("BBRI.JK")
    assert calls[1]["url"] == "https://api.sectors.app/v2/foreign-flow/BBRI/"

    with pytest.raises(ValueError):
        c.broker_summary("not a symbol!!")


# -------------------------------------------- docs-verified param windows (Q2 fix)


def test_optional_date_windows_and_day_scoped_keys(make_client, budget, monkeypatch):
    """Param-less calls use API defaults (params None); explicit dates flow into
    params — and therefore into the sha1 cache key, making keys day-scoped."""
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"data": []}))
    c = make_client()

    c.broker_summary("BBRI")
    assert calls[0]["params"] is None                       # API default window

    c.broker_summary("BBRI", start="2026-09-22", end="2026-10-06")
    assert calls[1]["params"] == {"start": "2026-09-22", "end": "2026-10-06"}

    c.foreign_flow("BBRI", start="2026-07-08", end="2026-10-06")
    assert calls[2]["params"] == {"start": "2026-07-08", "end": "2026-10-06"}

    # param-less and dated calls are distinct cache entries
    assert len(list((c.cache_dir / "broker_summary").glob("*.json"))) == 2
    assert len(list((c.cache_dir / "foreign_flow").glob("*.json"))) == 1
    assert budget.status()["total_spent"] == 3


def test_window_caps_rejected_preflight(make_client, budget, monkeypatch):
    calls = install_fake_get(monkeypatch, response=FakeResponse(200, {"data": []}))
    c = make_client()

    with pytest.raises(ValueError, match="exceeds 14d"):
        c.broker_summary("BBRI", start="2026-09-01", end="2026-10-06")   # 35d > 14d

    with pytest.raises(ValueError, match="exceeds 90d"):
        c.foreign_flow("BBRI", start="2026-01-01", end="2026-10-06")     # 278d > 90d

    with pytest.raises(ValueError, match="ISO date"):
        c.foreign_flow("BBRI", start="01/09/2026")                        # bad format

    assert calls == []                                                   # no HTTP
    assert budget.status()["total_spent"] == 0                           # 0 cr

    c.broker_summary("BBRI", start="2026-09-22", end="2026-10-06")       # exactly 14d OK
    assert len(calls) == 1


def test_daily_close_params_and_pagination_guards(make_client, budget, monkeypatch):
    calls = install_fake_get(
        monkeypatch, response=FakeResponse(200, {"results": [], "pagination": {}})
    )
    c = make_client()

    c.daily_close("2026-10-02", limit=30, offset=30)
    assert calls[0]["url"] == "https://api.sectors.app/v2/close/"
    assert calls[0]["params"] == {"date": "2026-10-02", "limit": 30, "offset": 30}

    with pytest.raises(ValueError):
        c.daily_close("2026-10-02", limit=31)      # per-page max is 30
    with pytest.raises(ValueError):
        c.daily_close("2026-10-02", offset=-1)
    with pytest.raises(ValueError):
        c.daily_close("not-a-date")

    assert len(calls) == 1                          # guards are pre-flight, no HTTP
    assert budget.status()["total_spent"] == 1      # only the real call billed
