"""M3 memory tests (§7 + §10 prompt: upsert + delta query tests; FR6, D6, D7).

0 credits: no network (socket-blocked), engine integration runs on the seed cache
in offline mode.

Run: python -m pytest test_m3.py -v
"""

from __future__ import annotations

import json
import socket
from datetime import date
from pathlib import Path

import pytest

from budget import CreditBudget
from client import SectorsClient
from engine import score_ticker
from memory import ScoreMemory, build_delta, render_delta

REPO = Path(__file__).resolve().parent
CACHE_DIR = REPO / "cache"
SEED_AS_OF = date(2026, 10, 6)


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M3 tests — must be 0 live calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)
    monkeypatch.delenv("BANDAR_WATCHLIST", raising=False)


@pytest.fixture()
def mem(tmp_path):
    m = ScoreMemory(db_path=tmp_path / "bandar.db",
                    watchlist_path=tmp_path / "watchlist.json")
    yield m
    m.close()


@pytest.fixture()
def offline_client(tmp_path):
    budget = CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")
    return SectorsClient(api_key="TESTKEY", mode="offline", budget=budget,
                         cache_dir=CACHE_DIR)


def make_result(symbol="BBRI", as_of="2026-10-05", score=4, denom=5,
                label="AGGRESSIVE", gate_override=None, factors=None, gates=None):
    decision = {"label": label, "deploy_pct": "60-80%", "bias": "BULLISH"}
    if gate_override:
        decision["gate_override"] = gate_override
    return {
        "symbol": symbol, "as_of": as_of, "status": "ok", "bars": 234,
        "score": score, "denominator": denom,
        "factors": factors if factors is not None else {"f5_broker": None},
        "gates": gates if gates is not None else {"regime": "bullish_above_ma200"},
        "decision": decision, "trade_plan": None, "trace": [],
    }


# ------------------------------------------------------- §7 LOCKED schema exact


def test_schema_matches_spec(mem):
    cols = mem.conn.execute("PRAGMA table_info(score_history)").fetchall()
    got = [(c["name"], c["type"], c["pk"]) for c in cols]
    assert got == [
        ("ticker", "TEXT", 1),
        ("as_of", "DATE", 2),            # composite PRIMARY KEY (ticker, as_of)
        ("score", "INTEGER", 0),
        ("denominator", "INTEGER", 0),
        ("decision", "TEXT", 0),
        ("factors_json", "TEXT", 0),
        ("gates_json", "TEXT", 0),
        ("created_at", "TIMESTAMP", 0),
    ]


# ---------------------------------------------------------------- upsert (D7)


def test_upsert_same_key_updates_in_place(mem):
    """D7: any run upserts by (ticker, as_of) — re-scoring a day never duplicates."""
    mem.record_score(make_result(score=2, denom=4, label="SCALP"))
    mem.record_score(make_result(score=4, denom=5, label="AGGRESSIVE"))

    rows = mem.conn.execute("SELECT * FROM score_history").fetchall()
    assert len(rows) == 1                            # one row, not two
    row = rows[0]
    assert (row["score"], row["denominator"], row["decision"]) == (4, 5, "AGGRESSIVE")


def test_upsert_different_dates_insert(mem):
    mem.record_score(make_result(as_of="2026-10-02", score=1))
    mem.record_score(make_result(as_of="2026-10-05", score=3))
    assert len(mem.get_history("BBRI")) == 2


def test_history_newest_first_with_limit(mem):
    for d, s in [("2026-10-01", 1), ("2026-10-03", 2), ("2026-10-05", 4)]:
        mem.record_score(make_result(as_of=d, score=s))
    hist = mem.get_history("bbri.jk")                # symbol normalization on read
    assert [h["as_of"] for h in hist] == ["2026-10-05", "2026-10-03", "2026-10-01"]
    assert [h["as_of"] for h in mem.get_history("BBRI", limit=2)] == ["2026-10-05", "2026-10-03"]


def test_json_snapshots_roundtrip_with_honest_nulls(mem):
    """D6: factors_json + gates_json stored per run; null f5 stays null (FR2)."""
    result = make_result(factors={"f1_macd": {"pass": True}, "f5_broker": None},
                         gates={"no_chase": {"triggered": False}})
    mem.record_score(result)
    row = mem.get_history("BBRI")[0]
    assert row["factors"]["f5_broker"] is None
    assert '"f5_broker": null' in row["factors_json"]
    assert row["gates"]["no_chase"] == {"triggered": False}


def test_insufficient_result_stores_nulls(mem):
    result = make_result(score=None, denom=None, label="WAIT")
    mem.record_score(result)
    row = mem.get_history("BBRI")[0]
    assert row["score"] is None and row["denominator"] is None
    assert row["decision"] == "WAIT"


def test_persistence_across_instances(tmp_path):
    m1 = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    m1.record_score(make_result())
    m1.close()
    m2 = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    assert len(m2.get_history("BBRI")) == 1
    m2.close()


# ------------------------------------------------------------ delta queries (FR6)


def test_get_prev_picks_latest_prior_not_oldest(mem):
    """§7 LOCKED: delta = LATEST row with as_of < current as_of."""
    for d in ("2026-10-01", "2026-10-03"):
        mem.record_score(make_result(as_of=d))
    prev = mem.get_prev("BBRI", "2026-10-05")
    assert prev is not None and prev["as_of"] == "2026-10-03"


def test_get_prev_none_on_first_run_never_invented(mem):
    assert mem.get_prev("BBRI", "2026-10-05") is None
    delta = build_delta(make_result(), mem.get_prev("BBRI", "2026-10-05"))
    assert delta["has_prev"] is False
    assert delta["prev_date"] is None and delta["score_old"] is None
    assert "first scored" in render_delta(delta)


def test_record_run_returns_old_to_new_deltas(mem):
    """Two consecutive runs: delta shows old->new score/decision + prior date (FR6)."""
    day1 = [make_result(as_of="2026-10-02", score=1, denom=5, label="WAIT"),
            make_result(symbol="ANTM", as_of="2026-10-02", score=3, denom=4, label="DEFENSIVE")]
    deltas1 = mem.record_run(day1)
    assert all(d["has_prev"] is False for d in deltas1)      # first run: honest no-delta

    day2 = [make_result(as_of="2026-10-05", score=4, denom=5, label="AGGRESSIVE"),
            make_result(symbol="ANTM", as_of="2026-10-05", score=3, denom=4,
                        label="WAIT", gate_override="no_chase")]
    deltas2 = mem.record_run(day2)

    bbri = next(d for d in deltas2 if d["ticker"] == "BBRI")
    assert bbri["has_prev"] is True and bbri["prev_date"] == "2026-10-02"
    assert (bbri["score_old"], bbri["score_new"]) == (1, 4)
    assert (bbri["decision_old"], bbri["decision_new"]) == ("WAIT", "AGGRESSIVE")
    assert bbri["changed"] is True
    assert "1/5 WAIT -> 4/5 AGGRESSIVE" in render_delta(bbri)
    assert "prior 2026-10-02" in render_delta(bbri)

    antm = next(d for d in deltas2 if d["ticker"] == "ANTM")
    assert antm["gate_override"] == "no_chase"               # §9 gate-override column
    assert antm["changed"] is True                           # same score, decision changed


def test_delta_changed_false_when_identical(mem):
    mem.record_score(make_result(as_of="2026-10-02", score=4, label="AGGRESSIVE"))
    deltas = mem.record_run([make_result(as_of="2026-10-05", score=4, label="AGGRESSIVE")])
    assert deltas[0]["has_prev"] is True and deltas[0]["changed"] is False


# ------------------------------------------------------------------- watchlist


def test_watchlist_env_seed_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDAR_WATCHLIST", "bbri.jk, BBCA ,bbri")   # normalize + dedupe
    m = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    assert m.load_watchlist() == ["BBRI", "BBCA"]
    assert tmp_path.joinpath("w.json").exists()                     # persisted on seed

    m2 = ScoreMemory(db_path=tmp_path / "b.db", watchlist_path=tmp_path / "w.json")
    monkeypatch.setenv("BANDAR_WATCHLIST", "TOTALLY,DIFFERENT")     # file wins over env
    assert m2.load_watchlist() == ["BBRI", "BBCA"]
    m.close(); m2.close()


def test_watchlist_rw_ops(mem):
    """Engine side of the M4 watchlist_rw tool (0 cr): add / remove / replace."""
    mem.save_watchlist(["BBRI", "BBCA"])
    assert mem.update_watchlist("add", ["antm.jk", "BBRI"]) == ["BBRI", "BBCA", "ANTM"]
    assert mem.update_watchlist("remove", ["BBCA"]) == ["BBRI", "ANTM"]
    assert mem.update_watchlist("replace", ["MDKA"]) == ["MDKA"]
    with pytest.raises(ValueError):
        mem.update_watchlist("nope", ["BBRI"])


# ------------------------------------------------- engine integration (0 credits)


def test_engine_to_memory_roundtrip(mem, offline_client):
    """Real cached score -> record -> re-score same day -> upsert keeps one row."""
    r1 = score_ticker("BBRI", client=offline_client, as_of=SEED_AS_OF)
    deltas = mem.record_run([r1])
    assert deltas[0]["has_prev"] is False                     # first ever run for BBRI

    row = mem.get_history("BBRI")[0]
    assert row["as_of"] == r1["as_of"] == "2026-10-05"
    assert row["score"] == r1["score"] and row["denominator"] == r1["denominator"]
    assert row["decision"] == r1["decision"]["label"]
    assert row["factors"]["f1_macd"] is not None              # snapshot parseable

    r2 = score_ticker("BBRI", client=offline_client, as_of=SEED_AS_OF)  # deterministic
    mem.record_run([r2])
    assert len(mem.get_history("BBRI")) == 1                  # upserted, not duplicated

    # a "next day" run against the stored snapshot produces a real delta
    tomorrow = dict(r2, as_of="2026-10-06")
    deltas2 = mem.record_run([tomorrow])
    assert deltas2[0]["prev_date"] == "2026-10-05"
    assert deltas2[0]["changed"] is False                     # identical scores

    assert offline_client.budget.status()["total_spent"] == 0  # still 0 credits
