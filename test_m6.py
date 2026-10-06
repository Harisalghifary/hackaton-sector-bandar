"""M6 tests — telegram_push.py (§9, FR11).

FR11 parity: the push message must carry the SAME engine numbers as the UI
foreground (top pick card fields, watchlist lines with gate overrides, deltas
with prior dates, credits used). Deterministic formatter => assert exact strings.

0 credits: BANDAR_AS_OF pins the seeded day (warm cache), BANDAR_STATE_DIR
isolates the ledger, sockets are blocked, Telegram sends are mocked except the
one deliberate live verification (manual, outside pytest).

Run: python -m pytest test_m6.py -v
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

import telegram_push as tp

REPO = Path(__file__).resolve().parent

PLAN = {"entry_zone": [9050.0, 9275.0], "stop_close": 8600.0, "lots": 12,
        "tick_valid": True}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M6 tests — push must be cache-only")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDAR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BANDAR_AS_OF", "2026-10-06")
    monkeypatch.setenv("BANDAR_WATCHLIST", "BBRI,BMRI,DSSA")
    monkeypatch.setenv("SECTORS_API_KEY", "TESTKEY")     # cache-first, warm -> 0 calls
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:TESTTOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    return tmp_path


def mk_score(sym, score, denom, label, plan=None, gate=False, flags=None,
             as_of="2026-10-05"):
    return {"symbol": sym, "as_of": as_of, "status": "ok", "bars": 234,
            "score": score, "denominator": denom, "factors": {},
            "gates": {"no_chase": {"triggered": gate}, "regime": "bullish"},
            "decision": {"label": label, "deploy_pct": "30-40%", "bias": "NEUTRAL",
                         "tier": 2 if plan else 0},
            "trade_plan": plan, "trace": [], "risk_flags": flags or []}


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {"ok": True, "result": {}}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


# ---------------------------------------------------------------- gather (0 cr)


def test_gather_scores_from_cache_at_zero_cost(env):
    res = tp.gather()
    assert {r["symbol"] for r in res["scores"]} == {"BBRI", "BMRI", "DSSA"}
    assert all(r["status"] == "ok" for r in res["scores"])
    assert res["push_cost"] == 0                        # every pull was a cache hit
    assert res["as_of"] == "2026-10-06"
    assert len(res["deltas"]) == 3                      # first run -> no prev
    assert all(d["has_prev"] is False for d in res["deltas"])
    state = env / "state" / "budget_state.json"
    assert not state.exists() or '"total_spent": 0' in state.read_text()


# ------------------------------------------------- build_message / FR11 parity


def test_message_matches_foreground_numbers(env):
    res = tp.gather()
    msg = tp.build_message(res["scores"], res["deltas"], res["budget"],
                           res["as_of"], res["push_cost"])

    ranked = res["scores"]
    actionable = [r for r in ranked if tp._is_actionable(r)]
    assert actionable, "seed day must have an actionable pick (DSSA)"
    pick = actionable[0]
    assert pick["symbol"] == "DSSA"                     # seed ranking top actionable

    # TOP PICK line carries the exact engine fields (FR11)
    tp_line = [ln for ln in msg.splitlines() if "TOP PICK" in ln][0]
    assert f"DSSA — {pick['score']}/{pick['denominator']}" in tp_line
    assert pick["decision"]["label"] in tp_line
    ez = pick["trade_plan"]["entry_zone"]
    assert f"Entry {ez[0]:g}–{ez[1]:g}" in msg
    assert f"Stop (close) {pick['trade_plan']['stop_close']:g}" in msg
    assert f"Lots {pick['trade_plan']['lots']}" in msg
    assert "Why:" in msg and "regime" in msg

    # every watchlist name appears with its score + decision line
    for r in ranked:
        assert any(ln.startswith(f"· {r['symbol']} {r['score']}/{r['denominator']}")
                   for ln in msg.splitlines())

    # BMRI's no-chase gate override is visible (D3 — never actionable)
    bmri = next(r for r in ranked if r["symbol"] == "BMRI")
    if bmri["gates"]["no_chase"]["triggered"]:
        bmri_line = [ln for ln in msg.splitlines() if ln.startswith("· BMRI")][0]
        assert "⛔ no-chase" in bmri_line

    # "N others actionable" + credits + FR9 disclaimer (§9 content contract)
    n_others = len(actionable) - 1
    assert f"➕ {n_others} others actionable" in msg
    assert f"Credits: {res['budget']['total_spent']}/{res['budget']['grant']}" in msg
    assert "this push 0 cr" in msg
    assert "never executes trades (FR9)" in msg


def test_message_when_nothing_actionable(env):
    scores = [mk_score("BBRI", 0, 5, "WAIT"),
              mk_score("BMRI", 2, 5, "WAIT", gate=True)]
    msg = tp.build_message(scores, [], {"total_spent": 49, "grant": 1000},
                           "2026-10-06", 0)
    assert "No actionable setup today" in msg
    assert "best score BBRI 0/5 (WAIT)" in msg
    assert "Entry" not in msg and "Lots" not in msg     # no plan -> no levels shown
    assert "➕ 0 others actionable" in msg


def test_message_delta_and_gate_marks(env):
    scores = [mk_score("DSSA", 3, 5, "DEFENSIVE", plan=PLAN),
              mk_score("BMRI", 2, 5, "WAIT", gate=True)]
    deltas = [{"ticker": "DSSA", "has_prev": True, "prev_date": "2026-10-05",
               "score_old": 2, "denominator_old": 5, "score_new": 3,
               "denominator_new": 5, "decision_old": "SCALP",
               "decision_new": "DEFENSIVE", "gate_override": None},
              {"ticker": "BMRI", "has_prev": False, "prev_date": None,
               "score_old": None, "denominator_old": None, "score_new": 2,
               "denominator_new": 5, "decision_old": None,
               "decision_new": "WAIT", "gate_override": "no_chase"}]
    msg = tp.build_message(scores, deltas, {"total_spent": 50, "grant": 1000},
                           "2026-10-06", 2)
    assert "Δ since 2026-10-05: 2/5 SCALP → 3/5 DEFENSIVE" in msg   # FR6 old->new + date
    assert "(was 2/5 SCALP on 2026-10-05)" not in msg               # only pick gets Δ line
    bmri_line = [ln for ln in msg.splitlines() if ln.startswith("· BMRI")][0]
    assert "⛔ no-chase" in bmri_line and "(was" not in bmri_line   # honest: no prev
    assert "★" in msg                                               # top pick marked
    assert "this push 2 cr" in msg


def test_message_truncated_to_telegram_limit(env):
    scores = [mk_score(f"T{i:03d}", 1, 5, "WAIT") for i in range(300)]
    msg = tp.build_message(scores, [], {"total_spent": 0, "grant": 1000},
                           "2026-10-06", 0)
    assert len(msg) <= tp.TELEGRAM_MAX_LEN
    assert msg.endswith("…(truncated)")


# ----------------------------------------------------------------- transport


def test_send_telegram_posts_to_bot_api(monkeypatch):
    calls = {}

    def fake_post(url, json=None, timeout=None):
        calls.update(url=url, payload=json)
        return FakeResp()

    monkeypatch.setattr(tp.requests, "post", fake_post)
    out = tp.send_telegram("hello", "123:TOKEN", "42")
    assert out["ok"] is True
    assert calls["url"] == "https://api.telegram.org/bot123:TOKEN/sendMessage"
    assert calls["payload"]["chat_id"] == "42"
    assert calls["payload"]["text"] == "hello"


def test_send_telegram_raises_loudly(monkeypatch):
    monkeypatch.setattr(tp.requests, "post",
                        lambda *a, **k: FakeResp(status=400, payload=None,
                                                 text="Bad Request: chat not found"))
    with pytest.raises(tp.TelegramPushError, match="HTTP 400"):
        tp.send_telegram("x", "t", "c")

    monkeypatch.setattr(tp.requests, "post",
                        lambda *a, **k: FakeResp(status=200, payload={"ok": False,
                                                                      "description": "nope"}))
    with pytest.raises(tp.TelegramPushError, match="rejected"):
        tp.send_telegram("x", "t", "c")


# -------------------------------------------------------------------- main()


def test_main_dry_run_end_to_end_from_cache(env, capsys):
    rc = tp.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOP PICK: DSSA" in out and "[dry-run]" in out
    # memory was recorded (D7) even in dry-run — the push mirrors a real run
    db = env / "state" / "bandar.db"
    assert db.exists() and db.stat().st_size > 0
    state = env / "state" / "budget_state.json"
    assert not state.exists() or '"total_spent": 0' in state.read_text()  # 0 cr


def test_main_sends_single_telegram_post(env, monkeypatch, capsys):
    sent = []

    def fake_post(url, json=None, timeout=None):
        sent.append((url, json))
        return FakeResp()

    monkeypatch.setattr(tp.requests, "post", fake_post)
    rc = tp.main([])
    assert rc == 0
    assert len(sent) == 1                                # exactly one send
    assert sent[0][0].endswith("/sendMessage")
    printed = capsys.readouterr().out
    assert sent[0][1]["text"].splitlines()[0] == "📈 BANDAR — pre-market desk note"
    assert "[bandar-push] sent ✓" in printed


def test_main_without_telegram_config_fails_loud(env, monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    monkeypatch.delenv("TELEGRAM_CHAT_ID")
    rc = tp.main([])
    assert rc == 2
    assert "cannot send" in capsys.readouterr().out


def test_main_data_failure_returns_1(env, monkeypatch, capsys):
    monkeypatch.setattr(tp, "run_tool",
                        lambda name, args, ctx: {"ok": False, "error": "SectorsNotFoundError",
                                                 "message": "no data"})
    rc = tp.main(["--dry-run"])
    assert rc == 1
    assert "data failure" in capsys.readouterr().out
