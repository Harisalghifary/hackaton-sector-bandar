"""M5 UI tests (§9) — headless via streamlit.testing.v1.AppTest.

LOCKED rules under test: API calls only behind buttons, results in session_state,
credit meter (st.metric), stream brief (st.write_stream), st.status trace rows,
collapsed workings, and the DO-NOT-BUILD list (no auth/charts/settings/execution).

0 credits: BANDAR_AS_OF pins the seeded day (warm cache), BANDAR_STATE_DIR
isolates the ledger, sockets are blocked, run_agent is mocked.

Run: python -m pytest test_m5.py -v
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parent

MOCK_BRIEF = {
    "type": "brief",
    "brief": {
        "symbol": "BBRI",
        "extracted": ["score 0/5, decision WAIT", "close 3120 as of 2026-10-05"],
        "interpretation": "No confluence while distribution persists.",
        "action_plan": ["Stay in cash; re-check after the next session."],
        "risk_flags": ["foreign distribution"],
    },
    "dropped_figures": [],
    "intent": {"intent": "score_ticker", "symbols": ["BBRI"], "question": "q"},
    "plan": {"steps": [{"tool": "score_ticker", "args": {"symbol": "BBRI"}, "reason": "r"}]},
    "tool_results": [],
    "memory_context": [],
    "truncated": False,
    "llm_calls": ["intent", "plan", "synthesis"],
    "trace": ["ohlcv:/v2/daily/BBRI/ [cache]"],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M5 tests — must be 0 live calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated ledger/DB + pinned seed day + watchlist. Cache stays the real warm one."""
    monkeypatch.setenv("BANDAR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BANDAR_AS_OF", "2026-10-06")
    monkeypatch.setenv("BANDAR_WATCHLIST", "BBRI,BMRI")
    monkeypatch.setenv("SECTORS_API_KEY", "TESTKEY")     # cache-first; warm cache -> 0 calls
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return tmp_path


def boot(env):
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=60)
    return at.run()


def all_text(at) -> str:
    parts = []
    for name in ("title", "header", "subheader", "markdown", "caption", "info",
                 "warning", "error", "success"):
        for el in getattr(at, name, []):
            parts.append(str(getattr(el, "value", el)))
    for m in at.metric:
        parts.append(f"{m.label}={m.value}")
    return "\n".join(parts)


def find_button(at, label_part):
    for b in at.button:
        if label_part.lower() in str(b.label).lower():
            return b
    raise AssertionError(f"button {label_part!r} not found; have: "
                         f"{[b.label for b in at.button]}")


# --------------------------------------------------------------------- boot state


def test_boot_renders_without_any_api_call(env):
    at = boot(env)
    assert not at.exception
    text = all_text(at)
    assert "BANDAR" in text and "no trade execution" in text.lower()
    # fresh isolated ledger -> full 800 spendable shown in the credit meter
    assert any(m.value == "800" for m in at.metric)
    assert "No scores yet" in text                      # nothing scored until a button
    # budget state file was never even created (no charge happened)
    assert not (env / "state" / "budget_state.json").exists() or \
        '"total_spent": 0' in (env / "state" / "budget_state.json").read_text()


def test_locked_controls_present(env):
    at = boot(env)
    assert find_button(at, "Force Live Refresh") is not None
    downloads = at.get("download_button")
    assert len(downloads) == 1 and "Download Brief" in downloads[0].label
    assert downloads[0].disabled                          # no brief yet
    assert len(at.chat_input) == 1                        # free-text ask (FR12)


def test_no_execution_path_anywhere(env):
    """FR9 / DO-NOT-BUILD: no buy/sell/order/execution control may exist."""
    at = boot(env)
    find_button(at, "Force Live Refresh")                 # these two may exist…
    at.get("download_button")
    for b in at.button:
        low = str(b.label).lower()
        assert not any(w in low for w in ("buy", "sell", "order", "execut", "trade"))
    text = all_text(at).lower()
    assert "never executes" in text or "no trade execution" in text


# ---------------------------------------------------------------- refresh button


def test_force_refresh_scores_watchlist_from_cache(env):
    at = boot(env)
    at = find_button(at, "Force Live Refresh").click().run()
    assert not at.exception

    scores = at.session_state["scores"]
    assert len(scores) == 2                                # BBRI + BMRI from env watchlist
    assert {s["symbol"] for s in scores} == {"BBRI", "BMRI"}
    assert all(s["status"] == "ok" and s["bars"] >= 200 for s in scores)

    deltas = at.session_state["deltas"]
    assert len(deltas) == 2 and all(d["has_prev"] is False for d in deltas)  # first run

    # all pulls were cache hits -> isolated ledger still at 0 (0 credits)
    assert (env / "state" / "budget_state.json").exists() is False or \
        '"total_spent": 0' in (env / "state" / "budget_state.json").read_text()

    text = all_text(at)
    assert "TOP PICK" in text and ("BBRI" in text or "BMRI" in text)
    assert "WATCHLIST DELTAS" in text
    assert '<table class="bd-table"' in text             # themed deltas table rendered
    assert "Score change (prior → now)" in text          # N: informative delta header


def test_refresh_is_idempotent_and_reruns_keep_session(env):
    at = boot(env)
    at = find_button(at, "Force Live Refresh").click().run()
    first = at.session_state["scores"]
    at = find_button(at, "Force Live Refresh").click().run()   # second press, same day
    assert at.session_state["scores"] == first                 # deterministic, cached
    assert '"total_spent": 0' in (env / "state" / "budget_state.json").read_text()


# ------------------------------------------------------------------- ask flow


def test_ask_streams_brief_and_fills_session(env, monkeypatch):
    import agent.executor as executor_mod
    calls = []

    def fake_run_agent(question, ctx, **kw):
        calls.append(question)
        return dict(MOCK_BRIEF)

    monkeypatch.setattr(executor_mod, "run_agent", fake_run_agent)
    at = boot(env)
    at.chat_input[0].set_value("how is BBRI?").run()

    assert calls == ["how is BBRI?"]                       # ask triggers the agent run
    assert at.session_state["agent_out"]["type"] == "brief"
    assert at.session_state["agent_out"]["question"] == "how is BBRI?"    # L: context
    assert at.session_state["brief_md"].startswith("# BANDAR brief — BBRI")
    assert "You asked" in at.session_state["brief_md"]     # download carries the question
    text = all_text(at)
    assert "No confluence while distribution persists." in text     # streamed (write_stream)
    assert "score 0/5, decision WAIT" in text
    assert "Answer" in text
    assert any("You asked" in str(c.value) and "how is BBRI?" in str(c.value)
               for c in at.caption)                        # answer tied to the question
    # download button now enabled with the brief
    downloads = at.get("download_button")
    assert not downloads[0].disabled


def test_ask_fallback_rendered_gracefully(env, monkeypatch):
    import agent.executor as executor_mod
    monkeypatch.setattr(executor_mod, "run_agent", lambda q, ctx, **kw: {
        "type": "fallback", "message": "Outside Bandar's scope: I never execute trades.",
        "llm_calls": ["intent"],
    })
    at = boot(env)
    at.chat_input[0].set_value("buy BBRI now").run()
    assert not at.exception
    assert any("Outside Bandar" in str(w.value) for w in at.warning)


def test_workings_expander_collapsed_with_traces_and_plan(env, monkeypatch):
    import agent.executor as executor_mod
    monkeypatch.setattr(executor_mod, "run_agent", lambda q, ctx, **kw: MOCK_BRIEF)
    at = boot(env)
    at = find_button(at, "Force Live Refresh").click().run()
    at.chat_input[0].set_value("how is BBRI?").run()

    exps = [e for e in at.expander if "Workings" in str(getattr(e, "label", ""))]
    assert exps, "collapsed workings expander must exist (§9)"
    # collapsed by default: app.py passes expanded=False (D9 — expanded once on camera only)
    text = all_text(at)
    assert "Agent plan" in text and "score_ticker" in text
    assert "[cache]" in text                               # cache/live tags visible
    assert "Per-factor math" in text and "Credit meter" in text


# ------------------------------------------------------------------ boot restore


def test_boot_restores_last_run_from_memory(env):
    at = boot(env)
    at = find_button(at, "Force Live Refresh").click().run()   # writes memory (D7)
    assert at.session_state["scores"]

    at2 = boot(env)                                            # fresh session, same state dir
    text = all_text(at2)
    assert "last recorded run from memory" in text             # 0-cr restore notice
    assert "BBRI" in text or "BMRI" in text


def test_theme_defaults_dark_and_switches_via_env(env, monkeypatch):
    at = boot(env)                                             # BANDAR_THEME unset -> dark
    css = "\n".join(str(getattr(el, "value", el)) for el in at.markdown)
    assert "#0f0f1a" in css and "bd-card" in css

    monkeypatch.setenv("BANDAR_THEME", "light")
    at2 = boot(env)
    css2 = "\n".join(str(getattr(el, "value", el)) for el in at2.markdown)
    assert "#ffffff" in css2 and "#0f0f1a" not in css2

    monkeypatch.setenv("BANDAR_THEME", "nonsense")             # invalid -> falls back dark
    at3 = boot(env)
    css3 = "\n".join(str(getattr(el, "value", el)) for el in at3.markdown)
    assert "#0f0f1a" in css3


def _md(at) -> str:
    return "\n".join(str(getattr(el, "value", el)) for el in at.markdown)


def _shows(md: str, theme: str) -> bool:
    """Active palette = last-injected <style>. Markers unique per palette:
    #0f0f1a exists only in dark, #eaeef2 only in light (#ffffff is dark's TEXT)."""
    mine, other = ("#0f0f1a", "#eaeef2") if theme == "dark" else ("#eaeef2", "#0f0f1a")
    return mine in md and md.rfind(mine) > md.rfind(other)


def _toggle(at, icon):
    return next(b for b in at.button if f"material/{icon}" in str(b.label))


def test_theme_toggle_button_flips_session(env):
    at = boot(env)                                             # dark default
    assert _shows(_md(at), "dark")
    at = _toggle(at, "dark_mode").click().run().run()          # moon shown -> go light
    assert _shows(_md(at), "light")
    at = _toggle(at, "light_mode").click().run().run()         # sun shown -> back to dark
    assert _shows(_md(at), "dark")


def test_theme_session_toggle_beats_env(env, monkeypatch):
    monkeypatch.setenv("BANDAR_THEME", "light")
    at = boot(env)
    assert _shows(_md(at), "light")                            # env default honored
    at = _toggle(at, "light_mode").click().run().run()         # session override wins
    assert _shows(_md(at), "dark")


EMOJI_BANNED = ["\U0001F3AF", "\U0001F4CA", "\U0001F527", "\u2705", "\u274C", "\U0001F504", "\u2B07", "\U0001F4C8", "\U0001F4AC", "\u26D4", "\u2796"]


def test_ui_chrome_is_emoji_free(env):
    """P1 regression: section icons are inline SVG, checks are ✓/✕ glyphs —
    no OS-dependent emoji anywhere in the UI chrome (video + cross-platform)."""
    at = boot(env)
    at = find_button(at, "Force Live Refresh").click().run()
    text = all_text(at)
    labels = " ".join(str(b.label) for b in at.button)
    for e in EMOJI_BANNED:
        assert e not in text, f"emoji {e} leaked into UI text"
        assert e not in labels, f"emoji {e} leaked into a button label"
    assert "bd-dots" in text                                   # P2 score dots rendered
    assert "bd-brand" in text and "bd-wordmark" in text        # P5 brand header
    assert "✕" in text and ("bd-ok" in text or "bd-bad" in text)   # glyph factor checks


def test_p4_context_panel_chip_and_p6_meter_style(env):
    at = boot(env)
    css = "\n".join(str(getattr(el, "value", el)) for el in at.markdown)
    assert "stMetricValue" in css and "stHorizontalBlock" in css   # P6 dynamic style
    assert "not scored yet" in css                                  # P4 chip, empty state

    at = find_button(at, "Force Live Refresh").click().run()
    text = all_text(at)
    assert "Desk context" in text and "Actionable" in text         # P4 side panel
    assert "Scored" in text and "2/2" in text
    assert "last scored:" in text                                  # chip now has a time
    assert '<th style="width:10%">Ticker</th>' in text             # P4 compact columns
    assert 'width:15%">Score change (prior → now)' in text        # N: delta column header
    assert "bd-same" in text or "bd-up" in text or "bd-down" in text
    assert "Action:" in text and "bd-pill blue" in text            # action line + bias badge
    assert "Below MA200 — bearish tape" in text                    # pass-5 human regime
    assert any(":material/sync:" in str(b.label) for b in at.button)


def test_workings_validator_audit_lists_removed_figures(env, monkeypatch):
    """O3: 'N figure(s) removed' is inspectable — Workings lists section, figures, item."""
    import agent.executor as executor_mod
    out = dict(MOCK_BRIEF, dropped_figures=[
        {"section": "extracted", "item": "cumulative net buying Rp 12.3B",
         "figures": ["12.3"]}])
    monkeypatch.setattr(executor_mod, "run_agent", lambda q, ctx, **kw: out)
    at = boot(env)
    at.chat_input[0].set_value("whos accumulating BBRI?").run()
    text = all_text(at)
    assert "Validator audit" in text and "12.3" in text
    assert "cumulative net buying" in text
    assert "anti-fabrication trace validator" in text       # M2 warning wording


def test_deltas_table_survives_null_scores(env):
    """Prior snapshot with a null score (or unscored new row) renders honest '—',
    never crashes the comparison (TypeError: None > int)."""
    at = boot(env)
    at.session_state["deltas"] = [{
        "ticker": "BBRI", "as_of": "2026-10-08", "has_prev": True,
        "prev_date": "2026-10-07", "score_old": None, "score_new": 3,
        "denominator_old": 5, "denominator_new": 5,
        "decision_old": None, "decision_new": "WAIT",
        "gate_override": False, "changed": True,
    }, {
        "ticker": "BMRI", "as_of": "2026-10-08", "has_prev": False,
        "prev_date": None, "score_old": None, "score_new": None,
        "denominator_old": None, "denominator_new": 5,
        "decision_old": None, "decision_new": None,
        "gate_override": False, "changed": False,
    }]
    at.run()
    assert not at.exception
    text = all_text(at)
    assert "WATCHLIST DELTAS" in text
    assert "None" not in text.split("WATCHLIST DELTAS", 1)[1].split("Ask Bandar", 1)[0]
