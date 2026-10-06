"""BANDAR Streamlit app (§9, [LOCKED] D9).

Foreground: header + credit meter · TOP PICK card · WATCHLIST DELTAS table ·
Download Brief / Force Live Refresh buttons · free-text ask (FR12 intents).
Workings (collapsed expander): agent plan · data pulls with cache/live tags
(st.status) · per-factor math · credit meter detail.

Rules honored:
- API calls ONLY behind explicit buttons / chat submit; results in st.session_state.
- st.metric credit meter · st.write_stream brief · st.status trace rows.
- DO NOT BUILD list: no auth, no multi-user, no charts, no settings page, no
  risk-profile switcher, no backtesting, NO execution path (FR9).
- Env anchors: BANDAR_AS_OF (replay a seeded day deterministically), BANDAR_STATE_DIR
  (isolate ledger/DB — used by tests).

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from agent.executor import run_agent                      # noqa: E402
from agent.tools import ToolContext, run_tool             # noqa: E402
from budget import CreditBudget                           # noqa: E402
from client import SectorsClient, SectorsError            # noqa: E402
from memory import ScoreMemory, build_delta               # noqa: E402

WIB = ZoneInfo("Asia/Jakarta")
STATE_DIR = Path(os.environ.get("BANDAR_STATE_DIR", "state"))
FACTOR_LABELS = {"f1_macd": "F1 MACD", "f2_volume": "F2 Vol", "f3_ad": "F3 A/D",
                 "f4_ma_stack": "F4 MAs", "f5_broker": "F5 Broker"}

st.set_page_config(page_title="BANDAR", page_icon="📈", layout="wide")


# --------------------------------------------------------------------- resources

def make_resources():
    budget = CreditBudget(state_path=STATE_DIR / "budget_state.json",
                          log_path=STATE_DIR / "budget_log.jsonl")
    api_key = os.environ.get("SECTORS_API_KEY", "")
    client = SectorsClient(api_key=api_key, mode="cache-first" if api_key else "offline",
                           budget=budget)
    memory = ScoreMemory(db_path=STATE_DIR / "bandar.db",
                         watchlist_path=STATE_DIR / "watchlist.json")
    return budget, client, memory


def anchor_date():
    env = os.environ.get("BANDAR_AS_OF")
    return datetime.fromisoformat(env).date() if env else datetime.now(WIB).date()


budget, client, memory = make_resources()
AS_OF = anchor_date()

for key, default in (("scores", []), ("deltas", []), ("agent_out", None),
                     ("brief_md", ""), ("last_error", "")):
    st.session_state.setdefault(key, default)


# ------------------------------------------------------------------ header + meter

st.title("📈 BANDAR — pre-market desk note")
st.caption(f"IDX swing-trading analyst · as-of {AS_OF.isoformat()} (WIB) · "
           "analysis only — **no trade execution, ever** (FR9)")

m1, m2, m3 = st.columns(3)
stt = budget.status()
m1.metric("Sectors credits left", stt["remaining_total"], help="of 800 spendable (200 reserve locked)")
m2.metric("Today (WIB)", f"{stt['daily_spent']} / 200", help="soft warn at 200/day")
m3.metric("This run", f"{stt['run_spent']} / 60", help="hard cap 60/run")

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])


# ----------------------------------------------------------------- actions (buttons)

def do_refresh() -> None:
    """Force Live Refresh: re-score the watchlist anchored on today (cache-first —
    a new day's windows fetch live (~3 cr/name), same-day re-clicks are 0 cr)."""
    budget.reset_run()
    ctx = ToolContext(client=client, memory=memory, budget=budget, as_of=AS_OF)
    with st.spinner("Scoring watchlist…"):
        out = run_tool("rank_watchlist", {}, ctx)
    if not out["ok"]:
        st.session_state["last_error"] = f"Refresh failed ({out['error']}): {out.get('message','')}"
        return
    results = out["data"]["ranked"]
    st.session_state["scores"] = results
    st.session_state["deltas"] = memory.record_run(results)   # D7 upsert + FR6 deltas
    st.session_state["last_error"] = ""


def brief_markdown(out: dict) -> str:
    b = out["brief"]
    lines = [f"# BANDAR brief — {b['symbol']} ({datetime.now(WIB).date().isoformat()} WIB)", "",
             b["interpretation"], "", "**Extracted (engine figures):**"]
    lines += [f"- {x}" for x in b["extracted"]]
    lines += ["", "**Action plan:**"] + [f"- {x}" for x in b["action_plan"]]
    lines += ["", "**Risk flags:**"] + [f"- {x}" for x in (b["risk_flags"] or ["none"])]
    if out.get("dropped_figures"):
        lines += ["", f"_Validator dropped {len(out['dropped_figures'])} unverifiable figure(s)._"]
    stt = budget.status()
    lines += ["", f"_Sectors credits used: {stt['total_spent']}/1000 · LLM calls: {out.get('llm_calls')}_"]
    return "\n".join(lines)


def do_ask(question: str) -> None:
    budget.reset_run()
    ctx = ToolContext(client=client, memory=memory, budget=budget, as_of=AS_OF)
    with st.spinner("Bandar is thinking (max 3 LLM calls)…"):
        try:
            out = run_agent(question, ctx)
        except SectorsError as exc:                     # product breaks loudly without Sectors (§14)
            st.session_state["last_error"] = f"Sectors data failure: {exc}"
            return
    st.session_state["agent_out"] = out
    if out["type"] == "brief":
        st.session_state["brief_md"] = brief_markdown(out)
        scored = []
        for t in out.get("tool_results", []):
            if t["ok"] and t["tool"] == "score_ticker":
                scored.append(t["data"])
            elif t["ok"] and t["tool"] == "rank_watchlist":
                scored.extend(t["data"]["ranked"])
        if scored:
            st.session_state["scores"] = sorted(scored, key=lambda r: -(r["score"] or 0))
            st.session_state["deltas"] = [
                build_delta(r, memory.get_prev(r["symbol"], r["as_of"])) for r in scored]
    st.session_state["last_error"] = ""


# Ask first (script order matters): do_ask may fill brief_md, and the buttons
# below must render from the UPDATED session_state within the same rerun.
# (st.chat_input docks at the bottom of the page regardless of script position.)
prompt = st.chat_input("Ask Bandar… (e.g. 'who's accumulating BMRI?', 'top pick today?')")
if prompt:
    do_ask(prompt)

col_a, col_b, _ = st.columns([1, 1, 3])
with col_a:
    if st.button("🔄 Force Live Refresh", help="Fetch today's windows and re-score the watchlist"):
        do_refresh()
with col_b:
    st.download_button(
        "⬇️ Download Brief", data=st.session_state["brief_md"] or "_no brief yet_",
        file_name=f"bandar_brief_{AS_OF.isoformat()}.md", mime="text/markdown",
        disabled=not st.session_state["brief_md"],
    )


# ------------------------------------------------------------------ TOP PICK card

def restore_from_memory() -> list[dict]:
    """0-cr boot: show the last known run from SQLite when session is empty."""
    rows = []
    for sym in memory.load_watchlist():
        latest = memory.latest(sym)
        if latest:
            rows.append(latest)
    return rows


scores = st.session_state["scores"]
st.subheader("🎯 TOP PICK")
if scores:
    top = scores[0] if isinstance(scores[0], dict) and "score" in scores[0] else None
    if top:
        card = st.container(border=True)
        with card:
            c1, c2, c3, c4 = st.columns([1.2, 2.2, 1.4, 2.2])
            c1.metric(f"{top['symbol']}", f"{top['score']}/{top['denominator']}")
            checks = []
            for fk, label in FACTOR_LABELS.items():
                f = (top.get("factors") or {}).get(fk)
                mark = "➖" if f is None else ("✅" if f.get("pass") else "❌")
                checks.append(f"{mark} {label}")
            c2.markdown(" · ".join(checks))
            d = top["decision"]
            c3.metric("Decision", d["label"], d["deploy_pct"])
            tp = top.get("trade_plan")
            if tp:
                ez = tp["entry_zone"]
                c4.markdown(f"**Entry** {ez[0]}–{ez[1]} · **Stop (close)** {tp['stop_close']} · "
                            f"**Lots** {tp['lots']} · tick_valid {'✅' if tp['tick_valid'] else '❌'}")
            else:
                c4.markdown("**No entry** — wait for a better setup.")
            gate = (top.get("gates") or {}).get("no_chase", {})
            why = (f"no-chase gate triggered — WAIT for pullback" if gate.get("triggered")
                   else f"{top['score']}/{top['denominator']} confluence · regime "
                        f"{(top.get('gates') or {}).get('regime', 'n/a')}"
                        + (f" · flags: {', '.join(top['risk_flags'])}" if top.get("risk_flags") else ""))
            st.caption(f"**Why:** {why} · as of {top['as_of']}")
else:
    prior = restore_from_memory()
    if prior:
        st.info("Showing last recorded run from memory (0 cr). Press **Force Live Refresh** "
                "for today's data.")
        best = max(prior, key=lambda r: (r.get("score") or 0))
        st.markdown(f"**{best['ticker']}** — {best['score']}/{best['denominator']} · "
                    f"{best['decision']} · as of {best['as_of']}")
    else:
        st.info("No scores yet — press **Force Live Refresh** to score the watchlist, "
                "or ask a question below.")


# ------------------------------------------------------------- WATCHLIST DELTAS

st.subheader("📊 WATCHLIST DELTAS")
deltas = st.session_state["deltas"]
if deltas:
    rows = [{
        "Ticker": d["ticker"],
        "Score": (f"{d['score_old']}/{d['denominator_old']} → {d['score_new']}/{d['denominator_new']}"
                  if d["has_prev"] else f"new: {d['score_new']}/{d['denominator_new']}"),
        "Decision": (f"{d['decision_old']} → {d['decision_new']}"
                     if d["has_prev"] else d["decision_new"]),
        "Gate override": d.get("gate_override") or "",
        "Prior date": d.get("prev_date") or "—",
    } for d in deltas]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.caption("Deltas appear after a refresh or an ask that scores tickers (old→new + prior date).")


# ------------------------------------------------------------------ brief answer

agent_out = st.session_state["agent_out"]
if agent_out is not None:
    st.subheader("💬 Answer")
    if agent_out["type"] == "brief":
        b = agent_out["brief"]

        def _stream():
            yield f"**{b['symbol']}** — {b['interpretation']}\n\n"
            for x in b["extracted"]:
                yield f"- {x}\n"
            yield "\n**Action plan:**\n"
            for x in b["action_plan"]:
                yield f"- {x}\n"
            if b["risk_flags"]:
                yield "\n**Risk flags:**\n"
                for x in b["risk_flags"]:
                    yield f"- {x}\n"
            if agent_out.get("dropped_figures"):
                yield (f"\n_⚠️ {len(agent_out['dropped_figures'])} figure(s) dropped by the "
                       "numeric trace validator (unverifiable)._\n")

        st.write_stream(_stream)
    else:
        st.warning(f"⚠️ {agent_out['message']}")


# ------------------------------------------------------------- workings (collapsed)

with st.expander("🔧 Workings — plan · data pulls · factor math · credits", expanded=False):
    if agent_out:
        st.markdown("**Agent plan** (LLM calls: "
                    f"{', '.join(agent_out.get('llm_calls', [])) or '—'})")
        for i, s in enumerate(agent_out.get("plan", {}).get("steps", []), 1):
            st.markdown(f"{i}. `{s['tool']}({json.dumps(s.get('args', {}))})` — {s.get('reason','')}")
        if agent_out.get("truncated"):
            st.warning("Run truncated by budget — narrated honestly (policy 4).")

    st.markdown("**Data pulls + routing** (cache/live tags)")
    traces = []
    for r in scores or []:
        traces.extend(r.get("trace") or [])
    if agent_out:
        traces.extend(agent_out.get("trace") or [])
    if traces:
        with st.status(f"{len(traces)} pull(s)", expanded=False) as status:
            for t in traces:
                st.write(f"`{t}`")
            status.update(label=f"{len(traces)} pull(s) — "
                          f"{sum('[cache]' in t for t in traces)} cache / "
                          f"{sum('[live' in t for t in traces)} live", state="complete")
    else:
        st.caption("no pulls yet")

    st.markdown("**Per-factor math**")
    for r in scores or []:
        if not isinstance(r, dict) or "factors" not in r or not r.get("factors"):
            continue
        with st.expander(f"{r.get('symbol')} — {r.get('score')}/{r.get('denominator')} "
                         f"{r.get('decision', {}).get('label', '')}", expanded=False):
            for fk, label in FACTOR_LABELS.items():
                f = (r.get("factors") or {}).get(fk)
                if f is None:
                    st.markdown(f"- {label}: ➖ null (honest — not fabricated)")
                else:
                    vals = ", ".join(f"{k}={v}" for k, v in (f.get("values") or {}).items())
                    st.markdown(f"- {label}: {'✅' if f.get('pass') else '❌'} · {vals}"
                                + (f" · flags: {f.get('flags')}" if f.get("flags") else ""))
            g = r.get("gates") or {}
            st.markdown(f"- Gates: no_chase={g.get('no_chase', {}).get('triggered')} · "
                        f"ara={g.get('ara', {}).get('triggered')} · regime={g.get('regime')}")

    st.markdown("**Credit meter**")
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("total spent", stt["total_spent"])
    cm2.metric("remaining spendable", stt["remaining_total"])
    cm3.metric("reserve (locked)", stt["reserve"])
    log_path = STATE_DIR / "budget_log.jsonl"
    if log_path.exists():
        tail = log_path.read_text().splitlines()[-5:]
        st.caption("Last live calls: " + (" | ".join(tail) if tail else "none"))

st.caption("_Bandar never executes trades (FR9). Numbers: deterministic engine only; "
           "the LLM narrates and every figure is trace-validated (FR10)._")
