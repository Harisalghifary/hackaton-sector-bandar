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
  (isolate ledger/DB — used by tests), BANDAR_THEME=dark|light (default dark —
  a UI preference only, NOT a settings page; set once per run/recording).

Run: streamlit run app.py            (dark terminal look)
     BANDAR_THEME=light streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

# --------------------------------------------------------------------- theming

THEME = os.environ.get("BANDAR_THEME", "dark").strip().lower()
if THEME not in ("dark", "light"):
    THEME = "dark"

PALETTES = {
    "dark": dict(scheme="dark", bg="#0d1117", panel="#161b22", text="#e6edf3",
                 muted="#8b949e", border="#30363d", green="#3fb950", amber="#d29922",
                 red="#f85149", accent="#58a6ff", on_accent="#0d1117",
                 rowhover="rgba(240,246,252,0.05)", green_bg="rgba(63,185,80,0.14)",
                 amber_bg="rgba(210,153,34,0.14)", red_bg="rgba(248,81,73,0.14)",
                 mono='"SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'),
    "light": dict(scheme="light", bg="#ffffff", panel="#f6f8fa", text="#1f2328",
                  muted="#57606a", border="#d0d7de", green="#1a7f37", amber="#9a6700",
                  red="#cf222e", accent="#0969da", on_accent="#ffffff",
                  rowhover="rgba(31,35,40,0.04)", green_bg="rgba(26,127,55,0.10)",
                  amber_bg="rgba(154,103,0,0.10)", red_bg="rgba(207,34,46,0.10)",
                  mono='"SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'),
}

CSS_TEMPLATE = """
<style>
:root { color-scheme: @scheme@; }
html, body, [data-testid="stAppViewContainer"], .stApp { background: @bg@ !important; }
[data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label { color: @text@; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: @muted@ !important; }

/* compact credit-meter chips */
[data-testid="stMetric"] { background: @panel@; border: 1px solid @border@;
  border-radius: 12px; padding: 6px 14px; }
[data-testid="stMetricValue"] { color: @text@ !important; font-family: @mono@;
  font-size: 1.3rem !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: @muted@ !important; font-size: .7rem !important;
  text-transform: uppercase; letter-spacing: .06em; }

/* buttons */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  background: @panel@; color: @text@; border: 1px solid @border@;
  border-radius: 10px; font-weight: 600; }
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: @accent@; color: @accent@; }
[data-testid="stBaseButton-primary"] { background: @accent@ !important;
  border-color: @accent@ !important; color: @on_accent@ !important; }

/* workings expander + status + code */
[data-testid="stExpander"] { background: @panel@; border: 1px solid @border@;
  border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
  color: @text@ !important; }
[data-testid="stExpander"] code, [data-testid="stStatusWidget"] code {
  background: @bg@; color: @accent@; }
[data-testid="stStatusWidget"] { background: @bg@; border: 1px solid @border@;
  border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 12px; }

/* chat input + fixed chrome (header bar / bottom dock) */
[data-testid="stHeader"] { background: @bg@ !important; }
[data-testid="stBottom"], [data-testid="stBottomContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div { background: @bg@ !important; }
[data-testid="stDeployButton"] button, [data-testid="stDeployButton"] p,
[data-testid="stMainMenu"] button { color: @muted@ !important; }
[data-testid="stMainMenu"] svg { fill: @muted@; }
[data-testid="stChatInput"] { background: transparent; }
[data-testid="stChatInput"] > div { background: transparent !important; }
[data-testid="stChatInputTextArea"] { background: @panel@ !important;
  color: @text@ !important; border: 1px solid @border@ !important;
  border-radius: 12px; }
[data-testid="stChatInputTextArea"]::placeholder { color: @muted@; }
[data-testid="stChatInput"] button { background: @panel@ !important;
  color: @text@ !important; border: 1px solid @border@ !important; }

/* ---- bandar components ---- */
.bd-card { background: @panel@; border: 1px solid @border@; border-radius: 16px;
  padding: 18px 22px; margin: 6px 0 10px; }
.bd-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.bd-sym { font-size: 2rem; font-weight: 800; letter-spacing: .04em;
  color: @text@; font-family: @mono@; }
.bd-mono, .bd-kv b { font-family: @mono@; }
.bd-pill { display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-weight: 700; font-size: .8rem; font-family: @mono@; }
.bd-pill.green { background: @green_bg@; color: @green@; border: 1px solid @green@; }
.bd-pill.amber { background: @amber_bg@; color: @amber@; border: 1px solid @amber@; }
.bd-pill.red   { background: @red_bg@;   color: @red@;   border: 1px solid @red@; }
.bd-badge { color: @muted@; font-size: .75rem; border: 1px dashed @border@;
  border-radius: 999px; padding: 2px 10px; }
.bd-checks { margin: 10px 0 4px; color: @muted@; font-size: .85rem; }
.bd-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px; margin: 12px 0 4px; }
.bd-kv { background: @bg@; border: 1px solid @border@; border-radius: 10px;
  padding: 8px 12px; }
.bd-kv span { display: block; color: @muted@; font-size: .68rem;
  text-transform: uppercase; letter-spacing: .06em; }
.bd-kv b { color: @text@; font-size: 1.05rem; }
.bd-noentry { margin: 12px 0 4px; color: @muted@; font-style: italic; }
.bd-why { margin-top: 10px; color: @text@; font-size: .9rem; }
.bd-why b { color: @accent@; }
.bd-table { width: 100%; border-collapse: collapse; font-size: .88rem; margin-top: 4px; }
.bd-table th { text-align: left; color: @muted@; font-size: .68rem;
  text-transform: uppercase; letter-spacing: .08em; padding: 6px 10px;
  border-bottom: 1px solid @border@; }
.bd-table td { padding: 9px 10px; border-bottom: 1px solid @border@; color: @text@; }
.bd-table tbody tr:hover td { background: @rowhover@; }
</style>
"""


def build_css(palette: dict) -> str:
    css = CSS_TEMPLATE
    for key, val in palette.items():
        css = css.replace(f"@{key}@", val)
    return css


st.set_page_config(page_title="BANDAR", page_icon="📈", layout="wide")
st.markdown(build_css(PALETTES[THEME]), unsafe_allow_html=True)


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
    if st.button("🔄 Force Live Refresh", type="primary",
                 help="Fetch today's windows and re-score the watchlist"):
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


def _fmt_price(p) -> str:
    return f"{p:g}"


def _pill(text: str, kind: str) -> str:
    return f'<span class="bd-pill {kind}">{text}</span>'


def _decision_kind(label: str, gate_triggered: bool) -> str:
    if gate_triggered:
        return "amber"
    return "red" if label == "WAIT" else "green"


def _score_kind(score) -> str:
    if score is None:
        return "red"
    return "green" if score >= 3 else ("amber" if score == 2 else "red")


def pick_card(r: dict, badge: str | None = None) -> str:
    """The §9 TOP PICK card as themed HTML: score badge · decision pill ·
    five factor checks · entry/stop/lots grid · one-line why."""
    sym = r.get("symbol") or r.get("ticker")
    d = r.get("decision") or {}
    label = d["label"] if isinstance(d, dict) else str(d)
    deploy = d.get("deploy_pct") if isinstance(d, dict) else None
    gate = (r.get("gates") or {}).get("no_chase", {}) or {}
    score, denom = r.get("score"), r.get("denominator")

    head = [f'<span class="bd-sym">{sym}</span>',
            _pill(f"{score}/{denom}" if score is not None else "—", _score_kind(score)),
            _pill(label + (f" · {deploy}" if deploy else ""),
                  _decision_kind(label, bool(gate.get("triggered"))))]
    if badge:
        head.append(f'<span class="bd-badge">{badge}</span>')
    head.append(f'<span class="bd-badge">as of {r.get("as_of")}</span>')

    checks = []
    for fk, flabel in FACTOR_LABELS.items():
        f = (r.get("factors") or {}).get(fk)
        mark = "➖" if f is None else ("✅" if f.get("pass") else "❌")
        checks.append(f"{mark} {flabel}")

    tp = r.get("trade_plan")
    if tp:
        ez = tp["entry_zone"]
        body = ('<div class="bd-grid">'
                f'<div class="bd-kv"><span>Entry zone</span><b>{_fmt_price(ez[0])}–{_fmt_price(ez[1])}</b></div>'
                f'<div class="bd-kv"><span>Stop (close)</span><b>{_fmt_price(tp["stop_close"])}</b></div>'
                f'<div class="bd-kv"><span>Lots · 0.5% risk</span><b>{tp["lots"]}</b></div>'
                "</div>")
    else:
        body = ('<div class="bd-noentry">No disciplined entry today — wait for a '
                "better setup (close-based rules only, FR4).</div>")

    why = ("no-chase gate triggered — WAIT for pullback (never chase, D3)"
           if gate.get("triggered") else
           f"{score}/{denom} confluence · regime {(r.get('gates') or {}).get('regime', 'n/a')}"
           + (f" · flags: {', '.join(r.get('risk_flags') or [])}" if r.get("risk_flags") else ""))

    return ('<div class="bd-card">'
            f'<div class="bd-row">{"".join(head)}</div>'
            f'<div class="bd-checks">{" · ".join(checks)}</div>'
            f"{body}"
            f'<div class="bd-why"><b>Why:</b> {why}</div>'
            "</div>")


scores = st.session_state["scores"]
st.subheader("🎯 TOP PICK")
if scores:
    top = scores[0] if isinstance(scores[0], dict) and "score" in scores[0] else None
    if top:
        st.markdown(pick_card(top), unsafe_allow_html=True)
else:
    prior = restore_from_memory()
    if prior:
        best = max(prior, key=lambda r: (r.get("score") or 0))
        st.markdown(pick_card(best, badge="last recorded run from memory (0 cr)"),
                    unsafe_allow_html=True)
        st.caption("Press **Force Live Refresh** for today's data.")
    else:
        st.info("No scores yet — press **Force Live Refresh** to score the watchlist, "
                "or ask a question below.")


# ------------------------------------------------------------- WATCHLIST DELTAS

def deltas_table(deltas: list[dict]) -> str:
    rows = []
    for d in deltas:
        dec_new = d["decision_new"] or "—"
        gate = d.get("gate_override")
        kind = "amber" if gate else ("red" if dec_new == "WAIT" else "green")
        score_cell = (f'{d["score_old"]}/{d["denominator_old"]} → '
                      f'{d["score_new"]}/{d["denominator_new"]}'
                      if d["has_prev"] else f'new: {d["score_new"]}/{d["denominator_new"]}')
        dec_cell = (f'{d["decision_old"]} → ' if d["has_prev"] else "") + _pill(dec_new, kind)
        gate_cell = _pill("⛔ no-chase", "amber") if gate else ""
        rows.append(f'<tr><td class="bd-mono">{d["ticker"]}</td>'
                    f'<td class="bd-mono">{score_cell}</td>'
                    f"<td>{dec_cell}</td><td>{gate_cell}</td>"
                    f'<td class="bd-mono">{d.get("prev_date") or "—"}</td></tr>')
    return ('<table class="bd-table"><thead><tr><th>Ticker</th><th>Score</th>'
            "<th>Decision</th><th>Gate override</th><th>Prior date</th></tr></thead>"
            f'<tbody>{"".join(rows)}</tbody></table>')


st.subheader("📊 WATCHLIST DELTAS")
deltas = st.session_state["deltas"]
if deltas:
    st.markdown(deltas_table(deltas), unsafe_allow_html=True)
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
                                + (f" · flags: {f.get('flags')}" if f.get('flags') else ""))
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
