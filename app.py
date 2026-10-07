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
  per-process default for camera-safe recordings; the in-UI sun/moon toggle
  overrides it per session via st.session_state, NOT a settings page).

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

def resolve_theme() -> str:
    """Session toggle > BANDAR_THEME env > dark. Invalid -> dark."""
    t = (st.session_state.get("theme") or os.environ.get("BANDAR_THEME", "dark"))
    t = str(t).strip().lower()
    return t if t in ("dark", "light") else "dark"


PALETTES = {
    # "Bloomberg terminal meets Telegram" night scheme (pass-4 audit adoption):
    # bg #0f0f1a · card #1a1a2e · highlight #16213e · card border #0f3460
    # status: success #4ade80 · warning #fbbf24 · danger #f87171 · info #3b82f6
    "dark": dict(scheme="dark", bg="#0f0f1a", panel="#1a1a2e", highlight="#16213e",
                 cardborder="#0f3460", text="#ffffff", muted="#888888",
                 border="#2b2b45", green="#4ade80", amber="#fbbf24",
                 red="#f87171", accent="#3b82f6", accent2="#6366f1", on_accent="#ffffff",
                 glow="rgba(59,130,246,0.35)", markfill="#1a1a2e", markstroke="#0f3460",
                 rowhover="rgba(255,255,255,0.05)", green_bg="rgba(74,222,128,0.14)",
                 amber_bg="rgba(251,191,36,0.14)", red_bg="rgba(248,113,113,0.14)",
                 mono='"SF Mono", "Monaco", "Inconsolata", ui-monospace, Menlo, monospace'),
    "light": dict(scheme="light", bg="#ffffff", panel="#f6f8fa", highlight="#eaeef2",
                  cardborder="#b6c2cf", text="#1f2328", muted="#57606a",
                  border="#d0d7de", green="#1a7f37", amber="#9a6700",
                  red="#cf222e", accent="#0969da", accent2="#4f46e5", on_accent="#ffffff",
                  glow="rgba(9,105,218,0.18)", markfill="#eaeef2", markstroke="#94a3b8",
                  rowhover="rgba(31,35,40,0.04)", green_bg="rgba(26,127,55,0.10)",
                  amber_bg="rgba(154,103,0,0.10)", red_bg="rgba(207,34,46,0.10)",
                  mono='"SF Mono", "Monaco", "Inconsolata", ui-monospace, Menlo, monospace'),
}

CSS_TEMPLATE = """
<style>
:root { color-scheme: @scheme@; --bd-bg: @bg@; --bd-panel: @panel@; --bd-text: @text@;
  --bd-muted: @muted@; --bd-border: @border@; --bd-green: @green@; --bd-amber: @amber@;
  --bd-red: @red@; --bd-accent: @accent@; --bd-markfill: @markfill@;
  --bd-markstroke: @markstroke@; }
html, body, [data-testid="stAppViewContainer"], .stApp,
section[data-testid="stApp"], [data-testid="stAppScrollToBottomContainer"] {
  background: @bg@ !important; }
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
  font-variant-numeric: tabular-nums; font-size: 1.3rem !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: @muted@ !important; font-size: .7rem !important;
  text-transform: uppercase; letter-spacing: .06em; }

/* buttons */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  background: @panel@; color: @text@; border: 1px solid @border@;
  border-radius: 10px; font-weight: 600; }
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: @accent@; color: @accent@; }
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, @accent@ 0%, @accent2@ 100%) !important;
  border-color: @accent@ !important; color: @on_accent@ !important;
  box-shadow: 0 4px 14px @glow@; }
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px);
  box-shadow: 0 6px 18px @glow@; }
/* pass-5: the actions row reads as one designed toolbar band */
[data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-primary"]) {
  background: @panel@; border: 1px solid @border@; border-radius: 14px;
  padding: 10px 14px; }
/* pass-5: session theme toggle — compact square icon button */
[data-testid="stColumn"]:has([data-testid="stMetric"]) button:first-of-type {
  width: 38px; height: 38px; padding: 0; display: flex; align-items: center;
  justify-content: center; border-radius: 10px; }

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
/* One cohesive input pill: the container is the border; textarea + send button sit
   inside it transparently. Single brand-colored focus ring via :focus-within (no
   double red+blue border, no detached arrow). */
[data-testid="stChatInput"] { background: @panel@; border: 1px solid @border@;
  border-radius: 12px; padding: 4px 6px 4px 14px;
  display: flex; align-items: center; gap: 6px; }
[data-testid="stChatInput"]:focus-within { border-color: @accent@;
  box-shadow: 0 0 0 3px @glow@; }
[data-testid="stChatInput"] > div { background: transparent !important; flex: 1; }
[data-testid="stChatInputTextArea"] { background: transparent !important;
  color: @text@ !important; border: none !important; box-shadow: none !important;
  outline: none !important; }
[data-testid="stChatInputTextArea"]::placeholder { color: @muted@; }
[data-testid="stChatInput"] button { background: transparent !important;
  color: @text@ !important; border: none !important; }

/* ---- bandar components ---- */
.bd-card { background: linear-gradient(135deg, @panel@ 0%, @highlight@ 100%);
  border: 1px solid @cardborder@; border-radius: 16px;
  padding: 18px 22px; margin: 6px 0 10px; }
.bd-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.bd-sym { font-size: 2rem; font-weight: 800; letter-spacing: .04em;
  color: @text@; font-family: @mono@; }
.bd-mono, .bd-kv b { font-family: @mono@; font-variant-numeric: tabular-nums; }
.bd-pill { display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-weight: 700; font-size: .8rem; font-family: @mono@; }
.bd-pill.green { background: @green_bg@; color: @green@; border: 1px solid @green@; }
.bd-pill.amber { background: @amber_bg@; color: @amber@; border: 1px solid @amber@; }
.bd-pill.red   { background: @red_bg@;   color: @red@;   border: 1px solid @red@; }
.bd-pill.blue  { background: rgba(59,130,246,0.14); color: @accent@;
  border: 1px solid @accent@; }
.bd-action { margin: 10px 0 2px; font-weight: 700; font-size: .95rem;
  font-family: @mono@; }
.bd-action.green { color: @green@; }
.bd-action.amber { color: @amber@; }
.bd-action.red { color: @red@; }
.bd-badge { color: @muted@; font-size: .75rem; border: 1px dashed @border@;
  border-radius: 999px; padding: 2px 10px; }
.bd-row .bd-badge:last-child { margin-left: auto; }   /* as-of badge right-anchored */
.bd-checks { margin: 10px 0 4px; color: @muted@; font-size: .85rem; }
.bd-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px; margin: 12px 0 4px; }
.bd-kv { background: @bg@; border: 1px solid @border@; border-radius: 10px;
  padding: 8px 12px; }
.bd-kv span { display: block; color: @muted@; font-size: .68rem;
  text-transform: uppercase; letter-spacing: .06em; }
.bd-kv b { color: @text@; font-size: 1.05rem; }
.bd-kv.stop b { color: @red@; }
.bd-noentry { margin: 12px 0 4px; color: @muted@; font-style: italic; }
.bd-why { margin-top: 12px; background: @bg@; border-left: 3px solid @green@;
  padding: 10px 14px; border-radius: 8px; color: @text@; font-size: .9rem; }
.bd-why b { color: @accent@; }
.bd-up { color: @green@; font-family: @mono@; font-weight: 600; }
.bd-down { color: @red@; font-family: @mono@; font-weight: 600; }
.bd-same { color: @muted@; font-family: @mono@; font-weight: 600; }
.bd-trace { color: @muted@; font-family: @mono@; font-size: .78rem; }
.bd-tag { font-family: @mono@; font-size: .75rem; font-weight: 600; }
.bd-tag.cache { color: @green@; }
.bd-tag.live { color: @amber@; }
.bd-table { width: 100%; border-collapse: collapse; font-size: .88rem; margin-top: 4px; }
.bd-table th { text-align: left; color: @muted@; font-size: .68rem;
  text-transform: uppercase; letter-spacing: .08em; padding: 6px 10px;
  border-bottom: 1px solid @border@; }
.bd-table td { padding: 9px 10px; border-bottom: 1px solid @border@; color: @text@;
  font-variant-numeric: tabular-nums; }
.bd-table tbody tr:hover td { background: @rowhover@; }

/* ---- brand header (P5) ---- */
.bd-brand { display: flex; align-items: center; gap: 14px; margin: 4px 0 2px; }
.bd-mark { width: 40px; height: 40px; flex: none; }
.bd-wordmark { font-size: 1.9rem; font-weight: 800; letter-spacing: .14em;
  color: @text@; font-family: @mono@; }
.bd-brandtag { color: @muted@; font-size: .95rem;
  border-left: 1px solid @border@; padding-left: 14px; }

/* ---- section headings with inline SVG icons (P1) ---- */
.bd-h { display: flex; align-items: center; gap: 10px; font-size: 1.15rem;
  font-weight: 700; letter-spacing: .03em; margin: 18px 0 8px; color: @text@; }
.bd-ic { width: 18px; height: 18px; color: @accent@; flex: none; }

/* ---- factor check glyphs (P1) ---- */
.bd-ok { color: @green@; font-weight: 700; }
.bd-bad { color: @red@; font-weight: 700; }
.bd-null { color: @muted@; }

/* ---- score dots (P2) ---- */
.bd-dots { display: inline-flex; gap: 3px; margin-left: 8px; vertical-align: middle; }
.bd-dots i { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  border: 1px solid @muted@; }
.bd-dots.green i.on { background: @green@; border-color: @green@; }
.bd-dots.amber i.on { background: @amber@; border-color: @amber@; }
.bd-dots.red i.on { background: @red@; border-color: @red@; }

/* ---- motion polish (P3) ---- */
.bd-card, [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
.bd-table td { transition: border-color .15s ease, background .15s ease,
  transform .15s ease, color .15s ease; }
.bd-card:hover { border-color: @accent@; transform: translateY(-1px); }
[data-testid="stButton"] button:focus-visible,
[data-testid="stDownloadButton"] button:focus-visible {
  outline: 2px solid @accent@; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .bd-card, [data-testid="stButton"] button,
  [data-testid="stDownloadButton"] button { transition: none; transform: none; }
}

/* hide Streamlit heading anchor links (video noise) */
[data-testid="stAppViewContainer"] h1 a, [data-testid="stAppViewContainer"] h2 a,
[data-testid="stAppViewContainer"] h3 a, [data-testid="stAppViewContainer"] h4 a {
  display: none; }

/* ---- P4: desk-context side panel + last-scored chip ---- */
.bd-side { background: @panel@; border: 1px solid @border@; border-radius: 16px;
  padding: 16px 18px; height: 100%; }
.bd-side-h { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: @muted@; margin-bottom: 10px; }
.bd-kvrow { display: flex; justify-content: space-between; gap: 12px;
  padding: 7px 0; border-bottom: 1px dashed @border@; font-size: .85rem; }
.bd-kvrow:last-child { border-bottom: none; }
.bd-kvrow span { color: @muted@; }
.bd-kvrow b { color: @text@; font-family: @mono@; font-weight: 600; }
.bd-chip-right { display: flex; justify-content: flex-end; color: @muted@;
  font-size: .75rem; font-family: @mono@; padding-top: 8px; }
</style>
"""


def build_css(palette: dict) -> str:
    css = CSS_TEMPLATE
    for key, val in palette.items():
        css = css.replace(f"@{key}@", val)
    return css


# Inline SVG icon system (P1): crisp, platform-independent, theme-aware via
# currentColor / CSS vars. No emoji anywhere in the UI chrome.
IC_MARK = ('<svg class="bd-mark" viewBox="0 0 32 32" fill="none" aria-hidden="true">'
           '<rect x="1" y="1" width="30" height="30" rx="8" '
           'style="fill:var(--bd-markfill, #1a1a2e); stroke:var(--bd-markstroke, #0f3460)"/>'
           '<path d="M8 22l5.5-7 4 3.2L24 9" style="stroke:var(--bd-accent, #3b82f6)" '
           'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
           '<circle cx="24" cy="9" r="2.6" style="fill:var(--bd-green, #4ade80)"/></svg>')
IC_TARGET = ('<svg class="bd-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/>'
             '<circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1.2" '
             'fill="currentColor" stroke="none"/></svg>')
IC_BARS = ('<svg class="bd-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
           'stroke-width="2" stroke-linecap="round" aria-hidden="true">'
           '<path d="M5 20V11"/><path d="M12 20V4"/><path d="M19 20v-6"/></svg>')
IC_CHAT = ('<svg class="bd-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
           'stroke-width="2" stroke-linejoin="round" aria-hidden="true">'
           '<path d="M4 5h16v11H10l-6 4z"/></svg>')

BRAND_HTML = ('<div class="bd-brand">' + IC_MARK +
              '<span class="bd-wordmark">BANDAR</span>'
              '<span class="bd-brandtag">pre-market desk note</span></div>')


def h2(icon: str, text: str) -> str:
    return f'<h2 class="bd-h">{icon}{text}</h2>'


# Pass-5: trader-facing regime copy. Raw engine codes stay in Workings only.
REGIME_LABELS = {
    "bearish_below_ma200": "Below MA200 — bearish tape",
    "bullish_above_ma200": "Above MA200 — bullish tape",
    "neutral_insufficient": "Regime unknown — insufficient data",
}


def regime_label(code) -> str:
    return REGIME_LABELS.get(code, str(code).replace("_", " "))


def dots_html(score, denom, kind: str) -> str:
    """P2: ●●●○○ confluence at a glance (denom slots, `score` filled)."""
    if score is None or denom is None:
        return ""
    filled = max(0, min(int(score), int(denom)))
    cells = "".join('<i class="on"></i>' if i < filled else '<i></i>'
                    for i in range(int(denom)))
    return f'<span class="bd-dots {kind}" title="{score}/{denom} confluence">{cells}</span>'


# P6: semantic meter colors. The three header st.metric chips (LOCKED widget) are
# targeted positionally; the workings meter sits inside <details> so it is excluded.
# Browsers without :has() ignore the rules -> meters stay neutral (graceful).
METER_SEL = ('[data-testid="stMainBlockContainer"] '
             '[data-testid="stHorizontalBlock"]:not(details *)'
             ':has(> div [data-testid="stMetric"])')


def meter_style(stt: dict, palette: dict) -> str:
    def col(val, amber_at, red_at, invert=False):
        hit = (val <= red_at) if invert else (val >= red_at)
        warn = (val <= amber_at) if invert else (val >= amber_at)
        return palette["red"] if hit else (palette["amber"] if warn else palette["green"])
    c1 = col(stt["remaining_total"], 200, 60, invert=True)   # spendable left
    c2 = col(stt["daily_spent"], 150, 200)                   # today vs 200 soft warn
    c3 = col(stt["run_spent"], 40, 60)                       # run vs 60 hard cap
    rules = "".join(
        f'{METER_SEL} > div:nth-child({i}) [data-testid="stMetricValue"] '
        f'{{ color: {c} !important; }}'
        for i, c in ((1, c1), (2, c2), (3, c3)))
    return f"<style>{rules}</style>"


st.set_page_config(page_title="BANDAR", page_icon="📈", layout="wide")
THEME = resolve_theme()
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

stt = budget.status()
hL, hR = st.columns([2.2, 3], gap="large")
with hL:
    st.markdown(BRAND_HTML, unsafe_allow_html=True)
    st.caption(f"IDX swing-trading analyst · as-of {AS_OF.isoformat()} (WIB) · "
               "analysis only — **no trade execution, ever** (FR9)")
with hR:
    tcol, tstamp = st.columns([1, 6])
    with tcol:
        if st.button(":material/dark_mode:" if THEME == "dark" else ":material/light_mode:",
                     key="theme_toggle",
                     help="Current theme — click to switch for this session "
                          "(default: BANDAR_THEME env)"):
            _new = "light" if THEME == "dark" else "dark"
            st.session_state["theme"] = _new
            # <style> is document-global: re-injecting here repaints the current
            # rerun immediately (the top-of-script injection ran pre-handler).
            st.markdown(build_css(PALETTES[_new]), unsafe_allow_html=True)
    with tstamp:
        st.markdown(f'<div class="bd-chip-right">{datetime.now(WIB).strftime("%d %b %H:%M")} '
                    "WIB</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sectors credits left", stt["remaining_total"], help="of 800 spendable (200 reserve locked)")
    m2.metric("Today (WIB)", f"{stt['daily_spent']} / 200", help="soft warn at 200/day")
    m3.metric("This run", f"{stt['run_spent']} / 60", help="hard cap 60/run")
    st.markdown(meter_style(stt, PALETTES[resolve_theme()]), unsafe_allow_html=True)

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

col_a, col_b, col_c = st.columns([1, 1, 2.4])
with col_a:
    if st.button(":material/sync: Force Live Refresh", type="primary",
                 help="Fetch today's windows and re-score the watchlist"):
        do_refresh()
with col_b:
    st.download_button(
        ":material/download: Download Brief", data=st.session_state["brief_md"] or "_no brief yet_",
        file_name=f"bandar_brief_{AS_OF.isoformat()}.md", mime="text/markdown",
        disabled=not st.session_state["brief_md"],
    )
with col_c:                                                  # P4: fill dead space
    # computed HERE (after the refresh handler) so the chip updates same-rerun
    _last_rows = [memory.latest(s) for s in memory.load_watchlist()]
    _last_ts = max((r["created_at"] for r in _last_rows if r), default=None)
    st.markdown('<div class="bd-chip-right">'
                + (f"last scored: {_last_ts[:16].replace('T', ' ')} WIB" if _last_ts
                   else "not scored yet")
                + "</div>", unsafe_allow_html=True)


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

    kind = _decision_kind(label, bool(gate.get("triggered")))
    bias = d.get("bias") if isinstance(d, dict) else None
    head = [f'<span class="bd-sym">{sym}</span>',
            _pill(f"{score}/{denom}" if score is not None else "—", _score_kind(score))
            + dots_html(score, denom, _score_kind(score)),
            _pill(label + (f" · {deploy}" if deploy else ""), kind)]
    if bias:
        head.append(_pill(bias, "blue"))
    if badge:
        head.append(f'<span class="bd-badge">{badge}</span>')
    head.append(f'<span class="bd-badge">as of {r.get("as_of")}</span>')
    action = (f'<div class="bd-action {kind}">Action: {label}'
              + (f" ({deploy} deploy)" if deploy else "") + "</div>")

    checks = []
    for fk, flabel in FACTOR_LABELS.items():
        f = (r.get("factors") or {}).get(fk)
        if f is None:
            mark = '<span class="bd-null">○</span>'
        elif f.get("pass"):
            mark = '<span class="bd-ok">✓</span>'
        else:
            mark = '<span class="bd-bad">✕</span>'
        checks.append(f"{mark} {flabel}")

    tp = r.get("trade_plan")
    if tp:
        ez = tp["entry_zone"]
        body = ('<div class="bd-grid">'
                f'<div class="bd-kv"><span>Entry zone</span><b>{_fmt_price(ez[0])}–{_fmt_price(ez[1])}</b></div>'
                f'<div class="bd-kv stop"><span>Stop (close)</span><b>{_fmt_price(tp["stop_close"])}</b></div>'
                f'<div class="bd-kv"><span>Lots · 0.5% risk</span><b>{tp["lots"]}</b></div>'
                "</div>")
    else:
        body = ('<div class="bd-noentry">No disciplined entry today — wait for a '
                "better setup (close-based rules only, FR4).</div>")

    why = ("no-chase gate triggered — WAIT for pullback (never chase, D3)"
           if gate.get("triggered") else
           f"{score}/{denom} confluence · "
           f"{regime_label((r.get('gates') or {}).get('regime', 'n/a'))}"
           + (f" · flags: {', '.join(r.get('risk_flags') or [])}" if r.get('risk_flags') else ""))

    return ('<div class="bd-card">'
            f'<div class="bd-row">{"".join(head)}</div>'
            f'<div class="bd-checks">{" · ".join(checks)}</div>'
            f"{action}"
            f"{body}"
            f'<div class="bd-why"><b>Why:</b> {why}</div>'
            "</div>")


scores = st.session_state["scores"]
st.markdown(h2(IC_TARGET, "TOP PICK"), unsafe_allow_html=True)


def _is_actionable(r: dict) -> bool:
    d = r.get("decision") or {}
    label = d.get("label") if isinstance(d, dict) else str(d)
    return bool(r.get("trade_plan")) and label != "WAIT"


def context_panel(rows: list[dict], as_of) -> str:
    """P4: desk-context side panel — all values engine/memory-derived, 0 calls."""
    data_date = max((r.get("as_of") or "" for r in rows), default="—")
    watch_n = len(memory.load_watchlist())
    act = sum(1 for r in rows if _is_actionable(r))
    top = rows[0] if rows else None
    regime = ((top.get("gates") or {}).get("regime", "—") if top else "—")
    kv = [("Anchor (WIB)", as_of.isoformat()), ("Data as of", data_date),
          ("Scored", f"{len(rows)}/{watch_n}"), ("Actionable", str(act)),
          ("Top regime", regime_label(regime) if regime != "—" else "—")]
    body = "".join(f'<div class="bd-kvrow"><span>{k}</span><b>{v}</b></div>'
                   for k, v in kv)
    return f'<div class="bd-side"><div class="bd-side-h">Desk context</div>{body}</div>'


prior = [] if scores else restore_from_memory()
shown = scores or prior
hero_l, hero_r = st.columns([2, 1], gap="medium")
with hero_r:
    st.markdown(context_panel(shown, AS_OF), unsafe_allow_html=True)
with hero_l:
    if scores:
        top = scores[0] if isinstance(scores[0], dict) and "score" in scores[0] else None
        if top:
            st.markdown(pick_card(top), unsafe_allow_html=True)
    elif prior:
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
        score_cell = f'{d["score_new"]}/{d["denominator_new"]}'
        score_cell += dots_html(d["score_new"], d["denominator_new"], kind)
        if d["has_prev"]:
            old, new = d["score_old"], d["score_new"]
            if new > old:
                delta_cell = f'<span class="bd-up">▲ {old}→{new}</span>'
            elif new < old:
                delta_cell = f'<span class="bd-down">▼ {old}→{new}</span>'
            else:
                delta_cell = f'<span class="bd-same">= {old}→{new}</span>'
            prior = d.get("prev_date") or "—"
            try:
                prior += f" ({(AS_OF - datetime.fromisoformat(prior).date()).days}d)"
            except (ValueError, TypeError):
                pass
        else:
            delta_cell = '<span class="bd-same">new</span>'
            prior = "—"
        dec_cell = (f'{d["decision_old"]} → ' if d["has_prev"] else "") + _pill(dec_new, kind)
        gate_cell = _pill("⊘ no-chase", "amber") if gate else '<span class="bd-null">—</span>'
        rows.append(f'<tr><td class="bd-mono">{d["ticker"]}</td>'
                    f'<td class="bd-mono">{score_cell}</td>'
                    f"<td>{delta_cell}</td><td>{dec_cell}</td><td>{gate_cell}</td>"
                    f'<td class="bd-mono">{prior}</td></tr>')
    return ('<table class="bd-table"><thead><tr>'
            '<th style="width:10%">Ticker</th><th style="width:18%">Score</th>'
            '<th style="width:12%">Δ</th><th style="width:22%">Decision</th>'
            '<th style="width:18%">Gate</th><th style="width:20%">Prior date</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


st.markdown(h2(IC_BARS, "WATCHLIST DELTAS"), unsafe_allow_html=True)
deltas = st.session_state["deltas"]
if deltas:
    st.markdown(deltas_table(deltas), unsafe_allow_html=True)
else:
    st.caption("Deltas appear after a refresh or an ask that scores tickers (old→new + prior date).")


# ------------------------------------------------------------------ brief answer

agent_out = st.session_state["agent_out"]
if agent_out is not None:
    st.markdown(h2(IC_CHAT, "Answer"), unsafe_allow_html=True)
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

with st.expander("Show Workings (Audit Trail)", expanded=False):
    if agent_out:
        st.markdown("**Agent plan** (LLM calls: "
                    f"{', '.join(agent_out.get('llm_calls', [])) or '—'})")
        _intent = agent_out.get("intent") or {}
        if _intent:
            _iline = f"**Intent:** `{_intent.get('intent')}`"
            if _intent.get("symbols"):
                _iline += f" · symbols: {', '.join(_intent['symbols'])}"
            if _intent.get("hint"):
                _iline += " · deterministic fast-path (0 LLM)"
            st.caption(_iline)
            if _intent.get("reason"):
                st.caption(f"_intent reason:_ {_intent['reason'][:300]}")
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
                if "[" in t:
                    pre, tag = t.split("[", 1)
                    tag = "[" + tag
                    cls = "live" if tag.startswith("[live") else "cache"
                    st.markdown(f'<span class="bd-trace">{pre.strip()}</span> '
                                f'<span class="bd-tag {cls}">{tag}</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="bd-trace">{t}</span>',
                                unsafe_allow_html=True)
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
                    st.markdown(f"- {label}: – null (honest — not fabricated)")
                else:
                    vals = ", ".join(f"{k}={v}" for k, v in (f.get("values") or {}).items())
                    st.markdown(f"- {label}: {'✓' if f.get('pass') else '✕'} · {vals}"
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
