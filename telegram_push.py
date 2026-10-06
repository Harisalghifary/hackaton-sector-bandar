"""M6 — Telegram 08:30 WIB morning push (§9, FR11).

Deterministic and engine-only: the cron message is built from the same
rank_watchlist results that drive the UI foreground (FR11 "message matches
foreground"). NO LLM call is involved in this path — the push cannot fail on
model quota and can never fabricate a number (every figure is an engine result,
FR10 in spirit).

§9 content contract:
  top pick + entry/stop/size + why · watchlist status lines ·
  "N others actionable" · credits used.

Spec also says: "GitHub Actions cron; validate a few times, don't run daily" —
see .github/workflows/bandar-push.yml (schedule intentionally commented out;
a cold-cache run fetches every window live, ~48 cr, guarded by the 60/run cap).

Usage:
  python telegram_push.py             # score (cache-first) -> record -> send
  python telegram_push.py --dry-run   # everything except the send (prints it)

Env anchors (shared with app.py):
  BANDAR_AS_OF     replay a specific day deterministically (0 cr on warm cache)
  BANDAR_STATE_DIR isolate ledger/DB (used by tests)
Secrets via .env: SECTORS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.

Exit codes: 0 sent (or dry-run) · 1 data/scoring failure · 2 send/config failure.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from agent.tools import ToolContext, run_tool          # noqa: E402
from budget import CreditBudget                        # noqa: E402
from client import SectorsClient                       # noqa: E402
from memory import ScoreMemory                         # noqa: E402

WIB = ZoneInfo("Asia/Jakarta")
TELEGRAM_MAX_LEN = 4096                # Bot API hard message limit
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramPushError(Exception):
    """Send failure — the product breaks loudly without its channels (§14)."""


# ------------------------------------------------------------------ gathering

def anchor_date() -> str:
    env = os.environ.get("BANDAR_AS_OF")
    if env:
        return datetime.fromisoformat(env).date().isoformat()
    return datetime.now(WIB).date().isoformat()


def gather() -> dict:
    """Score the watchlist (cache-first), record memory + deltas (D7/FR6).

    Same resources/anchors as app.py; results go straight into the message.
    """
    state_dir = Path(os.environ.get("BANDAR_STATE_DIR", "state"))
    budget = CreditBudget(state_path=state_dir / "budget_state.json",
                          log_path=state_dir / "budget_log.jsonl")
    api_key = os.environ.get("SECTORS_API_KEY", "")
    client = SectorsClient(api_key=api_key,
                           mode="cache-first" if api_key else "offline",
                           budget=budget)
    memory = ScoreMemory(db_path=state_dir / "bandar.db",
                         watchlist_path=state_dir / "watchlist.json")
    as_of = anchor_date()

    budget.reset_run()                              # this push = one run (60-cr cap)
    ctx = ToolContext(client=client, memory=memory, budget=budget,
                      as_of=datetime.fromisoformat(as_of).date())
    out = run_tool("rank_watchlist", {}, ctx)
    if not out["ok"]:
        raise TelegramPushError(f"rank_watchlist failed ({out['error']}): "
                                f"{out.get('message', '')}")
    scores = out["data"]["ranked"]
    deltas = memory.record_run(scores)              # upsert + FR6 deltas
    stt = budget.status()
    return {"scores": scores, "deltas": deltas, "as_of": as_of,
            "push_cost": stt["run_spent"], "budget": stt}


# ------------------------------------------------------------------ message

def _fmt_price(p) -> str:
    return f"{p:g}"


def _is_actionable(r: dict) -> bool:
    """A name is actionable only with a disciplined plan (gate-WAIT has none, D3)."""
    return bool(r.get("trade_plan")) and r["decision"]["label"] != "WAIT"


def _why_line(r: dict) -> str:
    gate = (r.get("gates") or {}).get("no_chase", {})
    if gate.get("triggered"):
        return "no-chase gate — extended; WAIT for pullback (never chase, D3)"
    why = (f"{r['score']}/{r['denominator']} confluence · regime "
           f"{(r.get('gates') or {}).get('regime', 'n/a')}")
    if r.get("risk_flags"):
        why += f" · flags: {', '.join(r['risk_flags'])}"
    return why


def build_message(scores: list[dict], deltas: list[dict], budget: dict,
                  as_of: str, push_cost: int) -> str:
    """Pure formatter: engine results -> the §9 push text. No I/O, no LLM."""
    data_date = max((r.get("as_of") or "" for r in scores), default="?")
    dmap = {d["ticker"]: d for d in deltas}
    lines = ["📈 BANDAR — pre-market desk note",
             f"{as_of} (WIB) · data as of {data_date}", ""]

    actionable = [r for r in scores if _is_actionable(r)]
    pick = actionable[0] if actionable else None

    if pick:
        tp = pick["trade_plan"]
        ez = tp["entry_zone"]
        lines += [
            f"🎯 TOP PICK: {pick['symbol']} — {pick['score']}/{pick['denominator']} · "
            f"{pick['decision']['label']} ({pick['decision']['deploy_pct']})",
            f"Entry {_fmt_price(ez[0])}–{_fmt_price(ez[1])} · "
            f"Stop (close) {_fmt_price(tp['stop_close'])} · Lots {tp['lots']}",
            f"Why: {_why_line(pick)}",
        ]
        d = dmap.get(pick["symbol"])
        if d and d.get("has_prev"):
            lines.append(f"Δ since {d['prev_date']}: {d['score_old']}/{d['denominator_old']} "
                         f"{d['decision_old']} → {d['score_new']}/{d['denominator_new']} "
                         f"{d['decision_new']}")
    else:
        best = scores[0] if scores else None
        if best:
            lines.append(f"🎯 No actionable setup today — best score "
                         f"{best['symbol']} {best['score']}/{best['denominator']} "
                         f"({best['decision']['label']}). {_why_line(best)}")
        else:
            lines.append("🎯 No scored names — watchlist empty or data unavailable.")

    if scores:
        lines += ["", "📋 Watchlist:"]
        for r in scores:
            d = dmap.get(r["symbol"], {})
            mark = " ★" if pick and r["symbol"] == pick["symbol"] else ""
            gate = " ⛔ no-chase" if (r.get("gates") or {}).get("no_chase", {}).get("triggered") else ""
            delta = ""
            if d.get("has_prev"):
                delta = f" (was {d['score_old']}/{d['denominator_old']} on {d['prev_date']})"
            lines.append(f"· {r['symbol']} {r['score']}/{r['denominator']} "
                         f"{r['decision']['label']}{delta}{gate}{mark}")

    n_others = max(0, len(actionable) - (1 if pick else 0))
    lines += ["", f"➕ {n_others} others actionable" if n_others
              else "➕ 0 others actionable"]
    lines += ["", f"🧠 Credits: {budget['total_spent']}/{budget['grant']} total · "
                  f"this push {push_cost} cr",
              "Analysis only — Bandar never executes trades (FR9)."]

    text = "\n".join(lines)
    if len(text) > TELEGRAM_MAX_LEN:               # honest truncation, never mid-number
        text = text[:TELEGRAM_MAX_LEN - 14].rstrip() + "\n…(truncated)"
    return text


# ------------------------------------------------------------------ sending

def send_telegram(text: str, bot_token: str, chat_id: str, timeout: float = 30.0) -> dict:
    """Plain requests.post to the Bot API (LOCKED stack: no SDK)."""
    resp = requests.post(TELEGRAM_API.format(token=bot_token),
                         json={"chat_id": chat_id, "text": text,
                               "disable_web_page_preview": True},
                         timeout=timeout)
    if resp.status_code != 200:
        raise TelegramPushError(f"telegram HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise TelegramPushError(f"telegram response not JSON: {exc}") from exc
    if not payload.get("ok"):
        raise TelegramPushError(f"telegram rejected send: {payload!r:.200}")
    return payload


# ------------------------------------------------------------------ entry

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BANDAR morning Telegram push (§9)")
    ap.add_argument("--dry-run", action="store_true",
                    help="score + build + print, but do not send")
    args = ap.parse_args(argv)

    try:
        res = gather()
    except TelegramPushError as exc:
        print(f"[bandar-push] data failure: {exc}")
        return 1

    message = build_message(res["scores"], res["deltas"], res["budget"],
                            res["as_of"], res["push_cost"])
    print(message)

    if args.dry_run:
        print("\n[dry-run] message built, not sent.")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[bandar-push] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing — cannot send.")
        return 2
    try:
        send_telegram(message, token, chat_id)
    except TelegramPushError as exc:
        print(f"[bandar-push] send failed: {exc}")
        return 2
    print("\n[bandar-push] sent ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
