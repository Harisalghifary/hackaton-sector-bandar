"""M1 seed snapshot (§15) — spends REAL credits.

Per watchlist symbol: 4 chunked ohlcv calls (~360 calendar days -> ~240 trading
bars, comfortably 200+ for MA200; note 3 chunks would only yield ~185 bars) +
broker_summary (14d window) + foreign_flow (90d window) = ~6 cr/symbol.
8 names -> ~48 cr, inside the 60-cr run cap.

Idempotent: cache-first mode — re-running only fetches what's missing.

Outputs state/seed_report.json with, per symbol:
- bar count, first/last date, zero-volume days (liquidity sanity)
- price-jump anomalies |daily move| > 35% (possible unadjusted split/CA in window)
- top-3 broker buys + concentration ratio = top3_buy/total_buy  <- raw data for
  the broker_concentration_threshold [TUNE] decision (§15)
- foreign flow: rows + net inflow last 5 sessions

Run: .venv/bin/python seed_snapshot.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from budget import BudgetError, CreditBudget
from client import SectorsClient, SectorsError, SectorsAPIError, SectorsNotFoundError

WIB = ZoneInfo("Asia/Jakarta")
TODAY = None  # set in main()
REPORT_PATH = Path("state") / "seed_report.json"

# 4 chunks of ~89 days each (~360 calendar days total), each within the 90d API cap.
# IMPORTANT: the most recent chunk ends at T-1 — the API rejects end=today with a 400
# before the session exists (learned the hard way, for free: 400 bills 0 cr).
# Chunks 2-4 keep the boundaries from the first run so cached entries are reused.
CHUNK_OFFSETS = [(89, 1), (179, 90), (269, 180), (359, 270)]  # (start_ago, end_ago)

PACE_S = 1.2          # pause before every call — 429s arrived after ~36 rapid requests
RETRY_WAITS = (5, 15, 45)  # 429 backoff (429 is free, so retries are credit-safe)


def paced(fn, *args, **kwargs):
    """Rate-limit-aware call wrapper: pace, and retry 429 with exponential backoff."""
    for attempt in range(len(RETRY_WAITS) + 1):
        time.sleep(PACE_S)
        try:
            return fn(*args, **kwargs)
        except SectorsAPIError as exc:
            if exc.status_code == 429 and attempt < len(RETRY_WAITS):
                wait = RETRY_WAITS[attempt]
                print(f"   429 rate limit — backing off {wait}s "
                      f"(free, retry {attempt + 1}/{len(RETRY_WAITS)})")
                time.sleep(wait)
                continue
            raise


def seed_symbol(c: SectorsClient, sym: str, today) -> dict:
    entry: dict = {"symbol": sym}

    # --- OHLCV: 4 chunks -> merged bar list -------------------------------
    bars: list[dict] = []
    for start_ago, end_ago in CHUNK_OFFSETS:
        start = (today - timedelta(days=start_ago)).isoformat()
        end = (today - timedelta(days=end_ago)).isoformat()
        try:
            rows = paced(c.ohlcv, sym, start, end)
            bars.extend(rows if isinstance(rows, list) else [])
        except SectorsError as exc:
            entry.setdefault("ohlcv_errors", []).append(f"{start}..{end}: {exc!r}")
    bars.sort(key=lambda r: r.get("date", ""))
    entry["bars"] = len(bars)
    entry["ma200_ok"] = len(bars) >= 200
    if bars:
        entry["first_date"] = bars[0].get("date")
        entry["last_date"] = bars[-1].get("date")
        entry["last_close"] = bars[-1].get("close")
        entry["zero_volume_days"] = sum(1 for b in bars if not b.get("volume"))
        jumps = []
        for prev, cur in zip(bars, bars[1:]):
            pc, cc = prev.get("close"), cur.get("close")
            if pc and cc and abs(cc / pc - 1) > 0.35:
                jumps.append({"date": cur.get("date"), "move_pct": round((cc / pc - 1) * 100, 1)})
        entry["price_jump_anomalies"] = jumps  # possible unadjusted split/corporate action

    # --- Broker summary: latest day top-3 buys + concentration -------------
    try:
        bs = paced(c.broker_summary, sym, start=(today - timedelta(days=14)).isoformat(),
                   end=(today - timedelta(days=1)).isoformat())
        days = [d for d in bs.get("data", []) if d.get("summary")]
        if days:
            latest = max(days, key=lambda d: d["date"])
            rows = latest["summary"]
            total_buy = sum(r.get("bval") or 0 for r in rows)
            top3 = sorted(rows, key=lambda r: r.get("bval") or 0, reverse=True)[:3]
            top3_buy = sum(r.get("bval") or 0 for r in top3)
            entry["broksum"] = {
                "date": latest["date"],
                "brokers_active": len(rows),
                "total_buy_idr": total_buy,
                "top3": [{"code": r["broker_code"], "bval_idr": r.get("bval")} for r in top3],
                "concentration": round(top3_buy / total_buy, 4) if total_buy else None,
            }
        else:
            entry["broksum"] = {"date": None, "note": "empty window (200, billed)"}
    except SectorsNotFoundError:
        entry["broksum"] = None  # honest-null F5 candidate (denominator 4) — 1 cr billed
        entry["broksum_404"] = True
    except SectorsError as exc:
        entry["broksum_error"] = repr(exc)

    # --- Foreign flow: 90d window -------------------------------------------
    try:
        ff = paced(c.foreign_flow, sym, start=(today - timedelta(days=90)).isoformat(),
                   end=(today - timedelta(days=1)).isoformat())
        pts = [p for p in ff.get("data", []) if p.get("net_foreign_inflow") is not None]
        entry["flow"] = {
            "rows": len(ff.get("data", [])),
            "net_last5_idr": sum(p["net_foreign_inflow"] for p in pts[-5:]),
        }
    except SectorsNotFoundError:
        entry["flow"] = None
    except SectorsError as exc:
        entry["flow_error"] = repr(exc)

    return entry


def main() -> None:
    global TODAY
    load_dotenv(dotenv_path=".env")
    watchlist = [s.strip().upper() for s in os.environ.get("BANDAR_WATCHLIST", "").split(",") if s.strip()]
    if not watchlist:
        raise SystemExit("BANDAR_WATCHLIST is empty — set it in .env first")

    TODAY = datetime.now(WIB).date()
    budget = CreditBudget()
    budget.reset_run()  # fresh run budget for the seed (60-cr cap)
    c = SectorsClient(mode="cache-first", budget=budget)  # key from SECTORS_API_KEY env

    print(f"Seeding {len(watchlist)} symbols as of {TODAY} (WIB). "
          f"Budget before: {budget.status()['total_spent']} cr spent total.\n")

    report = {"as_of": TODAY.isoformat(), "symbols": []}
    try:
        for sym in watchlist:
            entry = seed_symbol(c, sym, TODAY)
            report["symbols"].append(entry)
            conc = (entry.get("broksum") or {}).get("concentration")
            print(f"{sym:5s} bars={entry['bars']:3d} ma200_ok={entry['ma200_ok']} "
                  f"close={entry.get('last_close')} conc={conc} "
                  f"spent_so_far={budget.status()['total_spent']} cr")
    except BudgetError as exc:
        # Hard cap hit mid-seed — narrate honestly, persist partial report (§8 policy 4).
        print(f"\nBUDGET ABORT: {exc!r} — partial report kept.")
        report["budget_abort"] = repr(exc)

    # --- [TUNE] broker_concentration_threshold distribution ----------------
    concs = [e["broksum"]["concentration"] for e in report["symbols"]
             if e.get("broksum") and e["broksum"].get("concentration") is not None]
    if concs:
        report["concentration_stats"] = {
            "n": len(concs), "min": min(concs), "median": statistics.median(concs),
            "max": max(concs), "values": sorted(concs),
        }
        print(f"\nconcentration distribution (top3_buy/total_buy, latest day): "
              f"min={min(concs):.3f} median={statistics.median(concs):.3f} max={max(concs):.3f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    st = budget.status()
    print(f"\nReport -> {REPORT_PATH}")
    print(f"Budget: run={st['run_spent']} daily={st['daily_spent']} "
          f"total={st['total_spent']} (remaining spendable {st['remaining_total']})")


if __name__ == "__main__":
    main()
