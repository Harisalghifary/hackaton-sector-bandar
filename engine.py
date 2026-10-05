"""BANDAR scoring engine (§6) — deterministic 5-factor confluence, NO LLM. [LOCKED]

The LLM never computes a number (§1): every figure here comes from pandas math on
Sectors data, and every output number traces to a tool result (§14/FR10).

Factor contracts, aggregation, no-chase gate, overlays, decision matrix and sizing
follow §6 exactly. Where §6 leaves a detail open (entry-zone width, resistance
proximity, low-liquidity threshold, masked-broker handling), the choice is a named
constant below, documented, and unit-tested — no hidden magic numbers.

Indicators use pandas-ta (§3 LOCKED stack) with parity-tested formulas:
EMA = ewm(span, adjust=False) · ATR14 = Wilder RMA · AD = cumsum(MFM*V) (§6 formula).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_ta as ta

from client import SectorsClient, SectorsError, SectorsNotFoundError

WIB = ZoneInfo("Asia/Jakarta")

# ------------------------------------------------------------------ config (§6 LOCKED)

ENGINE_CONFIG = {
    "max_extension_atr": 0.5,                 # LOCKED (D3)
    "ara_limit_pct": 0.10,                    # proxy; verify per price tier if time allows
    "broker_concentration_threshold": 0.6,    # [TUNE] validated on seed snapshot (p75 ~ 0.61)
    "min_volume_ratio": 1.5,
    "min_bars_to_score": 60,
    "risk_per_trade_pct": 0.5,
}

# Engine choices where §6 is silent — named, documented, unit-tested (M2 review notes).
CLIMAX_VOLUME_RATIO = 2.5        # §6 F2 flag: climax_volume (>= 2.5)
FRESH_CROSS_MAX_AGE = 3          # §6 F1 flag: fresh_cross (<= 3 bars)
AD_SLOPE_WINDOW = 10             # §6 F3: slope = linreg last 10
DIVERGENCE_HALF_WINDOW = 5       # F3 bearish_divergence: compare last 5 bars vs prior 5
MIN_BROKERS_ACTIVE = 5           # fewer active brokers => data considered masked (MDKA '--' case)
MASKED_BROKER_CODE = "--"        # IDX masks broker details on certain days (FCA/UMA review)
THIN_TURNOVER_IDR = 5e9          # F2 low_liquidity: SMA20(volume*close) below this (IDR/day)
FOREIGN_DIVERGENCE_IDR = 500e9   # §6 F5 flag: net foreign SELL > 500B over last 5 sessions
F5_SESSIONS = 5                  # aggregate last 5 valid broksum sessions (robust vs 1-day noise)
RESISTANCE_LOOKBACK = 60         # "climax at resistance": close >= 97% of 60-bar high
RESISTANCE_PROXIMITY = 0.97
ENTRY_ZONE_ATR_FRAC = 0.25       # entry zone: [close, close + 0.25*ATR14], tick-rounded
STOP_LOOKBACK = 10               # stop = min of last 10 CLOSES (close-based only, FR4)
STOP_LOOKBACK_TIGHT = 5          # tightened to last 5 closes under distribution overlay
STOP_ATR_FALLBACK = 1.5          # degenerate stop >= entry: stop = entry - 1.5*ATR14

# OHLCV chunk windows (<= 90d API cap each), anchored on the analysis date; identical
# to seed_snapshot.py so cache keys match and re-scoring costs 0 cr.
CHUNK_OFFSETS = ((89, 1), (179, 90), (269, 180), (359, 270))  # (start_ago, end_ago)
BROKSUM_WINDOW_DAYS = 14
FLOW_WINDOW_DAYS = 90

# Flags that count as "flags" for the decision matrix row "4/5+flags -> DEFENSIVE 40-60%".
# fresh_cross / MA200_bull are informational-bullish and deliberately excluded.
WARN_FLAGS = frozenset({
    "momentum_fading", "climax_volume", "low_liquidity",
    "bearish_divergence", "foreign_divergence", "distribution",
})

# Decision tiers (ladder for "downgrade one tier" overlay). FR13: matrix exact.
TIER_AGGRESSIVE, TIER_DEF_HIGH, TIER_DEF_LOW, TIER_SCALP, TIER_WAIT = 4, 3, 2, 1, 0
TIER_META = {
    TIER_AGGRESSIVE: ("AGGRESSIVE", "60-80%"),
    TIER_DEF_HIGH: ("DEFENSIVE", "40-60%"),
    TIER_DEF_LOW: ("DEFENSIVE", "30-40%"),
    TIER_SCALP: ("SCALP", "15-20%"),
    TIER_WAIT: ("WAIT", "0%"),
}

DEFAULT_EQUITY_IDR = 100_000_000  # §4


# ------------------------------------------------------------------ ticks (§6, FR5)

def tick_size(price: float) -> int:
    """IDX tick table per §6 (>5,000->25 · 2,000-5,000->10 · 500-2,000->5 ·
    200-500->2 · 50-200->1). Below 50: tick 1 (defensive; verify current table §15)."""
    p = float(price)
    if p > 5000:
        return 25
    if p >= 2000:
        return 10
    if p >= 500:
        return 5
    if p >= 200:
        return 2
    return 1


def round_to_tick(price: float, mode: str = "nearest") -> int:
    """Round a price to its tick band. mode: nearest | down (stops) | up."""
    p = float(price)
    t = tick_size(p)
    if mode == "down":
        rounded = int(math.floor(p / t) * t)
    elif mode == "up":
        rounded = int(math.ceil(p / t) * t)
    else:
        rounded = int(round(p / t) * t)
    # Re-check band after rounding (e.g. 1999 -> 2000 changes band); 2000 % 10 == 0 holds.
    t2 = tick_size(rounded)
    if rounded % t2 != 0:
        rounded = int(math.floor(rounded / t2) * t2) if mode == "down" else int(round(rounded / t2) * t2)
    return rounded


def tick_valid(price: float) -> bool:
    p = float(price)
    return p > 0 and p % tick_size(p) == 0


# ------------------------------------------------------------------ indicators

def _ema(s: pd.Series, span: int) -> pd.Series:
    return ta.ema(s, length=span)


def _sma(s: pd.Series, n: int) -> pd.Series:
    return ta.sma(s, length=n)


def macd_lines(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """§6 F1: MACD = EMA12-EMA26; Signal = EMA9(MACD); Hist = MACD-Signal."""
    m = ta.macd(closes, fast=12, slow=26, signal=9)
    macd = m[f"MACD_12_26_9"]
    signal = m[f"MACDs_12_26_9"]
    hist = m[f"MACDh_12_26_9"]
    return macd, signal, hist


def atr14(df: pd.DataFrame) -> pd.Series:
    """Wilder ATR14 over high/low/close."""
    return ta.atr(df["high"], df["low"], df["close"], length=14)


# ------------------------------------------------------------------ factors (§6 table)

def f1_macd(df: pd.DataFrame) -> dict | None:
    """Point=1 when MACD>Signal, Hist>=0, Hist rising. Min 35 bars."""
    if len(df) < 35:
        return None
    closes = df["close"]
    macd, signal, hist = macd_lines(closes)
    h, h_prev = float(hist.iat[-1]), float(hist.iat[-2])
    m, s = float(macd.iat[-1]), float(signal.iat[-1])
    if math.isnan(h) or math.isnan(h_prev):
        return None
    passed = (m > s) and (h >= 0) and (h > h_prev)

    flags = []
    diff = macd - signal
    cross_age = None
    for age in range(0, min(FRESH_CROSS_MAX_AGE, len(diff) - 2) + 1):
        i = len(diff) - 1 - age
        if float(diff.iat[i]) > 0 and float(diff.iat[i - 1]) <= 0:
            cross_age = age
            break
    if cross_age is not None:
        flags.append("fresh_cross")
    if m > s and h < h_prev:
        flags.append("momentum_fading")

    return {
        "pass": bool(passed),
        "values": {"macd": m, "signal": s, "hist": h, "hist_prev": h_prev,
                   "hist_rising": bool(h > h_prev),
                   "cross_age_bars": cross_age},
        "flags": flags,
    }


def f2_volume(df: pd.DataFrame) -> dict | None:
    """vol_ratio = V / SMA20(V); point when >= 1.5. Min 20 bars (need SMA20 + current)."""
    if len(df) < 21:
        return None
    vol = df["volume"]
    sma20 = _sma(vol, 20)
    v, avg = float(vol.iat[-1]), float(sma20.iat[-1])
    if math.isnan(avg) or avg <= 0:
        ratio = None
        passed = False
    else:
        ratio = v / avg
        passed = ratio >= ENGINE_CONFIG["min_volume_ratio"]

    flags = []
    if ratio is not None and ratio >= CLIMAX_VOLUME_RATIO:
        flags.append("climax_volume")
    turnover20 = float(_sma(df["volume"] * df["close"], 20).iat[-1])
    if not math.isnan(turnover20) and turnover20 < THIN_TURNOVER_IDR:
        flags.append("low_liquidity")

    return {
        "pass": bool(passed),
        "values": {"volume": v, "sma20_volume": avg, "vol_ratio": ratio},
        "flags": flags,
    }


def f3_ad(df: pd.DataFrame) -> dict | None:
    """MFM=((C-L)-(H-C))/(H-L), 0 if H=L; AD=cumsum(MFM*V); slope=linreg last 10.
    Point when slope>0. Min 11 bars."""
    if len(df) < 11:
        return None
    hl = df["high"] - df["low"]
    mfm = pd.Series(0.0, index=df.index)
    nz = hl > 0
    mfm[nz] = ((df["close"][nz] - df["low"][nz]) - (df["high"][nz] - df["close"][nz])) / hl[nz]
    ad = (mfm * df["volume"]).cumsum()

    win = ad.iloc[-AD_SLOPE_WINDOW:]
    x = np.arange(len(win), dtype=float)
    slope = float(np.polyfit(x, win.to_numpy(dtype=float), 1)[0])
    passed = slope > 0

    flags = []
    if len(df) >= 2 * DIVERGENCE_HALF_WINDOW:
        k = DIVERGENCE_HALF_WINDOW
        price_hh = float(df["close"].iloc[-k:].max()) > float(df["close"].iloc[-2 * k:-k].max())
        ad_lh = float(ad.iloc[-k:].max()) < float(ad.iloc[-2 * k:-k].max())
        if price_hh and ad_lh:
            flags.append("bearish_divergence")

    return {
        "pass": bool(passed),
        "values": {"ad": float(ad.iat[-1]), "slope10": slope, "mfm": float(mfm.iat[-1])},
        "flags": flags,
    }


def f4_ma_stack(df: pd.DataFrame) -> dict | None:
    """stack = (C>MA9)+(MA9>MA26)+(MA26>MA50)+(MA50>MA200); point when stack>=3.
    60-199 bars: MA200 null, 3-term stack, point if >=2, regime neutral."""
    if len(df) < 50:
        return None
    closes = df["close"]
    c = float(closes.iat[-1])
    ma9 = float(_ema(closes, 9).iat[-1])
    ma26 = float(_ema(closes, 26).iat[-1])
    ma50 = float(_sma(closes, 50).iat[-1])
    ma200_series = _sma(closes, 200)
    ma200 = float(ma200_series.iat[-1]) if len(df) >= 200 and not math.isnan(float(ma200_series.iat[-1])) else None

    terms = [c > ma9, ma9 > ma26, ma26 > ma50]
    if ma200 is not None:
        terms.append(ma50 > ma200)
        stack = int(sum(terms))
        passed = stack >= 3
        regime = "bullish_above_ma200" if c >= ma200 else "bearish_below_ma200"
    else:
        stack = int(sum(terms))
        passed = stack >= 2          # §6: 3-term stack, point if >= 2
        regime = "neutral_insufficient"

    flags = ["MA200_bull"] if (ma200 is not None and c > ma200) else []
    return {
        "pass": bool(passed),
        "values": {"close": c, "ma9": ma9, "ma26": ma26, "ma50": ma50,
                   "ma200": ma200, "stack": stack, "stack_terms": len(terms)},
        "flags": flags,
        "regime": regime,
    }


def _valid_broksum_days(broksum: dict) -> list[dict]:
    """Days whose broker rows are unmasked: >= MIN_BROKERS_ACTIVE and no '--' code."""
    days = [d for d in broksum.get("data", []) if d.get("summary")]
    return [d for d in days
            if len(d["summary"]) >= MIN_BROKERS_ACTIVE
            and not any(r.get("broker_code") == MASKED_BROKER_CODE for r in d["summary"])]


def foreign_divergence_check(flow: dict | None) -> tuple[bool, int | None]:
    """§6 F5 flag: net foreign SELL > 500B over the last 5 flow sessions."""
    if not flow:
        return False, None
    pts = [p for p in flow.get("data", []) if p.get("net_foreign_inflow") is not None]
    if not pts:
        return False, None
    pts.sort(key=lambda p: p["date"])
    net5 = int(sum(p["net_foreign_inflow"] for p in pts[-5:]))
    return net5 < -FOREIGN_DIVERGENCE_IDR, net5


def f5_broker(broksum: dict | None, flow: dict | None, close: float) -> tuple[dict | None, list[str]]:
    """Point when net_foreign>0 AND concentration>=thr AND avg_buy<=C.
    Missing/masked data -> None (honest-null, denominator 4, never fabricated — FR2)."""
    vol_flags: list[str] = []
    div_flag, net5 = foreign_divergence_check(flow)

    if broksum is None:
        return None, vol_flags

    all_days = [d for d in broksum.get("data", []) if d.get("summary")]
    valid = _valid_broksum_days(broksum)
    if all_days and not any(d["date"] == max(x["date"] for x in all_days) for d in valid):
        vol_flags.append("broker_data_masked")   # latest session masked (IDX FCA/UMA-style)

    use = sorted(valid, key=lambda d: d["date"])[-F5_SESSIONS:]
    if not use:
        return None, vol_flags

    per_broker: dict[str, int] = {}
    total_buy = 0
    total_lots = 0
    for d in use:
        for r in d["summary"]:
            per_broker[r["broker_code"]] = per_broker.get(r["broker_code"], 0) + (r.get("bval") or 0)
            total_buy += r.get("bval") or 0
            total_lots += r.get("blot") or 0
    if total_buy <= 0 or total_lots <= 0:
        return None, vol_flags

    top3 = sorted(per_broker.values(), reverse=True)[:3]
    concentration = sum(top3) / total_buy
    avg_buy = total_buy / (total_lots * 100)   # blot is in lots (1 lot = 100 shares)

    session_dates = {d["date"] for d in use}
    net_foreign = None
    if flow:
        pts = [p for p in flow.get("data", [])
               if p.get("date") in session_dates and p.get("net_foreign_inflow") is not None]
        if pts:
            net_foreign = int(sum(p["net_foreign_inflow"] for p in pts))
    if net_foreign is None:
        # No flow data for the sessions used -> honest null, never fabricated (FR2).
        return None, vol_flags

    passed = (net_foreign > 0
              and concentration >= ENGINE_CONFIG["broker_concentration_threshold"]
              and avg_buy <= close)

    flags = ["foreign_divergence"] if div_flag else []
    return {
        "pass": bool(passed),
        "values": {"net_foreign": net_foreign, "concentration": concentration,
                   "avg_buy": avg_buy, "sessions_used": len(use),
                   "net5_all_sessions": net5},
        "flags": flags,
    }, vol_flags


# ------------------------------------------------------------------ gates (§6, D3)

def no_chase_gate(df: pd.DataFrame) -> dict:
    """LOCKED (D3):
        extended = close_D > high_{D-1} + max_extension_atr * ATR14_D
        ara_day  = close_D >= close_{D-1} * (1 + ara_limit_pct)
        gate_triggered = extended or ara_day  -> forced WAIT-pullback, NEVER actionable.
    """
    out = {"no_chase": {"triggered": False, "extension_atr": None,
                        "threshold_atr": ENGINE_CONFIG["max_extension_atr"],
                        "extended": False},
           "ara": {"triggered": False, "limit_pct": ENGINE_CONFIG["ara_limit_pct"],
                   "move_pct": None}}
    if len(df) < 2:
        return out
    close_d = float(df["close"].iat[-1])
    close_prev = float(df["close"].iat[-2])
    high_prev = float(df["high"].iat[-2])

    ara = close_d >= close_prev * (1 + ENGINE_CONFIG["ara_limit_pct"])
    out["ara"]["triggered"] = bool(ara)
    out["ara"]["move_pct"] = (close_d / close_prev - 1) if close_prev else None

    atr_s = atr14(df)
    atr_d = float(atr_s.iat[-1]) if not math.isnan(float(atr_s.iat[-1])) else None
    extended = False
    if atr_d and atr_d > 0:
        ext_atr = (close_d - high_prev) / atr_d
        out["no_chase"]["extension_atr"] = ext_atr
        extended = close_d > high_prev + ENGINE_CONFIG["max_extension_atr"] * atr_d
    out["no_chase"]["extended"] = bool(extended)
    out["no_chase"]["triggered"] = bool(extended or ara)
    return out


# ------------------------------------------------------------------ decision (§6 matrix, FR13)

def compute_bias(score: int, denom: int) -> str:
    """§6: BULLISH if bull>=3 and bull>bear; BEARISH if bear>=3; else NEUTRAL."""
    bull, bear = score, denom - score
    if bull >= 3 and bull > bear:
        return "BULLISH"
    if bear >= 3:
        return "BEARISH"
    return "NEUTRAL"


def decide(score: int, denom: int, bias: str, warn_flags: set[str],
           below_ma200: bool, gate_override: str | None = None) -> dict:
    """Decision matrix, exact per §6 (FR13), rows evaluated specific-over-general:
      4/4 or 4-5/5 BULLISH            -> AGGRESSIVE 60-80%
      3/4 or 4/5+flags                -> DEFENSIVE 40-60%
      2/4 or 3/5                      -> DEFENSIVE 30-40%
      2 NEUTRAL                       -> SCALP 15-20%
      <=1 BEARISH                     -> WAIT/CASH 0%
    Documented resolutions of matrix gaps/overlaps (M2 review notes):
      - score 2 NEUTRAL matches both row 3 and row 4; the more specific row 4 wins.
      - score 2 with denom 5 (bias BEARISH) matches no row literally; mapped to
        DEFENSIVE 30-40% (row 3 generalized to 'score 2, not NEUTRAL').
      - score <= 1 always implies bear >= 3 => BEARISH, so the bias qualifier is
        redundant and WAIT/CASH applies.
    Overlays (change decision, never score), in order:
      - no-chase gate: forced WAIT-pullback, NEVER actionable even at 4/4 (D3).
      - regime close < MA200: cap at DEFENSIVE (40-60%).
      - foreign_divergence: downgrade one tier.
    """
    if gate_override == "no_chase":
        label, deploy = TIER_META[TIER_WAIT]
        return {"label": label, "deploy_pct": deploy, "bias": bias,
                "tier": TIER_WAIT, "gate_override": "no_chase"}

    if score <= 1:
        tier = TIER_WAIT
    elif score == 2:
        tier = TIER_SCALP if bias == "NEUTRAL" else TIER_DEF_LOW
    elif score == 3:
        tier = TIER_DEF_HIGH if denom == 4 else TIER_DEF_LOW
    elif score == 4:
        if denom == 4:
            tier = TIER_AGGRESSIVE
        else:
            tier = TIER_DEF_HIGH if warn_flags else (
                TIER_AGGRESSIVE if bias == "BULLISH" else TIER_DEF_LOW)
    else:  # score == 5 (denom 5)
        tier = TIER_AGGRESSIVE if bias == "BULLISH" else TIER_DEF_HIGH

    if below_ma200:                       # regime cap: never above DEFENSIVE 40-60%
        tier = min(tier, TIER_DEF_HIGH)
    if "foreign_divergence" in warn_flags:  # downgrade one tier
        tier = max(tier - 1, TIER_WAIT)

    label, deploy = TIER_META[tier]
    return {"label": label, "deploy_pct": deploy, "bias": bias, "tier": tier}


# ------------------------------------------------------------------ sizing (§6, FR4/FR5)

def build_trade_plan(df: pd.DataFrame, tier: int, equity_idr: int,
                     distribution: bool) -> dict | None:
    """Entry zone / close-based stop / 0.5%-risk lots, tick-rounded (FR4, FR5).

    - entry_zone = [tick(close), tick_up(close + 0.25*ATR14)]
    - stop_close = tick_down(min(last 10 closes)); tightened to last 5 closes under
      the distribution overlay; ATR fallback if degenerate. CLOSE levels only (FR4).
    - lots = floor((equity * risk%) / ((entry_ref - stop) * 100)), entry_ref = zone mid.
    """
    if tier <= TIER_WAIT or len(df) < 15:
        return None
    close = float(df["close"].iat[-1])
    atr_s = atr14(df)
    atr_d = float(atr_s.iat[-1])
    if math.isnan(atr_d) or atr_d <= 0:
        return None

    entry_low = round_to_tick(close, "nearest")
    entry_high = round_to_tick(close + ENTRY_ZONE_ATR_FRAC * atr_d, "up")
    if entry_high <= entry_low:
        entry_high = round_to_tick(entry_low + tick_size(entry_low), "up")

    n = STOP_LOOKBACK_TIGHT if distribution else STOP_LOOKBACK
    stop = round_to_tick(float(df["close"].iloc[-n:].min()), "down")
    if stop >= entry_low:
        stop = round_to_tick(entry_low - STOP_ATR_FALLBACK * atr_d, "down")
    if stop >= entry_low:
        return None  # no disciplined plan exists -> caller keeps WAIT semantics

    entry_ref = (entry_low + entry_high) / 2
    risk_idr = equity_idr * ENGINE_CONFIG["risk_per_trade_pct"] / 100
    per_lot_risk = (entry_ref - stop) * 100
    lots = int(risk_idr // per_lot_risk) if per_lot_risk > 0 else 0

    return {
        "entry_zone": [entry_low, entry_high],
        "stop_close": stop,
        "lots": lots,
        "tick_valid": all(tick_valid(p) for p in (entry_low, entry_high, stop)),
    }


# ------------------------------------------------------------------ assembly

def _clean(obj: Any) -> Any:
    """JSON-safe: numpy -> python, NaN/inf -> None, floats rounded (determinism)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    return obj


def score_frame(df: pd.DataFrame, broksum: dict | None, flow: dict | None,
                symbol: str, equity_idr: int = DEFAULT_EQUITY_IDR,
                as_of: str | None = None) -> dict:
    """Pure scoring core: bars + broker/flow payloads -> §8 score_ticker contract.
    No I/O, no LLM, fully deterministic."""
    n = len(df)
    as_of = as_of or (str(df["date"].iat[-1]) if n else None)

    if n < ENGINE_CONFIG["min_bars_to_score"]:
        return _clean({
            "symbol": symbol, "as_of": as_of, "status": "insufficient_data",
            "bars": n, "score": None, "denominator": None,
            "factors": None, "gates": {"regime": "neutral_insufficient"},
            "decision": {"label": "WAIT", "deploy_pct": "0%", "bias": None,
                         "tier": TIER_WAIT, "reason": "insufficient_data"},
            "trade_plan": None, "trace": [],
        })

    f1 = f1_macd(df)
    f2 = f2_volume(df)
    f3 = f3_ad(df)
    f4 = f4_ma_stack(df)
    close = float(df["close"].iat[-1])

    core = sum(int(f["pass"]) for f in (f1, f2, f3, f4) if f is not None)
    core_denom = sum(1 for f in (f1, f2, f3, f4) if f is not None)

    f5, vol_flags = f5_broker(broksum, flow, close)
    if f5 is not None:
        score, denom = core + int(f5["pass"]), core_denom + 1
    else:
        score, denom = core, core_denom   # honest null -> denominator 4, never fabricated

    div_flag, net5 = foreign_divergence_check(flow)
    gates = no_chase_gate(df)
    regime = f4["regime"] if f4 else "neutral_insufficient"
    below_ma200 = regime == "bearish_below_ma200"

    # distribution overlay: climax_volume at resistance (§6 overlay gates)
    distribution = False
    if f2 and "climax_volume" in f2["flags"]:
        prior_highs = df["high"].iloc[-(RESISTANCE_LOOKBACK + 1):-1]
        if len(prior_highs) and close >= RESISTANCE_PROXIMITY * float(prior_highs.max()):
            distribution = True
    gates["regime"] = regime
    gates["distribution"] = {"triggered": bool(distribution)}
    gates["volatility_flags"] = sorted(set(vol_flags))
    gates["foreign_flow"] = {"net5_idr": net5, "divergence": bool(div_flag)}

    flags: set[str] = set()
    for f in (f1, f2, f3, f4, f5):
        if f:
            flags.update(f["flags"])
    if distribution:
        flags.add("distribution")
    if div_flag and flow is not None:
        flags.add("foreign_divergence")
    warn_flags = flags & WARN_FLAGS

    bias = compute_bias(score, denom)
    gate_override = "no_chase" if gates["no_chase"]["triggered"] else None
    decision = decide(score, denom, bias, warn_flags, below_ma200, gate_override)

    plan = None
    if decision["tier"] > TIER_WAIT:
        plan = build_trade_plan(df, decision["tier"], equity_idr, distribution)
        if plan is None:
            decision = {"label": "WAIT", "deploy_pct": "0%", "bias": bias,
                        "tier": TIER_WAIT, "reason": "no_disciplined_plan"}

    factors = {"f1_macd": f1, "f2_volume": f2, "f3_ad": f3,
               "f4_ma_stack": f4, "f5_broker": f5}

    return _clean({
        "symbol": symbol, "as_of": as_of, "status": "ok", "bars": n,
        "score": score, "denominator": denom,
        "factors": factors, "gates": gates, "decision": decision,
        "trade_plan": plan, "trace": [],
        "risk_flags": sorted(warn_flags),
    })


# ------------------------------------------------------------------ data access + API

def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_history(client: SectorsClient, symbol: str, as_of) -> pd.DataFrame:
    """Merge the 4 cached chunk windows (~360 days -> ~240 bars, 200+ for MA200)."""
    rows: list[dict] = []
    for start_ago, end_ago in CHUNK_OFFSETS:
        start = (as_of - timedelta(days=start_ago)).isoformat()
        end = (as_of - timedelta(days=end_ago)).isoformat()
        try:
            chunk = client.ohlcv(symbol, start, end)
            if isinstance(chunk, list):
                rows.extend(chunk)
        except SectorsError:
            continue  # honest gaps: fewer bars, never fabricated
    return _to_frame(rows)


def score_ticker(symbol: str, client: SectorsClient | None = None,
                 as_of=None, equity_idr: int | None = None) -> dict:
    """Full pipeline: cache-first data pull -> deterministic score (§8 contract).

    as_of: analysis anchor date (default today WIB). Windows end at as_of-1 because
    the API rejects end=today before the session exists. Cache keys match
    seed_snapshot.py, so re-scoring a seeded day costs 0 credits.
    """
    client = client or SectorsClient(mode="cache-first")
    if as_of is None:
        as_of = datetime.now(WIB).date()
    elif isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of).date()
    if equity_idr is None:
        equity_idr = int(os.environ.get("EQUITY_IDR", str(DEFAULT_EQUITY_IDR)))

    trace_start = len(client.trace_log)
    df = fetch_history(client, symbol, as_of)

    broksum = None
    try:
        broksum = client.broker_summary(
            symbol, start=(as_of - timedelta(days=BROKSUM_WINDOW_DAYS)).isoformat(),
            end=(as_of - timedelta(days=1)).isoformat())
    except SectorsNotFoundError:
        broksum = None            # honest-null F5 (FR2); trace shows the 404
    except SectorsError:
        broksum = None

    flow = None
    try:
        flow = client.foreign_flow(
            symbol, start=(as_of - timedelta(days=FLOW_WINDOW_DAYS)).isoformat(),
            end=(as_of - timedelta(days=1)).isoformat())
    except SectorsError:
        flow = None

    data_as_of = str(df["date"].iat[-1]) if len(df) else as_of.isoformat()
    result = score_frame(df, broksum, flow, symbol=symbol.upper().replace(".JK", ""),
                         equity_idr=equity_idr, as_of=data_as_of)
    result["trace"] = client.trace_log[trace_start:]
    return result


def score_watchlist(client: SectorsClient | None = None, as_of=None,
                    equity_idr: int | None = None) -> list[dict]:
    """Score BANDAR_WATCHLIST (env) — engine side of the M4 rank_watchlist tool."""
    syms = [s.strip().upper() for s in os.environ.get("BANDAR_WATCHLIST", "").split(",") if s.strip()]
    return [score_ticker(s, client=client, as_of=as_of, equity_idr=equity_idr) for s in syms]
