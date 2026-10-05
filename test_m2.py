"""M2 scoring-engine tests (§6 + §10 prompt + FR3/FR4/FR5/FR13).

Spec-mandated proofs:
- no-chase gate: assert 4/4 + extended => WAIT (FR3, D3)
- reproducible scores on cached data (0 credits — offline mode + socket blocker)

Run: python -m pytest test_m2.py -v
"""

from __future__ import annotations

import json
import math
import socket
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import engine
from budget import CreditBudget
from client import SectorsClient
from engine import (
    ENGINE_CONFIG,
    TIER_AGGRESSIVE,
    TIER_WAIT,
    build_trade_plan,
    compute_bias,
    decide,
    f1_macd,
    f2_volume,
    f3_ad,
    no_chase_gate,
    round_to_tick,
    score_frame,
    score_ticker,
    tick_size,
    tick_valid,
)

REPO = Path(__file__).resolve().parent
CACHE_DIR = REPO / "cache"
SEED_AS_OF = date(2026, 10, 6)      # anchor used by seed_snapshot.py
WATCHLIST = ["BBRI", "BBCA", "BMRI", "ANTM", "DSSA", "PTBA", "MDKA", "ICBP"]


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """0 credits, guaranteed: any socket use fails the suite."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access during M2 tests — must be 0 live calls")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.delenv("SECTORS_API_KEY", raising=False)


@pytest.fixture()
def budget(tmp_path):
    return CreditBudget(state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl")


@pytest.fixture()
def offline_client(budget):
    """Offline + real seed cache: any miss raises instead of spending."""
    return SectorsClient(api_key="TESTKEY", mode="offline", budget=budget,
                         cache_dir=CACHE_DIR)


def make_df(closes, volumes=None, hi=1.004, lo=0.988, end="2026-10-05"):
    n = len(closes)
    closes = [float(c) for c in closes]
    dates = pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "symbol": ["TEST"] * n, "date": dates,
        "open": closes, "high": [c * hi for c in closes],
        "low": [c * lo for c in closes], "close": closes,
        "volume": list(volumes) if volumes is not None else [5_000_000] * n,
    })


def uptrend(n=120):
    """Accelerating uptrend: F1-F4 all pass on 120 bars (3-term F4)."""
    closes = [1000 + 5 * i + 0.02 * i * i for i in range(n)]
    vols = [5_000_000] * (n - 1) + [8_000_000]        # last-bar ratio ~1.55
    return make_df(closes, vols)


def broksum(days):
    """days: list of (date, [(code, bval, blot), ...]) -> API-shaped payload."""
    return {"symbol": "TEST.JK", "start": days[0][0] if days else None,
            "end": days[-1][0] if days else None,
            "data": [{"date": d,
                      "summary": [{"broker_code": c, "bval": v, "blot": l} for c, v, l in rows]}
                     for d, rows in days]}


BULL_ROWS = [("AA", 60e9, 600_000), ("BB", 30e9, 300_000), ("CC", 20e9, 200_000),
             ("DD", 5e9, 50_000), ("EE", 5e9, 50_000)]
VALID_DATES = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
MASKED_DAY = ("2026-10-05", [("--", 1e9, 10_000)])


def flow(dates, per_day_idr):
    return {"symbol": "TEST.JK", "data": [
        {"date": d, "net_foreign_inflow": per_day_idr,
         "foreign_buy_idr": 0, "foreign_sell_idr": 0, "foreign_share": 0.5}
        for d in dates]}


# ------------------------------------------------- §10 required: no-chase gate


def test_no_chase_44_extended_forces_wait():
    """THE spec test (FR3/D3): 4/4 + extended => WAIT-pullback, NEVER actionable."""
    df = uptrend()
    base = score_frame(df, None, None, symbol="TEST")
    assert base["status"] == "ok"
    assert base["score"] == 4 and base["denominator"] == 4      # genuine 4/4
    assert base["gates"]["no_chase"]["triggered"] is False
    assert base["decision"]["label"] == "AGGRESSIVE"             # actionable without gate

    ext = uptrend()
    ext.loc[ext.index[-1], "close"] = float(ext["close"].iat[-2]) * 1.06   # +6%: extended, not ARA
    r = score_frame(ext, None, None, symbol="TEST")
    assert r["score"] == 4 and r["denominator"] == 4             # gate changes decision, NEVER score
    assert r["gates"]["no_chase"]["extended"] is True
    assert r["gates"]["ara"]["triggered"] is False
    assert r["gates"]["no_chase"]["triggered"] is True
    assert r["decision"]["label"] == "WAIT"
    assert r["decision"]["deploy_pct"] == "0%"
    assert r["decision"]["gate_override"] == "no_chase"
    assert r["trade_plan"] is None                               # never actionable


def test_no_chase_ara_hard_block():
    df = uptrend()
    df.loc[df.index[-1], "close"] = float(df["close"].iat[-2]) * 1.105   # >= +10% ARA proxy
    r = score_frame(df, None, None, symbol="TEST")
    assert r["gates"]["ara"]["triggered"] is True
    assert r["gates"]["no_chase"]["triggered"] is True
    assert r["decision"]["label"] == "WAIT"


def test_no_chase_extension_math():
    """extended = close_D > high_{D-1} + 0.5 * ATR14_D — boundary behavior."""
    df = uptrend()
    g = no_chase_gate(df)
    assert g["no_chase"]["threshold_atr"] == ENGINE_CONFIG["max_extension_atr"] == 0.5
    assert g["no_chase"]["extension_atr"] < 0.5 and not g["no_chase"]["extended"]

    atr_last = float(engine.atr14(df).iat[-1])
    df2 = uptrend()
    # place close exactly at prev high + 0.5*ATR -> NOT extended (strict >)
    prev_high = float(df2["high"].iat[-2])
    df2.loc[df2.index[-1], "close"] = prev_high + 0.5 * atr_last
    assert no_chase_gate(df2)["no_chase"]["extended"] is False
    df2.loc[df2.index[-1], "close"] = prev_high + 0.51 * atr_last + 1
    assert no_chase_gate(df2)["no_chase"]["extended"] is True


# ------------------------------------------------- FR13: decision matrix exact


@pytest.mark.parametrize("score,denom,bias,flags,below,override,label,deploy", [
    (4, 4, "BULLISH", set(), False, None, "AGGRESSIVE", "60-80%"),
    (5, 5, "BULLISH", set(), False, None, "AGGRESSIVE", "60-80%"),
    (4, 5, "BULLISH", set(), False, None, "AGGRESSIVE", "60-80%"),
    (4, 5, "BULLISH", {"climax_volume"}, False, None, "DEFENSIVE", "40-60%"),
    (3, 4, "BULLISH", set(), False, None, "DEFENSIVE", "40-60%"),
    (3, 5, "BULLISH", set(), False, None, "DEFENSIVE", "30-40%"),
    (2, 4, "NEUTRAL", set(), False, None, "SCALP", "15-20%"),
    (2, 5, "BEARISH", set(), False, None, "DEFENSIVE", "30-40%"),
    (1, 4, "BEARISH", set(), False, None, "WAIT", "0%"),
    (0, 5, "BEARISH", set(), False, None, "WAIT", "0%"),
    # overlays: regime cap, foreign_divergence downgrade, no-chase override
    (4, 4, "BULLISH", set(), True, None, "DEFENSIVE", "40-60%"),
    (4, 4, "BULLISH", {"foreign_divergence"}, False, None, "DEFENSIVE", "40-60%"),
    (3, 4, "BULLISH", {"foreign_divergence"}, False, None, "DEFENSIVE", "30-40%"),
    (2, 4, "NEUTRAL", {"foreign_divergence"}, False, None, "WAIT", "0%"),
    (4, 4, "BULLISH", set(), False, "no_chase", "WAIT", "0%"),
    (5, 5, "BULLISH", set(), True, "no_chase", "WAIT", "0%"),
])
def test_decision_matrix_exact(score, denom, bias, flags, below, override, label, deploy):
    d = decide(score, denom, bias, set(flags), below, override)
    assert (d["label"], d["deploy_pct"]) == (label, deploy)


def test_bias_rule():
    assert compute_bias(4, 4) == "BULLISH"
    assert compute_bias(3, 4) == "BULLISH"      # bull>=3 and bull>bear
    assert compute_bias(2, 4) == "NEUTRAL"
    assert compute_bias(3, 5) == "BULLISH"      # 3>2
    assert compute_bias(2, 5) == "BEARISH"      # bear 3
    assert compute_bias(1, 4) == "BEARISH"


# ------------------------------------------------------ aggregation & honest null


def test_denominator_4_honest_null():
    r = score_frame(uptrend(), None, None, symbol="TEST")
    assert r["denominator"] == 4 and r["score"] == 4
    assert r["factors"]["f5_broker"] is None
    assert '"f5_broker": null' in json.dumps(r["factors"])   # never fabricated (FR2)


def test_denominator_5_with_broksum():
    bs = broksum([(d, BULL_ROWS) for d in VALID_DATES])
    fl = flow(VALID_DATES, 100e9)                             # net_foreign > 0
    r = score_frame(uptrend(), bs, fl, symbol="TEST")
    f5 = r["factors"]["f5_broker"]
    assert r["denominator"] == 5
    assert f5 is not None and f5["pass"] is True
    assert f5["values"]["sessions_used"] == 5
    assert math.isclose(f5["values"]["concentration"], round(110 / 120, 4), rel_tol=1e-9)
    assert math.isclose(f5["values"]["avg_buy"], 1000.0, rel_tol=1e-6)
    assert r["score"] == 5 and r["decision"]["label"] == "AGGRESSIVE"


def test_f5_skips_masked_broker_days():
    """MDKA artifact: latest day masked ('--', 1 broker) -> fall back to valid sessions."""
    days = [(d, BULL_ROWS) for d in VALID_DATES] + [MASKED_DAY]
    bs = broksum(days)
    fl = flow(VALID_DATES, 100e9)
    r = score_frame(uptrend(), bs, fl, symbol="TEST")
    f5 = r["factors"]["f5_broker"]
    assert f5 is not None
    assert f5["values"]["sessions_used"] == 5                 # masked day NOT used
    assert "broker_data_masked" in r["gates"]["volatility_flags"]


def test_f5_all_masked_is_null():
    bs = broksum([MASKED_DAY, ("2026-10-02", [("--", 2e9, 20_000)])])
    r = score_frame(uptrend(), bs, flow(["2026-10-05"], 1e9), symbol="TEST")
    assert r["factors"]["f5_broker"] is None                  # honest null
    assert r["denominator"] == 4
    assert "broker_data_masked" in r["gates"]["volatility_flags"]


def test_insufficient_data_below_60_bars():
    r = score_frame(uptrend(59), None, None, symbol="TEST")
    assert r["status"] == "insufficient_data"
    assert r["score"] is None and r["denominator"] is None
    assert r["decision"]["label"] == "WAIT" and r["decision"]["deploy_pct"] == "0%"
    assert r["trade_plan"] is None


def test_60_to_199_bars_3term_stack_neutral_regime():
    r = score_frame(uptrend(100), None, None, symbol="TEST")
    f4 = r["factors"]["f4_ma_stack"]
    assert r["status"] == "ok"
    assert f4["values"]["ma200"] is None
    assert f4["values"]["stack_terms"] == 3
    assert f4["pass"] is True                                  # 3-term, point if >= 2
    assert r["gates"]["regime"] == "neutral_insufficient"


def test_200plus_bars_full_stack_and_regime():
    r = score_frame(uptrend(210), None, None, symbol="TEST")
    f4 = r["factors"]["f4_ma_stack"]
    assert f4["values"]["ma200"] is not None
    assert f4["values"]["stack_terms"] == 4 and f4["values"]["stack"] == 4
    assert r["gates"]["regime"] == "bullish_above_ma200"
    assert "MA200_bull" in f4["flags"]


# ---------------------------------------------------------------- factor details


def test_f2_climax_and_low_liquidity():
    climax = uptrend()
    climax.loc[climax.index[-1], "volume"] = 15_000_000        # ratio ~2.9
    f2 = f2_volume(climax)
    assert f2["pass"] is True and "climax_volume" in f2["flags"]

    thin = make_df([1000 + i for i in range(60)], [10_000] * 60)  # ~1e7 IDR/day turnover
    f2t = f2_volume(thin)
    assert "low_liquidity" in f2t["flags"]


def test_f3_mfm_zero_when_high_equals_low():
    df = uptrend()
    df.loc[df.index[-5], ["high", "low", "close"]] = [1500.0, 1500.0, 1500.0]  # H == L
    f3 = f3_ad(df)
    assert f3 is not None and math.isfinite(f3["values"]["ad"])
    assert math.isfinite(f3["values"]["slope10"])              # no div-by-zero, no NaN


def test_f1_fresh_cross_and_momentum_fading():
    crossed = make_df([1000.0] * 60 + [1020.0, 1045.0, 1075.0])
    f1 = f1_macd(crossed)
    assert f1["pass"] is True and "fresh_cross" in f1["flags"]
    assert f1["values"]["cross_age_bars"] <= 3

    # decelerating rise: MACD still above Signal, but hist shrinking -> momentum_fading
    ramp = [1000 + 0.2 * i * i for i in range(50)]
    fading = make_df(ramp + [ramp[-1] + 2, ramp[-1] + 4])
    f1f = f1_macd(fading)
    assert "momentum_fading" in f1f["flags"]
    assert f1f["values"]["macd"] > f1f["values"]["signal"]
    assert f1f["pass"] is False                                # pass requires hist rising


# ------------------------------------------------------------------ ticks & size


@pytest.mark.parametrize("price,expected", [
    (5210, 25), (5000.01, 25), (5000, 10), (3127, 10), (2000, 10),
    (1999, 5), (1145, 5), (500, 5), (499, 2), (350, 2), (200, 2),
    (199, 1), (175, 1), (50, 1), (49, 1),
])
def test_tick_size_table(price, expected):
    assert tick_size(price) == expected


def test_round_to_tick_modes():
    assert round_to_tick(5210, "nearest") == 5200
    assert round_to_tick(5235, "up") == 5250
    assert round_to_tick(3127, "nearest") == 3130
    assert round_to_tick(1146.4, "nearest") == 1145
    assert round_to_tick(5118, "down") == 5100
    assert all(tick_valid(p) for p in (5200, 5250, 5100, 3130, 1145, 350, 175))
    assert not tick_valid(5205)     # >5000 band tick 25
    assert not tick_valid(3127)     # 2000-5000 band tick 10


def test_sizing_matches_spec_example():
    """§8 example: entry [5200,5250], stop 5100, equity 100M -> exactly 40 lots."""
    closes = [5000 + 5 * i for i in range(60)] + \
             [5118, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5210]
    df = make_df(closes, hi=1.012, lo=0.988)                   # TR ~2.4% -> ATR ~125
    plan = build_trade_plan(df, TIER_AGGRESSIVE, 100_000_000, distribution=False)
    assert plan["entry_zone"] == [5200, 5250]
    assert plan["stop_close"] == 5100                          # min(last 10 closes), tick-down
    assert plan["lots"] == 40                                  # floor(500k / (125*100))
    assert plan["tick_valid"] is True


def test_sizing_floor_and_risk():
    closes = [5000 + 5 * i for i in range(60)] + \
             [5118, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5210]
    df = make_df(closes, hi=1.012, lo=0.988)
    plan = build_trade_plan(df, TIER_AGGRESSIVE, 99_000_000, distribution=False)
    assert plan["lots"] == 39                                  # floor(495k / 12.5k)
    # lots formula invariant: floor((equity*0.5%) / ((entry_ref - stop) * 100))
    lo, hi = plan["entry_zone"]
    ref = (lo + hi) / 2
    assert plan["lots"] == int((99_000_000 * 0.005) // ((ref - plan["stop_close"]) * 100))


def test_sizing_distribution_tightens_stop():
    closes = [5000 + 5 * i for i in range(60)] + \
             [5118, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5210]
    df = make_df(closes, hi=1.012, lo=0.988)
    normal = build_trade_plan(df, TIER_AGGRESSIVE, 100_000_000, distribution=False)
    tight = build_trade_plan(df, TIER_AGGRESSIVE, 100_000_000, distribution=True)
    assert tight["stop_close"] > normal["stop_close"]          # 5150 vs 5100: stop is CLOSER
    assert tight["lots"] > normal["lots"]                      # tighter stop -> less risk/lot
    # ...so constant 0.5% risk sizes the position UP (66 vs 40 lots) — by design (FR4/FR5)
    assert normal["stop_close"] == 5100 and tight["stop_close"] == 5150


def test_no_plan_when_wait():
    df = uptrend()
    assert build_trade_plan(df, TIER_WAIT, 100_000_000, False) is None


# --------------------------------------- §10 required: reproducible cached scores


def test_reproducible_scores_on_cached_data(offline_client, budget):
    r1 = score_ticker("BBRI", client=offline_client, as_of=SEED_AS_OF)
    r2 = score_ticker("bbri.jk", client=offline_client, as_of="2026-10-06")  # normalization + str date
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)  # exact reproducibility

    assert r1["status"] == "ok" and r1["bars"] == 234
    assert r1["symbol"] == "BBRI" and r1["as_of"] == "2026-10-05"
    assert r1["denominator"] in (4, 5) and 0 <= r1["score"] <= r1["denominator"]
    assert all(tag.endswith("[cache]") for tag in r1["trace"])               # no live calls
    assert not any("[live" in tag for tag in r1["trace"])
    assert budget.status()["total_spent"] == 0                               # 0 credits


def test_full_watchlist_on_cache(offline_client, budget):
    results = [score_ticker(s, client=offline_client, as_of=SEED_AS_OF) for s in WATCHLIST]
    assert len(results) == 8
    for r in results:
        assert r["status"] == "ok", r["symbol"]
        assert r["bars"] >= 200
        assert r["denominator"] in (4, 5)
        assert 0 <= r["score"] <= r["denominator"]
        assert r["decision"]["label"] in {"AGGRESSIVE", "DEFENSIVE", "SCALP", "WAIT"}
        if r["trade_plan"]:
            assert r["trade_plan"]["tick_valid"] is True
            assert isinstance(r["trade_plan"]["lots"], int) and r["trade_plan"]["lots"] >= 0
            lo, hi = r["trade_plan"]["entry_zone"]
            assert r["trade_plan"]["stop_close"] < lo <= hi
        assert all(t.endswith("[cache]") for t in r["trace"])
    assert budget.status()["total_spent"] == 0                               # whole pass: 0 cr


def test_cached_real_data_edge_cases(offline_client):
    """Real seeded data: BMRI foreign_divergence fires; MDKA masked day handled."""
    bmri = score_ticker("BMRI", client=offline_client, as_of=SEED_AS_OF)
    assert bmri["gates"]["foreign_flow"]["divergence"] is True    # seed showed -530B/5d
    assert "foreign_divergence" in bmri["risk_flags"]

    mdka = score_ticker("MDKA", client=offline_client, as_of=SEED_AS_OF)
    assert "broker_data_masked" in mdka["gates"]["volatility_flags"]   # Oct 5 '--' day
    f5 = mdka["factors"]["f5_broker"]
    assert f5 is None or f5["values"]["sessions_used"] >= 1
    if f5 is not None:
        assert "--" not in json.dumps(f5["values"])


def test_score_output_is_json_clean(offline_client):
    r = score_ticker("ANTM", client=offline_client, as_of=SEED_AS_OF)
    s = json.dumps(r)                      # raises on NaN with allow_nan=False
    json.dumps(r, allow_nan=False)
    assert "NaN" not in s


# ------------------------------------------------------------------ stack parity


def test_pandas_ta_parity():
    """Pin indicator interpretations (§3 stack: pandas-ta). pandas-ta seeds EMAs
    with an SMA warm-up, so parity is asserted on converged tails, not warm-up."""
    rng = np.random.default_rng(42)
    closes = pd.Series(1000 + np.cumsum(rng.normal(0, 10, 250)))
    highs = closes * 1.01
    lows = closes * 0.99

    macd_manual = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    signal_manual = macd_manual.ewm(span=9, adjust=False).mean()
    m_ta, s_ta, _ = engine.macd_lines(closes)
    # pandas-ta's SMA-seeded warm-up leaves a small residual (~1e-3 rel) that decays;
    # converged tails must agree tightly. pandas-ta is the pinned implementation.
    assert np.allclose(m_ta.values[100:], macd_manual.values[100:], rtol=5e-3)
    assert np.allclose(s_ta.values[100:], signal_manual.values[100:], rtol=5e-3)

    prev = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev).abs(), (lows - prev).abs()], axis=1).max(axis=1)
    atr_manual = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_ta = engine.atr14(pd.DataFrame({"high": highs, "low": lows, "close": closes}))
    assert np.allclose(atr_ta.values[100:], atr_manual.values[100:], rtol=1e-3)
