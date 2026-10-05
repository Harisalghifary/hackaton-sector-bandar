# BANDAR — Build Spec & OpenCode Pack v2.0
Supersedes PRD v1 (22 Sep 2026). Track 1 (AI Agents & Assistants) · Sectors Hackathon 2026 · Submission due 30 Sep 2026, 23:59 WIB (submit early — submission freezes the repo).

## 0. How to work with this doc in OpenCode
- **Single source of truth.** If OpenCode's output conflicts with this doc, this doc wins.
- Markers: `[LOCKED]` = decided, do not revisit or "improve" · `[TUNE]` = set from seed snapshot data · `[TODO-USER]` = needs human input.
- **Brainstorm mode** (before each milestone): ask OpenCode to critique the relevant section, e.g. "Critique §6 against FR1–FR5 and list ambiguities", "Red-team §5 for credit-leak paths", "Propose unit tests for the no-chase gate". Apply only changes that respect [LOCKED] items.
- **Build mode**: milestone-gated prompts (§10). One milestone per session. Never say "build the whole app".
- Schedule is relative: Day 1 = first build session … Day 8 = submit.

## 1. Product & strategy [LOCKED]
- **One-liner (also the submission problem statement):** Bandar is an autonomous pre-market AI analyst for IDX swing traders that scores your watchlist with a deterministic 5-factor confluence engine and hands you a disciplined, act-on-able brief — entry, close-based stop, 0.5%-risk size, and what changed since yesterday — via a clean Streamlit app and a free 08:30 WIB Telegram push.
- **Architecture:** deterministic scoring engine (all math) wrapped in a genuine LLM agent (plan, route, remember, narrate). **The LLM never computes a number.**
- **Kill test:** remove the agent layer → only a screener remains that cannot plan, remember, or answer free-text.
- **Rubric play:** usability 40% (trader-first UI), video 30% (trader's story), technical 30% (repo carries the engineering story; UI stays clean).

## 2. Decisions log [LOCKED]
| # | Decision | Choice |
|---|---|---|
| D1 | Track | 1 (AI Agents), hybrid engine+agent |
| D2 | Risk profiles | Single preservation-first profile; `max_extension_atr` is a named config key (profiles = post-hackathon) |
| D3 | No-chase gate | ATR extension (0.5) + ARA hard block combined |
| D4 | Adaptivity location | Policy rules in planner system prompt; executor enforces budget |
| D5 | Synthesis output | Required boxed JSON (`submit_brief`), not free-form |
| D6 | Memory snapshots | Store `factors_json` + `gates_json` per run (delta-with-reason) |
| D7 | History writes | Any run upserts by `(ticker, as_of)` |
| D8 | Runtime LLM | Gemini Flash 3.8 primary; Claude Sonnet 4.5 hot-swap fallback |
| D9 | UI | Answer foreground, workings collapsed; no charts/auth/settings |
| D10 | Dev workflow | OpenCode, milestone-gated, ~10 iterations |

## 3. Stack & runtime config [LOCKED]
Python 3.11+ · Streamlit · Telegram Bot API (plain `requests.post`) · Sectors REST v2 · SQLite + JSON · raw tool-calling loop (no agent framework) · pandas + pandas-ta.
```python
RUNTIME_LLM = {
    "primary":  "gemini-flash-3.8",    # all tiers: intent, plan, synthesis
    "fallback": "claude-sonnet-4.5",   # config-only hot swap
    "enforce":  ["response_schema:plan", "response_schema:submit_brief",
                 "numeric_trace_validator", "plan_validator"],
    "temp": 0.2,
}
```
- Gemini controlled generation: `response_mime_type="application/json"` + `response_schema`.
- **Max 3 LLM calls per run:** intent → plan → synthesis. Stream synthesis via `st.write_stream`.

## 4. Credentials & env [TODO-USER]
| Secret / input | Env var | Needed at |
|---|---|---|
| Sectors API key (1,000-cr grant) | `SECTORS_API_KEY` | M1 |
| Watchlist 6–12 names | `BANDAR_WATCHLIST` | M1 |
| Equity in IDR (default 100,000,000) | `EQUITY_IDR` | M2 |
| Gemini API key | `GEMINI_API_KEY` | M4 |
| Anthropic key (fallback/dev) | `ANTHROPIC_API_KEY` | M4 (optional) |
| Telegram bot token + chat id | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | M6 |
`.env.example` committed; real `.env` gitignored with `cache/`, `state/` (keep `state/.gitkeep`), `__pycache__/`. CI uses GitHub Actions secrets. Secrets sweep before repo goes public.

## 5. Data layer contract (`budget.py`, `client.py`) [LOCKED]
**budget.py:** `precheck(cost)` raises if unaffordable; `charge(cost, status, endpoint)` only after billable response — **2xx bills stated cost, 404 bills 1; 400/401/403/429/5xx free**. Per-run hard cap **60** (abort); daily soft warn **200** (log); untouchable reserve **200**; grant 1000. Cache hits never touch the guard. Persists state; logs every live call.
**client.py:** modes `cache-first` (default), `live`, `offline` (miss raises, never spends). Key = sha1(endpoint + sorted params); JSON per namespace. Auth header `Authorization: <raw key>` — **no "Bearer"**. Base `https://api.sectors.app`.
| Method | Endpoint (v2, confirmed) | Cost |
|---|---|---|
| `ohlcv(sym,start,end)` | `GET /v2/daily/{symbol}/` (≤90d) | 1 |
| `foreign_flow(sym)` | `GET /v2/foreign-flow/{symbol}/` | 1 |
| `broker_summary(sym)` | `GET /v2/broker-summary/{symbol}/` (≤14d) | 1 |
| `fundamentals(sym)` | `GET /v2/company/report/{symbol}/` | 1 |
| `screener_structured` / `screener_nl` | `GET /v2/companies/` | 1 / 3 |
| `daily_close(date)` | `GET /v2/close/` (full universe) | 1 |
MA200 needs 200+ bars → seeding ≈ 3 daily calls/ticker; first score ≈5 cr; seeded 6-name watchlist ≈30 cr once; daily re-score ≈1–2 cr.

## 6. Scoring engine (`engine.py`) — deterministic, no LLM [LOCKED]
```python
ENGINE_CONFIG = {
    "max_extension_atr": 0.5,                 # LOCKED (D3)
    "ara_limit_pct": 0.10,                    # proxy; verify per price tier if time allows
    "broker_concentration_threshold": 0.6,    # [TUNE] from seed snapshot distribution
    "min_volume_ratio": 1.5,
    "min_bars_to_score": 60,
    "risk_per_trade_pct": 0.5,
}
```
| # | Computation | Point=1 when | Flags | Min bars |
|---|---|---|---|---|
| F1 MACD | EMA12−EMA26; Signal=EMA9(MACD); Hist=MACD−Signal | MACD>Signal, Hist≥0, Hist rising | fresh_cross (≤3 bars); momentum_fading | 35 |
| F2 Volume | vol_ratio = V / SMA20(V) | ≥1.5 | climax_volume (≥2.5); low_liquidity | 20 |
| F3 A/D | MFM=((C−L)−(H−C))/(H−L), 0 if H=L; AD=cumsum(MFM×V); slope=linreg last 10 | slope>0 | bearish_divergence (price HH + AD LH) | 11 |
| F4 MA stack | EMA9, EMA26, SMA50, SMA200; stack=(C>MA9)+(MA9>MA26)+(MA26>MA50)+(MA50>MA200) | stack≥3 | MA200_bull feeds regime | 200* |
| F5 Broker | net_foreign; concentration=top3_buy/total_buy; avg_buy vs C | net_foreign>0 AND concentration≥thr AND avg_buy≤C | foreign_divergence (bullish + net-sell >500B) | — |
Aggregation: core=F1..F4 (0–4); broksum present → +F5, denom 5; missing → **denom 4, F5 null, never fabricated**. <60 bars → `insufficient_data`. 60–199 bars: MA200 null, F4 3-term stack (point if ≥2), regime neutral. Bias: BULLISH if bull≥3 and bull>bear; BEARISH if bear≥3; else NEUTRAL/MIXED.
**No-chase gate (D3):**
```python
extended = close_D > high_{D-1} + ENGINE_CONFIG["max_extension_atr"] * ATR14_D
ara_day  = close_D >= close_{D-1} * (1 + ENGINE_CONFIG["ara_limit_pct"])
gate_triggered = extended or ara_day   # → decision forced WAIT-pullback; NEVER "actionable", even at 4/4
```
Overlay gates (change decision, never score): foreign_divergence → downgrade one tier; climax_volume at resistance → distribution flag, tighten stop; regime C<MA200 → cap at DEFENSIVE; FCA/PPK UMA-10 volatility flag.
**Decision matrix:** 4/4 or 4–5/5 BULLISH → AGGRESSIVE 60–80% · 3/4 or 4/5+flags → DEFENSIVE 40–60% · 2/4 or 3/5 → DEFENSIVE 30–40% · 2 NEUTRAL → scalp 15–20% · ≤1 BEARISH → WAIT/CASH 0%.
**Sizing:** stop = close level only. `lots = floor((equity × 0.005) / ((entry − stop) × 100))`, 1 lot = 100 shares. Round entry/stop/TP to IDX ticks: >5,000→25 · 2,000–5,000→10 · 500–2,000→5 · 200–500→2 · 50–200→1 (verify current table). `tick_valid` on every plan.

## 7. Memory [LOCKED] (D6, D7)
```sql
CREATE TABLE score_history (
  ticker TEXT, as_of DATE, score INTEGER, denominator INTEGER,
  decision TEXT, factors_json TEXT, gates_json TEXT, created_at TIMESTAMP,
  PRIMARY KEY (ticker, as_of)
);
```
Any run upserts by `(ticker, as_of)`. Delta = latest row with `as_of < current as_of`; render `old→new + prior date`. Watchlist = persistent JSON; run scratchpad ephemeral. Each run: read history first, report deltas, then score.

## 8. Agent layer [LOCKED] (D4, D5, D8)
**Tools (7):** `score_ticker(symbol)` ~2cr · `rank_watchlist()` 2×N · `score_history(symbol)` 0cr · `get_fundamentals(symbol)` 1 · `get_foreign_flow(symbol)` 1 · `screen(where|q)` 1/3 · `watchlist_rw(op,list)` 0cr.
**`score_ticker` return contract:**
```json
{"symbol":"BBRI.JK","as_of":"2026-09-21","score":4,"denominator":5,
 "factors":{"f1_macd":{"pass":true,"values":{"hist":12.5,"hist_rising":true},"flags":["fresh_cross"]},
            "f5_broker":{"pass":false,"values":{"concentration":0.45,"net_foreign":120000000000},"flags":[]}},
 "gates":{"no_chase":{"triggered":false,"extension_atr":0.2,"threshold_atr":0.5},"ara":{"triggered":false},"regime":"bullish_above_ma200"},
 "decision":{"label":"AGGRESSIVE","deploy_pct":"60-80%"},
 "trade_plan":{"entry_zone":[5200,5250],"stop_close":5100,"lots":40,"tick_valid":true},
 "trace":["ohlcv:/v2/daily/BBRI/ [cache]","broksum:/v2/broker-summary/BBRI/ [live, 1cr]"]}
```
Honest-null: broksum missing → `f5: null, denominator: 4`, trace shows the 404.
**Schemas:** plan `{steps:[{tool,args,reason}]}` · synthesis `submit_brief {symbol, extracted[], interpretation, action_plan[], risk_flags[]}` · fallback `{type:"fallback", message}`.
**Planner policy (system prompt):** (1) active-position/risk items before new candidates; (2) route by sub-question: price/TA→engine, valuation→fundamentals, smart-money→broksum/flow, universe→screener; (3) adaptivity: score≥3→fundamentals, score≤1→skip context, tie→broksum/flow tiebreak; (4) budget breach → abort step, narrate truncation honestly; (5) read history before scoring.
**Code-enforced validators:** numeric trace validator (every figure in output must appear in tool-result trace; else dropped/flagged) · plan validator (known tools, schema-valid args, watchlist symbols only).
**Voice:** desk-note, ≤2 sentences/section, quote values exactly, no derived arithmetic, no invented deltas.

## 9. Product surfaces [LOCKED] (D9)
**Foreground:** header + credit meter · TOP PICK card (score x/5, 5 factor ticks, decision, entry zone, close-based stop, lots, one-line why) · WATCHLIST DELTAS table (old→new, decision, gate override, prior date) · buttons Download Brief / Force Live Refresh.
**Workings (`st.expander`, collapsed):** agent plan · data pulls + routing + cache/live tags · per-factor math with computed values · credit meter. Expanded once on camera only.
**Telegram 08:30 WIB:** top pick + entry/stop/size + why · watchlist status lines · "N others actionable" · credits used. GitHub Actions cron; validate a few times, don't run daily.
**Streamlit:** `st.session_state` across reruns; API calls ONLY behind explicit buttons, results stored in session_state; `st.status` trace rows inside workings; `st.write_stream` brief; `st.metric` credits.
**DO NOT BUILD:** auth, multi-user, candlestick charts, settings pages, risk-profile switcher, backtesting, any execution path.

## 10. Milestones & OpenCode prompts (Day 1 → Day 8)
M1 data layer → M2 engine → M3 memory → M4 agent (4a tools+planner, 4b executor+validators) → M5 UI (5a skeleton, 5b wiring) → M6 Telegram+cron → M7 video (3-min judging + 1-min teaser) → M8 submit early + social post.
**M1 prompt (paste verbatim):**
```text
You are an expert Python engineer building the data layer for a financial app called "Bandar" for the Sectors Hackathon 2026.
Create two files: budget.py and client.py, per spec sections 5.
budget.py: class CreditBudget; state persisted to state/budget_state.json; tracks total_spent, daily_spent, run_spent; day-rollover resets daily_spent; precheck(cost) raises if run_spent+cost>60 or total_spent would exceed 800 (grant 1000, reserve 200); charge(cost,status,endpoint) bills ONLY on 2xx or 404 (400/401/403/429/5xx free) and logs; reset_run() zeroes run_spent.
client.py: class SectorsClient; base https://api.sectors.app; header Authorization: {api_key} (NO Bearer); modes cache-first/live/offline; cache cache/{namespace}/{sha1(endpoint+sorted params)}.json; _fetch checks cache (hit = 0 credits, no HTTP), offline miss raises; else precheck → GET → charge → save cache. Methods: ohlcv(symbol,start,end) GET /v2/daily/{symbol}/ cost 1; foreign_flow(symbol) GET /v2/foreign-flow/{symbol}/ cost 1; broker_summary(symbol) GET /v2/broker-summary/{symbol}/ cost 1.
Also write test_m1.py proving an offline cache miss raises and a cache hit spends 0. Stop and wait for review.
```
**M2–M6 prompts:** one sentence each — "Implement spec §6 exactly, with unit tests for the no-chase gate (assert 4/4 + extended ⇒ WAIT) and reproducible scores on cached data." / "Implement spec §7 with upsert + delta query tests." / "Implement spec §8 tools + planner with plan_validator tests." / "Implement spec §8 executor + validators + submit_brief synthesis." / "Implement spec §9 UI; all API calls behind buttons, results in st.session_state." / "Implement telegram_push.py + Actions cron per spec §9."

## 11. Model gate (`tests/llm_gate.py`, land Day 1, run before rehearsal)
Fixtures `tests/fixtures/engine_objects.json`: BBRI (happy path), ANTM (no-chase triggered), BBCA (denom 4, f5 null). 20 prompts: 5 plan, 8 synth, 4 freeform, 3 compare. **Pass = 20/20 schema-valid, 0 fabricated numbers, 0 NO_CHASE_VIOLATION.** Fail → flip `RUNTIME_LLM["primary"]` to fallback and re-run.

## 12. Rules compliance [verified vs official rules page]
No automated trade execution (FR9) · Sectors REST = core data source · working end-to-end MVP · public repo + videos suffice (no live deploy) · assets: public repo, 1-min teaser, ≤3-min judging video, one-sentence problem statement (§1), track=1, team names, social post · project exclusive to this hackathon, fresh repo · **complete Sectors App onboarding (not just API key) — verify today** · submission before 30 Sep 23:59 WIB.

## 13. FR1–FR13 acceptance checklist
FR1 factor values+pass/fail · FR2 denom-4 fallback, no fabricated flow · FR3 extended/ARA never "actionable" (unit-test at 4/4) · FR4 close-based stop only · FR5 whole lots + valid ticks · FR6 delta old→new + prior date · FR7 abort at 60, free on 4xx/5xx except 404 · FR8 cache hit 0 cr, offline miss raises · FR9 no execution path · FR10 numeric trace validator green · FR11 cron message matches foreground · FR12 all 5 intents + graceful fallback · FR13 decision matrix exact.

## 14. Non-negotiables
No trade execution ever · close-based stops only · every output number traces to a tool result · 60-credit cap per run · cache-first in dev · product must break without Sectors.

## 15. Open items [TODO]
- [TODO-USER] Provide §4 credentials/inputs (Sectors key, watchlist, equity).
- [ ] M1 seed snapshot (~30 cr) → log raw top-3 shares → set `broker_concentration_threshold` [TUNE].
- [ ] Verify current IDX tick-size table (§6).
- [ ] Confirm Sectors App onboarding complete (§12).
- [ ] Draft social post + teaser storyboard (M7).