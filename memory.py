"""BANDAR memory layer (§7, [LOCKED] D6/D7).

- score_history table: exact §7 schema, PRIMARY KEY (ticker, as_of).
- Any run upserts by (ticker, as_of) (D7).
- Delta = latest row with as_of < current as_of; rendered old->new + prior date (FR6).
- Snapshots store factors_json + gates_json per run (D6, delta-with-reason).
- Watchlist = persistent JSON (state/watchlist.json); run scratchpad is ephemeral
  (never persisted — by design, nothing here stores one).
- Each run: read history first, report deltas, then score (§7; orchestration in M4).

0 credits: this module never touches the network.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from client import _normalize_symbol

WIB = ZoneInfo("Asia/Jakarta")

DEFAULT_DB_PATH = os.path.join("state", "bandar.db")
DEFAULT_WATCHLIST_PATH = os.path.join("state", "watchlist.json")

# §7 [LOCKED] — exact schema, do not add/remove columns.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS score_history (
  ticker TEXT, as_of DATE, score INTEGER, denominator INTEGER,
  decision TEXT, factors_json TEXT, gates_json TEXT, created_at TIMESTAMP,
  PRIMARY KEY (ticker, as_of)
);
"""

UPSERT_SQL = """
INSERT INTO score_history
  (ticker, as_of, score, denominator, decision, factors_json, gates_json, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(ticker, as_of) DO UPDATE SET
  score = excluded.score,
  denominator = excluded.denominator,
  decision = excluded.decision,
  factors_json = excluded.factors_json,
  gates_json = excluded.gates_json,
  created_at = excluded.created_at
"""


def _now_wib() -> str:
    return datetime.now(WIB).isoformat(timespec="seconds")


class ScoreMemory:
    """SQLite-backed score history + persistent watchlist."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH,
                 watchlist_path: str | Path = DEFAULT_WATCHLIST_PATH) -> None:
        self.db_path = Path(db_path)
        self.watchlist_path = Path(watchlist_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ScoreMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ history writes

    def record_score(self, result: dict) -> None:
        """Upsert one §8 score_ticker result by (ticker, as_of) — D6/D7.

        Stores factors_json + gates_json snapshots verbatim (honest nulls kept:
        a null f5_broker is stored as JSON null, never fabricated).
        """
        decision = result.get("decision") or {}
        self.conn.execute(UPSERT_SQL, (
            result["symbol"],
            result["as_of"],
            result.get("score"),                     # NULL for insufficient_data
            result.get("denominator"),               # NULL for insufficient_data
            decision.get("label"),
            json.dumps(result.get("factors")),
            json.dumps(result.get("gates")),
            _now_wib(),
        ))
        self.conn.commit()

    def record_run(self, results: list[dict]) -> list[dict]:
        """Record a whole run and return delta rows (read history FIRST, per §7).

        For each result: fetch the prior snapshot (latest as_of < current), upsert
        the new one, then build the delta row for the §9 WATCHLIST DELTAS table.
        """
        deltas = []
        for r in results:
            prev = self.get_prev(r["symbol"], r["as_of"])
            self.record_score(r)
            deltas.append(build_delta(r, prev))
        return deltas

    # ------------------------------------------------------------ history reads

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for key in ("factors_json", "gates_json"):
            if d.get(key) is not None:
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except json.JSONDecodeError:
                    d[key.replace("_json", "")] = None
        return d

    def get_history(self, ticker: str, limit: int | None = None) -> list[dict]:
        """All snapshots for a ticker, newest as_of first (M4 score_history tool: 0 cr)."""
        sql = "SELECT * FROM score_history WHERE ticker = ? ORDER BY as_of DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, (_normalize_symbol(ticker),)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_prev(self, ticker: str, as_of: str) -> dict | None:
        """§7 [LOCKED]: delta = latest row with as_of < current as_of. None if no prior
        run — a first-ever score has NO delta and none is invented (FR6 honest)."""
        row = self.conn.execute(
            "SELECT * FROM score_history WHERE ticker = ? AND as_of < ? "
            "ORDER BY as_of DESC LIMIT 1",
            (_normalize_symbol(ticker), as_of),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def latest(self, ticker: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM score_history WHERE ticker = ? ORDER BY as_of DESC LIMIT 1",
            (_normalize_symbol(ticker),),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    # ---------------------------------------------------------------- watchlist

    def load_watchlist(self) -> list[str]:
        """Persistent JSON watchlist; seeded once from BANDAR_WATCHLIST env (§4)."""
        if self.watchlist_path.exists():
            try:
                data = json.loads(self.watchlist_path.read_text(encoding="utf-8"))
                syms = [_normalize_symbol(s) for s in data.get("symbols", [])]
                return list(dict.fromkeys(syms))
            except (json.JSONDecodeError, ValueError, OSError):
                pass  # fall through to env seed
        env = os.environ.get("BANDAR_WATCHLIST", "")
        syms = list(dict.fromkeys(
            _normalize_symbol(s.strip()) for s in env.split(",") if s.strip()
        ))
        if syms:
            self.save_watchlist(syms)
        return syms

    def save_watchlist(self, symbols: list[str]) -> None:
        syms = list(dict.fromkeys(_normalize_symbol(s) for s in symbols if str(s).strip()))
        self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        self.watchlist_path.write_text(
            json.dumps({"symbols": syms, "updated_at": _now_wib()}, indent=2),
            encoding="utf-8",
        )

    def update_watchlist(self, op: str, symbols: list[str]) -> list[str]:
        """Engine side of the M4 watchlist_rw tool (0 cr). op: add | remove | replace."""
        current = self.load_watchlist()
        norm = list(dict.fromkeys(_normalize_symbol(s) for s in symbols if str(s).strip()))
        if op == "add":
            merged = list(dict.fromkeys(current + norm))
        elif op == "remove":
            merged = [s for s in current if s not in set(norm)]
        elif op == "replace":
            merged = norm
        else:
            raise ValueError(f"unknown watchlist op {op!r} — use add/remove/replace")
        self.save_watchlist(merged)
        return merged


# -------------------------------------------------------------------- deltas (FR6)

def build_delta(result: dict, prev: dict | None) -> dict:
    """Delta row for the §9 WATCHLIST DELTAS table: old->new + prior date.

    No prior run -> has_prev False and old fields None (never invented, FR6).
    """
    decision = result.get("decision") or {}
    return {
        "ticker": result.get("symbol"),
        "as_of": result.get("as_of"),
        "has_prev": prev is not None,
        "prev_date": prev.get("as_of") if prev else None,
        "score_old": prev.get("score") if prev else None,
        "score_new": result.get("score"),
        "denominator_old": prev.get("denominator") if prev else None,
        "denominator_new": result.get("denominator"),
        "decision_old": prev.get("decision") if prev else None,
        "decision_new": decision.get("label"),
        "gate_override": decision.get("gate_override"),
        "changed": bool(prev) and (
            prev.get("score") != result.get("score")
            or prev.get("decision") != decision.get("label")
        ),
    }


def render_delta(delta: dict) -> str:
    """Human one-liner: 'BBRI 2/5 WAIT (Oct 3: 4/5 DEFENSIVE)' — old->new + prior date."""
    def pair(which: str) -> str:
        s, dn = delta.get(f"score_{which}"), delta.get(f"denominator_{which}")
        return f"{s}/{dn}" if isinstance(s, (int, float)) else "—"   # honest null

    if not delta.get("has_prev"):
        return f"{delta['ticker']} {pair('new')} " \
               f"{delta['decision_new']} (first scored {delta['as_of']})"
    return (f"{delta['ticker']} {pair('old')} "
            f"{delta['decision_old']} -> {pair('new')} "
            f"{delta['decision_new']} (prior {delta['prev_date']})")
