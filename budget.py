"""BANDAR credit budget guard (§5, [LOCKED]).

Rules enforced here:
- 2xx bills the stated cost; 404 bills exactly 1; all other 4xx/5xx are free (FR7).
- Per-run hard cap: 60 credits (abort via precheck).
- Daily soft warn: 200 credits (log only, never raises).
- Untouchable reserve: 200 of the 1000-cr grant -> spendable total is 800.
- Cache hits never touch this guard (client.py returns before precheck/charge).
- State persists to state/budget_state.json; every live call is logged (JSONL).
- Day boundary is WIB (Asia/Jakarta); daily_spent resets on rollover.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("bandar.budget")

GRANT = 1000            # total granted credits
RESERVE = 200           # untouchable reserve
SPENDABLE = GRANT - RESERVE  # 800 — total_spent may land exactly on 800, never exceed
RUN_CAP = 60            # per-run hard cap (abort)
DAILY_WARN = 200        # daily soft warn (log only)

DEFAULT_STATE_PATH = os.path.join("state", "budget_state.json")
DEFAULT_LOG_PATH = os.path.join("state", "budget_log.jsonl")
DEFAULT_TZ = "Asia/Jakarta"  # WIB


class BudgetError(Exception):
    """Base class for all budget violations."""


class RunCapExceeded(BudgetError):
    """Raised when a call would push run_spent past the 60-cr per-run hard cap."""


class ReserveExceeded(BudgetError):
    """Raised when a call would push total_spent past 800 (reserve is untouchable)."""


def billable_amount(cost: int, status: int) -> int:
    """How much a response with this HTTP status bills (§5 + FR7).

    2xx -> stated cost · 404 -> exactly 1 · every other 4xx/5xx -> 0.
    Unlisted statuses (1xx/3xx — requests follows redirects, so these
    should not surface) bill 0 defensively.
    """
    if 200 <= status < 300:
        return cost
    if status == 404:
        return 1
    return 0


class CreditBudget:
    """Tracks total/daily/run spend with persistence and logging."""

    def __init__(
        self,
        state_path: str | Path = DEFAULT_STATE_PATH,
        log_path: str | Path = DEFAULT_LOG_PATH,
        tz: str = DEFAULT_TZ,
    ) -> None:
        self.state_path = Path(state_path)
        self.log_path = Path(log_path)
        self.tz = ZoneInfo(tz)
        self._state = {
            "total_spent": 0,
            "daily_spent": 0,
            "run_spent": 0,
            "day": str(self._today()),
            "daily_warned": False,
        }
        self._load()

    # ------------------------------------------------------------------ time

    def _today(self):
        """Current date in the configured timezone (WIB by default)."""
        return datetime.now(self.tz).date()

    def _rollover(self) -> None:
        """Reset daily_spent when the WIB day changes."""
        today = str(self._today())
        if self._state["day"] != today:
            logger.info(
                "budget day rollover %s -> %s: daily_spent %d -> 0",
                self._state["day"], today, self._state["daily_spent"],
            )
            self._state["day"] = today
            self._state["daily_spent"] = 0
            self._state["daily_warned"] = False
            self._save()

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                for key in self._state:
                    if key in loaded:
                        self._state[key] = loaded[key]
            except (json.JSONDecodeError, OSError) as exc:
                # Corrupt state must not silently grant fresh credits — surface it.
                raise BudgetError(f"corrupt budget state at {self.state_path}: {exc}") from exc
        self._rollover()

    def _save(self) -> None:
        """Atomic write: tmp file + os.replace so state never corrupts mid-write."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        os.replace(tmp, self.state_path)

    def _append_log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ guard

    def precheck(self, cost: int) -> None:
        """Raise before any HTTP if this cost is unaffordable (§5).

        - run_spent + cost > 60            -> RunCapExceeded (abort the run)
        - total_spent + cost > 800         -> ReserveExceeded (reserve untouchable)

        Boundary semantics: landing exactly on 60 (run) or 800 (total) is allowed.
        """
        if cost < 0:
            raise ValueError(f"cost must be >= 0, got {cost}")
        self._rollover()
        if self._state["run_spent"] + cost > RUN_CAP:
            raise RunCapExceeded(
                f"run cap: run_spent={self._state['run_spent']} + cost={cost} > {RUN_CAP}"
            )
        if self._state["total_spent"] + cost > SPENDABLE:
            raise ReserveExceeded(
                f"reserve: total_spent={self._state['total_spent']} + cost={cost} > {SPENDABLE}"
            )

    def charge(self, cost: int, status: int, endpoint: str) -> int:
        """Bill ONLY after a billable response (§5). Returns the amount billed.

        2xx -> cost · 404 -> 1 · 400/401/403/429/5xx (and any other 4xx) -> 0.
        Every live call is logged, billed or free.
        """
        self._rollover()
        billed = billable_amount(cost, status)
        if billed == 0 and status not in (400, 401, 403, 429) and not (500 <= status < 600):
            logger.warning("charge: unlisted status %d for %s — billed 0", status, endpoint)

        self._state["total_spent"] += billed
        self._state["daily_spent"] += billed
        self._state["run_spent"] += billed
        self._save()

        self._append_log({
            "ts": datetime.now(self.tz).isoformat(timespec="seconds"),
            "endpoint": endpoint,
            "status": status,
            "cost": cost,
            "billed": billed,
            "total_spent": self._state["total_spent"],
            "daily_spent": self._state["daily_spent"],
            "run_spent": self._state["run_spent"],
        })

        if billed and self._state["run_spent"] > RUN_CAP:
            # precheck should make this unreachable; log loudly if state drifted.
            logger.error("run_spent %d exceeds cap %d after charge", self._state["run_spent"], RUN_CAP)

        if (
            not self._state["daily_warned"]
            and self._state["daily_spent"] >= DAILY_WARN
        ):
            self._state["daily_warned"] = True
            self._save()
            logger.warning(
                "daily soft warn: daily_spent=%d >= %d", self._state["daily_spent"], DAILY_WARN
            )
        return billed

    def reset_run(self) -> None:
        """Zero run_spent (start of a new run). total/daily untouched."""
        self._state["run_spent"] = 0
        self._save()

    # ------------------------------------------------------------------ views

    def remaining_run(self) -> int:
        self._rollover()
        return max(0, RUN_CAP - self._state["run_spent"])

    def remaining_total(self) -> int:
        self._rollover()
        return max(0, SPENDABLE - self._state["total_spent"])

    def status(self) -> dict:
        """Snapshot for the M5 credit meter (st.metric)."""
        self._rollover()
        return {
            "total_spent": self._state["total_spent"],
            "daily_spent": self._state["daily_spent"],
            "run_spent": self._state["run_spent"],
            "remaining_run": self.remaining_run(),
            "remaining_total": self.remaining_total(),
            "grant": GRANT,
            "reserve": RESERVE,
            "run_cap": RUN_CAP,
            "day": self._state["day"],
        }
