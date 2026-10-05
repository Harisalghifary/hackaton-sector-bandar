"""BANDAR Sectors REST v2 client (§5, [LOCKED]).

- Base: https://api.sectors.app · Auth header: `Authorization: <raw key>` — NO "Bearer".
- Modes: cache-first (default) · live (always HTTP, rewrites cache) · offline (cache or raise, never spends).
- Cache: cache/{namespace}/{sha1(endpoint + sorted params)}.json — cache hit = 0 credits,
  no HTTP, budget guard never touched.
- Live path order (LOCKED): precheck -> GET -> charge -> save cache (2xx only; no negative caching).
- Billing (charge): 2xx bills stated cost · 404 bills exactly 1 · everything else free (FR7).

Param names VERIFIED against docs.sectors.app (v2): /v2/daily/ takes `start`/`end`;
/v2/broker-summary/ takes optional `broker_code`/`start`/`end` (default: last 14 days
ending today); /v2/foreign-flow/ takes optional `start`/`end` (default: last 90 days
ending today); /v2/close/ takes `date`/`limit`/`offset` (paginated: 1 cr PER PAGE,
max 30 tickers/page, full universe ~32 pages ~32 cr — NOT for daily watchlist refresh).

Because broker-summary/foreign-flow default their window server-side to "today",
param-less calls have stable cache keys and would replay stale windows across days
in cache-first mode. Pass explicit `start`/`end` for day-scoped cache keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, date as date_cls
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from budget import CreditBudget

BASE_URL = "https://api.sectors.app"
VALID_MODES = ("cache-first", "live", "offline")
MAX_OHLCV_DAYS = 90    # §5: GET /v2/daily/{symbol}/ is capped at <= 90 days per call
MAX_BROKER_DAYS = 14   # docs: /v2/broker-summary/{symbol}/ window <= 14 days
MAX_FLOW_DAYS = 90     # docs: /v2/foreign-flow/{symbol}/ window <= 90 days


class SectorsError(Exception):
    """Base class for all client errors."""


class OfflineCacheMissError(SectorsError):
    """Offline mode + cache miss: raises, never spends (§5/FR8)."""


class SectorsNotFoundError(SectorsError):
    """Live 404. Exactly 1 cr has been charged. Honest-null signal for F5 (FR2)."""


class SectorsAPIError(SectorsError):
    """Any other 4xx/5xx. 0 cr charged. Carries .status_code."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class SectorsConnectionError(SectorsError):
    """Timeout / connection failure. charge() was never called — 0 cr."""


# Short labels for §8-style trace rows: "ohlcv:/v2/daily/BBRI/ [cache]"
TRACE_LABELS = {
    "ohlcv": "ohlcv",
    "foreign_flow": "flow",
    "broker_summary": "broksum",
    "fundamentals": "fund",
    "screener": "screen",
    "daily_close": "close",
}


def _normalize_symbol(symbol: str) -> str:
    """'bbri.jk' / 'BBRI.JK' / 'BBRI' -> 'BBRI' (matches §8 trace /v2/daily/BBRI/)."""
    if not isinstance(symbol, str):
        raise ValueError(f"symbol must be a string, got {type(symbol).__name__}")
    sym = symbol.strip().upper()
    if sym.endswith(".JK"):
        sym = sym[:-3]
    if not re.fullmatch(r"[A-Z0-9]{1,10}", sym):
        raise ValueError(f"invalid IDX symbol: {symbol!r}")
    return sym


def _iso_date(value: str, field: str) -> str:
    """Validate/normalize a YYYY-MM-DD string; raise ValueError otherwise."""
    try:
        return date_cls.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO date YYYY-MM-DD, got {value!r}") from exc


def _check_window(start: str, end: str, max_days: int) -> None:
    """Validate a date window: parseable, ordered, span <= max_days (§5 caps).

    Pre-flight rejection — no HTTP, no precheck, 0 cr. (The API clamps wider
    windows instead of erroring, but silent clamping would corrupt seeding math.)
    """
    try:
        d_start = date_cls.fromisoformat(start)
        d_end = date_cls.fromisoformat(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"start/end must be ISO dates YYYY-MM-DD: {exc}") from exc
    if d_end < d_start:
        raise ValueError(f"end {end} is before start {start}")
    if (d_end - d_start).days > max_days:
        raise ValueError(
            f"range {(d_end - d_start).days}d exceeds {max_days}d cap — chunk the call"
        )


def _cache_key(endpoint: str, params: dict | None) -> str:
    """sha1(endpoint + sorted params) — LOCKED formula.

    Canonical string: endpoint + '?' + 'k=v' joined by '&' in sorted-key order
    (no '?...' suffix when there are no params). Deterministic regardless of
    dict insertion order.
    """
    raw = endpoint
    if params:
        raw += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class SectorsClient:
    """Cache-first HTTP client for Sectors REST v2 with credit-budget guarding."""

    BASE_URL = BASE_URL

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "cache-first",
        budget: CreditBudget | None = None,
        cache_dir: str | Path = "cache",
        timeout: float = 15.0,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.api_key = api_key if api_key is not None else os.environ.get("SECTORS_API_KEY", "")
        if mode != "offline" and not self.api_key:
            raise ValueError(
                "SECTORS_API_KEY missing — required for cache-first/live modes "
                "(offline mode never calls the API)"
            )
        self.mode = mode
        self.budget = budget if budget is not None else CreditBudget()
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.trace_log: list[str] = []  # §8 trace rows, appended by _fetch

    def reset_trace(self) -> None:
        self.trace_log = []

    def _trace(self, namespace: str, endpoint: str, tag: str) -> None:
        label = TRACE_LABELS.get(namespace, namespace)
        self.trace_log.append(f"{label}:{endpoint} [{tag}]")

    # ---------------------------------------------------------------- cache

    def _cache_path(self, namespace: str, endpoint: str, params: dict | None) -> Path:
        return self.cache_dir / namespace / f"{_cache_key(endpoint, params)}.json"

    def _read_cache(self, namespace: str, endpoint: str, params: dict | None):
        """Return cached payload or None. Envelope: {endpoint, params, fetched_at, data}."""
        path = self._cache_path(namespace, endpoint, params)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                envelope = json.load(fh)
            return envelope["data"]
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupt cache entry = miss (will be rewritten on next live 2xx).
            return None

    def _write_cache(self, namespace: str, endpoint: str, params: dict | None, data) -> None:
        path = self._cache_path(namespace, endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "endpoint": endpoint,
            "params": params or {},
            "fetched_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(timespec="seconds"),
            "data": data,
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(envelope, fh)
        os.replace(tmp, path)

    # ------------------------------------------------------------ live path

    def _fetch(self, namespace: str, endpoint: str, params: dict | None = None, cost: int = 1):
        """LOCKED order: cache check -> precheck -> GET -> charge -> save cache.

        - cache-first: hit returns immediately — 0 credits, no HTTP, guard never touched.
        - offline: miss raises OfflineCacheMissError, never spends.
        - live: always HTTP (still rewrites cache on 2xx).
        """
        if self.mode in ("cache-first", "offline"):
            cached = self._read_cache(namespace, endpoint, params)
            if cached is not None:
                self._trace(namespace, endpoint, "cache")
                return cached
            if self.mode == "offline":
                raise OfflineCacheMissError(
                    f"offline cache miss: {namespace} {endpoint} {params or {}}"
                )

        # Live path from here — the guard is only touched for live calls.
        self.budget.precheck(cost)  # raises RunCapExceeded / ReserveExceeded before any HTTP
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Authorization": self.api_key, "Accept": "application/json"}  # NO "Bearer"
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            # charge() is never called on network failure — 0 cr.
            self._trace(namespace, endpoint, "live, 0cr, conn-error")
            raise SectorsConnectionError(f"network failure on {endpoint}: {exc}") from exc

        billed = self.budget.charge(cost, resp.status_code, endpoint)  # every live call billed/logged here

        if 200 <= resp.status_code < 300:
            data = resp.json()
            self._write_cache(namespace, endpoint, params, data)  # 2xx only — no negative caching
            self._trace(namespace, endpoint, f"live, {billed}cr")
            return data
        if resp.status_code == 404:
            self._trace(namespace, endpoint, f"live, {billed}cr, 404")
            raise SectorsNotFoundError(f"404 on {endpoint} (1 cr billed) — honest-null signal")
        self._trace(namespace, endpoint, f"live, {billed}cr, http-{resp.status_code}")
        raise SectorsAPIError(
            f"HTTP {resp.status_code} on {endpoint} (0 cr billed)", resp.status_code
        )

    # ------------------------------------------------------------ endpoints

    @staticmethod
    def _optional_window(start: str | None, end: str | None, max_days: int) -> dict | None:
        """Build query params from optional start/end; None when both omitted (API defaults)."""
        if start is None and end is None:
            return None
        params: dict = {}
        if start is not None:
            params["start"] = _iso_date(start, "start")
        if end is not None:
            params["end"] = _iso_date(end, "end")
        if start is not None and end is not None:
            _check_window(params["start"], params["end"], max_days)
        return params

    def ohlcv(self, symbol: str, start: str, end: str) -> dict:
        """GET /v2/daily/{symbol}/ — cost 1. Range must be <= 90 days (§5).

        MA200 seeding needs 200+ bars ≈ 3 chunked calls per ticker (§5).
        Response is a bare array of {symbol, date, close, open, high, low, volume, market_cap}.
        """
        _check_window(_iso_date(start, "start"), _iso_date(end, "end"), MAX_OHLCV_DAYS)
        sym = _normalize_symbol(symbol)
        endpoint = f"/v2/daily/{sym}/"
        return self._fetch("ohlcv", endpoint, {"start": start, "end": end}, cost=1)

    def foreign_flow(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        """GET /v2/foreign-flow/{symbol}/ — cost 1.

        start/end optional (omitted -> API default: last 90 days ending today).
        Pass explicit dates for day-scoped cache keys (see module docstring).
        `symbol="IHSG"` returns the market-wide series (useful for regime context).
        """
        sym = _normalize_symbol(symbol)
        params = self._optional_window(start, end, MAX_FLOW_DAYS)
        return self._fetch("foreign_flow", f"/v2/foreign-flow/{sym}/", params, cost=1)

    def broker_summary(self, symbol: str, start: str | None = None, end: str | None = None) -> dict:
        """GET /v2/broker-summary/{symbol}/ — cost 1. 404 here = honest-null F5 (FR2).

        start/end optional (omitted -> API default: last 14 days ending today).
        Pass explicit dates for day-scoped cache keys (see module docstring).
        """
        sym = _normalize_symbol(symbol)
        params = self._optional_window(start, end, MAX_BROKER_DAYS)
        return self._fetch("broker_summary", f"/v2/broker-summary/{sym}/", params, cost=1)

    def fundamentals(self, symbol: str) -> dict:
        """GET /v2/company/report/{symbol}/ — cost 1."""
        sym = _normalize_symbol(symbol)
        return self._fetch("fundamentals", f"/v2/company/report/{sym}/", None, cost=1)

    def screener_structured(self, where: dict) -> dict:
        """GET /v2/companies/ with structured filters — cost 1."""
        params = {str(k): v for k, v in dict(where).items()}
        return self._fetch("screener", "/v2/companies/", params, cost=1)

    def screener_nl(self, q: str) -> dict:
        """GET /v2/companies/ natural-language screen — cost 3."""
        if not isinstance(q, str) or not q.strip():
            raise ValueError("screener_nl requires a non-empty query string")
        return self._fetch("screener", "/v2/companies/", {"q": q.strip()}, cost=3)

    def daily_close(self, day: str, limit: int | None = None, offset: int | None = None) -> dict:
        """GET /v2/close/ — full-universe close for one trading day — cost 1 PER PAGE.

        Docs: paginated feed, max 30 tickers/page; full ~950-ticker universe is
        ~32 pages (~32 cr). NOT suitable for daily watchlist refresh — per-symbol
        ohlcv is far cheaper for <= 12 names. Use for universe-level scans only.
        """
        params: dict = {"date": _iso_date(day, "day")}
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 30):
                raise ValueError("limit must be an int in 1..30 (per-page max)")
            params["limit"] = limit
        if offset is not None:
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise ValueError("offset must be a non-negative int")
            params["offset"] = offset
        return self._fetch("daily_close", "/v2/close/", params, cost=1)
