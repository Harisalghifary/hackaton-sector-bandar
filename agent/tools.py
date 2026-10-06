"""BANDAR agent tools (§8, [LOCKED]: exactly 7 tools).

| Tool                    | Credit cost (Sectors)          |
|-------------------------|--------------------------------|
| score_ticker(symbol)    | ~2 (0 on warm cache)           |
| rank_watchlist()        | 2×N                            |
| score_history(symbol)   | 0                              |
| get_fundamentals(symbol)| 1                              |
| get_foreign_flow(symbol)| 1                              |
| screen(where|q)         | 1 / 3                          |
| watchlist_rw(op,list)   | 0                              |

Tools never raise to the LLM loop: run_tool wraps everything into
{"tool", "ok", "data"|"error"|"message", "trace"} — budget breaches abort the
step with an honest narration (planner policy 4), 404s stay honest-null (FR2),
and every data line carries its [cache]/[live, Ncr] trace (§8, FR10 input).
"""

from __future__ import annotations

import engine
from budget import BudgetError
from client import SectorsError, SectorsNotFoundError
from memory import ScoreMemory

TOOL_NAMES = ("score_ticker", "rank_watchlist", "score_history", "get_fundamentals",
              "get_foreign_flow", "screen", "watchlist_rw")


class ToolContext:
    """Shared per-run context. Scratchpad is ephemeral (§7) — a plain dict."""

    def __init__(self, client, memory: ScoreMemory, budget=None,
                 as_of=None, equity_idr: int | None = None):
        self.client = client
        self.memory = memory
        self.budget = budget if budget is not None else client.budget
        self.as_of = as_of
        self.equity_idr = equity_idr
        self.scratchpad: dict = {}   # ephemeral run state, never persisted


# ------------------------------------------------------------------ tool bodies

def tool_score_ticker(args: dict, ctx: ToolContext) -> dict:
    return engine.score_ticker(args["symbol"], client=ctx.client,
                               as_of=ctx.as_of, equity_idr=ctx.equity_idr)


def tool_rank_watchlist(args: dict, ctx: ToolContext) -> dict:
    symbols = ctx.memory.load_watchlist()
    results = [engine.score_ticker(s, client=ctx.client, as_of=ctx.as_of,
                                   equity_idr=ctx.equity_idr) for s in symbols]
    ranked = sorted(results, key=lambda r: (-(r["score"] or 0), r["symbol"]))
    return {"watchlist": symbols, "ranked": ranked}


def tool_score_history(args: dict, ctx: ToolContext) -> dict:
    limit = args.get("limit", 10)
    return {"symbol": args["symbol"], "history": ctx.memory.get_history(args["symbol"], limit=limit)}


def tool_get_fundamentals(args: dict, ctx: ToolContext) -> dict:
    return {"symbol": args["symbol"], "report": ctx.client.fundamentals(args["symbol"])}


def tool_get_foreign_flow(args: dict, ctx: ToolContext) -> dict:
    return {"symbol": args["symbol"], "flow": ctx.client.foreign_flow(args["symbol"])}


def tool_screen(args: dict, ctx: ToolContext) -> dict:
    if "q" in args:                       # NL screen: 3 cr
        return {"mode": "nl", "q": args["q"], "results": ctx.client.screener_nl(args["q"])}
    return {"mode": "structured", "where": args["where"],
            "results": ctx.client.screener_structured(args["where"])}   # 1 cr


def tool_watchlist_rw(args: dict, ctx: ToolContext) -> dict:
    symbols = ctx.memory.update_watchlist(args["op"], args["list"])
    return {"op": args["op"], "watchlist": symbols}


TOOLS = {
    "score_ticker": tool_score_ticker,
    "rank_watchlist": tool_rank_watchlist,
    "score_history": tool_score_history,
    "get_fundamentals": tool_get_fundamentals,
    "get_foreign_flow": tool_get_foreign_flow,
    "screen": tool_screen,
    "watchlist_rw": tool_watchlist_rw,
}


# ------------------------------------------------------------------ safe runner

def run_tool(name: str, args: dict | None, ctx: ToolContext) -> dict:
    """Execute one plan step. Never raises — honest structured errors instead."""
    if name not in TOOLS:
        return {"tool": name, "ok": False, "error": "unknown_tool",
                "message": f"no such tool: {name}", "trace": []}
    trace_start = len(ctx.client.trace_log)
    try:
        data = TOOLS[name](args or {}, ctx)
        return {"tool": name, "ok": True, "data": data,
                "trace": list(ctx.client.trace_log[trace_start:])}
    except BudgetError as exc:
        # Planner policy 4: budget breach -> abort step, narrate honestly.
        return {"tool": name, "ok": False, "error": "budget_abort", "message": str(exc),
                "trace": list(ctx.client.trace_log[trace_start:])}
    except SectorsNotFoundError as exc:
        return {"tool": name, "ok": False, "error": "not_found", "message": str(exc),
                "trace": list(ctx.client.trace_log[trace_start:])}
    except SectorsError as exc:
        return {"tool": name, "ok": False, "error": "data_error", "message": str(exc),
                "trace": list(ctx.client.trace_log[trace_start:])}
    except (ValueError, KeyError) as exc:
        return {"tool": name, "ok": False, "error": "bad_args", "message": str(exc),
                "trace": list(ctx.client.trace_log[trace_start:])}
