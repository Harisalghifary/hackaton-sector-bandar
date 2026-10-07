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


def _flow_summary(rows) -> dict:
    """Deterministic engine aggregates over the fetched flow window (O1).

    Signs are the information: net_foreign_inflow > 0 = foreign money ENTERING
    the stock (accumulation pressure); < 0 = money LEAVING (outflow/distribution).
    The engine computes these sums/counts here so synthesis can quote cumulative
    figures without deriving arithmetic itself (FR10: every figure traces).
    """
    nets = sorted(
        ((str(r.get("date")), r.get("net_foreign_inflow")) for r in rows or []
         if isinstance(r, dict)
         and isinstance(r.get("net_foreign_inflow"), (int, float))
         and not isinstance(r.get("net_foreign_inflow"), bool)),
        key=lambda t: t[0],
    )
    if not nets:
        return {}
    vals = [v for _, v in nets]
    total = sum(vals)
    inflow = [(d, v) for d, v in nets if v > 0]
    outflow = [(d, v) for d, v in nets if v < 0]
    summary = {
        "window_start": nets[0][0], "window_end": nets[-1][0], "n_days": len(nets),
        "net_total": total,                       # signed cumulative flow over window
        "net_last_5d": sum(vals[-5:]),
        "net_last_20d": sum(vals[-20:]),
        "inflow_days": len(inflow),               # accumulation sessions
        "outflow_days": len(outflow),             # distribution sessions
        "bias": ("net inflow" if total > 0 else
                 "net outflow" if total < 0 else "balanced"),
    }
    if inflow:
        summary["top_inflow_day"], summary["top_inflow_value"] = max(inflow, key=lambda t: t[1])
    if outflow:
        summary["top_outflow_day"], summary["top_outflow_value"] = min(outflow, key=lambda t: t[1])
    return summary


def tool_get_foreign_flow(args: dict, ctx: ToolContext) -> dict:
    flow = ctx.client.foreign_flow(args["symbol"])
    rows = flow if isinstance(flow, list) else (flow or {}).get("data") or []
    return {"symbol": args["symbol"], "flow": flow, "summary": _flow_summary(rows)}


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
