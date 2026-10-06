"""Code-enforced validators (§8, [LOCKED] enforce list; FR10, FR12).

plan_validator: known tools · schema-valid args · watchlist symbols only.
numeric_trace_validator: every figure in synthesis output must appear in the
tool-result trace; else dropped/flagged (lands with the M4b executor per §10).

The symbol restriction applies to per-symbol data tools (score_ticker,
score_history, get_fundamentals, get_foreign_flow). screen queries the universe
by design, and watchlist_rw must be able to ADD new names — both unrestricted.
"""

from __future__ import annotations

from client import _normalize_symbol

# Arg schemas per tool (§8 signatures). "symbol_restricted" => watchlist only.
ARG_RULES: dict[str, dict] = {
    "score_ticker":     {"required": {"symbol": str}, "symbol_restricted": True},
    "rank_watchlist":   {"required": {}},
    "score_history":    {"required": {"symbol": str}, "optional": {"limit": int},
                         "symbol_restricted": True},
    "get_fundamentals": {"required": {"symbol": str}, "symbol_restricted": True},
    "get_foreign_flow": {"required": {"symbol": str}, "symbol_restricted": True},
    "screen":           {"one_of": {"where": dict, "q": str}},
    "watchlist_rw":     {"required": {"op": str, "list": list},
                         "enums": {"op": {"add", "remove", "replace"}}},
}


class PlanValidationError(Exception):
    """Plan rejected by the code-enforced validator (never sent to executor)."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _check_args(tool: str, args, watchlist_norm: set[str], where: str) -> list[str]:
    errors: list[str] = []
    rules = ARG_RULES[tool]

    if not isinstance(args, dict):
        return [f"{where}: args must be an object, got {type(args).__name__}"]

    if "one_of" in rules:
        present = [k for k in rules["one_of"] if k in args]
        if len(present) != 1:
            errors.append(f"{where}: exactly one of {sorted(rules['one_of'])} required, "
                          f"got {present or 'none'}")
        for key in present:
            if not isinstance(args[key], rules["one_of"][key]):
                errors.append(f"{where}: '{key}' must be {rules['one_of'][key].__name__}")
        extra = set(args) - set(rules["one_of"])
        if extra:
            errors.append(f"{where}: unexpected args {sorted(extra)}")
        return errors

    for key, typ in rules.get("required", {}).items():
        if key not in args:
            errors.append(f"{where}: missing required arg '{key}'")
        elif not isinstance(args[key], typ) or isinstance(args[key], bool):
            errors.append(f"{where}: arg '{key}' must be {typ.__name__}")
    for key, typ in rules.get("optional", {}).items():
        if key in args and (not isinstance(args[key], typ) or isinstance(args[key], bool)):
            errors.append(f"{where}: arg '{key}' must be {typ.__name__}")
    for key, allowed in rules.get("enums", {}).items():
        if key in args and args[key] not in allowed:
            errors.append(f"{where}: arg '{key}' must be one of {sorted(allowed)}")
    unexpected = set(args) - set(rules.get("required", {})) - set(rules.get("optional", {}))
    if unexpected:
        errors.append(f"{where}: unexpected args {sorted(unexpected)}")

    if rules.get("symbol_restricted") and isinstance(args.get("symbol"), str):
        sym = _normalize_symbol(args["symbol"]) if args["symbol"].strip() else ""
        if sym not in watchlist_norm:
            errors.append(f"{where}: symbol '{args['symbol']}' is not on the watchlist "
                          f"({sorted(watchlist_norm)}) — watchlist symbols only (§8)")
    return errors


def validate_plan(plan, watchlist: list[str]) -> list[str]:
    """Return a list of violation reasons — empty list means the plan is valid."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return ["plan.steps must be an array"]
    if not steps:
        errors.append("plan.steps is empty — nothing to execute")

    watchlist_norm = {_normalize_symbol(s) for s in watchlist}
    for i, step in enumerate(steps):
        where = f"steps[{i}]"
        if not isinstance(step, dict):
            errors.append(f"{where}: step must be an object")
            continue
        for key in ("tool", "args", "reason"):
            if key not in step:
                errors.append(f"{where}: missing '{key}'")
        if "reason" in step and (not isinstance(step["reason"], str) or not step["reason"].strip()):
            errors.append(f"{where}: 'reason' must be a non-empty string")
        tool = step.get("tool")
        if tool not in ARG_RULES:
            errors.append(f"{where}: unknown tool {tool!r} — known: {sorted(ARG_RULES)}")
            continue
        errors.extend(_check_args(tool, step.get("args"), watchlist_norm, where))
    return errors


def validate_plan_or_raise(plan, watchlist: list[str]) -> dict:
    errors = validate_plan(plan, watchlist)
    if errors:
        raise PlanValidationError(errors)
    return plan
