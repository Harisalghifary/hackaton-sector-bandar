"""Code-enforced validators (§8, [LOCKED] enforce list; FR10, FR12).

plan_validator: known tools · schema-valid args · watchlist symbols only.
numeric_trace_validator: every figure in synthesis output must appear in the
tool-result trace; else dropped/flagged (lands with the M4b executor per §10).

The symbol restriction applies to per-symbol data tools (score_ticker,
score_history, get_fundamentals, get_foreign_flow). screen queries the universe
by design, and watchlist_rw must be able to ADD new names — both unrestricted.
"""

from __future__ import annotations

import re

from client import _normalize_symbol

# Numeric token: digits with optional thousands separators / decimal part.
NUM_TOKEN_RE = re.compile(r"\d[\d,]*\.?\d*")

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


# ------------------------------------------------- numeric trace validator (FR10)
# §8: "every figure in output must appear in tool-result trace; else dropped/flagged"
# §14 non-negotiable: every output number traces to a tool result.
# §8 voice: no derived arithmetic. Unit reformatting is allowed ONLY through the
# deterministic M/B/T alias set below (computed from real corpus values), so a
# human-readable "Rp 152.1B" still traces to 152110100000 in the tool results.


def _canonical_number_forms(value) -> set[str]:
    """All exact textual forms a JSON numeric value may be quoted as.

    Besides exact forms, large values (|v| >= 1e6) also allow deterministic
    human-unit aliases (M/B/T, 1-2 decimals, integer form when |scaled| >= 10)
    in BOTH signs, so synthesis may write "Rp 152.1B", "Rp 85B", or quote a
    distribution day's magnitude as "Rp 85.4B" for -85432100000. The numeric
    token regex strips signs, so unsigned forms must be allowed too. Every
    alias is COMPUTED from a real corpus value, so FR10 still holds: an
    allowed token always maps back to a tool result — nothing can be invented
    through the alias set. Values below 1e6 get no aliases (keeps small
    integers like scores/lots from leaking generic tokens such as "0.23").
    """
    forms: set[str] = set()
    if isinstance(value, bool):
        return forms

    def add(text: str) -> None:
        forms.add(text)
        forms.add(text.lstrip("-"))      # NUM_TOKEN_RE strips the sign from tokens

    if isinstance(value, int):
        add(str(value))
    elif isinstance(value, float):
        r = round(value, 4)
        add(repr(r))
        if r == int(r):
            add(str(int(r)))             # 3120.0 quotable as "3120"
    if isinstance(value, (int, float)) and abs(value) >= 1_000_000:
        for scale in (1_000_000, 1_000_000_000, 1_000_000_000_000):
            s = value / scale
            if not 0.1 <= abs(s) < 1000:     # natural range for that unit
                continue
            for nd in (1, 2):
                add(str(round(s, nd)))
                add(f"{s:.{nd}f}")
            if abs(s) >= 10:
                add(str(int(round(s))))      # "Rp 85B" style
    return forms


def _string_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in NUM_TOKEN_RE.findall(text):
        out.add(tok)
        stripped = tok.replace(",", "")
        out.add(stripped)
        if "." not in stripped:
            out.add(str(int(stripped)))  # "05" (dates) quotable as "5"
    return out


def _walk_corpus(node, allowed: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            allowed.update(_string_tokens(str(k)))
            _walk_corpus(v, allowed)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk_corpus(item, allowed)
    elif isinstance(node, str):
        allowed.update(_string_tokens(node))
    elif isinstance(node, (int, float)):
        allowed.update(_canonical_number_forms(node))


def collect_allowed_numbers(corpus) -> set[str]:
    """Every exact numeric token appearing in tool results / memory context."""
    allowed: set[str] = set()
    _walk_corpus(corpus, allowed)
    return allowed


def _token_ok(token: str, allowed: set[str]) -> bool:
    return token in allowed or token.replace(",", "") in allowed


def validate_brief_numbers(brief: dict, allowed: set[str]) -> tuple[dict, list[dict]]:
    """Enforce FR10 on a submit_brief: figures not in the corpus are dropped/flagged.

    - extracted / action_plan / risk_flags: whole items containing unverifiable
      figures are DROPPED (recorded in the returned list — honest narration).
    - interpretation: unverifiable figure tokens are replaced with "[removed]"
      and listed in interpretation_flag (FLAGGED, never silently kept).
    """
    cleaned = dict(brief)
    dropped: list[dict] = []

    for section in ("extracted", "action_plan", "risk_flags"):
        kept = []
        for item in cleaned.get(section) or []:
            text = str(item)
            bad = sorted({t for t in NUM_TOKEN_RE.findall(text) if not _token_ok(t, allowed)})
            if bad:
                dropped.append({"section": section, "item": text, "figures": bad})
            else:
                kept.append(item)
        cleaned[section] = kept

    interp = str(cleaned.get("interpretation") or "")
    bad_interp = sorted({t for t in NUM_TOKEN_RE.findall(interp) if not _token_ok(t, allowed)})
    if bad_interp:
        cleaned["interpretation"] = NUM_TOKEN_RE.sub(
            lambda m: m.group(0) if _token_ok(m.group(0), allowed) else "[removed]", interp)
        cleaned["interpretation_flag"] = bad_interp
        dropped.append({"section": "interpretation", "figures": bad_interp})
    return cleaned, dropped
