"""Minimal LLM transport (§3, D8): Gemini controlled generation primary,
Claude hot-swap fallback — plain requests.post, no SDK, no framework.

Gemini: response_mime_type="application/json" + response_schema (§3 LOCKED).
Claude: no native JSON-schema enforcement -> the schema is embedded in the system
prompt and output parsed as JSON (same contract for callers).

Tests mock this module's requests.post — no live LLM calls in CI.
"""

from __future__ import annotations

import json
import os
import time

import requests

from agent.config import RUNTIME_LLM

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_VERSION = "2023-06-01"

# Spec (D8) model names -> real provider API ids. RUNTIME_LLM stays byte-for-byte
# as the LOCKED §3 block; this map is a transport-layer naming translation only,
# so the config-only hot swap (D8) is unaffected.
# - gemini-3.8-flash verified against ListModels on 2026-10-06 ("Gemini 3.8 Flash").
# - claude-sonnet-4-5 STILL UNVERIFIED: an Anthropic key exists (2026-10-06) but the
#   account has zero credit balance — HTTP 400 arrives before model resolution, so
#   the id could not be confirmed. Re-verify once the account is funded. The D8
#   degradation path itself was validated live: 400 -> LLMError -> honest fallback.
MODEL_IDS = {
    "gemini-flash-3.8": "gemini-3.8-flash",
    "claude-sonnet-4.5": "claude-sonnet-4-5",
}


def resolve_model(spec_name: str) -> str:
    """Spec name -> provider API id; unknown names pass through unchanged."""
    return MODEL_IDS.get(spec_name, spec_name)


# Transient-failure retry: a 429/5xx transport attempt never produced a completion,
# so retrying does not consume a §3 "LLM call" slot. Bounded: 3 POSTs max per call.
RETRY_STATUSES = (429, 500, 502, 503)
# Longer backoff so a transient 503 "high demand" spike can clear before we give up.
# These retries never produced a completion, so they do NOT consume a §3 LLM-call slot.
RETRY_WAITS = [0, 5, 15, 30]


def _is_quota(resp) -> bool:
    """A 429 that is a hard quota/billing limit (not transient rate-limit)."""
    return resp is not None and resp.status_code == 429 \
        and "quota" in (resp.text or "").lower()


def _post_with_retry(url: str, headers: dict, body: dict, timeout: float):
    resp = None
    for wait in RETRY_WAITS:
        if wait:
            time.sleep(wait)
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code not in RETRY_STATUSES:
            return resp
        if _is_quota(resp):
            return resp          # quota/billing limit: retrying cannot help — fail fast
    return resp


class LLMError(Exception):
    """Transport/parse failure. Executor converts to the §8 fallback schema."""


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 temperature: float | None = None, timeout: float = 60.0):
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model = model or RUNTIME_LLM["primary"]        # spec name (LOCKED)
        self.model_id = resolve_model(self.model)           # real API id
        self.temperature = RUNTIME_LLM["temp"] if temperature is None else temperature
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY missing")

    def complete(self, system: str, user: str, schema: dict | None = None,
                 purpose: str = "") -> dict:
        """One controlled-generation call -> parsed JSON dict (§3: max 3 per run)."""
        generation_config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
        }
        if schema is not None:
            generation_config["response_schema"] = schema
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        resp = _post_with_retry(
            GEMINI_URL.format(model=self.model_id),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            body=body, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise LLMError(f"gemini HTTP {resp.status_code} ({purpose}): {resp.text[:300]}")
        return _parse_gemini(resp.json(), purpose)


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 temperature: float | None = None, timeout: float = 90.0,
                 max_tokens: int = 4096):
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or RUNTIME_LLM["fallback"]       # spec name (LOCKED)
        self.model_id = resolve_model(self.model)           # real API id
        self.temperature = RUNTIME_LLM["temp"] if temperature is None else temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY missing")

    def complete(self, system: str, user: str, schema: dict | None = None,
                 purpose: str = "") -> dict:
        if schema is not None:
            system += ("\n\nRespond with ONLY a single JSON object conforming to this"
                       f" JSON Schema (no prose, no markdown):\n{json.dumps(schema)}")
        body = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = _post_with_retry(
            CLAUDE_URL,
            headers={"x-api-key": self.api_key, "anthropic-version": CLAUDE_VERSION,
                     "Content-Type": "application/json"},
            body=body, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise LLMError(f"claude HTTP {resp.status_code} ({purpose}): {resp.text[:300]}")
        try:
            text = resp.json()["content"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"claude response parse failure ({purpose}): {exc}") from exc


def _parse_gemini(payload: dict, purpose: str) -> dict:
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        raise LLMError(f"gemini response parse failure ({purpose}): {exc}") from exc


def make_llm(which: str = "primary", **kwargs):
    """Config-only hot swap (D8): which='primary'|'fallback' -> matching transport."""
    model = RUNTIME_LLM[which]
    if model.startswith("gemini"):
        return GeminiLLM(model=model, **kwargs)
    return ClaudeLLM(model=model, **kwargs)
