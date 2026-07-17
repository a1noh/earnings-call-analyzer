"""Anthropic Claude wrapper for structured, grounded analysis calls.

Design goals:
* **One client**, reused across all nodes.
* **Structured output** via ``output_config.format`` when the installed SDK/model
  supports it, with a robust prompt-JSON fallback so the app still works on older
  SDK builds instead of crashing on an unknown kwarg or a 400.
* **Prompt caching** of the shared per-quarter transcript context, so nodes 2-4
  reuse the cached prefix (~0.1x input cost).
* **Refusal-safe**: always check ``stop_reason`` before reading content.
"""
from __future__ import annotations

import json
from typing import Optional

import anthropic

from config import ANALYSIS_MODEL, MAX_TOKENS

_client: Optional[anthropic.Anthropic] = None


class ClaudeError(Exception):
    """Raised when Claude cannot return usable structured output."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Reads ANTHROPIC_API_KEY from the environment.
        _client = anthropic.Anthropic()
    return _client


def _json_from_text(text: str) -> dict:
    """Extract a JSON object from model text, tolerating code fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ClaudeError("Model did not return valid JSON.")


def _text_from_response(resp) -> str:
    if getattr(resp, "stop_reason", None) == "refusal":
        raise ClaudeError("Claude declined to answer this request (safety refusal).")
    parts = [
        block.text
        for block in resp.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts)


def structured_call(
    system_context: str,
    instruction: str,
    schema: dict,
    model: str = ANALYSIS_MODEL,
) -> dict:
    """Run a grounded analysis call and return a parsed JSON dict.

    Args:
        system_context: Shared grounding text (retrieved chunks). Cached.
        instruction: The node-specific task and rules.
        schema: JSON schema the output must satisfy.
        model: Claude model id.

    Returns:
        The parsed JSON object.

    Raises:
        ClaudeError: on refusal or unparseable output.
    """
    client = _get_client()

    system_blocks = [
        {
            "type": "text",
            "text": system_context,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # The schema is embedded in the prompt so the fallback path also produces JSON.
    user = (
        f"{instruction}\n\n"
        "Respond with ONLY a single JSON object (no prose, no code fences) that "
        "conforms to this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    messages = [{"role": "user", "content": user}]

    try:
        # Tier 1: native structured outputs (best fidelity when supported).
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_blocks,
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            return _json_from_text(_text_from_response(resp))
        except TypeError:
            pass  # installed SDK doesn't accept output_config
        except anthropic.BadRequestError:
            pass  # model/endpoint rejected structured outputs; fall back

        # Tier 2: plain call, rely on the in-prompt schema instruction.
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=messages,
        )
        return _json_from_text(_text_from_response(resp))
    except ClaudeError:
        raise
    except anthropic.APIError as exc:
        # Auth, not-found, rate-limit, overloaded, etc.
        raise ClaudeError(f"Claude API error: {exc}") from exc
    except Exception as exc:
        # Anything else (e.g. a non-ASCII character in the API key breaking
        # header encoding) -> handled ClaudeError so the graph degrades
        # gracefully instead of crashing the whole app.
        raise ClaudeError(f"Claude request failed: {exc}") from exc
