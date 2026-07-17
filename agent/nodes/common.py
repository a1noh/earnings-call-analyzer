"""Shared helpers for analysis nodes.

Keeps each node focused on its prompt + schema by centralizing grounding, the
Claude call with uniform error handling, and reasoning-step construction. Each
node returns a partial ``AnalyzerState`` update.
"""
from __future__ import annotations

from typing import Any

from agent import rag
from data.models import RetrievedChunk
from llm import claude

# Base rules shared by every analysis node — enforces grounding and honesty.
BASE_RULES = (
    "You are a meticulous equity research analyst. Analyze ONLY the transcript "
    "excerpts provided. Every claim must be supported by a verbatim quote copied "
    "exactly from the excerpts. If the excerpts do not contain the information, say "
    "so explicitly (e.g. 'not disclosed') rather than guessing. Do not use outside "
    "knowledge."
)


def reasoning_step(
    node_label: str, queries: list[str], chunks: list[RetrievedChunk], summary: str
) -> dict[str, Any]:
    """Build a visible reasoning step for the UI."""
    return {
        "node": node_label,
        "queries": queries,
        "sections_used": sorted({c.section for c in chunks}),
        "num_chunks": len(chunks),
        "summary": summary,
    }


def run_node(
    state,
    *,
    node_label: str,
    result_key: str,
    queries: list[str],
    instruction: str,
    schema: dict,
    summarize,
) -> dict[str, Any]:
    """Ground, call Claude, and package a partial state update.

    Args:
        summarize: callable(result_dict) -> short summary string for the UI.
    """
    ticker, year, quarter = state["ticker"], state["year"], state["quarter"]
    chunks = rag.retrieve(ticker, year, quarter, queries)
    context = rag.format_context(chunks)
    full_instruction = f"{BASE_RULES}\n\n{instruction}"
    evidence = [c.text for c in chunks]

    try:
        result = claude.structured_call(context, full_instruction, schema)
        summary = summarize(result)
    except claude.ClaudeError as exc:
        step = reasoning_step(node_label, queries, chunks, f"error: {exc}")
        return {
            result_key: {},
            "reasoning": [step],
            "evidence": evidence,
            "errors": [f"{node_label}: {exc}"],
        }

    step = reasoning_step(node_label, queries, chunks, summary)
    return {result_key: result, "reasoning": [step], "evidence": evidence}
