"""Sentiment analyzer node.

Scores management's tone as bullish, neutral, or cautious with a numeric score and
evidence-backed drivers.
"""
from __future__ import annotations

from agent.nodes.common import run_node
from llm.schemas import SENTIMENT_SCHEMA

QUERIES = [
    "management confidence optimism strong momentum record growth",
    "tone outlook pleased excited disappointed cautious",
    "demand trends performance expectations sentiment",
]

INSTRUCTION = (
    "Assess management's overall tone on this call.\n"
    "- label: bullish, neutral, or cautious.\n"
    "- score: a number from -1.0 (very cautious) to 1.0 (very bullish).\n"
    "- drivers: the specific statements driving your read, each with a verbatim quote.\n"
    "- rationale: one or two sentences explaining the classification."
)


def _summary(result: dict) -> str:
    label = result.get("label", "unknown")
    score = result.get("score", "?")
    return f"Tone: {label} (score {score})."


def sentiment_node(state):
    """Return {sentiment, reasoning, evidence} for the current quarter."""
    return run_node(
        state,
        node_label="Sentiment Analyzer",
        result_key="sentiment",
        queries=QUERIES,
        instruction=INSTRUCTION,
        schema=SENTIMENT_SCHEMA,
        summarize=_summary,
    )
