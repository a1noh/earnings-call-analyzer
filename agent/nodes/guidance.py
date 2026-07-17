"""Revenue guidance extractor node.

Pulls specific revenue/EPS guidance numbers, the guidance direction, and
forward-looking statements — each backed by a verbatim quote.
"""
from __future__ import annotations

from agent.nodes.common import run_node
from llm.schemas import GUIDANCE_SCHEMA

QUERIES = [
    "revenue guidance outlook next quarter full year forecast",
    "earnings per share EPS operating margin guidance",
    "management expects forward-looking outlook full-year",
]

INSTRUCTION = (
    "Extract management's forward guidance from this earnings call:\n"
    "- revenue_guidance: specific revenue figures/percentages given, or 'not disclosed'.\n"
    "- eps_guidance: specific EPS or margin guidance, or 'not disclosed'.\n"
    "- direction: whether guidance was raised, maintained, lowered, or not_provided "
    "vs. prior expectations mentioned in the excerpts.\n"
    "- forward_statements: notable forward-looking statements, each with a verbatim quote."
)


def _summary(result: dict) -> str:
    direction = result.get("direction", "unknown")
    n = len(result.get("forward_statements", []) or [])
    return f"Guidance direction: {direction}; {n} forward-looking statement(s)."


def guidance_node(state):
    """Return {guidance, reasoning, evidence} for the current quarter."""
    return run_node(
        state,
        node_label="Revenue Guidance Extractor",
        result_key="guidance",
        queries=QUERIES,
        instruction=INSTRUCTION,
        schema=GUIDANCE_SCHEMA,
        summarize=_summary,
    )
