"""Risk factor detector node.

Identifies risks management raised, flags which are new or escalating, assigns a
severity, and attaches a verbatim quote to each.
"""
from __future__ import annotations

from agent.nodes.common import run_node
from llm.schemas import RISK_SCHEMA

QUERIES = [
    "risk uncertainty headwind challenge macroeconomic pressure",
    "supply chain demand weakness competition regulatory",
    "caution concern impact softness decline slowdown",
]

INSTRUCTION = (
    "Identify the risks and headwinds management discussed on this call.\n"
    "For each: state the risk, assign severity (low/medium/high) based on the "
    "language used, mark is_new_or_escalating true if the excerpts frame it as new "
    "or worsening, and include a verbatim quote. Only include risks actually "
    "mentioned in the excerpts."
)


def _summary(result: dict) -> str:
    risks = result.get("risks", []) or []
    escalating = sum(1 for r in risks if r.get("is_new_or_escalating"))
    return f"{len(risks)} risk(s) detected; {escalating} new/escalating."


def risk_node(state):
    """Return {risks, reasoning, evidence} for the current quarter."""
    return run_node(
        state,
        node_label="Risk Factor Detector",
        result_key="risks",
        queries=QUERIES,
        instruction=INSTRUCTION,
        schema=RISK_SCHEMA,
        summarize=_summary,
    )
