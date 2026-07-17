"""Shared LangGraph state for the analysis pipeline.

Keys annotated with an ``operator.add`` reducer accumulate across nodes (reasoning
steps, retrieval evidence, errors); all other keys are last-write-wins. ``total=False``
lets each node return only the slice of state it produces.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AnalyzerState(TypedDict, total=False):
    # Inputs / context
    ticker: str
    year: int
    quarter: str
    source: str
    source_label: str

    # Per-node results
    guidance: dict[str, Any]
    risks: dict[str, Any]
    sentiment: dict[str, Any]
    qoq: dict[str, Any]
    eval: dict[str, Any]

    # Accumulated across nodes
    reasoning: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
