"""LangGraph wiring for the analysis pipeline.

Builds a sequential StateGraph:

    START -> guidance -> risk -> sentiment -> qoq -> evaluator -> END

Sequential order lets the UI stream each node's completion as a live progress step
(not a spinner) and lets the evaluator score the accumulated results at the end.
"""
from __future__ import annotations

from typing import Iterator

from langgraph.graph import END, START, StateGraph

from agent.nodes.evaluator import evaluator_node
from agent.nodes.guidance import guidance_node
from agent.nodes.qoq import qoq_node
from agent.nodes.risk import risk_node
from agent.nodes.sentiment import sentiment_node
from agent.state import AnalyzerState

# Human-readable labels for each graph node (for the UI progress stream).
# Node ids intentionally differ from state keys (LangGraph forbids collisions).
NODE_LABELS = {
    "extract_guidance": "Revenue Guidance Extractor",
    "detect_risks": "Risk Factor Detector",
    "analyze_sentiment": "Sentiment Analyzer",
    "compare_qoq": "Quarter-over-Quarter Comparator",
    "evaluate": "Eval Scorer",
}

# The order nodes execute in — used by the UI to render a live checklist.
NODE_ORDER = list(NODE_LABELS.keys())


def build_graph():
    """Compile and return the analysis StateGraph."""
    g = StateGraph(AnalyzerState)
    g.add_node("extract_guidance", guidance_node)
    g.add_node("detect_risks", risk_node)
    g.add_node("analyze_sentiment", sentiment_node)
    g.add_node("compare_qoq", qoq_node)
    g.add_node("evaluate", evaluator_node)

    g.add_edge(START, "extract_guidance")
    g.add_edge("extract_guidance", "detect_risks")
    g.add_edge("detect_risks", "analyze_sentiment")
    g.add_edge("analyze_sentiment", "compare_qoq")
    g.add_edge("compare_qoq", "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


# Compile once at import time (nodes/tools are stateless).
GRAPH = build_graph()


def stream_analysis(
    ticker: str, year: int, quarter: str, source: str, source_label: str
) -> Iterator[tuple[str, dict]]:
    """Run the graph, yielding (node_name, cumulative_state) after each node.

    The caller (Streamlit) uses this to render step-by-step progress and, at the
    end, the final results + eval. State accumulates via the graph's reducers.
    """
    initial: AnalyzerState = {
        "ticker": ticker.upper(),
        "year": year,
        "quarter": quarter,
        "source": source,
        "source_label": source_label,
        "reasoning": [],
        "evidence": [],
        "errors": [],
    }
    accumulated: dict = dict(initial)
    for update in GRAPH.stream(initial, stream_mode="updates"):
        # `update` is {node_name: partial_state}; merge into our running copy.
        for node_name, partial in update.items():
            for key, value in (partial or {}).items():
                if key in ("reasoning", "evidence", "errors"):
                    accumulated[key] = accumulated.get(key, []) + value
                else:
                    accumulated[key] = value
            yield node_name, dict(accumulated)


def run_analysis(ticker: str, year: int, quarter: str, source: str, source_label: str) -> dict:
    """Run the full graph and return the final accumulated state (no streaming)."""
    final: dict = {}
    for _, state in stream_analysis(ticker, year, quarter, source, source_label):
        final = state
    return final
