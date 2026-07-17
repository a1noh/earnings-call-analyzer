"""Quarter-over-quarter comparator node.

Compares the current quarter against the immediately prior indexed quarter, pulling
grounded excerpts from BOTH quarters so the model can point to what changed. If no
prior quarter is indexed, it honestly reports that no comparison basis exists rather
than fabricating one.
"""
from __future__ import annotations

from agent import rag
from agent.nodes.common import BASE_RULES, reasoning_step
from ingest import vectorstore
from llm import claude
from llm.schemas import QOQ_SCHEMA

QUERIES = [
    "revenue growth margin change versus prior quarter",
    "guidance outlook change sequentially quarter over quarter",
    "improved worsened accelerated decelerated compared previous",
]

NODE_LABEL = "Quarter-over-Quarter Comparator"


def _prior_quarter(ticker: str, year: int, quarter: str):
    """Return the (year, quarter) immediately preceding the current one, if indexed."""
    collection = vectorstore.get_collection(ticker)
    quarters = vectorstore.indexed_quarters(collection, ticker)  # sorted desc
    current = (year, quarter)
    if current not in quarters:
        return None
    idx = quarters.index(current)
    return quarters[idx + 1] if idx + 1 < len(quarters) else None


def qoq_node(state):
    """Return {qoq, reasoning, evidence} comparing the quarter to its predecessor."""
    ticker, year, quarter = state["ticker"], state["year"], state["quarter"]
    prior = _prior_quarter(ticker, year, quarter)

    if prior is None:
        result = {
            "comparison_available": False,
            "current_quarter": f"{quarter} {year}",
            "prior_quarter": "none indexed",
            "metric_changes": [],
            "narrative_shift": "No prior quarter is available for comparison.",
        }
        step = reasoning_step(NODE_LABEL, QUERIES, [], "No prior quarter to compare.")
        return {"qoq": result, "reasoning": [step], "evidence": []}

    prior_year, prior_q = prior
    cur_chunks = rag.retrieve(ticker, year, quarter, QUERIES)
    prior_chunks = rag.retrieve(ticker, prior_year, prior_q, QUERIES)

    context = (
        rag.format_context(cur_chunks, header=f"CURRENT QUARTER ({quarter} {year})")
        + "\n\n"
        + rag.format_context(prior_chunks, header=f"PRIOR QUARTER ({prior_q} {prior_year})")
    )
    instruction = (
        f"{BASE_RULES}\n\n"
        f"Compare {quarter} {year} against {prior_q} {prior_year}. Identify what "
        "changed meaningfully — metrics, guidance, and narrative/tone. Each metric "
        "change MUST cite a verbatim quote from the current quarter (quote_current) "
        "AND one from the prior quarter (quote_prior). Set comparison_available true. "
        "narrative_shift should summarize the most important change in one to three "
        "sentences, referencing both quarters."
    )
    evidence = [c.text for c in cur_chunks] + [c.text for c in prior_chunks]

    try:
        result = claude.structured_call(context, instruction, QOQ_SCHEMA)
        n = len(result.get("metric_changes", []) or [])
        summary = f"Compared vs {prior_q} {prior_year}; {n} metric change(s)."
    except claude.ClaudeError as exc:
        step = reasoning_step(NODE_LABEL, QUERIES, cur_chunks, f"error: {exc}")
        return {
            "qoq": {},
            "reasoning": [step],
            "evidence": evidence,
            "errors": [f"{NODE_LABEL}: {exc}"],
        }

    step = reasoning_step(
        NODE_LABEL, QUERIES, cur_chunks + prior_chunks, summary
    )
    return {"qoq": result, "reasoning": [step], "evidence": evidence}
