"""Eval node: score the analysis on groundedness, completeness, and consistency.

* **Groundedness (rule-based):** every model quote must appear (normalized) in the
  retrieved evidence — catches fabricated quotes deterministically and for free.
* **Completeness (rule-based):** did all four analysis dimensions produce output?
* **Consistency (rule-based):** does QoQ reference both quarters, and does sentiment
  agree with guidance direction?
* **Groundedness judge (LLM, optional):** a cheap Haiku pass; skipped silently on error.

Runs last in the graph and writes ``state['eval']``.
"""
from __future__ import annotations

from statistics import mean

from config import JUDGE_MODEL
from llm import claude
from llm.schemas import GROUNDEDNESS_JUDGE_SCHEMA


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _collect_quotes(state) -> dict[str, list[str]]:
    """Gather quotes grouped by the node that produced them."""
    guidance = state.get("guidance") or {}
    risks = state.get("risks") or {}
    sentiment = state.get("sentiment") or {}
    qoq = state.get("qoq") or {}

    quotes = {
        "guidance": [
            fs.get("quote", "") for fs in guidance.get("forward_statements", []) or []
        ],
        "risks": [r.get("quote", "") for r in risks.get("risks", []) or []],
        "sentiment": [d.get("quote", "") for d in sentiment.get("drivers", []) or []],
        "qoq": [
            q
            for mc in qoq.get("metric_changes", []) or []
            for q in (mc.get("quote_current", ""), mc.get("quote_prior", ""))
        ],
    }
    return {k: [q for q in v if q] for k, v in quotes.items()}


def _groundedness(state) -> tuple[dict, dict[str, float | None]]:
    evidence_blob = " || ".join(_norm(e) for e in state.get("evidence", []) or [])
    per_node: dict[str, float | None] = {}
    supported = total = 0
    unsupported: list[str] = []

    for node, quotes in _collect_quotes(state).items():
        if not quotes:
            per_node[node] = None
            continue
        node_ok = 0
        for q in quotes:
            total += 1
            if _norm(q) and _norm(q) in evidence_blob:
                supported += 1
                node_ok += 1
            else:
                unsupported.append(q)
        per_node[node] = round(node_ok / len(quotes), 2)

    score = round(supported / total, 2) if total else 0.0
    return (
        {"score": score, "supported": supported, "total": total, "unsupported": unsupported[:8]},
        per_node,
    )


def _completeness(state) -> dict:
    dims = {
        "guidance": bool(state.get("guidance")),
        "risks": bool(state.get("risks")),
        "sentiment": bool(state.get("sentiment")),
        "qoq": bool(state.get("qoq")),
    }
    covered = sum(dims.values())
    missing = [k for k, v in dims.items() if not v]
    return {"score": round(covered / 4, 2), "dimensions_covered": covered, "missing": missing}


def _consistency(state) -> dict:
    guidance = state.get("guidance") or {}
    sentiment = state.get("sentiment") or {}
    qoq = state.get("qoq") or {}
    flags: list[str] = []

    label, direction = sentiment.get("label"), guidance.get("direction")
    if label == "bullish" and direction == "lowered":
        flags.append("Bullish tone but guidance was lowered.")
    if label == "cautious" and direction == "raised":
        flags.append("Cautious tone but guidance was raised.")

    components = [1.0 if not flags else 0.0]
    qoq_available = bool(qoq.get("comparison_available"))
    changes = qoq.get("metric_changes", []) or []
    qoq_both = (
        qoq_available
        and len(changes) > 0
        and all(mc.get("quote_current") and mc.get("quote_prior") for mc in changes)
    )
    if qoq_available:
        components.append(1.0 if qoq_both else 0.0)

    return {
        "score": round(mean(components), 2),
        "flags": flags,
        "qoq_references_both": qoq_both,
    }


def _judge(state) -> dict | None:
    """Optional LLM-as-judge groundedness pass on the cheaper model."""
    evidence = " ".join(state.get("evidence", []) or [])[:6000]
    if not evidence:
        return None
    quotes = _collect_quotes(state)
    claims = "\n".join(f"- {q}" for qs in quotes.values() for q in qs)
    if not claims:
        return None
    instruction = (
        "You are grading an equity analysis. Below is the model's extracted claims "
        "(as quotes). Using ONLY the transcript excerpts in the context, rate how "
        "well-supported these claims are.\n\nCLAIMS:\n" + claims
    )
    try:
        return claude.structured_call(
            evidence, instruction, GROUNDEDNESS_JUDGE_SCHEMA, model=JUDGE_MODEL
        )
    except Exception:  # judge is best-effort; never break the run
        return None


def evaluator_node(state):
    """Return {eval: {...}} scoring the completed analysis."""
    groundedness, per_node = _groundedness(state)
    completeness = _completeness(state)
    consistency = _consistency(state)
    overall = round(
        mean([groundedness["score"], completeness["score"], consistency["score"]]), 2
    )
    return {
        "eval": {
            "groundedness": groundedness,
            "completeness": completeness,
            "consistency": consistency,
            "per_node_groundedness": per_node,
            "judge": _judge(state),
            "overall": overall,
        }
    }
