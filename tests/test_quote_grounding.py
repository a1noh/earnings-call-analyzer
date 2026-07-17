"""Unit test for the rule-based groundedness eval (no network).

The LLM judge inside evaluator_node is best-effort and swallows errors, so this
test runs offline: with no API key it simply returns judge=None while the
deterministic rule-based scores are still computed.
"""
from agent.nodes.evaluator import evaluator_node


def _state():
    return {
        "evidence": ["We expect revenue to grow ten percent next year."],
        "guidance": {
            "revenue_guidance": "10%",
            "eps_guidance": "not disclosed",
            "direction": "raised",
            "forward_statements": [
                {
                    "statement": "Management expects growth.",
                    "quote": "We expect revenue to grow ten percent next year.",
                }
            ],
        },
        "risks": {
            "risks": [
                {
                    "risk": "macro headwind",
                    "severity": "low",
                    "is_new_or_escalating": False,
                    "quote": "a totally fabricated quote not present anywhere",
                }
            ]
        },
        "sentiment": {"label": "bullish", "score": 0.5, "drivers": [], "rationale": "x"},
        "qoq": {
            "comparison_available": False,
            "current_quarter": "Q3 2024",
            "prior_quarter": "none",
            "metric_changes": [],
            "narrative_shift": "no basis",
        },
    }


def test_grounding_catches_fabricated_quote():
    out = evaluator_node(_state())
    ev = out["eval"]
    g = ev["groundedness"]
    assert g["total"] == 2
    assert g["supported"] == 1
    assert any("fabricated" in u for u in g["unsupported"])
    # per-node badge: guidance fully grounded, risks not
    assert ev["per_node_groundedness"]["guidance"] == 1.0
    assert ev["per_node_groundedness"]["risks"] == 0.0


def test_completeness_and_consistency_present():
    out = evaluator_node(_state())
    ev = out["eval"]
    assert 0.0 <= ev["completeness"]["score"] <= 1.0
    assert 0.0 <= ev["consistency"]["score"] <= 1.0
    assert "overall" in ev
