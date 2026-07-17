"""JSON schemas for each analysis node's structured output.

Each schema is strict (``additionalProperties: false`` with every property
``required``) so that, when the SDK/model supports structured outputs, the model
is constrained to exactly this shape. The same schema is also embedded in the
prompt as a belt-and-suspenders instruction for the prompt-JSON fallback path.
"""
from __future__ import annotations

GUIDANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "revenue_guidance": {
            "type": "string",
            "description": "Specific revenue guidance numbers/percentages, or 'not disclosed'.",
        },
        "eps_guidance": {
            "type": "string",
            "description": "Specific EPS/margin guidance, or 'not disclosed'.",
        },
        "direction": {
            "type": "string",
            "enum": ["raised", "maintained", "lowered", "not_provided"],
        },
        "forward_statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string", "description": "Verbatim supporting quote."},
                },
                "required": ["statement", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["revenue_guidance", "eps_guidance", "direction", "forward_statements"],
    "additionalProperties": False,
}

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "is_new_or_escalating": {"type": "boolean"},
                    "quote": {"type": "string", "description": "Verbatim supporting quote."},
                },
                "required": ["risk", "severity", "is_new_or_escalating", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["risks"],
    "additionalProperties": False,
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["bullish", "neutral", "cautious"]},
        "score": {
            "type": "number",
            "description": "Tone score from -1.0 (very cautious) to 1.0 (very bullish).",
        },
        "drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "driver": {"type": "string"},
                    "quote": {"type": "string", "description": "Verbatim supporting quote."},
                },
                "required": ["driver", "quote"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["label", "score", "drivers", "rationale"],
    "additionalProperties": False,
}

QOQ_SCHEMA = {
    "type": "object",
    "properties": {
        "comparison_available": {"type": "boolean"},
        "current_quarter": {"type": "string"},
        "prior_quarter": {"type": "string"},
        "metric_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "change": {"type": "string"},
                    "quote_current": {"type": "string"},
                    "quote_prior": {"type": "string"},
                },
                "required": ["metric", "change", "quote_current", "quote_prior"],
                "additionalProperties": False,
            },
        },
        "narrative_shift": {"type": "string"},
    },
    "required": [
        "comparison_available",
        "current_quarter",
        "prior_quarter",
        "metric_changes",
        "narrative_shift",
    ],
    "additionalProperties": False,
}

GROUNDEDNESS_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness": {
            "type": "number",
            "description": "0.0-1.0: fraction of claims well-supported by the provided context.",
        },
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["groundedness", "unsupported_claims", "rationale"],
    "additionalProperties": False,
}
