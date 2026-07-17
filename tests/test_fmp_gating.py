"""Unit tests for FMP gating detection (no network)."""
from data.fmp_client import FMPClient


def test_http_forbidden_is_gated():
    assert FMPClient._classify_payload(403, None) == "fmp_gated"
    assert FMPClient._classify_payload(402, None) == "fmp_gated"


def test_error_message_mentioning_plan_is_gated():
    payload = {"Error Message": "This endpoint requires a paid subscription plan."}
    assert FMPClient._classify_payload(200, payload) == "fmp_gated"


def test_generic_error_message_is_error():
    payload = {"Error Message": "Something odd happened."}
    assert FMPClient._classify_payload(200, payload) == "fmp_error"


def test_empty_payload():
    assert FMPClient._classify_payload(200, []) == "fmp_empty"


def test_usable_payload_returns_none():
    assert FMPClient._classify_payload(200, [{"content": "hello"}]) is None


def test_no_key_short_circuits():
    client = FMPClient(api_key=None)
    result = client.get_transcript("AAPL", 2024, "Q3")
    assert result.ok is False
    assert result.reason == "fmp_no_key"
