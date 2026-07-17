"""Unit tests for EDGAR helpers (no network)."""
from data.edgar_client import EdgarClient


def test_period_to_quarter():
    assert EdgarClient._period_to_quarter("2024-03-31") == (2024, "Q1")
    assert EdgarClient._period_to_quarter("2024-06-30") == (2024, "Q2")
    assert EdgarClient._period_to_quarter("2024-09-30") == (2024, "Q3")
    assert EdgarClient._period_to_quarter("2024-12-31") == (2024, "Q4")


def test_strip_html_removes_scripts_and_keeps_text():
    html = (
        "<html><body><p>Record revenue this quarter.</p>"
        "<script>var x = 1;</script>"
        "<style>.a{color:red}</style>"
        "<p>Guidance raised.</p></body></html>"
    )
    text = EdgarClient._strip_html(html)
    assert "Record revenue this quarter." in text
    assert "Guidance raised." in text
    assert "var x" not in text
    assert "color:red" not in text
