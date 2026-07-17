"""Unit tests for the section-aware chunker (no network)."""
from data.models import Document
from ingest.chunker import chunk_document


def test_fmp_transcript_splits_prepared_and_qa():
    prepared = "CEO: Revenue grew strongly this quarter.\n" * 40
    qa = "Analyst: What about margins next year?\n" * 40
    text = prepared + "\nQuestion-and-Answer Session\n" + qa
    doc = Document(
        ticker="AAPL", year=2024, quarter="Q3", source="fmp_transcript", text=text
    )
    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert "prepared_remarks" in sections
    assert "qa" in sections
    assert all(c.id.startswith("AAPL_2024_Q3_") for c in chunks)
    # ids are unique and index-ordered
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_edgar_press_release_tagged():
    text = "MSFT reported record cloud revenue growth.\n" * 60
    doc = Document(
        ticker="MSFT", year=2024, quarter="Q2", source="edgar_8k", text=text
    )
    chunks = chunk_document(doc)
    assert chunks
    assert all(c.section == "press_release" for c in chunks)


def test_speaker_detection():
    text = "Tim Cook: We are very pleased with these results.\n" * 30
    doc = Document(
        ticker="AAPL", year=2024, quarter="Q1", source="fmp_transcript", text=text
    )
    chunks = chunk_document(doc)
    assert any(c.speaker == "Tim Cook" for c in chunks)
