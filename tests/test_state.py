"""Tests for state schema definitions."""

from src.state import (
    ErrorEntry,
    NodeCost,
    Report,
    ResearchState,
    SearchResult,
    Source,
    Summary,
)


def test_search_result_creation():
    r = SearchResult(url="https://example.com", title="Test", content="Body", score=0.9)
    assert r["url"] == "https://example.com"
    assert r["score"] == 0.9


def test_source_creation():
    s = Source(url="https://x.com", title="X", content="Content", relevance_score=0.75)
    assert s["relevance_score"] == 0.75


def test_summary_creation():
    s = Summary(url="https://x.com", title="X", summary="A great summary.")
    assert "summary" in s


def test_report_creation():
    r = Report(
        query="test query",
        executive_summary="Summary text",
        key_findings=["finding 1"],
        sources=[],
        metadata={"confidence": "high"},
    )
    assert r["query"] == "test query"
    assert len(r["key_findings"]) == 1


def test_node_cost():
    c = NodeCost(node="planner", tokens_in=100, tokens_out=50, cost_usd=0.001)
    assert c["cost_usd"] == 0.001


def test_error_entry():
    e = ErrorEntry(node="searcher", error="timeout", attempt=1)
    assert e["attempt"] == 1
