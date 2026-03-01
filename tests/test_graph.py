"""Integration tests for the LangGraph StateGraph structure."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.graph import _APPROVAL_TOKENS, _route_after_human_review, _route_after_searcher, build_graph
from src.state import ResearchState


def _make_state(**kwargs) -> dict:
    base = {
        "query": "test query",
        "session_id": "test-session",
        "plan": [],
        "search_queries": [],
        "search_results": [],
        "extracted_sources": [],
        "summaries": [],
        "final_report": None,
        "human_feedback": None,
        "retry_counts": {},
        "error_log": [],
        "node_costs": [],
        "status": "running",
    }
    base.update(kwargs)
    return base


def test_route_after_human_review_approved():
    for token in ["approved", "Approved", "ok", "yes", "  "]:
        state = _make_state(human_feedback=token)
        assert _route_after_human_review(state) == "searcher"


def test_route_after_human_review_redirect():
    state = _make_state(human_feedback="Focus on medical applications please")
    assert _route_after_human_review(state) == "planner"


def test_route_after_searcher_success():
    state = _make_state(status="running", search_results=[{"url": "x", "title": "t", "content": "c", "score": 0.9}])
    assert _route_after_searcher(state) == "extractor"


def test_route_after_searcher_failure():
    from langgraph.graph import END
    state = _make_state(status="failed")
    assert _route_after_searcher(state) == END


def test_build_graph_without_checkpointer():
    """Graph should build successfully without a checkpointer (no HITL)."""
    graph = build_graph(checkpointer=None)
    assert graph is not None


def test_build_graph_with_memory_checkpointer():
    """Graph should build with MemorySaver for HITL support."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph(checkpointer=None)
    node_names = set(graph.nodes.keys())
    expected = {"planner", "human_review", "searcher", "extractor", "summarizer", "synthesizer", "storage_agent"}
    assert expected.issubset(node_names)
