"""Shared pytest fixtures for the multi-agent workflow tests."""

from __future__ import annotations

import os
import uuid

import pytest

# Point to test environment defaults before any app imports
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/research_db_test")
os.environ.setdefault("SYNC_DATABASE_URL", "postgresql://postgres:password@localhost:5432/research_db_test")


@pytest.fixture
def sample_query() -> str:
    return "What are the latest advances in quantum computing?"


@pytest.fixture
def sample_session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_search_results() -> list[dict]:
    return [
        {
            "url": "https://example.com/quantum-1",
            "title": "Quantum Computing Breakthroughs 2024",
            "content": "Researchers achieved a new milestone in quantum error correction, reducing error rates by 40%.",
            "score": 0.92,
        },
        {
            "url": "https://example.com/quantum-2",
            "title": "IBM Quantum Roadmap",
            "content": "IBM unveiled a 1000-qubit processor and announced plans for fault-tolerant quantum computing.",
            "score": 0.87,
        },
        {
            "url": "https://example.com/quantum-3",
            "title": "Google Quantum AI Update",
            "content": "Google demonstrated quantum advantage on optimization problems using Sycamore processor.",
            "score": 0.81,
        },
    ]


@pytest.fixture
def sample_sources(sample_search_results) -> list[dict]:
    return [
        {
            "url": r["url"],
            "title": r["title"],
            "content": r["content"],
            "relevance_score": r["score"],
        }
        for r in sample_search_results
    ]


@pytest.fixture
def sample_summaries(sample_sources) -> list[dict]:
    return [
        {
            "url": s["url"],
            "title": s["title"],
            "summary": f"Summary of {s['title']}: {s['content'][:100]}",
        }
        for s in sample_sources
    ]


@pytest.fixture
def base_state(sample_query, sample_session_id) -> dict:
    return {
        "query": sample_query,
        "session_id": sample_session_id,
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
