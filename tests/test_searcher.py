"""Tests for the searcher agent (mocked Tavily calls)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.searcher import searcher_node
from src.state import SearchResult


@pytest.mark.asyncio
async def test_searcher_success(base_state, sample_search_results):
    base_state["search_queries"] = ["quantum computing breakthroughs"]

    with patch("src.agents.searcher.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [SearchResult(**r) for r in sample_search_results]
        result = await searcher_node(base_state)

    assert result["status"] == "running"
    assert len(result["search_results"]) == len(sample_search_results)


@pytest.mark.asyncio
async def test_searcher_deduplicates_urls(base_state):
    base_state["search_queries"] = ["query 1", "query 2"]

    duplicate_results = [
        SearchResult(url="https://example.com/dup", title="Dup", content="X", score=0.8),
        SearchResult(url="https://example.com/dup", title="Dup", content="X", score=0.9),
        SearchResult(url="https://example.com/unique", title="U", content="Y", score=0.7),
    ]

    with patch("src.agents.searcher.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = duplicate_results
        result = await searcher_node(base_state)

    urls = [r["url"] for r in result["search_results"]]
    assert len(urls) == len(set(urls)), "Duplicate URLs should be deduplicated"


@pytest.mark.asyncio
async def test_searcher_no_queries(base_state):
    base_state["search_queries"] = []
    result = await searcher_node(base_state)
    assert result["status"] == "failed"
    assert len(result["search_results"]) == 0


@pytest.mark.asyncio
async def test_searcher_all_queries_fail(base_state):
    base_state["search_queries"] = ["bad query"]

    with patch("src.agents.searcher.search_web", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = Exception("API error")
        result = await searcher_node(base_state)

    assert result["status"] == "failed"
    assert len(result["error_log"]) > 0
