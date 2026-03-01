"""Tests for the extractor agent."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.extractor import _compute_relevance, extractor_node
from src.state import SearchResult


def test_compute_relevance_high_overlap():
    result = SearchResult(
        url="https://x.com",
        title="Quantum Computing",
        content="quantum computing breakthroughs 2024 error correction advances",
        score=0.8,
    )
    score = _compute_relevance(result, "quantum computing advances")
    assert score > 0.6


def test_compute_relevance_no_overlap():
    result = SearchResult(
        url="https://x.com",
        title="Cooking Recipes",
        content="pasta sauce garlic olive oil basil",
        score=0.1,
    )
    score = _compute_relevance(result, "quantum computing")
    assert score < 0.2


@pytest.mark.asyncio
async def test_extractor_selects_top_n(base_state, sample_search_results):
    base_state["search_results"] = sample_search_results

    with patch("src.agents.extractor.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = None  # No additional scraping needed
        result = await extractor_node(base_state)

    assert result["status"] == "running"
    # Should select up to max_sources_to_extract (default 5)
    assert len(result["extracted_sources"]) <= 5
    assert len(result["extracted_sources"]) == len(sample_search_results)


@pytest.mark.asyncio
async def test_extractor_no_results(base_state):
    base_state["search_results"] = []
    result = await extractor_node(base_state)
    assert result["status"] == "failed"
    assert result["extracted_sources"] == []


@pytest.mark.asyncio
async def test_extractor_enriches_short_content(base_state):
    base_state["search_results"] = [
        {
            "url": "https://example.com/short",
            "title": "Short",
            "content": "Very short.",  # < 300 chars → triggers scraping
            "score": 0.9,
        }
    ]

    with patch("src.agents.extractor.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = "Much longer content from actual page " * 20
        result = await extractor_node(base_state)

    source = result["extracted_sources"][0]
    assert len(source["content"]) > 100
    mock_scrape.assert_called_once()
