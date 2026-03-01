"""Extractor agent: scores source relevance and enriches content via scraping."""

from __future__ import annotations

import asyncio
import logging

from config import get_settings
from src.state import ResearchState, SearchResult, Source
from src.tools.web_scraper import scrape_url

logger = logging.getLogger(__name__)


def _compute_relevance(result: SearchResult, query: str) -> float:
    """Heuristic relevance score combining Tavily score with keyword overlap."""
    tavily_score = result.get("score", 0.0)
    query_words = set(query.lower().split())
    content_words = set(result.get("content", "").lower().split())
    overlap = len(query_words & content_words) / max(len(query_words), 1)
    # Weighted average: 70% Tavily score, 30% keyword overlap
    return round(0.7 * tavily_score + 0.3 * overlap, 4)


async def extractor_node(state: ResearchState) -> dict:
    """Select top N sources, optionally enrich via scraping, assign relevance."""
    settings = get_settings()
    search_results = state.get("search_results", [])
    query = state.get("query", "")
    max_sources = settings.max_sources_to_extract

    if not search_results:
        return {"extracted_sources": [], "status": "failed"}

    # Score and rank
    scored = sorted(
        search_results,
        key=lambda r: _compute_relevance(r, query),
        reverse=True,
    )
    top_results = scored[:max_sources]

    async def enrich(result: SearchResult) -> Source:
        relevance = _compute_relevance(result, query)
        content = result.get("content", "")

        # Try to get richer content via scraping if Tavily content is short
        if len(content) < 300:
            scraped = await scrape_url(result["url"])
            if scraped:
                content = scraped

        return Source(
            url=result["url"],
            title=result.get("title", ""),
            content=content,
            relevance_score=relevance,
        )

    sources = await asyncio.gather(*[enrich(r) for r in top_results])
    return {"extracted_sources": list(sources), "status": "running"}
