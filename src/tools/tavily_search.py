"""Async Tavily web search with retry + DuckDuckGo fallback."""

from __future__ import annotations

import asyncio
import logging

import httpx
from tavily import AsyncTavilyClient

from config import get_settings
from src.state import SearchResult

logger = logging.getLogger(__name__)


async def _duckduckgo_fallback(query: str, max_results: int) -> list[SearchResult]:
    """Minimal DuckDuckGo fallback via instant-answer API (no API key required)."""
    results: list[SearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": "1"},
            )
            data = resp.json()
            # RelatedTopics contains result snippets
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "FirstURL" in topic and "Text" in topic:
                    results.append(
                        SearchResult(
                            url=topic["FirstURL"],
                            title=topic.get("Text", "")[:100],
                            content=topic.get("Text", ""),
                            score=0.5,
                        )
                    )
    except Exception as exc:
        logger.warning("DuckDuckGo fallback failed: %s", exc)
    return results


async def search_web(
    query: str,
    *,
    max_results: int | None = None,
    max_retries: int | None = None,
) -> list[SearchResult]:
    """Search the web via Tavily with exponential-backoff retry and DDG fallback.

    Returns a list of SearchResult dicts sorted by relevance score descending.
    """
    settings = get_settings()
    max_results = max_results or settings.max_search_results
    max_retries = max_retries or settings.max_retries

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.search(
                query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
            )
            raw = response.get("results", [])
            return [
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                )
                for r in raw
            ]
        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            logger.warning(
                "Tavily search attempt %d/%d failed (%s). Retrying in %ds…",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            await asyncio.sleep(wait)

    logger.error("All Tavily retries exhausted. Using DuckDuckGo fallback.")
    return await _duckduckgo_fallback(query, max_results)
