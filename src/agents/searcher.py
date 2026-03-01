"""Searcher agent: executes web searches via Tavily for all search queries."""

from __future__ import annotations

import asyncio
import logging

from config import get_settings
from src.state import ErrorEntry, ResearchState, SearchResult
from src.tools.tavily_search import search_web

logger = logging.getLogger(__name__)


async def searcher_node(state: ResearchState) -> dict:
    """Run all search queries concurrently and collect results."""
    settings = get_settings()
    search_queries = state.get("search_queries", [])
    retry_counts: dict[str, int] = dict(state.get("retry_counts") or {})
    error_log: list[ErrorEntry] = []

    if not search_queries:
        logger.warning("Searcher: no search queries provided.")
        return {
            "search_results": [],
            "status": "failed",
            "error_log": [
                ErrorEntry(node="searcher", error="No search queries generated", attempt=0)
            ],
        }

    # Fan out: run all queries concurrently
    tasks = [
        search_web(q, max_results=settings.max_search_results)
        for q in search_queries
    ]

    results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[SearchResult] = []
    for query, result in zip(search_queries, results_per_query):
        if isinstance(result, Exception):
            logger.error("Search failed for query '%s': %s", query, result)
            retry_counts["searcher"] = retry_counts.get("searcher", 0) + 1
            error_log.append(
                ErrorEntry(
                    node="searcher",
                    error=str(result),
                    attempt=retry_counts["searcher"],
                )
            )
        else:
            all_results.extend(result)

    # Deduplicate by URL (keep highest score)
    seen: dict[str, SearchResult] = {}
    for r in all_results:
        url = r["url"]
        if url not in seen or r["score"] > seen[url]["score"]:
            seen[url] = r

    deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    status = "failed" if not deduped else "running"

    return {
        "search_results": deduped,
        "retry_counts": retry_counts,
        "status": status,
        "error_log": error_log,
    }
