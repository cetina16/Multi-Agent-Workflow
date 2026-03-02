"""Web search via DuckDuckGo — free, no API key required."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from config import get_settings
from src.state import SearchResult

logger = logging.getLogger(__name__)

# Dedicated single-worker executor so only one DDG thread runs at a time.
# asyncio.wait_for cannot cancel running threads, so we must prevent thread
# pool exhaustion by ensuring searches never run concurrently.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddg")


async def search_web(
    query: str,
    *,
    max_results: int | None = None,
    max_retries: int | None = None,
) -> list[SearchResult]:
    """Search the web using DuckDuckGo (free, no API key).

    DDGS has a built-in HTTP timeout (default 10 s). We rely on that rather
    than asyncio.wait_for, because wait_for cannot cancel an already-running
    thread — it only cancels the asyncio Future, leaving the thread alive and
    blocking the pool.
    """
    settings = get_settings()
    max_results = max_results or settings.max_search_results
    max_retries = max_retries or settings.max_retries

    def _sync_search() -> list[dict]:
        from ddgs import DDGS
        # timeout= caps each HTTP request; DDGS raises an exception if exceeded
        with DDGS(timeout=10) as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    loop = asyncio.get_event_loop()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = await loop.run_in_executor(_EXECUTOR, _sync_search)
            results = [
                SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    score=round(1.0 - (i * 0.05), 2),
                )
                for i, r in enumerate(raw)
                if r.get("href")
            ]
            logger.info("DuckDuckGo returned %d results for '%s'", len(results), query)
            return results
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "DDG search attempt %d/%d failed (%s). Retrying in %ds…",
                attempt + 1, max_retries, exc, wait,
            )
            await asyncio.sleep(wait)

    logger.error("All search retries exhausted for '%s': %s", query, last_exc)
    return []
