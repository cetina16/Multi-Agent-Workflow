"""Lightweight async web scraper for extracting clean text from URLs."""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://github.com/research-bot)"
    )
}
_TIMEOUT = 15
_MAX_CONTENT_CHARS = 8_000


async def scrape_url(url: str) -> str | None:
    """Fetch a URL and return cleaned text content, or None on failure."""
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove boilerplate tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Prefer <article> or <main>, fall back to <body>
            main = soup.find("article") or soup.find("main") or soup.find("body")
            if main is None:
                return None

            text = main.get_text(separator=" ", strip=True)
            # Collapse excessive whitespace
            text = " ".join(text.split())
            return text[:_MAX_CONTENT_CHARS]

    except Exception as exc:
        logger.warning("Failed to scrape %s: %s", url, exc)
        return None
