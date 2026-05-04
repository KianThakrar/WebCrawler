"""Web crawler for quotes.toscrape.com."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL: str = "https://quotes.toscrape.com/"

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(value: str) -> str:
    """Return *value* with collapsed whitespace and surrounding quote marks stripped.

    Args:
        value: Raw string extracted from HTML.

    Returns:
        Normalised plain-text string.
    """
    normalized = " ".join(value.split())
    return normalized.strip(" \t\r\n\"'“”")

MIN_POLITENESS_DELAY_SECONDS: float = 6.0
_REQUEST_TIMEOUT: int = 10


def parse_quote_page(html: str, url: str) -> dict:
    """Parse one Quotes to Scrape HTML page into structured data.

    Args:
        html: Raw HTML source of the page.
        url:  Absolute URL of the page (used to resolve relative links).

    Returns:
        A dict with keys: url, title, quotes, content, next_url.
    """
    soup = BeautifulSoup(html or "", "html.parser")

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""

    quotes: list[dict] = []
    content_parts: list[str] = []

    for quote_el in soup.select("div.quote"):
        text_el = quote_el.select_one(".text")
        author_el = quote_el.select_one(".author")
        if text_el is None:
            continue

        text = clean_text(text_el.get_text(" ", strip=True))
        author = clean_text(author_el.get_text(" ", strip=True)) if author_el else ""
        tags = [
            clean_text(tag.get_text(" ", strip=True))
            for tag in quote_el.select("a.tag")
        ]
        tags = [t for t in tags if t]

        quotes.append({"text": text, "author": author, "tags": tags})
        content_parts.extend([text, author] + tags)

    next_url: str | None = None
    next_li = soup.select_one("li.next > a")
    if next_li and next_li.get("href"):
        next_url = urljoin(url, next_li["href"])

    return {
        "url": url,
        "title": title,
        "quotes": quotes,
        "content": " ".join(content_parts),
        "next_url": next_url,
    }


def crawl(
    start_url: str = BASE_URL,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    max_pages: int | None = None,
) -> list[dict]:
    """Crawl quotes.toscrape.com starting from *start_url*.

    Follows pagination links in BFS order.  Deduplicates visited URLs and
    enforces MIN_POLITENESS_DELAY_SECONDS between requests.

    Args:
        start_url:   First URL to fetch.
        on_progress: Optional callback ``(url, pages_done, estimated_total)``.
        max_pages:   Stop after this many pages (useful for testing).

    Returns:
        List of page dicts as returned by :func:`parse_quote_page`.
    """
    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    pages: list[dict] = []

    while queue:
        if max_pages is not None and len(pages) >= max_pages:
            break

        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        if pages:
            time.sleep(MIN_POLITENESS_DELAY_SECONDS)

        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            continue

        page_data = parse_quote_page(response.text, url)
        pages.append(page_data)

        if on_progress is not None:
            on_progress(url, len(pages), len(pages))

        if page_data["next_url"] and page_data["next_url"] not in visited:
            queue.append(page_data["next_url"])

    return pages
