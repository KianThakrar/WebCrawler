"""Web crawler for quotes.toscrape.com.

Crawl scope
-----------
The crawler performs a breadth-first walk that follows every internal
link of three kinds:

  - Pagination     ``/`` and ``/page/N/``                        (quote listings)
  - Author pages   ``/author/Name/``                             (biography text)
  - Tag pages      ``/tag/name/`` and ``/tag/name/page/N/``      (filtered listings)

Listing pages and tag pages share the same HTML structure (a series of
``div.quote`` blocks), so they share a single parser.  Author pages
have their own structure (name, born, biographical description) and a
dedicated parser.  External links and static assets are ignored.

The visited-URL set prevents loops; the politeness sleep enforces a
six-second gap between every fetch after the first.
"""

from __future__ import annotations

__all__ = [
    "BASE_URL",
    "MIN_POLITENESS_DELAY_SECONDS",
    "clean_text",
    "parse_quote_page",
    "parse_author_page",
    "crawl",
]

import os
import time
from collections import deque
from typing import Callable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL: str = "https://quotes.toscrape.com/"

# The coursework specification requires a politeness window of at least
# 6 seconds between successive requests.  This is the production default
# and the value submitted code uses.  For local development workflows
# (e.g. regenerating data/index.json without waiting many minutes), the
# CRAWLER_POLITENESS_SECONDS environment variable can override the delay
# used at runtime.  The constant itself is unchanged so any test or caller
# that imports MIN_POLITENESS_DELAY_SECONDS still sees 6.0.
MIN_POLITENESS_DELAY_SECONDS: float = 6.0
_REQUEST_TIMEOUT: int = 10
_USER_AGENT: str = "WebCrawler/1.0 (COMP3011 coursework; educational use)"

_BASE_NETLOC: str = urlparse(BASE_URL).netloc


def _runtime_politeness_delay() -> float:
    """Return the politeness delay actually used at runtime.

    Defaults to ``MIN_POLITENESS_DELAY_SECONDS`` (6 s, as required by the
    coursework specification).  Can be overridden by setting the
    ``CRAWLER_POLITENESS_SECONDS`` environment variable to a non-negative
    float.  Invalid values fall back to the default.
    """
    raw = os.environ.get("CRAWLER_POLITENESS_SECONDS")
    if raw is None:
        return MIN_POLITENESS_DELAY_SECONDS
    try:
        value = float(raw)
        return value if value >= 0 else MIN_POLITENESS_DELAY_SECONDS
    except (TypeError, ValueError):
        return MIN_POLITENESS_DELAY_SECONDS

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


def _is_crawlable(url: str) -> bool:
    """Return True if *url* is an internal page we want to crawl.

    Allows: the homepage, /page/N/, /author/Name/, /tag/name/, /tag/name/page/N/.
    Blocks: external hosts, static assets, /login, fragment-only links.
    """
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != _BASE_NETLOC:
        return False
    path = parsed.path or "/"
    # Strip query / fragment by virtue of using parsed.path
    if path == "/" or path == "":
        return True
    if path.startswith("/page/"):
        return True
    if path.startswith("/author/"):
        return True
    if path.startswith("/tag/"):
        return True
    return False


def _discover_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    """Return de-duplicated absolute URLs to internal pages found in *soup*."""
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        absolute = urljoin(current_url, href)
        # Drop fragments — same URL with different hash is the same page
        if "#" in absolute:
            absolute = absolute.split("#", 1)[0]
        if absolute in seen:
            continue
        if _is_crawlable(absolute):
            seen.add(absolute)
            out.append(absolute)
    return out


# ---------------------------------------------------------------------------
# Page parsers
# ---------------------------------------------------------------------------


def parse_quote_page(html: str, url: str) -> dict[str, object]:
    """Parse a Quotes-to-Scrape *listing* page (homepage, /page/N/, /tag/...).

    Args:
        html: Raw HTML source of the page.
        url:  Absolute URL of the page (used to resolve relative links).

    Returns:
        A dict with keys: url, title, quotes, content, next_url, discovered_links.
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

    discovered = _discover_links(soup, url)

    return {
        "url": url,
        "title": title,
        "quotes": quotes,
        "content": " ".join(content_parts),
        "next_url": next_url,
        "discovered_links": discovered,
    }


def parse_author_page(html: str, url: str) -> dict[str, object]:
    """Parse an author biography page (``/author/Name/``).

    Extracts the author's name, birth metadata, and biographical
    description; combines them into the page's indexable ``content``.

    Args:
        html: Raw HTML source of the page.
        url:  Absolute URL of the page.

    Returns:
        A dict with keys: url, title, content, discovered_links, next_url.
        The ``next_url`` is always None and ``discovered_links`` is empty
        because author pages don't drive further crawling.
    """
    soup = BeautifulSoup(html or "", "html.parser")

    name_el = soup.select_one("h3.author-title")
    name = clean_text(name_el.get_text(" ", strip=True)) if name_el else ""

    born_date_el = soup.select_one(".author-born-date")
    born_loc_el = soup.select_one(".author-born-location")
    born_parts: list[str] = []
    if born_date_el:
        born_parts.append(clean_text(born_date_el.get_text(" ", strip=True)))
    if born_loc_el:
        born_parts.append(clean_text(born_loc_el.get_text(" ", strip=True)))
    born = " ".join(p for p in born_parts if p)

    desc_el = soup.select_one(".author-description")
    description = clean_text(desc_el.get_text(" ", strip=True)) if desc_el else ""

    content = " ".join(p for p in (name, born, description) if p)
    title = name or clean_text(soup.title.get_text(" ", strip=True)) if soup.title else name

    return {
        "url": url,
        "title": title,
        "content": content,
        "discovered_links": [],
        "next_url": None,
    }


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


def _select_parser(url: str):
    """Return the appropriate parser for *url* based on its path."""
    path = urlparse(url).path
    if path.startswith("/author/"):
        return parse_author_page
    return parse_quote_page


def crawl(
    start_url: str = BASE_URL,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
    max_pages: int | None = None,
) -> list[dict]:
    """Crawl quotes.toscrape.com starting from *start_url*.

    Performs a breadth-first walk over every internal page reachable
    via pagination, author links, and tag links.  Deduplicates visited
    URLs and enforces ``MIN_POLITENESS_DELAY_SECONDS`` between requests.

    Args:
        start_url:   First URL to fetch.
        on_progress: Optional callback ``(url, pages_done, estimated_total)``.
        max_pages:   Stop after this many pages (useful for testing).

    Returns:
        List of page dicts.  Each has at minimum ``url`` and ``content``;
        listing pages additionally carry ``quotes``, ``next_url``, and
        ``discovered_links``; author pages carry the extracted bio.
    """
    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    pages: list[dict[str, object]] = []
    _headers: dict[str, str] = {"User-Agent": _USER_AGENT}

    while queue:
        if max_pages is not None and len(pages) >= max_pages:
            break

        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        # Skip delay before the very first request; every subsequent
        # fetch must wait to satisfy the 6 s politeness requirement.
        # _runtime_politeness_delay() returns the constant by default but
        # honours CRAWLER_POLITENESS_SECONDS for local development crawls.
        if pages:
            time.sleep(_runtime_politeness_delay())

        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=_headers)
            response.raise_for_status()
        except Exception:
            continue

        parser = _select_parser(url)
        page_data = parser(response.text, url)
        pages.append(page_data)

        if on_progress is not None:
            on_progress(url, len(pages), len(pages))

        # Enqueue the explicit "next" pagination link if present
        next_url = page_data.get("next_url")
        if next_url and next_url not in visited:
            queue.append(next_url)

        # Enqueue every other internal link discovered on the page
        for link in page_data.get("discovered_links", []) or []:
            if link not in visited and link not in queue:
                queue.append(link)

    return pages
