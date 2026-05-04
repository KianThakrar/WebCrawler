"""Web crawler for quotes.toscrape.com."""

from __future__ import annotations

from urllib.parse import urljoin

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


def crawl(start_url: str = BASE_URL, **kwargs) -> list:  # type: ignore[empty-body]
    """Crawl the target site from *start_url*. (stub – not yet implemented)"""
    raise NotImplementedError
