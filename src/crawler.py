"""Web crawler for quotes.toscrape.com."""

from __future__ import annotations

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


def parse_quote_page(html: str, url: str) -> dict:  # type: ignore[empty-body]
    """Parse one page of quotes.toscrape.com. (stub – not yet implemented)"""
    raise NotImplementedError


def crawl(start_url: str = BASE_URL, **kwargs) -> list:  # type: ignore[empty-body]
    """Crawl the target site from *start_url*. (stub – not yet implemented)"""
    raise NotImplementedError
