"""Shared pytest fixtures available to all test modules."""

from __future__ import annotations

import pytest

from src.indexer import InvertedIndex, build_index
from src.search import SearchEngine


@pytest.fixture
def sample_pages() -> list[dict]:
    """Three synthetic pages covering a small vocabulary."""
    return [
        {
            "url": "https://quotes.toscrape.com/page/1/",
            "content": "wisdom brings great courage and wisdom again",
            "title": "Page One",
            "quotes": [],
        },
        {
            "url": "https://quotes.toscrape.com/page/2/",
            "content": "courage without wisdom fails every time",
            "title": "Page Two",
            "quotes": [],
        },
        {
            "url": "https://quotes.toscrape.com/page/3/",
            "content": "unique rare solitary term only present here",
            "title": "Page Three",
            "quotes": [],
        },
    ]


@pytest.fixture
def built_index(sample_pages: list[dict]) -> InvertedIndex:
    """Fully built and TF-IDF scored InvertedIndex over sample_pages."""
    return build_index(sample_pages)


@pytest.fixture
def search_engine(built_index: InvertedIndex) -> SearchEngine:
    """SearchEngine wrapping the built_index fixture."""
    return SearchEngine(built_index)
