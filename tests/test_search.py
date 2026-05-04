"""Tests for the search engine module.

Strategy
--------
All tests operate on a pre-built InvertedIndex constructed from synthetic
data so no crawling or file I/O is needed.
We cover:
  - SearchEngine: single-word lookup, multi-word AND, ranked results
  - print_entry: formatted output for the print command
  - Edge cases: empty query, unknown words, case insensitivity
"""

from __future__ import annotations

from src.indexer import InvertedIndex, build_index
from src.search import SearchEngine

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PAGE_A = {
    "url": "https://example.com/page/1/",
    "content": "wisdom brings great courage and wisdom again",
    "title": "Page One",
    "quotes": [],
}

PAGE_B = {
    "url": "https://example.com/page/2/",
    "content": "courage without wisdom fails",
    "title": "Page Two",
    "quotes": [],
}

PAGE_C = {
    "url": "https://example.com/page/3/",
    "content": "unique rare solitary term only here",
    "title": "Page Three",
    "quotes": [],
}


def _build() -> SearchEngine:
    idx = build_index([PAGE_A, PAGE_B, PAGE_C])
    return SearchEngine(idx)


# ---------------------------------------------------------------------------
# SearchEngine – single word
# ---------------------------------------------------------------------------

class TestSingleWordSearch:
    def test_find_known_word_returns_results(self) -> None:
        engine = _build()
        results = engine.find(["wisdom"])
        assert len(results) > 0

    def test_find_returns_list_of_tuples(self) -> None:
        engine = _build()
        results = engine.find(["wisdom"])
        assert isinstance(results, list)
        assert isinstance(results[0], tuple)

    def test_result_tuple_has_url_and_score(self) -> None:
        engine = _build()
        url, score = engine.find(["wisdom"])[0]
        assert url.startswith("http")
        assert isinstance(score, float)

    def test_unknown_word_returns_empty(self) -> None:
        engine = _build()
        assert engine.find(["xyznonexistent"]) == []

    def test_results_sorted_by_score_descending(self) -> None:
        engine = _build()
        results = engine.find(["wisdom"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_page_with_higher_frequency_scores_higher(self) -> None:
        # PAGE_A has "wisdom" twice; PAGE_B once — PAGE_A should rank first
        engine = _build()
        results = engine.find(["wisdom"])
        urls = [u for u, _ in results]
        assert urls[0] == PAGE_A["url"]

    def test_case_insensitive_lookup(self) -> None:
        engine = _build()
        lower = engine.find(["wisdom"])
        upper = engine.find(["WISDOM"])
        assert lower == upper


# ---------------------------------------------------------------------------
# SearchEngine – multi-word AND
# ---------------------------------------------------------------------------

class TestMultiWordSearch:
    def test_both_words_must_appear(self) -> None:
        engine = _build()
        # PAGE_C has neither "wisdom" nor "courage"
        results = engine.find(["wisdom", "courage"])
        urls = [u for u, _ in results]
        assert PAGE_C["url"] not in urls

    def test_multi_word_returns_pages_with_all_terms(self) -> None:
        engine = _build()
        results = engine.find(["wisdom", "courage"])
        # Both PAGE_A and PAGE_B contain courage AND wisdom
        assert len(results) == 2

    def test_score_is_sum_of_tfidf(self) -> None:
        engine = _build()
        results = engine.find(["wisdom", "courage"])
        assert all(score > 0 for _, score in results)

    def test_no_common_page_returns_empty(self) -> None:
        engine = _build()
        # "unique" only in PAGE_C; "wisdom" only in PAGE_A/B → no overlap
        assert engine.find(["unique", "wisdom"]) == []

    def test_single_result_when_one_page_has_all_terms(self) -> None:
        engine = _build()
        results = engine.find(["unique", "rare"])
        assert len(results) == 1
        assert results[0][0] == PAGE_C["url"]


# ---------------------------------------------------------------------------
# SearchEngine – print_entry
# ---------------------------------------------------------------------------

class TestPrintEntry:
    def test_print_known_word_returns_string(self) -> None:
        engine = _build()
        output = engine.print_entry("wisdom")
        assert isinstance(output, str)

    def test_print_contains_word(self) -> None:
        engine = _build()
        output = engine.print_entry("wisdom")
        assert "wisdom" in output.lower()

    def test_print_contains_url(self) -> None:
        engine = _build()
        output = engine.print_entry("wisdom")
        assert "https://" in output

    def test_print_contains_frequency(self) -> None:
        engine = _build()
        output = engine.print_entry("wisdom")
        assert "frequency" in output.lower()

    def test_print_contains_tfidf(self) -> None:
        engine = _build()
        output = engine.print_entry("wisdom")
        assert "tf_idf" in output.lower() or "tfidf" in output.lower() or "tf-idf" in output.lower()

    def test_print_unknown_word_returns_not_found_message(self) -> None:
        engine = _build()
        output = engine.print_entry("xyznonexistent")
        assert "not found" in output.lower() or "no results" in output.lower()

    def test_print_case_insensitive(self) -> None:
        engine = _build()
        lower = engine.print_entry("wisdom")
        upper = engine.print_entry("WISDOM")
        assert lower == upper


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query_returns_empty(self) -> None:
        engine = _build()
        assert engine.find([]) == []

    def test_empty_index_returns_empty(self) -> None:
        engine = SearchEngine(InvertedIndex())
        assert engine.find(["anything"]) == []

    def test_whitespace_only_term_treated_as_empty(self) -> None:
        engine = _build()
        assert engine.find([""]) == []
