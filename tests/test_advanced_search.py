"""Tests for advanced search features.

Covers:
  - Phrase search (exact consecutive token matching using positions)
  - Proximity scoring (bonus when query terms appear near each other)
  - BM25 ranking (Okapi BM25 as an alternative to plain TF-IDF)
  - Query suggestions (did-you-mean via Levenshtein edit distance)
"""

from __future__ import annotations

from src.indexer import build_index
from src.search import SearchEngine, suggest_terms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAGE_A = {
    "url": "https://example.com/1/",
    "content": "good friends make life worth living good times",
    "title": "Page One",
    "quotes": [],
}

PAGE_B = {
    "url": "https://example.com/2/",
    "content": "life worth living requires courage and good judgement",
    "title": "Page Two",
    "quotes": [],
}

PAGE_C = {
    "url": "https://example.com/3/",
    "content": "ancient wisdom knowledge philosophy brings peace",
    "title": "Page Three",
    "quotes": [],
}


def _build() -> SearchEngine:
    return SearchEngine(build_index([PAGE_A, PAGE_B, PAGE_C]))


# ---------------------------------------------------------------------------
# Phrase search
# ---------------------------------------------------------------------------

class TestPhraseSearch:
    def test_exact_phrase_returns_matching_page(self) -> None:
        engine = _build()
        # "good friends" appear consecutively in PAGE_A
        results = engine.find_phrase("good friends")
        urls = [u for u, _ in results]
        assert PAGE_A["url"] in urls

    def test_phrase_not_present_returns_empty(self) -> None:
        engine = _build()
        results = engine.find_phrase("friends ancient")
        assert results == []

    def test_phrase_order_matters(self) -> None:
        engine = _build()
        # "friends good" reversed — not consecutive in PAGE_A
        results = engine.find_phrase("friends good")
        assert results == []

    def test_single_word_phrase_same_as_find(self) -> None:
        engine = _build()
        phrase_results = engine.find_phrase("courage")
        find_results = engine.find(["courage"])
        assert {u for u, _ in phrase_results} == {u for u, _ in find_results}

    def test_phrase_results_ranked_by_score(self) -> None:
        engine = _build()
        results = engine.find_phrase("good")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Proximity scoring
# ---------------------------------------------------------------------------

class TestProximityScoring:
    def test_proximity_score_higher_when_terms_adjacent(self) -> None:
        # PAGE_A has "good friends" adjacent; PAGE_B has "good" far from nothing nearby
        # Build a corpus where proximity difference is unambiguous
        near_page = {
            "url": "https://example.com/near/",
            "content": "wisdom courage wisdom courage",
            "title": "",
            "quotes": [],
        }
        far_page = {
            "url": "https://example.com/far/",
            "content": "wisdom " + "filler " * 20 + "courage",
            "title": "",
            "quotes": [],
        }
        engine = SearchEngine(build_index([near_page, far_page]))
        results = engine.find_with_proximity(["wisdom", "courage"])
        assert results[0][0] == near_page["url"]

    def test_proximity_find_returns_list_of_tuples(self) -> None:
        engine = _build()
        results = engine.find_with_proximity(["good", "life"])
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], tuple)

    def test_proximity_unknown_word_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_with_proximity(["xyznonexistent"]) == []


# ---------------------------------------------------------------------------
# BM25 ranking
# ---------------------------------------------------------------------------

class TestBM25:
    def test_bm25_find_returns_results(self) -> None:
        engine = _build()
        results = engine.find_bm25(["good"])
        assert len(results) > 0

    def test_bm25_results_are_tuples(self) -> None:
        engine = _build()
        results = engine.find_bm25(["good"])
        url, score = results[0]
        assert isinstance(url, str)
        assert isinstance(score, float)

    def test_bm25_results_sorted_descending(self) -> None:
        engine = _build()
        results = engine.find_bm25(["good"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_unknown_word_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_bm25(["xyznonexistent"]) == []

    def test_bm25_multi_word_and_semantics(self) -> None:
        engine = _build()
        results = engine.find_bm25(["good", "life"])
        urls = {u for u, _ in results}
        # PAGE_C has neither "good" nor "life"
        assert PAGE_C["url"] not in urls

    def test_bm25_higher_freq_scores_higher_than_lower_freq(self) -> None:
        # PAGE_A has "good" twice; PAGE_B once
        engine = _build()
        results = engine.find_bm25(["good"])
        url_order = [u for u, _ in results]
        assert url_order[0] == PAGE_A["url"]


# ---------------------------------------------------------------------------
# Query suggestions
# ---------------------------------------------------------------------------

class TestQuerySuggestions:
    def test_suggest_returns_list(self) -> None:
        idx = build_index([PAGE_A, PAGE_B, PAGE_C])
        suggestions = suggest_terms(idx, "wisdon")  # typo of "wisdom"
        assert isinstance(suggestions, list)

    def test_suggest_finds_close_word(self) -> None:
        idx = build_index([PAGE_A, PAGE_B, PAGE_C])
        suggestions = suggest_terms(idx, "wisdon")
        assert "wisdom" in suggestions

    def test_suggest_exact_match_returns_empty_or_self(self) -> None:
        idx = build_index([PAGE_A])
        suggestions = suggest_terms(idx, "good")
        # exact match — either no suggestions needed or word itself returned
        assert isinstance(suggestions, list)

    def test_suggest_unknown_gibberish_returns_empty(self) -> None:
        idx = build_index([PAGE_A])
        suggestions = suggest_terms(idx, "zzzzqqqqxxxx")
        assert suggestions == []

    def test_suggest_respects_max_results(self) -> None:
        idx = build_index([PAGE_A, PAGE_B, PAGE_C])
        suggestions = suggest_terms(idx, "courg", max_results=2)
        assert len(suggestions) <= 2

    def test_suggest_case_insensitive(self) -> None:
        idx = build_index([PAGE_A])
        lower = suggest_terms(idx, "goo")
        upper = suggest_terms(idx, "GOO")
        assert lower == upper

    def test_suggest_empty_string_returns_empty(self) -> None:
        idx = build_index([PAGE_A])
        assert suggest_terms(idx, "") == []

    def test_suggest_whitespace_returns_empty(self) -> None:
        idx = build_index([PAGE_A])
        assert suggest_terms(idx, "   ") == []


# ---------------------------------------------------------------------------
# Guard-branch coverage for find / find_phrase / find_with_proximity / find_bm25
# ---------------------------------------------------------------------------

class TestGuardBranches:
    def test_find_second_term_missing_returns_empty(self) -> None:
        engine = _build()
        assert engine.find(["good", "xyznonexistent999"]) == []

    def test_find_phrase_empty_string_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_phrase("") == []

    def test_find_phrase_all_stopwords_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_phrase("the a an") == []

    def test_has_consecutive_positions_empty_terms(self) -> None:
        engine = _build()
        assert engine._has_consecutive_positions("https://example.com/1/", []) is False

    def test_proximity_missing_term_in_pair(self) -> None:
        engine = _build()
        # One term exists, one doesn't — min_distance should be inf, result empty
        result = engine.find_with_proximity(["good", "xyznonexistent999"])
        assert result == []

    def test_find_bm25_empty_terms_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_bm25([]) == []

    def test_find_bm25_missing_term_returns_empty(self) -> None:
        engine = _build()
        assert engine.find_bm25(["xyznonexistent999"]) == []
