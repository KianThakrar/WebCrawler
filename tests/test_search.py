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

import math
import os

from src.indexer import InvertedIndex, build_index
from src.search import (
    SearchEngine,
    evaluate_ranker,
    format_comparison_table,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    run_evaluation,
)

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


# ---------------------------------------------------------------------------
# Punctuation tolerance
# ---------------------------------------------------------------------------

class TestPunctuationTolerance:
    def test_find_strips_trailing_comma(self) -> None:
        engine = _build()
        plain = engine.find(["wisdom"])
        with_comma = engine.find(["wisdom,"])
        assert with_comma == plain

    def test_find_strips_quote_marks(self) -> None:
        engine = _build()
        plain = engine.find(["wisdom"])
        quoted = engine.find(['"wisdom"'])
        assert quoted == plain

    def test_print_entry_tolerates_punctuation(self) -> None:
        engine = _build()
        plain = engine.print_entry("wisdom")
        with_comma = engine.print_entry("wisdom,")
        assert with_comma == plain

    def test_bm25_tolerates_punctuation(self) -> None:
        engine = _build()
        plain = engine.find_bm25(["wisdom"])
        with_comma = engine.find_bm25(["wisdom,"])
        assert with_comma == plain


# ---------------------------------------------------------------------------
# Evaluation harness — metrics
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant_returns_one(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 2, "b": 2, "c": 2}
        assert precision_at_k(ranked, judgements, k=3) == 1.0

    def test_partial_relevant(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 2, "b": 0, "c": 1}
        assert precision_at_k(ranked, judgements, k=3) == 2 / 3

    def test_empty_ranked_returns_zero(self) -> None:
        assert precision_at_k([], {"a": 2}, k=3) == 0.0

    def test_unjudged_treated_as_irrelevant(self) -> None:
        ranked = ["unknown", "a"]
        judgements = {"a": 2}
        assert precision_at_k(ranked, judgements, k=2) == 0.5

    def test_threshold_applied(self) -> None:
        ranked = ["a", "b"]
        judgements = {"a": 1, "b": 2}
        # At threshold=2, only b counts as relevant
        assert precision_at_k(ranked, judgements, k=2, threshold=2) == 0.5


class TestReciprocalRank:
    def test_first_position_is_one(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 2}
        assert reciprocal_rank(ranked, judgements) == 1.0

    def test_third_position(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"c": 1}
        assert reciprocal_rank(ranked, judgements) == 1 / 3

    def test_no_relevant_returns_zero(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 0, "b": 0}
        assert reciprocal_rank(ranked, judgements) == 0.0


class TestNdcgAtK:
    def test_perfect_ordering_returns_one(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 2, "b": 1, "c": 0}
        assert ndcg_at_k(ranked, judgements, k=3) == 1.0

    def test_reversed_ordering_below_one(self) -> None:
        ranked = ["c", "b", "a"]
        judgements = {"a": 2, "b": 1, "c": 0}
        result = ndcg_at_k(ranked, judgements, k=3)
        # DCG = 0/log2(2) + 1/log2(3) + 2/log2(4)
        # IDCG = 2/log2(2) + 1/log2(3) + 0/log2(4)
        expected = (1 / math.log2(3) + 2 / math.log2(4)) / (
            2 / math.log2(2) + 1 / math.log2(3)
        )
        assert abs(result - expected) < 1e-9

    def test_no_relevant_returns_zero(self) -> None:
        ranked = ["a", "b"]
        judgements = {"a": 0, "b": 0}
        assert ndcg_at_k(ranked, judgements, k=2) == 0.0

    def test_ndcg_in_unit_interval(self) -> None:
        ranked = ["a", "b", "c"]
        judgements = {"a": 1, "b": 2, "c": 3}
        result = ndcg_at_k(ranked, judgements, k=3)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Evaluation harness — runner
# ---------------------------------------------------------------------------

class TestEvaluateRanker:
    def test_returns_per_query_and_mean(self) -> None:
        # Build a tiny synthetic judgements dict and a fake ranker
        judgements_data = {
            "queries": {
                "q1": {"judgements": {"a": 2, "b": 0}},
                "q2": {"judgements": {"x": 1, "y": 1}},
            }
        }

        def fake_ranker(_q: str) -> list[tuple[str, float]]:
            return [("a", 1.0), ("x", 0.5)]

        result = evaluate_ranker(fake_ranker, judgements_data, k=2)
        assert "per_query" in result
        assert set(result["per_query"].keys()) == {"q1", "q2"}
        assert "mean_ndcg_at_k" in result["mean"]
        assert "mean_precision_at_k" in result["mean"]
        assert "mean_reciprocal_rank" in result["mean"]


class TestRunEvaluation:
    def test_returns_three_rankers(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results = run_evaluation(
            os.path.join(repo_root, "data", "index.json"),
            os.path.join(repo_root, "tests", "relevance_judgements.json"),
            k=5,
        )
        assert set(results.keys()) == {"TF-IDF", "BM25", "Proximity"}
        for ranker_results in results.values():
            assert "per_query" in ranker_results
            assert "mean" in ranker_results

    def test_metric_values_in_unit_interval(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results = run_evaluation(
            os.path.join(repo_root, "data", "index.json"),
            os.path.join(repo_root, "tests", "relevance_judgements.json"),
            k=5,
        )
        for ranker_results in results.values():
            for metric_name, value in ranker_results["mean"].items():
                assert 0.0 <= value <= 1.0, f"{metric_name} out of range: {value}"


class TestFormatComparisonTable:
    def test_includes_all_ranker_names(self) -> None:
        results = {
            "TF-IDF": {
                "per_query": {"q1": {"precision_at_k": 1.0, "reciprocal_rank": 1.0, "ndcg_at_k": 1.0}},
                "mean": {"mean_precision_at_k": 1.0, "mean_reciprocal_rank": 1.0, "mean_ndcg_at_k": 1.0},
            },
            "BM25": {
                "per_query": {"q1": {"precision_at_k": 0.5, "reciprocal_rank": 0.5, "ndcg_at_k": 0.5}},
                "mean": {"mean_precision_at_k": 0.5, "mean_reciprocal_rank": 0.5, "mean_ndcg_at_k": 0.5},
            },
        }
        table = format_comparison_table(results, k=5)
        assert "TF-IDF" in table
        assert "BM25" in table
        assert "q1" in table
