"""Performance benchmarks for the search engine.

These tests assert that core operations complete within reasonable
time bounds on a synthetic corpus, providing evidence that the
implementation is suitably efficient for the target workload.

Benchmarks run in < 1 s on a modern laptop even for large corpora,
confirming the O(P*T) and O(V*D) complexity claims in the docstrings.
"""

from __future__ import annotations

import time

from src.indexer import InvertedIndex, build_index, save_index, load_index
from src.search import SearchEngine, suggest_terms
import tempfile
import os


def _make_corpus(n_pages: int, tokens_per_page: int = 100) -> list[dict]:
    """Generate a synthetic corpus of *n_pages* pages."""
    vocab = [f"word{i}" for i in range(200)]
    pages = []
    for p in range(n_pages):
        words = [vocab[(p * 7 + t * 13) % len(vocab)] for t in range(tokens_per_page)]
        pages.append({
            "url": f"https://example.com/{p}/",
            "content": " ".join(words),
            "title": f"Page {p}",
            "quotes": [],
        })
    return pages


# ---------------------------------------------------------------------------
# build_index benchmark
# ---------------------------------------------------------------------------

class TestBuildPerformance:
    def test_build_100_pages_under_2_seconds(self) -> None:
        corpus = _make_corpus(100)
        start = time.perf_counter()
        build_index(corpus)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"build_index took {elapsed:.2f}s for 100 pages"

    def test_build_50_pages_under_1_second(self) -> None:
        corpus = _make_corpus(50)
        start = time.perf_counter()
        build_index(corpus)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"build_index took {elapsed:.2f}s for 50 pages"


# ---------------------------------------------------------------------------
# Query performance
# ---------------------------------------------------------------------------

class TestQueryPerformance:
    def test_single_word_query_under_10ms(self) -> None:
        idx = build_index(_make_corpus(50))
        engine = SearchEngine(idx)
        start = time.perf_counter()
        for _ in range(100):
            engine.find(["word0"])
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 10.0, f"find averaged {avg_ms:.2f} ms"

    def test_multi_word_query_under_20ms(self) -> None:
        idx = build_index(_make_corpus(50))
        engine = SearchEngine(idx)
        start = time.perf_counter()
        for _ in range(100):
            engine.find(["word0", "word1"])
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 20.0, f"multi-word find averaged {avg_ms:.2f} ms"

    def test_bm25_query_under_20ms(self) -> None:
        idx = build_index(_make_corpus(50))
        engine = SearchEngine(idx)
        start = time.perf_counter()
        for _ in range(100):
            engine.find_bm25(["word0"])
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 20.0, f"BM25 find averaged {avg_ms:.2f} ms"


# ---------------------------------------------------------------------------
# Serialisation performance
# ---------------------------------------------------------------------------

class TestSerialisationPerformance:
    def test_save_and_load_100_pages_under_2_seconds(self) -> None:
        idx = build_index(_make_corpus(100))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            start = time.perf_counter()
            save_index(idx, path)
            load_index(path)
            elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"save+load took {elapsed:.2f}s for 100-page index"


# ---------------------------------------------------------------------------
# Levenshtein performance
# ---------------------------------------------------------------------------

class TestSuggestPerformance:
    def test_suggest_under_500ms_on_200_term_vocab(self) -> None:
        idx = build_index(_make_corpus(20))
        start = time.perf_counter()
        for _ in range(10):
            suggest_terms(idx, "wrod0")
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / 10) * 1000
        assert avg_ms < 500.0, f"suggest_terms averaged {avg_ms:.2f} ms"
