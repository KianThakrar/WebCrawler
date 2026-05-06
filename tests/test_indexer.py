"""Tests for the inverted index module.

Strategy
--------
No network I/O needed – all tests operate on synthetic page dicts.
We cover:
  - tokenise: lowercasing, punctuation stripping, stopword removal
  - InvertedIndex: frequency counting, position tracking
  - TF-IDF: term frequency, IDF, combined score
  - build_index: full pipeline from page list to populated index
  - save_index / load_index: JSON round-trip serialisation
"""

from __future__ import annotations

import json
import os
import tempfile

from src.indexer import (
    InvertedIndex,
    build_index,
    load_index,
    save_index,
    tokenise,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_A = {
    "url": "https://example.com/page/1/",
    "title": "Page One",
    "content": "The quick brown fox jumps over the lazy dog",
    "quotes": [],
}

PAGE_B = {
    "url": "https://example.com/page/2/",
    "title": "Page Two",
    "content": "The fox ran quickly across the field",
    "quotes": [],
}

PAGE_C = {
    "url": "https://example.com/page/3/",
    "title": "Page Three",
    "content": "unique word only here",
    "quotes": [],
}


# ---------------------------------------------------------------------------
# tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_tokens(self) -> None:
        assert "hello" in tokenise("Hello World")

    def test_splits_on_whitespace(self) -> None:
        assert tokenise("one two three") == ["one", "two", "three"]

    def test_strips_punctuation(self) -> None:
        tokens = tokenise("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation_from_hyphenated(self) -> None:
        tokens = tokenise("well-known")
        assert all("," not in t and "." not in t for t in tokens)

    def test_removes_stopwords(self) -> None:
        tokens = tokenise("the quick brown fox")
        assert "the" not in tokens

    def test_removes_empty_tokens(self) -> None:
        tokens = tokenise("  hello   world  ")
        assert "" not in tokens

    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenise("") == []

    def test_all_stopwords_returns_empty(self) -> None:
        assert tokenise("the a an is") == []

    def test_numbers_kept(self) -> None:
        tokens = tokenise("page 42")
        assert "42" in tokens

    def test_returns_list(self) -> None:
        assert isinstance(tokenise("hello"), list)


# ---------------------------------------------------------------------------
# InvertedIndex – structure and insertion
# ---------------------------------------------------------------------------

class TestInvertedIndexStructure:
    def test_starts_empty(self) -> None:
        idx = InvertedIndex()
        assert len(idx) == 0

    def test_add_document_increases_term_count(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        assert len(idx) == 2

    def test_term_has_url_entry(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        assert "https://a.com/" in idx["hello"]

    def test_frequency_counted(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello hello world")
        assert idx["hello"]["https://a.com/"]["frequency"] == 2

    def test_positions_tracked(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world hello")
        positions = idx["hello"]["https://a.com/"]["positions"]
        assert positions == [0, 2]

    def test_multiple_documents(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/1/", "hello world")
        idx.add_document("https://a.com/2/", "hello there")
        assert len(idx["hello"]) == 2

    def test_word_not_in_index_raises(self) -> None:
        idx = InvertedIndex()
        try:
            idx["nonexistent"]  # noqa: B018
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_document_count_tracked(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/1/", "hello")
        idx.add_document("https://a.com/2/", "world")
        assert idx.document_count == 2

    def test_add_same_document_twice_overwrites(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello")
        idx.add_document("https://a.com/", "hello hello")
        assert idx["hello"]["https://a.com/"]["frequency"] == 2

    def test_re_indexing_same_url_does_not_double_count_documents(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        assert idx.document_count == 1
        idx.add_document("https://a.com/", "hello there")
        assert idx.document_count == 1
        idx.add_document("https://b.com/", "another page")
        assert idx.document_count == 2


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

class TestTfIdf:
    def test_compute_scores_adds_tfidf_key(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        idx.add_document("https://b.com/", "hello there")
        idx.compute_tfidf()
        assert "tf_idf" in idx["hello"]["https://a.com/"]

    def test_tfidf_is_float(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        idx.compute_tfidf()
        assert isinstance(idx["hello"]["https://a.com/"]["tf_idf"], float)

    def test_rare_word_has_higher_tfidf(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world unique")
        idx.add_document("https://b.com/", "hello world")
        idx.compute_tfidf()
        # "unique" only appears in 1/2 docs → higher IDF than "hello"
        score_unique = idx["unique"]["https://a.com/"]["tf_idf"]
        score_hello = idx["hello"]["https://a.com/"]["tf_idf"]
        assert score_unique > score_hello

    def test_tfidf_positive(self) -> None:
        idx = InvertedIndex()
        idx.add_document("https://a.com/", "hello world")
        idx.compute_tfidf()
        assert idx["hello"]["https://a.com/"]["tf_idf"] >= 0.0

    def test_word_in_all_docs_has_lower_idf(self) -> None:
        idx = InvertedIndex()
        for i in range(5):
            idx.add_document(f"https://a.com/{i}/", "hello unique")
        idx.compute_tfidf()
        # hello appears in all 5; unique appears in all 5 — same IDF
        # a word appearing in only 1 of 5 should have higher IDF
        idx2 = InvertedIndex()
        for i in range(5):
            content = "hello" if i > 0 else "hello rare"
            idx2.add_document(f"https://b.com/{i}/", content)
        idx2.compute_tfidf()
        score_rare = idx2["rare"]["https://b.com/0/"]["tf_idf"]
        score_hello = idx2["hello"]["https://b.com/0/"]["tf_idf"]
        assert score_rare > score_hello


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_returns_inverted_index(self) -> None:
        idx = build_index([PAGE_A])
        assert isinstance(idx, InvertedIndex)

    def test_words_from_content_indexed(self) -> None:
        idx = build_index([PAGE_A])
        assert "fox" in idx

    def test_stopwords_excluded(self) -> None:
        idx = build_index([PAGE_A])
        assert "the" not in idx

    def test_multiple_pages_indexed(self) -> None:
        idx = build_index([PAGE_A, PAGE_B])
        assert idx.document_count == 2

    def test_tfidf_computed_after_build(self) -> None:
        idx = build_index([PAGE_A, PAGE_B])
        url = PAGE_A["url"]
        assert "tf_idf" in idx["fox"][url]

    def test_empty_page_list(self) -> None:
        idx = build_index([])
        assert len(idx) == 0

    def test_page_with_empty_content(self) -> None:
        page = {"url": "https://a.com/", "title": "", "content": "", "quotes": []}
        idx = build_index([page])
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_creates_file(self) -> None:
        idx = build_index([PAGE_A])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            assert os.path.exists(path)

    def test_load_returns_inverted_index(self) -> None:
        idx = build_index([PAGE_A])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)
        assert isinstance(loaded, InvertedIndex)

    def test_round_trip_preserves_terms(self) -> None:
        idx = build_index([PAGE_A, PAGE_B])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)
        assert "fox" in loaded

    def test_round_trip_preserves_frequency(self) -> None:
        idx = build_index([PAGE_A])
        url = PAGE_A["url"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)
        assert loaded["fox"][url]["frequency"] == idx["fox"][url]["frequency"]

    def test_round_trip_preserves_tfidf(self) -> None:
        idx = build_index([PAGE_A, PAGE_B])
        url = PAGE_A["url"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)
        assert abs(loaded["fox"][url]["tf_idf"] - idx["fox"][url]["tf_idf"]) < 1e-9

    def test_round_trip_preserves_document_count(self) -> None:
        idx = build_index([PAGE_A, PAGE_B, PAGE_C])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)
        assert loaded.document_count == 3

    def test_saved_file_is_valid_json(self) -> None:
        idx = build_index([PAGE_A])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            with open(path) as f:
                data = json.load(f)
        assert isinstance(data, dict)

    def test_load_missing_file_raises(self) -> None:
        try:
            load_index("/tmp/does_not_exist_xyz.json")
            assert False, "Should have raised"
        except FileNotFoundError:
            pass

    def test_load_corrupt_json_raises_value_error(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ not valid json }")
            path = f.name
        try:
            load_index(path)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_load_missing_keys_raises_value_error(self) -> None:
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"wrong_key": {}}, f)
            path = f.name
        try:
            load_index(path)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_save_creates_nested_directory(self) -> None:
        idx = build_index([PAGE_A])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "deep", "index.json")
            save_index(idx, path)
            assert os.path.exists(path)
