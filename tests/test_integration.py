"""Integration tests for the full crawl → index → search pipeline.

These tests exercise multiple modules together to verify correct
end-to-end behaviour.  Network calls are mocked; file I/O uses
temporary directories so no side-effects persist between tests.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

from src.crawler import BASE_URL
from src.indexer import build_index, load_index, save_index
from src.main import Shell
from src.search import SearchEngine

# ---------------------------------------------------------------------------
# Minimal multi-page HTML fixtures
# ---------------------------------------------------------------------------

PAGE_1_HTML = """
<html><head><title>Quotes – Page 1</title></head>
<body>
  <div class="quote">
    <span class="text">&#8220;The world is what we make it&#8221;</span>
    <small class="author">Einstein</small>
    <div class="tags"><a class="tag" href="/tag/world/">world</a></div>
  </div>
  <ul class="pager"><li class="next"><a href="/page/2/">Next</a></li></ul>
</body></html>
"""

PAGE_2_HTML = """
<html><head><title>Quotes – Page 2</title></head>
<body>
  <div class="quote">
    <span class="text">&#8220;Courage is the world&#8217;s greatest gift&#8221;</span>
    <small class="author">Hemingway</small>
    <div class="tags"><a class="tag" href="/tag/courage/">courage</a></div>
  </div>
</body></html>
"""


def _make_response(html: str, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = html
    r.raise_for_status = MagicMock()
    return r


# ---------------------------------------------------------------------------
# Crawl → index pipeline
# ---------------------------------------------------------------------------

class TestCrawlToIndex:
    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_two_pages_produce_populated_index(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(PAGE_1_HTML),
            _make_response(PAGE_2_HTML),
        ]
        from src.crawler import crawl
        pages = crawl(BASE_URL)
        idx = build_index(pages)
        assert len(idx) > 0

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_word_from_page_1_found_in_index(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(PAGE_1_HTML),
            _make_response(PAGE_2_HTML),
        ]
        from src.crawler import crawl
        pages = crawl(BASE_URL)
        idx = build_index(pages)
        assert "world" in idx

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_word_from_page_2_found_in_index(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(PAGE_1_HTML),
            _make_response(PAGE_2_HTML),
        ]
        from src.crawler import crawl
        pages = crawl(BASE_URL)
        idx = build_index(pages)
        assert "courage" in idx

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_tfidf_scores_computed(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(PAGE_1_HTML),
            _make_response(PAGE_2_HTML),
        ]
        from src.crawler import crawl
        pages = crawl(BASE_URL)
        idx = build_index(pages)
        url = f"{BASE_URL}"
        assert idx["world"][url]["tf_idf"] > 0


# ---------------------------------------------------------------------------
# Index → save → load round-trip
# ---------------------------------------------------------------------------

class TestIndexPersistence:
    def test_save_and_load_preserves_search_results(self) -> None:
        pages = [
            {"url": "https://a.com/1/", "content": "courage wisdom", "title": "", "quotes": []},
            {"url": "https://a.com/2/", "content": "wisdom knowledge", "title": "", "quotes": []},
        ]
        idx = build_index(pages)
        engine_before = SearchEngine(idx)
        results_before = engine_before.find(["wisdom"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)

        engine_after = SearchEngine(loaded)
        results_after = engine_after.find(["wisdom"])
        assert results_before == results_after

    def test_print_entry_same_before_and_after_load(self) -> None:
        pages = [{"url": "https://a.com/", "content": "hello world", "title": "", "quotes": []}]
        idx = build_index(pages)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            save_index(idx, path)
            loaded = load_index(path)

        assert SearchEngine(idx).print_entry("hello") == SearchEngine(loaded).print_entry("hello")


# ---------------------------------------------------------------------------
# Shell end-to-end
# ---------------------------------------------------------------------------

class TestShellEndToEnd:
    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_build_then_find_returns_results(
        self, mock_crawl: MagicMock, mock_build: MagicMock, mock_save: MagicMock
    ) -> None:
        pages = [{"url": "https://a.com/", "content": "wisdom courage", "title": "", "quotes": []}]
        mock_crawl.return_value = pages
        mock_build.return_value = build_index(pages)

        shell = Shell()
        shell.run_command("build")
        output = shell.run_command("find wisdom")
        assert "https://a.com/" in output

    @patch("src.main.load_index")
    def test_load_then_print_returns_entry(self, mock_load: MagicMock) -> None:
        pages = [{"url": "https://a.com/", "content": "wisdom courage", "title": "", "quotes": []}]
        mock_load.return_value = build_index(pages)

        shell = Shell()
        shell.run_command("load")
        output = shell.run_command("print wisdom")
        assert "wisdom" in output.lower()

    @patch("src.main.load_index")
    def test_load_then_find_multi_word(self, mock_load: MagicMock) -> None:
        pages = [{"url": "https://a.com/", "content": "wisdom courage knowledge", "title": "", "quotes": []}]
        mock_load.return_value = build_index(pages)

        shell = Shell()
        shell.run_command("load")
        output = shell.run_command("find wisdom courage")
        assert "https://a.com/" in output

    def test_find_before_load_returns_helpful_message(self) -> None:
        shell = Shell()
        output = shell.run_command("find wisdom")
        assert any(word in output.lower() for word in ("build", "load", "index"))

    def test_print_before_load_returns_helpful_message(self) -> None:
        shell = Shell()
        output = shell.run_command("print wisdom")
        assert any(word in output.lower() for word in ("build", "load", "index"))
