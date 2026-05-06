"""Tests for the web crawler module.

Strategy
--------
All network I/O is mocked so tests are fast and deterministic.
We cover:
  - HTML parsing helpers (clean_text, parse_quote_page)
  - Crawl logic (BFS, dedup, domain restriction, politeness delay)
  - Error handling (HTTP errors, network failures, malformed HTML)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.crawler import (
    BASE_URL,
    MIN_POLITENESS_DELAY_SECONDS,
    clean_text,
    crawl,
    parse_author_page,
    parse_quote_page,
)
from src.crawler import _is_crawlable, _discover_links  # type: ignore  # internal helpers tested directly


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_PAGE_HTML = """
<html>
<head><title>Test Page 1</title></head>
<body>
  <div class="quote">
    <span class="text">&#8220;The world as we have created it&#8221;</span>
    <small class="author">Albert Einstein</small>
    <div class="tags">
      <a class="tag" href="/tag/change/">change</a>
      <a class="tag" href="/tag/deep-thoughts/">deep-thoughts</a>
    </div>
  </div>
  <div class="quote">
    <span class="text">&#8220;It is our choices that show what we truly are&#8221;</span>
    <small class="author">J.K. Rowling</small>
    <div class="tags">
      <a class="tag" href="/tag/choices/">choices</a>
    </div>
  </div>
  <nav>
    <ul class="pager">
      <li class="next"><a href="/page/2/">Next</a></li>
    </ul>
  </nav>
</body>
</html>
"""

LAST_PAGE_HTML = """
<html>
<head><title>Test Page 2</title></head>
<body>
  <div class="quote">
    <span class="text">&#8220;A single quote on the final page&#8221;</span>
    <small class="author">Unknown</small>
  </div>
</body>
</html>
"""

MALFORMED_HTML = "<html><body><div class='quote'></div></body></html>"

EMPTY_HTML = ""


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_surrounding_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_strips_curly_quotes(self) -> None:
        assert clean_text("“hello”") == "hello"

    def test_strips_straight_quotes(self) -> None:
        assert clean_text('"hello"') == "hello"

    def test_strips_newlines(self) -> None:
        assert clean_text("\nhello\nworld\n") == "hello world"

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_only_whitespace(self) -> None:
        assert clean_text("   ") == ""


# ---------------------------------------------------------------------------
# parse_quote_page
# ---------------------------------------------------------------------------

class TestParseQuotePage:
    def test_extracts_page_title(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert result["title"] == "Test Page 1"

    def test_extracts_quotes(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert len(result["quotes"]) == 2

    def test_quote_has_text_author_tags(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        first = result["quotes"][0]
        assert "text" in first
        assert "author" in first
        assert "tags" in first

    def test_quote_text_cleaned(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert result["quotes"][0]["text"].startswith("The world")

    def test_tags_extracted(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert "change" in result["quotes"][0]["tags"]

    def test_next_page_url_resolved(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert result["next_url"] == "https://quotes.toscrape.com/page/2/"

    def test_no_next_url_on_last_page(self) -> None:
        result = parse_quote_page(LAST_PAGE_HTML, BASE_URL)
        assert result["next_url"] is None

    def test_content_text_non_empty(self) -> None:
        result = parse_quote_page(SIMPLE_PAGE_HTML, BASE_URL)
        assert len(result["content"]) > 0

    def test_malformed_html_returns_empty_quotes(self) -> None:
        result = parse_quote_page(MALFORMED_HTML, BASE_URL)
        assert result["quotes"] == []

    def test_empty_html_returns_defaults(self) -> None:
        result = parse_quote_page(EMPTY_HTML, BASE_URL)
        assert result["quotes"] == []
        assert result["next_url"] is None
        assert result["title"] == ""


# ---------------------------------------------------------------------------
# crawl
# ---------------------------------------------------------------------------

def _make_response(html: str, status_code: int = 200) -> MagicMock:
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


class TestCrawl:
    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_crawls_all_pages(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        mock_get.side_effect = [
            _make_response(SIMPLE_PAGE_HTML),
            _make_response(LAST_PAGE_HTML),
        ]
        pages = crawl(BASE_URL)
        assert len(pages) == 2

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_returns_page_data_with_url(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        mock_get.return_value = _make_response(LAST_PAGE_HTML)
        pages = crawl(BASE_URL)
        assert pages[0]["url"] == BASE_URL

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_politeness_sleep_called_between_pages(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(SIMPLE_PAGE_HTML),
            _make_response(LAST_PAGE_HTML),
        ]
        crawl(BASE_URL)
        # sleep must be called at least once with >= 6 seconds
        assert mock_sleep.call_count >= 1
        for c in mock_sleep.call_args_list:
            assert c.args[0] >= MIN_POLITENESS_DELAY_SECONDS

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_does_not_revisit_urls(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        # next_url on page 2 loops back to page 1 — should be ignored
        looping_html = SIMPLE_PAGE_HTML.replace("/page/2/", "/")
        mock_get.side_effect = [
            _make_response(looping_html),
        ]
        pages = crawl(BASE_URL)
        assert len(pages) == 1

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_http_error_skips_page(self, mock_get: MagicMock, _mock_sleep: MagicMock) -> None:
        mock_get.side_effect = [
            _make_response(SIMPLE_PAGE_HTML),
            _make_response("", status_code=404),
        ]
        # should not raise; bad page simply not included
        pages = crawl(BASE_URL)
        assert len(pages) == 1

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_network_exception_skips_page(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(SIMPLE_PAGE_HTML),
            Exception("Connection refused"),
        ]
        pages = crawl(BASE_URL)
        assert len(pages) == 1

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_progress_callback_called_per_page(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = [
            _make_response(SIMPLE_PAGE_HTML),
            _make_response(LAST_PAGE_HTML),
        ]
        progress_calls: list[str] = []
        crawl(BASE_URL, on_progress=lambda url, n, total: progress_calls.append(url))
        assert len(progress_calls) == 2

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_respects_max_pages_limit(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.return_value = _make_response(SIMPLE_PAGE_HTML)
        pages = crawl(BASE_URL, max_pages=1)
        assert len(pages) == 1

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_get_called_with_correct_url(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.return_value = _make_response(LAST_PAGE_HTML)
        crawl(BASE_URL)
        call_kwargs = mock_get.call_args
        assert call_kwargs.args[0] == BASE_URL
        assert call_kwargs.kwargs.get("timeout") == 10
        assert "User-Agent" in call_kwargs.kwargs.get("headers", {})

    @patch("src.crawler.time.sleep")
    @patch("src.crawler.requests.get")
    def test_empty_crawl_on_immediate_http_error(
        self, mock_get: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_get.side_effect = Exception("DNS failure")
        pages = crawl(BASE_URL)
        assert pages == []


# ---------------------------------------------------------------------------
# Internal-link filter (_is_crawlable)
# ---------------------------------------------------------------------------

class TestIsCrawlable:
    def test_homepage_is_crawlable(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/") is True

    def test_pagination_is_crawlable(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/page/3/") is True

    def test_author_page_is_crawlable(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/author/Albert-Einstein/") is True

    def test_tag_page_is_crawlable(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/tag/wisdom/") is True

    def test_paginated_tag_is_crawlable(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/tag/love/page/2/") is True

    def test_external_host_rejected(self) -> None:
        assert _is_crawlable("https://en.wikipedia.org/wiki/Albert_Einstein") is False

    def test_login_path_rejected(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/login") is False

    def test_static_asset_rejected(self) -> None:
        assert _is_crawlable("https://quotes.toscrape.com/static/main.css") is False


# ---------------------------------------------------------------------------
# Author page parser
# ---------------------------------------------------------------------------

AUTHOR_PAGE_HTML = """
<html>
<body>
  <div class="author-details">
    <h3 class="author-title">Albert Einstein</h3>
    <p>
      <span class="author-born-date">March 14, 1879</span>
      <span class="author-born-location">in Ulm, Germany</span>
    </p>
    <div class="author-description">
      Albert Einstein was a German-born theoretical physicist who developed
      the theory of relativity, one of the two pillars of modern physics.
    </div>
  </div>
</body>
</html>
"""


class TestParseAuthorPage:
    def test_extracts_name(self) -> None:
        result = parse_author_page(AUTHOR_PAGE_HTML, "https://quotes.toscrape.com/author/Albert-Einstein/")
        assert "Albert Einstein" in result["title"]

    def test_content_includes_name_and_bio(self) -> None:
        result = parse_author_page(AUTHOR_PAGE_HTML, "https://quotes.toscrape.com/author/Albert-Einstein/")
        content = result["content"]
        assert "Albert Einstein" in content
        assert "physicist" in content
        assert "relativity" in content

    def test_content_includes_birth_metadata(self) -> None:
        result = parse_author_page(AUTHOR_PAGE_HTML, "https://quotes.toscrape.com/author/Albert-Einstein/")
        assert "1879" in result["content"]
        assert "Germany" in result["content"]

    def test_no_discovered_links(self) -> None:
        result = parse_author_page(AUTHOR_PAGE_HTML, "https://quotes.toscrape.com/author/Albert-Einstein/")
        assert result["discovered_links"] == []

    def test_no_next_url(self) -> None:
        result = parse_author_page(AUTHOR_PAGE_HTML, "https://quotes.toscrape.com/author/Albert-Einstein/")
        assert result["next_url"] is None

    def test_empty_html_returns_empty_content(self) -> None:
        result = parse_author_page("", "https://quotes.toscrape.com/author/X/")
        assert result["content"] == ""

    def test_partial_author_page_handles_missing_fields(self) -> None:
        partial = "<html><body><h3 class='author-title'>Anon</h3></body></html>"
        result = parse_author_page(partial, "https://quotes.toscrape.com/author/Anon/")
        assert "Anon" in result["title"]


# ---------------------------------------------------------------------------
# Listing-page link discovery
# ---------------------------------------------------------------------------

LISTING_WITH_AUTHOR_AND_TAG_LINKS = """
<html>
<body>
  <div class="quote">
    <span class="text">&#8220;Quote A&#8221;</span>
    <small class="author">Albert Einstein</small>
    <a href="/author/Albert-Einstein">(about)</a>
    <div class="tags">
      <a class="tag" href="/tag/love/">love</a>
      <a class="tag" href="/tag/wisdom/">wisdom</a>
    </div>
  </div>
  <nav>
    <ul class="pager">
      <li class="next"><a href="/page/2/">Next</a></li>
    </ul>
  </nav>
</body>
</html>
"""


class TestListingPageLinkDiscovery:
    def test_discovers_author_links(self) -> None:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(LISTING_WITH_AUTHOR_AND_TAG_LINKS, "html.parser")
        links = _discover_links(soup, "https://quotes.toscrape.com/")
        assert any("/author/Albert-Einstein" in u for u in links)

    def test_discovers_tag_links(self) -> None:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(LISTING_WITH_AUTHOR_AND_TAG_LINKS, "html.parser")
        links = _discover_links(soup, "https://quotes.toscrape.com/")
        assert any("/tag/love/" in u for u in links)
        assert any("/tag/wisdom/" in u for u in links)

    def test_discovers_pagination(self) -> None:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(LISTING_WITH_AUTHOR_AND_TAG_LINKS, "html.parser")
        links = _discover_links(soup, "https://quotes.toscrape.com/")
        assert any("/page/2/" in u for u in links)

    def test_parse_quote_page_returns_discovered_links(self) -> None:
        result = parse_quote_page(LISTING_WITH_AUTHOR_AND_TAG_LINKS, "https://quotes.toscrape.com/")
        assert "discovered_links" in result
        assert any("/author/" in u for u in result["discovered_links"])
        assert any("/tag/" in u for u in result["discovered_links"])
