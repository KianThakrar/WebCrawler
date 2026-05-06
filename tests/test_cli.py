"""Tests for the interactive CLI shell.

Strategy
--------
The Shell class is tested in isolation by patching crawl, build_index,
save_index, load_index, and the SearchEngine so no network or file I/O
occurs.  We verify command dispatch, output messages, and error handling.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.indexer import InvertedIndex, build_index
from src.main import Shell

_JUDGEMENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "relevance_judgements.json",
)
_JUDGEMENTS_AVAILABLE = os.path.exists(_JUDGEMENTS_PATH)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAGE_A = {
    "url": "https://example.com/page/1/",
    "content": "wisdom brings great courage",
    "title": "Page One",
    "quotes": [],
}


def _shell_with_index() -> Shell:
    """Return a Shell pre-loaded with a small index."""
    shell = Shell()
    shell._index = build_index([PAGE_A])
    from src.search import SearchEngine
    shell._engine = SearchEngine(shell._index)
    return shell


# ---------------------------------------------------------------------------
# Shell initialisation
# ---------------------------------------------------------------------------

class TestShellInit:
    def test_shell_creates_instance(self) -> None:
        assert Shell() is not None

    def test_index_is_none_before_build_or_load(self) -> None:
        shell = Shell()
        assert shell._index is None

    def test_engine_is_none_before_build_or_load(self) -> None:
        shell = Shell()
        assert shell._engine is None


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------

class TestBuildCommand:
    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_build_calls_crawl(
        self, mock_crawl: MagicMock, mock_build: MagicMock, _mock_save: MagicMock
    ) -> None:
        mock_crawl.return_value = [PAGE_A]
        mock_build.return_value = InvertedIndex()
        shell = Shell()
        shell.run_command("build")
        mock_crawl.assert_called_once()

    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_build_calls_build_index(
        self, mock_crawl: MagicMock, mock_build: MagicMock, _mock_save: MagicMock
    ) -> None:
        mock_crawl.return_value = [PAGE_A]
        mock_build.return_value = InvertedIndex()
        shell = Shell()
        shell.run_command("build")
        mock_build.assert_called_once_with([PAGE_A])

    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_build_sets_engine(
        self, mock_crawl: MagicMock, mock_build: MagicMock, _mock_save: MagicMock
    ) -> None:
        mock_crawl.return_value = [PAGE_A]
        mock_build.return_value = build_index([PAGE_A])
        shell = Shell()
        shell.run_command("build")
        assert shell._engine is not None

    @patch("src.main.save_index")
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_build_output_mentions_pages(
        self, mock_crawl: MagicMock, mock_build: MagicMock, _mock_save: MagicMock
    ) -> None:
        mock_crawl.return_value = [PAGE_A]
        mock_build.return_value = InvertedIndex()
        shell = Shell()
        output = shell.run_command("build")
        assert "page" in output.lower() or "index" in output.lower()


# ---------------------------------------------------------------------------
# load command
# ---------------------------------------------------------------------------

class TestLoadCommand:
    @patch("src.main.load_index")
    def test_load_calls_load_index(self, mock_load: MagicMock) -> None:
        mock_load.return_value = build_index([PAGE_A])
        shell = Shell()
        with tempfile.TemporaryDirectory() as tmpdir:
            shell._index_path = os.path.join(tmpdir, "index.json")
            shell.run_command("load")
        mock_load.assert_called_once()

    @patch("src.main.load_index")
    def test_load_sets_engine(self, mock_load: MagicMock) -> None:
        mock_load.return_value = build_index([PAGE_A])
        shell = Shell()
        with tempfile.TemporaryDirectory() as tmpdir:
            shell._index_path = os.path.join(tmpdir, "index.json")
            shell.run_command("load")
        assert shell._engine is not None

    def test_load_missing_file_returns_error_message(self) -> None:
        shell = Shell()
        shell._index_path = "/tmp/does_not_exist_abc123.json"
        output = shell.run_command("load")
        assert "error" in output.lower() or "not found" in output.lower() or "no index" in output.lower()

    @patch("src.main.load_index")
    def test_load_corrupt_file_returns_error_message(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = ValueError("Index file is not valid JSON")
        shell = Shell()
        output = shell.run_command("load")
        assert "error" in output.lower()


# ---------------------------------------------------------------------------
# help command
# ---------------------------------------------------------------------------

class TestHelpCommand:
    def test_help_lists_build(self) -> None:
        assert "build" in Shell().run_command("help")

    def test_help_lists_find(self) -> None:
        assert "find" in Shell().run_command("help")

    def test_help_lists_phrase(self) -> None:
        assert "phrase" in Shell().run_command("help")

    def test_help_lists_bm25(self) -> None:
        assert "bm25" in Shell().run_command("help")


# ---------------------------------------------------------------------------
# print command
# ---------------------------------------------------------------------------

class TestPrintCommand:
    def test_print_known_word_returns_output(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("print wisdom")
        assert "wisdom" in output.lower()

    def test_print_unknown_word_returns_not_found(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("print xyznonexistent")
        assert "not found" in output.lower()

    def test_print_without_index_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("print wisdom")
        assert "load" in output.lower() or "build" in output.lower() or "index" in output.lower()

    def test_print_no_argument_returns_usage(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("print")
        assert "usage" in output.lower() or "word" in output.lower() or "provide" in output.lower()


# ---------------------------------------------------------------------------
# find command
# ---------------------------------------------------------------------------

class TestFindCommand:
    def test_find_known_word_returns_results(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find wisdom")
        assert "example.com" in output

    def test_find_unknown_word_returns_no_results_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find xyznonexistent")
        assert "no" in output.lower() or "not found" in output.lower()

    def test_find_multi_word_returns_output(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find wisdom courage")
        assert isinstance(output, str)

    def test_find_without_index_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("find wisdom")
        assert "load" in output.lower() or "build" in output.lower() or "index" in output.lower()

    def test_find_no_argument_returns_usage(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find")
        assert "usage" in output.lower() or "word" in output.lower() or "provide" in output.lower()

    def test_find_output_contains_url(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find wisdom")
        assert "https://" in output

    def test_find_output_contains_score(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find wisdom")
        assert "score" in output.lower() or "." in output

    def test_find_punctuation_only_returns_normalisation_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find ,,,")
        assert "no searchable terms" in output.lower()

    def test_find_stopwords_only_returns_normalisation_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("find the and of")
        assert "no searchable terms" in output.lower()

    def test_phrase_punctuation_only_returns_normalisation_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("phrase ,,, !!!")
        assert "no searchable terms" in output.lower()

    def test_bm25_stopwords_only_returns_normalisation_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("bm25 the and")
        assert "no searchable terms" in output.lower()


# ---------------------------------------------------------------------------
# Unknown / empty commands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# phrase command (advanced)
# ---------------------------------------------------------------------------

class TestPhraseCommand:
    def test_phrase_known_words_returns_output(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("phrase wisdom courage")
        assert isinstance(output, str)

    def test_phrase_without_index_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("phrase wisdom")
        assert any(w in output.lower() for w in ("build", "load", "index"))

    def test_phrase_no_argument_returns_usage(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("phrase")
        assert "usage" in output.lower() or "word" in output.lower()

    def test_phrase_not_found_returns_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("phrase xyznonexistent")
        assert "no" in output.lower() or "not found" in output.lower()


# ---------------------------------------------------------------------------
# bm25 command (advanced)
# ---------------------------------------------------------------------------

class TestBm25Command:
    def test_bm25_known_word_returns_results(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("bm25 wisdom")
        assert "example.com" in output

    def test_bm25_without_index_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("bm25 wisdom")
        assert any(w in output.lower() for w in ("build", "load", "index"))

    def test_bm25_no_argument_returns_usage(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("bm25")
        assert "usage" in output.lower() or "word" in output.lower()

    def test_bm25_not_found_returns_message(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("bm25 xyznonexistent")
        assert "no" in output.lower() or "not found" in output.lower()


class TestEvalCommand:
    @pytest.mark.skipif(
        not _JUDGEMENTS_AVAILABLE,
        reason="relevance_judgements.json is a local-only artefact (gitignored)",
    )
    def test_eval_returns_comparison_table(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("eval")
        # Output should be the markdown comparison table from format_comparison_table
        assert "TF-IDF" in output
        assert "BM25" in output
        assert "NDCG" in output

    def test_eval_with_missing_index_returns_error(self, tmp_path) -> None:
        from src.main import Shell
        shell = Shell(index_path=str(tmp_path / "nonexistent.json"))
        output = shell.run_command("eval")
        assert "error" in output.lower() or "cannot" in output.lower()

    def test_help_lists_eval(self) -> None:
        from src.main import Shell
        assert "eval" in Shell().run_command("help")


class TestUnknownCommands:
    def test_unknown_command_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("foobar")
        assert "unknown" in output.lower() or "command" in output.lower()

    def test_empty_input_returns_empty_or_prompt(self) -> None:
        shell = Shell()
        output = shell.run_command("")
        assert isinstance(output, str)

    def test_quit_returns_quit_sentinel(self) -> None:
        shell = Shell()
        assert shell.run_command("quit") is None or shell.run_command("quit") == ""

    def test_exit_returns_quit_sentinel(self) -> None:
        shell = Shell()
        assert shell.run_command("exit") is None or shell.run_command("exit") == ""


# ---------------------------------------------------------------------------
# main() REPL loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

class TestStatsCommand:
    def test_stats_returns_term_count(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("stats")
        assert "term" in output.lower()

    def test_stats_returns_doc_count(self) -> None:
        shell = _shell_with_index()
        output = shell.run_command("stats")
        assert "document" in output.lower() or "doc" in output.lower()

    def test_stats_without_index_returns_error(self) -> None:
        shell = Shell()
        output = shell.run_command("stats")
        assert any(w in output.lower() for w in ("build", "load", "index"))


class TestMainRepl:
    @patch("builtins.input", side_effect=["find wisdom", "quit"])
    @patch("src.main.build_index")
    @patch("src.main.crawl")
    def test_main_runs_and_exits_on_quit(
        self, _mock_crawl: MagicMock, _mock_build: MagicMock, _mock_input: MagicMock
    ) -> None:
        from src.main import main
        # Should return cleanly without raising
        main()

    @patch("builtins.input", side_effect=EOFError)
    def test_main_exits_on_eof(self, _mock_input: MagicMock) -> None:
        from src.main import main
        main()  # should not raise

    @patch("builtins.input", side_effect=["exit"])
    def test_main_exits_on_exit_command(self, _mock_input: MagicMock) -> None:
        from src.main import main
        main()

    def test_print_progress_outputs_url(self, capsys) -> None:
        Shell._print_progress("https://example.com/", 1, 0)
        captured = capsys.readouterr()
        assert "https://example.com/" in captured.out
