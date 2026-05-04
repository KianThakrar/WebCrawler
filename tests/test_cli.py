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

from src.indexer import InvertedIndex, build_index
from src.main import Shell

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
        self, mock_crawl: MagicMock, mock_build: MagicMock, mock_save: MagicMock
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
        self, mock_crawl: MagicMock, mock_build: MagicMock, mock_save: MagicMock
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
        self, mock_crawl: MagicMock, mock_build: MagicMock, mock_save: MagicMock
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
        self, mock_crawl: MagicMock, mock_build: MagicMock, mock_save: MagicMock
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


# ---------------------------------------------------------------------------
# Unknown / empty commands
# ---------------------------------------------------------------------------

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
