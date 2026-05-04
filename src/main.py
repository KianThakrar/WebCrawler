"""Interactive CLI shell for the search engine.

Commands
--------
build           – crawl the site, build the index, save to disk
load            – load a previously built index from disk
print <word>    – show the inverted index entry for a word
find <word(s)>  – find pages containing all given words, ranked by TF-IDF
quit / exit     – exit the shell
"""

from __future__ import annotations

import os

from src.crawler import crawl
from src.indexer import InvertedIndex, build_index, load_index, save_index
from src.search import SearchEngine, suggest_terms

_DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "index.json"
)


class Shell:
    """REPL shell that wraps crawl → index → search pipeline.

    Attributes:
        _index:      Currently loaded :class:`InvertedIndex`, or None.
        _engine:     :class:`SearchEngine` over *_index*, or None.
        _index_path: Filesystem path used for save/load.
    """

    def __init__(self, index_path: str = _DEFAULT_INDEX_PATH) -> None:
        self._index: InvertedIndex | None = None
        self._engine: SearchEngine | None = None
        self._index_path: str = index_path

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def run_command(self, line: str) -> str:
        """Parse *line* and execute the matching command.

        Args:
            line: Raw input string from the user.

        Returns:
            String output to display.  Returns empty string for blank
            input; returns ``None``-ish empty string for quit/exit.
        """
        parts = line.strip().split()
        if not parts:
            return ""

        cmd, *args = parts

        dispatch = {
            "build": self._cmd_build,
            "load": self._cmd_load,
            "print": self._cmd_print,
            "find": self._cmd_find,
            "phrase": self._cmd_phrase,
            "bm25": self._cmd_bm25,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = dispatch.get(cmd.lower())
        if handler is None:
            return f"Unknown command '{cmd}'. Type 'help' for available commands."
        return handler(args)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_build(self, _args: list[str]) -> str:
        print("Crawling https://quotes.toscrape.com/ …")
        pages = crawl(on_progress=self._print_progress)
        print(f"Crawl complete. {len(pages)} page(s) fetched.")

        self._index = build_index(pages)
        self._engine = SearchEngine(self._index)

        save_index(self._index, self._index_path)
        return (
            f"Index built. {len(pages)} page(s), "
            f"{len(self._index)} unique term(s).\n"
            f"Index saved to {self._index_path}"
        )

    def _cmd_load(self, _args: list[str]) -> str:
        try:
            self._index = load_index(self._index_path)
            self._engine = SearchEngine(self._index)
            return (
                f"Index loaded from {self._index_path} "
                f"({self._index.document_count} page(s), "
                f"{len(self._index)} term(s))"
            )
        except FileNotFoundError:
            return (
                f"Error: index file not found at '{self._index_path}'. "
                "Run 'build' first."
            )
        except ValueError as exc:
            return f"Error: could not load index – {exc}"

    def _cmd_print(self, args: list[str]) -> str:
        if not args:
            return "Usage: print <word>"
        if self._engine is None:
            return "No index loaded. Run 'build' or 'load' first."
        return self._engine.print_entry(args[0])

    def _cmd_find(self, args: list[str]) -> str:
        if not args:
            return "Usage: find <word> [word ...]"
        if self._engine is None:
            return "No index loaded. Run 'build' or 'load' first."

        results = self._engine.find(args)
        if not results:
            msg = f"No pages found containing: {' '.join(args)}"
            suggestions = [
                s for term in args
                for s in suggest_terms(self._index, term)
                if s != term.lower()
            ]
            if suggestions:
                unique = list(dict.fromkeys(suggestions))[:3]
                msg += f"\nDid you mean: {', '.join(unique)}?"
            return msg

        lines = [f"Results for: {' AND '.join(args)}"]
        for rank, (url, score) in enumerate(results, start=1):
            lines.append(f"  {rank}. {url}  (score: {score:.4f})")
        return "\n".join(lines)

    def _cmd_phrase(self, args: list[str]) -> str:
        if not args:
            return "Usage: phrase <word> [word ...]"
        if self._engine is None:
            return "No index loaded. Run 'build' or 'load' first."

        results = self._engine.find_phrase(" ".join(args))
        if not results:
            return f"No pages found containing exact phrase: {' '.join(args)}"

        lines = [f"Phrase results for: \"{' '.join(args)}\""]
        for rank, (url, score) in enumerate(results, start=1):
            lines.append(f"  {rank}. {url}  (score: {score:.4f})")
        return "\n".join(lines)

    def _cmd_bm25(self, args: list[str]) -> str:
        if not args:
            return "Usage: bm25 <word> [word ...]"
        if self._engine is None:
            return "No index loaded. Run 'build' or 'load' first."

        results = self._engine.find_bm25(args)
        if not results:
            return f"No pages found (BM25) for: {' '.join(args)}"

        lines = [f"BM25 results for: {' AND '.join(args)}"]
        for rank, (url, score) in enumerate(results, start=1):
            lines.append(f"  {rank}. {url}  (score: {score:.4f})")
        return "\n".join(lines)

    def _cmd_stats(self, _args: list[str]) -> str:
        if self._index is None:
            return "No index loaded. Run 'build' or 'load' first."

        n_terms = len(self._index)
        n_docs = self._index.document_count

        # Top-10 terms by document frequency (most widespread)
        postings = self._index._index
        by_df = sorted(postings.items(), key=lambda x: len(x[1]), reverse=True)
        top_terms = [f"{term} ({len(docs)} docs)" for term, docs in by_df[:10]]

        lines = [
            "Index statistics:",
            f"  Unique terms    : {n_terms}",
            f"  Documents       : {n_docs}",
            f"  Top terms by df : {', '.join(top_terms)}",
        ]
        return "\n".join(lines)

    def _cmd_help(self, _args: list[str]) -> str:
        return (
            "Available commands:\n"
            "  build               – crawl site and build index\n"
            "  load                – load saved index from disk\n"
            "  print <word>        – show posting list for a word\n"
            "  find  <word(s)>     – AND search, TF-IDF ranked\n"
            "  phrase <word(s)>    – exact phrase search\n"
            "  bm25  <word(s)>     – AND search, BM25 ranked\n"
            "  stats               – show index statistics\n"
            "  quit / exit         – exit the shell"
        )

    def _cmd_quit(self, _args: list[str]) -> str:
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_progress(url: str, done: int, _total: int) -> None:
        print(f"  [{done}] {url}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the interactive CLI shell."""
    shell = Shell()
    print("Search Engine – quotes.toscrape.com")
    print("Commands: build | load | print | find | phrase | bm25 | stats | help | quit\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if line.lower() in ("quit", "exit"):
            print("Bye.")
            break

        output = shell.run_command(line)
        if output:
            print(output)


if __name__ == "__main__":
    main()
