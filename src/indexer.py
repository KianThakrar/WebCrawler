"""Inverted index builder with TF-IDF scoring.

Design
------
The index maps  word → {url → {frequency, positions, tf_idf}}.
TF-IDF uses log-normalised term frequency and smoothed IDF to prevent
division-by-zero and reduce the dominance of very common terms.

Complexity
----------
- build_index:   O(P × T)   P = pages, T = tokens per page
- compute_tfidf: O(V × D)   V = vocabulary size, D = documents per term
- save / load:   O(V × D)   dominated by JSON serialisation

Public API
----------
tokenise(text)                  → list[str]
InvertedIndex                   class  (add_document, compute_tfidf, __len__, …)
build_index(pages)              → InvertedIndex
save_index(index, path)         → None
load_index(path)                → InvertedIndex
"""

from __future__ import annotations

__all__ = ["tokenise", "InvertedIndex", "build_index", "save_index", "load_index"]

import json
import math
import os
import re

# ---------------------------------------------------------------------------
# Stopwords (common English function words that carry no search signal)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "his", "her", "they", "their", "what", "which",
    "who", "not", "no", "so", "as", "if", "then", "than", "into", "up",
    "out", "about", "over", "after", "before",
})

_PUNCT_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """Maps terms to per-document posting lists.

    Structure::

        {
          "word": {
            "https://page.url/": {
              "frequency": int,
              "positions": [int, ...],
              "tf_idf":    float        # populated by compute_tfidf()
            }
          }
        }
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, dict]] = {}
        self._doc_count: int = 0

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of unique terms in the index."""
        return len(self._index)

    def __getitem__(self, term: str) -> dict:
        """Return the posting dict for *term*; raises KeyError if absent."""
        return self._index[term]

    def __contains__(self, term: object) -> bool:
        return term in self._index

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_document(self, url: str, content: str) -> None:
        """Tokenise *content* and add all postings for *url*.

        Calling this method twice with the same *url* overwrites the
        previous entry for that document.

        Args:
            url:     Canonical URL used as the document identifier.
            content: Plain-text content of the page.
        """
        tokens = tokenise(content)

        # Remove any existing postings for this URL so re-indexing is clean
        for postings in self._index.values():
            postings.pop(url, None)

        seen_urls_before = {
            u for postings in self._index.values() for u in postings
        }
        if url not in seen_urls_before:
            self._doc_count += 1

        for position, token in enumerate(tokens):
            if token not in self._index:
                self._index[token] = {}
            if url not in self._index[token]:
                self._index[token][url] = {"frequency": 0, "positions": [], "tf_idf": 0.0}
            self._index[token][url]["frequency"] += 1
            self._index[token][url]["positions"].append(position)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Total number of documents added to the index."""
        return self._doc_count

    def __repr__(self) -> str:
        return f"InvertedIndex(terms={len(self)}, documents={self._doc_count})"

    def compute_tfidf(self) -> None:
        """Compute and store TF-IDF scores for every posting in the index.

        Uses log-normalised TF and smoothed IDF::

            tf(t, d)  = 1 + log(frequency)   if frequency > 0, else 0
            idf(t)    = log((N + 1) / df(t))  smoothed to avoid zero division
            tf_idf    = tf * idf

        where N is the total number of indexed documents and df(t) is the
        number of documents containing term t.
        """
        n_docs = self._doc_count or 1  # guard against empty index
        for term, postings in self._index.items():
            df = len(postings)
            idf = math.log((n_docs + 1) / df)
            for url, entry in postings.items():
                tf = 1 + math.log(entry["frequency"]) if entry["frequency"] > 0 else 0.0
                entry["tf_idf"] = tf * idf


def build_index(pages: list[dict]) -> InvertedIndex:
    """Build an inverted index from a list of crawled page dicts.

    Args:
        pages: List of page dicts as returned by the crawler.

    Returns:
        Fully populated :class:`InvertedIndex` with TF-IDF scores computed.
    """
    idx = InvertedIndex()
    for page in pages:
        idx.add_document(page["url"], page.get("content", ""))
    idx.compute_tfidf()
    return idx


def save_index(index: InvertedIndex, path: str) -> None:
    """Serialise *index* to a JSON file at *path*.

    Args:
        index: Populated :class:`InvertedIndex` to persist.
        path:  Filesystem path for the output JSON file.
    """
    payload = {
        "document_count": index.document_count,
        "index": index._index,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_index(path: str) -> InvertedIndex:
    """Deserialise an :class:`InvertedIndex` from a JSON file at *path*.

    Args:
        path: Filesystem path to a JSON file previously written by
              :func:`save_index`.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Returns:
        Restored :class:`InvertedIndex`.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    idx = InvertedIndex()
    idx._index = payload["index"]
    idx._doc_count = payload["document_count"]
    return idx


def tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, and remove stopwords from *text*.

    Args:
        text: Raw plain-text string to tokenise.

    Returns:
        Ordered list of meaningful tokens.
    """
    lowered = str(text).lower()
    stripped = _PUNCT_RE.sub(" ", lowered)
    tokens = stripped.split()
    return [t for t in tokens if t and t not in _STOPWORDS]
