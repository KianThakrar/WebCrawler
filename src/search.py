"""Search engine query processing.

Design
------
SearchEngine wraps an InvertedIndex and exposes two operations:
  find(terms)       – AND query returning (url, score) pairs ranked by
                      combined TF-IDF, highest first.
  print_entry(word) – formatted string showing the full posting list
                      for a single word (used by the CLI `print` command).

Complexity
----------
- find (single term):  O(D)  D = documents containing the term
- find (k terms):      O(k * D + D * log D)  for intersection + sort
"""

from __future__ import annotations

from src.indexer import InvertedIndex


class SearchEngine:
    """Query interface over a populated :class:`InvertedIndex`.

    Args:
        index: A fully built and TF-IDF-scored :class:`InvertedIndex`.
    """

    def __init__(self, index: InvertedIndex) -> None:
        self._index = index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, terms: list[str]) -> list[tuple[str, float]]:
        """Return pages containing ALL *terms*, ranked by combined TF-IDF.

        The query is case-insensitive and uses AND semantics: only pages
        that contain every supplied term are returned.

        Args:
            terms: List of query words (pre-tokenised by the caller).

        Returns:
            List of ``(url, score)`` tuples sorted by score descending.
            Empty list if no terms are supplied or no page matches.
        """
        clean = [t.lower().strip() for t in terms if t.strip()]
        if not clean:
            return []

        # Start with the candidate set from the first term
        first = clean[0]
        if first not in self._index:
            return []
        candidate_urls: set[str] = set(self._index[first].keys())

        # Intersect with each subsequent term's document set
        for term in clean[1:]:
            if term not in self._index:
                return []
            candidate_urls &= set(self._index[term].keys())

        if not candidate_urls:
            return []

        # Score each candidate as the sum of TF-IDF across all query terms
        scored: list[tuple[str, float]] = []
        for url in candidate_urls:
            score = sum(
                self._index[term][url]["tf_idf"]
                for term in clean
                if term in self._index and url in self._index[term]
            )
            scored.append((url, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def print_entry(self, word: str) -> str:
        """Return a formatted string showing the posting list for *word*.

        Args:
            word: Term to look up (case-insensitive).

        Returns:
            Multi-line string with per-URL frequency, positions, and
            TF-IDF score, or a 'not found' message if the word is absent.
        """
        term = word.lower().strip()
        if term not in self._index:
            return f"'{word}' not found in index."

        lines: list[str] = [f"{term}:"]
        for url, entry in self._index[term].items():
            lines.append(f"  {url}")
            lines.append(f"    frequency : {entry['frequency']}")
            lines.append(f"    positions : {entry['positions']}")
            lines.append(f"    tf_idf    : {entry['tf_idf']:.4f}")
        return "\n".join(lines)
