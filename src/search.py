"""Search engine query processing.

Design
------
SearchEngine wraps an InvertedIndex and exposes:
  find(terms)              – AND query, TF-IDF ranked
  find_phrase(phrase)      – exact consecutive phrase, positionally verified
  find_with_proximity(ts)  – AND query with proximity bonus for near terms
  find_bm25(terms)         – AND query using Okapi BM25 ranking
  print_entry(word)        – formatted posting list for the `print` command

The module also exports:
  suggest_terms(index, word) – Levenshtein-based did-you-mean suggestions

Complexity
----------
- find (k terms):           O(k * D + D log D)  intersection + sort
- find_phrase (k terms):    O(k * D * P)  P = positions per posting
- find_bm25 (k terms):      O(k * D + D log D)  same structure as find
- suggest_terms:            O(V * W)  V = vocab size, W = word length
"""

from __future__ import annotations

__all__ = ["SearchEngine", "suggest_terms"]

import bisect
import math

from src.indexer import InvertedIndex, tokenise


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

    def find_phrase(self, phrase: str) -> list[tuple[str, float]]:
        """Return pages where *phrase* appears as consecutive tokens.

        Uses the stored position lists to verify adjacency rather than
        simply checking that all words are present — this distinguishes
        "good friends" from a page that has "good" and "friends" far apart.

        Args:
            phrase: Space-separated phrase string (case-insensitive).

        Returns:
            Ranked ``(url, score)`` list; score is the sum of TF-IDF for
            all phrase tokens in the matching document.
        """
        terms = tokenise(phrase)
        if not terms:
            return []

        # Candidate URLs must contain every token
        results = self.find(terms)
        if not results:
            return []

        matched: list[tuple[str, float]] = []
        for url, score in results:
            # Check whether all terms appear consecutively in this document
            if self._has_consecutive_positions(url, terms):
                matched.append((url, score))
        return matched

    def _has_consecutive_positions(self, url: str, terms: list[str]) -> bool:
        """Return True if *terms* appear in consecutive positions in *url*.

        For each starting position of terms[0] in the document, uses
        ``bisect.bisect_left`` (binary search) to check whether each
        subsequent term appears at the required offset — O(log P) per
        lookup rather than O(P) linear scan.

        Time complexity: O(P * k * log P) vs O(P * k) naïve — bisect
        pays off when position lists are long (dense documents).
        """
        if not terms:
            return False
        first_positions = self._index[terms[0]][url]["positions"]
        for start in first_positions:
            if all(
                terms[i] in self._index
                and url in self._index[terms[i]]
                and _position_exists(
                    self._index[terms[i]][url]["positions"], start + i
                )
                for i in range(1, len(terms))
            ):
                return True
        return False

    def find_with_proximity(self, terms: list[str]) -> list[tuple[str, float]]:
        """AND query with a proximity bonus for terms appearing near each other.

        Base score is the sum of TF-IDF weights (same as :meth:`find`).
        A proximity bonus is added that is inversely proportional to the
        minimum token distance between any pair of query terms in the document.

        bonus = 1 / (1 + min_distance)

        This rewards pages where query terms are co-located rather than
        scattered across the text.

        Args:
            terms: List of query words.

        Returns:
            Ranked ``(url, score)`` list with proximity bonus applied.
        """
        base_results = self.find(terms)
        if not base_results or len(terms) < 2:
            return base_results

        clean = [t.lower().strip() for t in terms if t.strip()]
        rescored: list[tuple[str, float]] = []
        for url, base_score in base_results:
            min_dist = self._min_term_distance(url, clean)
            bonus = 1.0 / (1.0 + min_dist)
            rescored.append((url, base_score + bonus))

        return sorted(rescored, key=lambda x: x[1], reverse=True)

    def _min_term_distance(self, url: str, terms: list[str]) -> float:
        """Return the minimum position distance between any two query terms.

        Compares the position lists of consecutive term pairs and finds the
        closest occurrence.  Returns infinity when fewer than two terms are
        present or positions are unavailable.
        """
        min_dist: float = float("inf")
        for i in range(len(terms) - 1):
            t1, t2 = terms[i], terms[i + 1]
            if t1 not in self._index or t2 not in self._index:
                continue
            p1 = self._index[t1].get(url, {}).get("positions", [])
            p2 = self._index[t2].get(url, {}).get("positions", [])
            for pos1 in p1:
                for pos2 in p2:
                    min_dist = min(min_dist, abs(pos1 - pos2))
        return min_dist

    def find_bm25(
        self,
        terms: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[str, float]]:
        """AND query using Okapi BM25 ranking.

        BM25 is the industry-standard probabilistic ranking function used
        by search engines such as Elasticsearch and Lucene.  It improves
        on TF-IDF by applying document-length normalisation (controlled
        by *b*) and term-frequency saturation (controlled by *k1*).

        Formula per term per document::

            idf  = log((N - df + 0.5) / (df + 0.5) + 1)
            tf_n = freq * (k1 + 1) / (freq + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_n

        Args:
            terms: List of query words.
            k1:    Term-frequency saturation parameter (default 1.5).
            b:     Document-length normalisation factor (default 0.75).

        Returns:
            Ranked ``(url, score)`` list sorted descending.
        """
        clean = [t.lower().strip() for t in terms if t.strip()]
        if not clean:
            return []

        n_docs = self._index.document_count or 1

        # Compute average document length (in tokens) across the corpus
        doc_lengths: dict[str, int] = {}
        for postings in self._index._index.values():
            for url, entry in postings.items():
                doc_lengths[url] = doc_lengths.get(url, 0) + entry["frequency"]
        avg_dl = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1.0

        # Intersect candidate set (AND semantics)
        candidate_urls: set[str] | None = None
        for term in clean:
            if term not in self._index:
                return []
            term_urls = set(self._index[term].keys())
            candidate_urls = term_urls if candidate_urls is None else candidate_urls & term_urls
        if not candidate_urls:
            return []

        scored: list[tuple[str, float]] = []
        for url in candidate_urls:
            dl = doc_lengths.get(url, 1)
            total = 0.0
            for term in clean:
                if term not in self._index or url not in self._index[term]:
                    continue
                freq = self._index[term][url]["frequency"]
                df = len(self._index[term])
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                tf_n = freq * (k1 + 1) / (freq + k1 * (1 - b + b * dl / avg_dl))
                total += idf * tf_n
            scored.append((url, total))

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _position_exists(positions: list[int], target: int) -> bool:
    """Return True if *target* is in the sorted *positions* list.

    Uses ``bisect.bisect_left`` for O(log P) binary search rather than
    O(P) linear scan — beneficial for long position lists.
    """
    idx = bisect.bisect_left(positions, target)
    return idx < len(positions) and positions[idx] == target


def _levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Compute the Levenshtein edit distance between strings *a* and *b*.

    Uses a row-at-a-time DP with early exit: if the minimum value in the
    current row already exceeds *max_dist*, the strings are guaranteed to
    be further apart than the threshold and ``max_dist + 1`` is returned
    immediately — avoiding unnecessary computation.

    Time complexity:  O(|a| * |b|) worst case; faster with small *max_dist*
    Space complexity: O(min(|a|, |b|))  — two rows, not full matrix
    """
    m, n = len(a), len(b)
    # Keep the shorter string in the inner loop for cache efficiency
    if m < n:
        a, b, m, n = b, a, n, m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        if max_dist is not None and min(curr) > max_dist:
            return max_dist + 1
        prev = curr
    return prev[n]


def suggest_terms(
    index: InvertedIndex,
    word: str,
    max_results: int = 5,
    max_distance: int = 2,
) -> list[str]:
    """Return vocabulary words similar to *word* using Levenshtein distance.

    Useful for did-you-mean suggestions when a query term is not found in
    the index.  Only words within *max_distance* edits are returned, sorted
    by ascending edit distance then alphabetically.

    Args:
        index:        Populated :class:`InvertedIndex` to search.
        word:         Misspelled or approximate query term.
        max_results:  Maximum number of suggestions to return.
        max_distance: Maximum edit distance to consider (default 2).

    Returns:
        List of up to *max_results* vocabulary words closest to *word*.
        Empty list if no word is within *max_distance* edits.
    """
    query = word.lower().strip()
    if not query:
        return []

    # Tokenise to apply the same normalisation as during indexing
    candidates: list[tuple[int, str]] = []
    for term in index._index:
        dist = _levenshtein(query, term, max_dist=max_distance)
        if 0 < dist <= max_distance:
            candidates.append((dist, term))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [t for _, t in candidates[:max_results]]
