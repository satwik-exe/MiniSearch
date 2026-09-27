"""The inverted index.

This is the core data structure that makes search fast: instead of scanning
every document's text for every query (O(N x |doc|) per query), we maintain

    term -> { doc_id: (term_freq, positions) }

so a query only ever touches the documents that actually contain a query term.
An in-memory dict lookup for a term is average-case O(1); the equivalent SQLite
lookup (`postings` has a B-tree index on `term`) is O(log P) where P is the
total number of posting rows -- NOT O(1), that claim in an earlier version of
this file was wrong and has been corrected. Either way, the cost of scoring a
query is proportional to the size of the candidate set (documents that share at
least one term with the query), not the size of the corpus.

Example (from the project spec):

    "database":
        doc1 -> TF 4
        doc3 -> TF 2
        doc7 -> TF 6

The index is built once at ingestion time and kept in memory (`InvertedIndex`)
for fast lookups during search, while being persisted to SQLite (`postings`,
`terms` tables) so it survives an application restart without a full rebuild.

We also maintain a small forward index, `doc_terms: doc_id -> {term: tf}`, the
mirror image of the postings map. Two things need it:
  1. Correct TF-IDF cosine similarity requires each candidate document's FULL
     vector norm (every term in the document, not just the query's terms) --
     see app/search/tfidf.py for why a query-restricted norm is not standard
     cosine similarity. Computing that on demand from `postings` would mean
     scanning the whole vocabulary per document; `doc_terms` makes it O(terms
     in that one document) instead.
  2. `remove_document` used to scan every posting list in the index (O(V), V =
     vocabulary size) to find which terms belonged to the deleted document.
     With `doc_terms` it only touches that document's own terms.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from typing import Dict, List

from app.search.tokenizer import preprocess


class InvertedIndex:
    """In-memory inverted index with SQLite persistence.

    Internal structure:
        self.postings:   term -> { doc_id: (term_freq, [positions]) }
        self.doc_terms:  doc_id -> { term: term_freq }   (forward index / mirror of postings)
        self.doc_freq:   term -> number of documents containing the term
        self.doc_lengths: doc_id -> token count (needed for BM25 length normalization)
    """

    def __init__(self) -> None:
        self.postings: Dict[str, Dict[int, tuple]] = defaultdict(dict)
        self.doc_terms: Dict[int, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_lengths: Dict[int, int] = {}
        self.total_tokens: int = 0
        self.num_docs: int = 0

    # -- Building ------------------------------------------------------------

    def add_document(self, doc_id: int, text: str, *, stem: bool = True) -> int:
        """Tokenize `text`, add it to the index, and return the token count.

        Safe to call again for a doc_id that's already indexed (e.g. a future
        "re-index this document" feature): any existing entry for that doc_id
        is removed first, so num_docs/total_tokens/doc_freq are never
        double-counted. An earlier version of this method did not guard
        against this and would silently corrupt corpus-level stats if ever
        called twice for the same doc_id.
        """
        if doc_id in self.doc_terms:
            self.remove_document(doc_id)

        tokens = preprocess(text, stem=stem)
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            term_positions[term].append(pos)

        doc_term_freqs: Dict[str, int] = {}
        for term, positions in term_positions.items():
            tf = len(positions)
            if doc_id not in self.postings[term]:
                self.doc_freq[term] += 1
            self.postings[term][doc_id] = (tf, positions)
            doc_term_freqs[term] = tf

        self.doc_terms[doc_id] = doc_term_freqs
        self.doc_lengths[doc_id] = len(tokens)
        self.total_tokens += len(tokens)
        self.num_docs += 1
        return len(tokens)

    def remove_document(self, doc_id: int) -> List[str]:
        """Remove a document from the index (used by DELETE and rebuild).

        O(terms in this document), not O(vocabulary size): we use doc_terms to
        know exactly which posting lists to touch, instead of scanning every
        term in the index checking "does this term's posting list contain
        doc_id?" (which is what an earlier version of this method did).

        Returns the list of terms that belonged to this document, so callers
        that persist the change incrementally (remove_document_postings) know
        exactly which rows to touch without re-scanning anything themselves.
        """
        if doc_id not in self.doc_lengths:
            return []
        self.total_tokens -= self.doc_lengths.pop(doc_id)
        self.num_docs -= 1

        affected_terms = list(self.doc_terms.pop(doc_id, {}).keys())
        for term in affected_terms:
            docs = self.postings.get(term)
            if docs is None or doc_id not in docs:
                continue
            del docs[doc_id]
            self.doc_freq[term] -= 1
            if not docs:
                del self.postings[term]
                del self.doc_freq[term]
        return affected_terms

    def clear(self) -> None:
        self.postings.clear()
        self.doc_terms.clear()
        self.doc_freq.clear()
        self.doc_lengths.clear()
        self.total_tokens = 0
        self.num_docs = 0

    # -- Lookup ----------------------------------------------------------------

    def get_posting_list(self, term: str) -> Dict[int, tuple]:
        """Return {doc_id: (term_freq, positions)} for a term, empty dict if unseen."""
        return self.postings.get(term, {})

    def get_document_terms(self, doc_id: int) -> Dict[str, int]:
        """Return {term: term_freq} for every term in a document -- the forward-index
        lookup used to compute a document's TRUE (not query-restricted) TF-IDF norm."""
        return self.doc_terms.get(doc_id, {})

    def get_document_frequency(self, term: str) -> int:
        return self.doc_freq.get(term, 0)

    def average_doc_length(self) -> float:
        if self.num_docs == 0:
            return 0.0
        return self.total_tokens / self.num_docs

    def vocabulary_size(self) -> int:
        return len(self.postings)

    # -- Persistence -------------------------------------------------------

    def upsert_document_postings(self, conn: sqlite3.Connection, doc_id: int) -> None:
        """Incrementally persist just the postings/term-frequency changes caused by
        adding ONE document, instead of rewriting the entire postings table.

        Now that self.doc_terms tracks exactly which terms belong to this document,
        we only touch those rows -- O(terms in the new document), not O(vocabulary
        size). An earlier version scanned every posting list in the whole index to
        find this document's entries, which was O(V) per upload for no reason.
        """
        doc_terms = self.doc_terms.get(doc_id, {})
        term_rows = [(term, self.doc_freq[term]) for term in doc_terms]
        posting_rows = [
            (term, doc_id, tf, json.dumps(self.postings[term][doc_id][1]))
            for term, tf in doc_terms.items()
        ]

        conn.executemany(
            "INSERT INTO terms (term, doc_freq) VALUES (?, ?) "
            "ON CONFLICT(term) DO UPDATE SET doc_freq = excluded.doc_freq",
            term_rows,
        )
        conn.executemany(
            "INSERT INTO postings (term, doc_id, term_freq, positions) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(term, doc_id) DO UPDATE SET term_freq = excluded.term_freq, "
            "positions = excluded.positions",
            posting_rows,
        )
        self._update_meta(conn)
        conn.commit()

    def remove_document_postings(
        self, conn: sqlite3.Connection, doc_id: int, affected_terms: List[str]
    ) -> None:
        """Incrementally remove one document's postings and update just the doc_freq
        rows that actually changed, without touching the rest of the terms table.

        `affected_terms` is the list returned by self.remove_document(doc_id) --
        pass it straight through. An earlier version re-upserted doc_freq for
        EVERY term in the whole vocabulary on every single deletion (O(V) writes
        for a change that only ever affects O(document's own terms) rows).
        """
        conn.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
        surviving = [(t, self.doc_freq[t]) for t in affected_terms if t in self.doc_freq]
        dropped = [(t,) for t in affected_terms if t not in self.doc_freq]
        if surviving:
            conn.executemany(
                "INSERT INTO terms (term, doc_freq) VALUES (?, ?) "
                "ON CONFLICT(term) DO UPDATE SET doc_freq = excluded.doc_freq",
                surviving,
            )
        if dropped:
            conn.executemany("DELETE FROM terms WHERE term = ?", dropped)
        self._update_meta(conn)
        conn.commit()

    def _update_meta(self, conn: sqlite3.Connection) -> None:
        meta = {
            "num_docs": str(self.num_docs),
            "vocabulary_size": str(self.vocabulary_size()),
            "total_tokens": str(self.total_tokens),
        }
        for k, v in meta.items():
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, v),
            )

    def save_to_db(self, conn: sqlite3.Connection) -> float:
        """Persist the current in-memory index to SQLite. Returns build time in ms."""
        start = time.perf_counter()
        conn.execute("DELETE FROM postings")
        conn.execute("DELETE FROM terms")

        term_rows = [(term, self.doc_freq[term]) for term in self.postings]
        conn.executemany("INSERT INTO terms (term, doc_freq) VALUES (?, ?)", term_rows)

        posting_rows = []
        for term, docs in self.postings.items():
            for doc_id, (tf, positions) in docs.items():
                posting_rows.append((term, doc_id, tf, json.dumps(positions)))
        conn.executemany(
            "INSERT INTO postings (term, doc_id, term_freq, positions) VALUES (?, ?, ?, ?)",
            posting_rows,
        )

        meta = {
            "num_docs": str(self.num_docs),
            "vocabulary_size": str(self.vocabulary_size()),
            "total_tokens": str(self.total_tokens),
            "last_rebuild_ts": _now_iso(),
        }
        for k, v in meta.items():
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, v),
            )
        conn.commit()
        elapsed_ms = (time.perf_counter() - start) * 1000
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES ('last_index_build_ms', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(elapsed_ms),),
        )
        conn.commit()
        return elapsed_ms

    def load_from_db(self, conn: sqlite3.Connection) -> None:
        """Rebuild the in-memory index from persisted SQLite tables (no re-tokenizing)."""
        self.clear()

        for row in conn.execute("SELECT term, doc_freq FROM terms"):
            self.doc_freq[row["term"]] = row["doc_freq"]

        for row in conn.execute("SELECT term, doc_id, term_freq, positions FROM postings"):
            positions = json.loads(row["positions"])
            self.postings[row["term"]][row["doc_id"]] = (row["term_freq"], positions)
            self.doc_terms.setdefault(row["doc_id"], {})[row["term"]] = row["term_freq"]

        doc_rows = conn.execute("SELECT id, token_count FROM documents").fetchall()
        for row in doc_rows:
            self.doc_lengths[row["id"]] = row["token_count"]
        self.num_docs = len(doc_rows)
        self.total_tokens = sum(self.doc_lengths.values())

    @classmethod
    def rebuild_from_documents(cls, conn: sqlite3.Connection, *, stem: bool = True) -> "InvertedIndex":
        """Full rebuild: re-tokenize every stored document from scratch.
        Used by POST /api/index/rebuild, and only there — normal startup uses
        load_from_db() which is much cheaper."""
        index = cls()
        rows = conn.execute("SELECT id, text FROM documents").fetchall()
        for row in rows:
            token_count = index.add_document(row["id"], row["text"], stem=stem)
            conn.execute(
                "UPDATE documents SET token_count = ? WHERE id = ?", (token_count, row["id"])
            )
        conn.commit()
        index.save_to_db(conn)
        return index


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()
