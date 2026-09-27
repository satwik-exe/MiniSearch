"""SearchEngine: the single object the API layer talks to.

Wraps the in-memory InvertedIndex + SQLite connection and exposes the
operations the API needs: add a document, delete a document, search, rebuild,
and stats. Kept as one class so main.py can hold a single instance in memory
for the app's lifetime (loaded once at startup from the persisted index).

Thread-safety: FastAPI/Starlette runs sync `def` route handlers in a thread
pool, so two uploads (or an upload and a rebuild) can genuinely execute
concurrently. The in-memory InvertedIndex is a set of plain dicts with
multi-step mutations (e.g. "check doc_freq, then update it, then write a
posting") that are NOT atomic under interleaving -- two concurrent writers
could corrupt doc_freq counts or leave postings out of sync with doc_terms.
An earlier version of this class had no locking at all. We use a single
`threading.Lock` around every mutation (and around read+mutate sequences like
search-after-rebuild) rather than anything fancier (e.g. per-document locks,
a reader/writer lock): at this project's scale -- a personal/portfolio
document collection, not a high-throughput service -- a single coarse lock is
easy to reason about and costs nothing measurable, and introducing finer-
grained locking here would be exactly the kind of unnecessary complexity this
project is explicitly trying to avoid.
"""
from __future__ import annotations

import datetime
import threading
import time
from typing import List, Optional

from app.database.db import get_connection, init_db
from app.documents.ingestion import IngestedDocument
from app.search.index import InvertedIndex
from app.search.ranker import rank, matched_terms_for_doc
from app.search.snippet import generate_snippet
from app.search.tokenizer import preprocess

STEM = True  # whether the index/query pipeline applies stemming


class SearchEngine:
    def __init__(self) -> None:
        self.index = InvertedIndex()
        self._lock = threading.Lock()

    def startup(self) -> None:
        """Called once at app startup: ensure schema exists, load index from DB."""
        conn = get_connection()
        init_db(conn)
        with self._lock:
            self.index.load_from_db(conn)
        conn.close()

    # -- Documents -----------------------------------------------------------

    def add_document(self, doc: IngestedDocument) -> dict:
        # Token count is computed up front (pure function of doc.text, no doc_id
        # needed) so the INSERT below is a single write with the correct value,
        # rather than an earlier version's INSERT-with-0-then-UPDATE two-step,
        # which left a moment where a concurrent reader could see token_count=0
        # for a fully-indexed document (skewing average_doc_length).
        token_count = len(preprocess(doc.text, stem=STEM))

        conn = get_connection()
        try:
            with self._lock:
                cur = conn.execute(
                    "INSERT INTO documents (filename, title, file_type, upload_ts, size_bytes, "
                    "text, word_count, token_count) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        doc.filename,
                        doc.title,
                        doc.file_type,
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        doc.size_bytes,
                        doc.text,
                        doc.word_count,
                        token_count,
                    ),
                )
                doc_id = cur.lastrowid
                # add_document re-tokenizes internally; the token_count computed
                # above is only used for the DB row so search's average_doc_length
                # reads a correct value even before the index write below completes.
                self.index.add_document(doc_id, doc.text, stem=STEM)
                conn.commit()
                self.index.upsert_document_postings(conn, doc_id)
                row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
                return dict(row)
        finally:
            conn.close()

    def delete_document(self, doc_id: int) -> bool:
        conn = get_connection()
        try:
            with self._lock:
                row = conn.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone()
                if row is None:
                    return False
                affected_terms = self.index.remove_document(doc_id)
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                self.index.remove_document_postings(conn, doc_id, affected_terms)
                return True
        finally:
            conn.close()

    def list_documents(self) -> List[dict]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT id, filename, title, file_type, upload_ts, size_bytes, word_count, "
                "token_count FROM documents ORDER BY upload_ts DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_document(self, doc_id: int) -> Optional[dict]:
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # -- Search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        ranking: str,
        top_k: int,
        page: int,
        *,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
    ) -> dict:
        # Only the ranking computation itself is measured under the lock+timer;
        # the DB round-trip to fetch title/snippet text for the page and the
        # separate latency-recording write happen outside the timed window (see
        # below), so search_time_ms reflects the actual search/ranking work, not
        # unrelated I/O -- this was an explicit audit point.
        with self._lock:
            start = time.perf_counter()
            scores = rank(query, ranking, self.index, stem=STEM, bm25_k1=bm25_k1, bm25_b=bm25_b)
            ranked_ids = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]
            matched_by_doc = {
                doc_id: matched_terms_for_doc(query, doc_id, self.index, stem=STEM)
                for doc_id in ranked_ids[: page * top_k]
            }
            elapsed_ms = (time.perf_counter() - start) * 1000

        total_matches = len(scores)
        offset = (page - 1) * top_k
        page_ids = ranked_ids[offset: offset + top_k]

        results = []
        if page_ids:
            conn = get_connection()
            try:
                placeholders = ",".join("?" * len(page_ids))
                rows = {
                    r["id"]: r
                    for r in conn.execute(
                        f"SELECT id, title, filename, text FROM documents WHERE id IN ({placeholders})",
                        page_ids,
                    ).fetchall()
                }
            finally:
                conn.close()

            for doc_id in page_ids:
                row = rows.get(doc_id)
                if row is None:
                    continue
                snippet = generate_snippet(row["text"], query, stem=STEM)
                results.append(
                    {
                        "document_id": doc_id,
                        "title": row["title"],
                        "filename": row["filename"],
                        "score": round(scores[doc_id], 6),
                        "snippet": snippet,
                        "matched_terms": matched_by_doc.get(doc_id, []),
                    }
                )

        self._record_latency(elapsed_ms)

        return {
            "query": query,
            "ranking": ranking,
            "results": results,
            "total_matches": total_matches,
            "page": page,
            "top_k": top_k,
            "search_time_ms": round(elapsed_ms, 3),
        }

    def _record_latency(self, ms: float) -> None:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES ('last_search_latency_ms', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(ms),),
            )
            conn.commit()
        finally:
            conn.close()

    # -- Index management --------------------------------------------------

    def rebuild(self) -> dict:
        conn = get_connection()
        try:
            with self._lock:
                start = time.perf_counter()
                self.index = InvertedIndex.rebuild_from_documents(conn, stem=STEM)
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "documents_indexed": self.index.num_docs,
                    "unique_terms": self.index.vocabulary_size(),
                    "total_tokens": self.index.total_tokens,
                    "build_time_ms": round(elapsed_ms, 3),
                }
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = get_connection()
        try:
            meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM index_meta")}
            db_size = 0
            try:
                import os
                from app.database.db import DB_PATH

                db_size = os.path.getsize(DB_PATH)
            except OSError:
                pass
            with self._lock:
                documents_indexed = self.index.num_docs
                unique_terms = self.index.vocabulary_size()
                total_tokens = self.index.total_tokens
                average_doc_length = round(self.index.average_doc_length(), 2)
            return {
                "documents_indexed": documents_indexed,
                "unique_terms": unique_terms,
                "total_tokens": total_tokens,
                "average_doc_length": average_doc_length,
                "index_size_bytes": db_size,
                "last_rebuild_ts": meta.get("last_rebuild_ts"),
                "last_index_build_ms": (
                    float(meta["last_index_build_ms"]) if "last_index_build_ms" in meta else None
                ),
                "last_search_latency_ms": (
                    float(meta["last_search_latency_ms"]) if "last_search_latency_ms" in meta else None
                ),
            }
        finally:
            conn.close()
