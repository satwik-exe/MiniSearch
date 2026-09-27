import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search.index import InvertedIndex


def test_posting_list_term_frequencies_correct(small_index):
    posting = small_index.get_posting_list("memory")
    assert posting[1][0] == 1
    assert posting[2][0] == 2
    assert posting[4][0] == 3


def test_document_frequency_correct(small_index):
    # "memory" appears in docs 1, 2, 4 (not 3)
    assert small_index.get_document_frequency("memory") == 3


def test_unseen_term_returns_empty(small_index):
    assert small_index.get_posting_list("nonexistentword") == {}
    assert small_index.get_document_frequency("nonexistentword") == 0


def test_vocabulary_and_totals(small_index):
    assert small_index.vocabulary_size() > 0
    assert small_index.num_docs == 4
    assert small_index.total_tokens > 0


def test_remove_document_updates_postings_and_df():
    idx = InvertedIndex()
    idx.add_document(1, "database database indexing", stem=False)
    idx.add_document(2, "database queries", stem=False)
    assert idx.get_document_frequency("database") == 2

    idx.remove_document(1)
    assert idx.get_document_frequency("database") == 1
    assert 1 not in idx.get_posting_list("database")
    assert idx.num_docs == 1


def test_duplicate_document_text_indexed_independently():
    idx = InvertedIndex()
    idx.add_document(1, "identical text twice", stem=False)
    idx.add_document(2, "identical text twice", stem=False)
    assert idx.get_document_frequency("identical") == 2
    assert idx.get_posting_list("identical")[1][0] == idx.get_posting_list("identical")[2][0] == 1


def test_empty_document_indexes_cleanly():
    idx = InvertedIndex()
    token_count = idx.add_document(1, "", stem=False)
    assert token_count == 0
    assert idx.doc_lengths[1] == 0


def test_very_short_document():
    idx = InvertedIndex()
    idx.add_document(1, "hi", stem=False)
    assert idx.get_document_frequency("hi") == 1


def test_persistence_round_trip(tmp_path):
    import sqlite3
    import app.database.db as db_module

    db_module.DB_PATH = tmp_path / "test.db"
    conn = db_module.get_connection()
    db_module.init_db(conn)

    idx = InvertedIndex()
    idx.add_document(1, "database systems are useful", stem=False)
    idx.add_document(2, "database indexing improves performance", stem=False)

    for doc_id, text in [(1, "x"), (2, "y")]:
        conn.execute(
            "INSERT INTO documents (id, filename, title, file_type, upload_ts, size_bytes, "
            "text, word_count, token_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, f"{doc_id}.txt", str(doc_id), "txt", "2026-01-01", 1, text, 1, idx.doc_lengths[doc_id]),
        )
    conn.commit()
    idx.save_to_db(conn)

    idx2 = InvertedIndex()
    idx2.load_from_db(conn)
    assert idx2.vocabulary_size() == idx.vocabulary_size()
    assert idx2.get_document_frequency("database") == idx.get_document_frequency("database")
    conn.close()


# --- Added during audit: forward index (doc_terms), incremental persistence,
# and the add_document re-add guard -------------------------------------------


def test_forward_index_returns_all_terms_in_a_document():
    idx = InvertedIndex()
    idx.add_document(1, "database systems store data efficiently", stem=False)
    terms = idx.get_document_terms(1)
    assert set(terms.keys()) == {"database", "systems", "store", "data", "efficiently"}
    assert all(tf == 1 for tf in terms.values())


def test_forward_index_empty_for_unknown_doc():
    idx = InvertedIndex()
    assert idx.get_document_terms(999) == {}


def test_remove_document_returns_affected_terms():
    idx = InvertedIndex()
    idx.add_document(1, "database indexing performance", stem=False)
    affected = idx.remove_document(1)
    assert set(affected) == {"database", "indexing", "performance"}
    # doc_terms entry itself must also be gone
    assert idx.get_document_terms(1) == {}


def test_remove_document_on_unknown_doc_returns_empty_list():
    idx = InvertedIndex()
    idx.add_document(1, "something", stem=False)
    assert idx.remove_document(999) == []


def test_remove_document_is_cheap_not_a_full_vocabulary_scan():
    """Functional proxy for the O(V) -> O(doc terms) fix: removing one small
    document from a large-vocabulary index should only touch THAT document's
    own terms, leaving every other document's postings completely untouched."""
    idx = InvertedIndex()
    idx.add_document(1, "alpha beta gamma", stem=False)
    for i in range(2, 52):
        idx.add_document(i, f"unique{i} word{i} term{i}", stem=False)
    before_vocab = idx.vocabulary_size()

    idx.remove_document(1)

    assert idx.get_document_frequency("alpha") == 0
    assert "alpha" not in idx.postings
    # every unrelated document's postings must be untouched
    for i in range(2, 52):
        assert idx.get_document_frequency(f"unique{i}") == 1
    assert idx.vocabulary_size() == before_vocab - 3  # alpha, beta, gamma removed


def test_add_document_called_twice_for_same_doc_id_does_not_double_count():
    """Regression test for a real bug found during audit: add_document had no
    guard against being called twice for the same doc_id, which silently
    double-counted num_docs/total_tokens/doc_freq."""
    idx = InvertedIndex()
    idx.add_document(99, "apple banana apple", stem=False)
    idx.add_document(99, "apple banana apple", stem=False)  # re-add, same doc_id

    assert idx.num_docs == 1
    assert idx.get_document_frequency("apple") == 1
    assert idx.total_tokens == 3
    assert idx.get_posting_list("apple")[99][0] == 2  # tf of "apple" in the doc


def test_add_document_reindex_with_different_text_replaces_old_terms():
    """A re-add with DIFFERENT text should fully replace the old term set, not
    merge the two (this is what the remove-then-add guard should produce)."""
    idx = InvertedIndex()
    idx.add_document(1, "original content about databases", stem=False)
    idx.add_document(1, "completely different content about networks", stem=False)

    assert idx.get_document_frequency("databases") == 0
    assert "databases" not in idx.postings
    assert idx.get_document_frequency("networks") == 1
    assert idx.num_docs == 1


def test_incremental_upsert_and_remove_persistence(tmp_path):
    """The incremental (non-full-rewrite) persistence path used by
    engine.add_document/delete_document: upsert_document_postings and
    remove_document_postings, exercised together against real SQLite."""
    import app.database.db as db_module

    db_module.DB_PATH = tmp_path / "incremental.db"
    conn = db_module.get_connection()
    db_module.init_db(conn)

    idx = InvertedIndex()
    idx.add_document(1, "database indexing systems", stem=False)
    conn.execute(
        "INSERT INTO documents (id, filename, title, file_type, upload_ts, size_bytes, "
        "text, word_count, token_count) VALUES (1,'a.txt','A','txt','2026-01-01',1,'x',1,?)",
        (idx.doc_lengths[1],),
    )
    conn.commit()
    idx.upsert_document_postings(conn, 1)

    row = conn.execute("SELECT doc_freq FROM terms WHERE term = 'database'").fetchone()
    assert row["doc_freq"] == 1
    row2 = conn.execute("SELECT term_freq FROM postings WHERE term='database' AND doc_id=1").fetchone()
    assert row2["term_freq"] == 1

    affected = idx.remove_document(1)
    conn.execute("DELETE FROM documents WHERE id = 1")
    conn.commit()
    idx.remove_document_postings(conn, 1, affected)

    remaining = conn.execute("SELECT COUNT(*) as c FROM postings WHERE doc_id = 1").fetchone()
    assert remaining["c"] == 0
    remaining_terms = conn.execute("SELECT COUNT(*) as c FROM terms WHERE term = 'database'").fetchone()
    assert remaining_terms["c"] == 0  # doc_freq hit 0, term row should be dropped
    conn.close()
