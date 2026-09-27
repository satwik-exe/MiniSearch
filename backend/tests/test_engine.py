import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def engine(tmp_path):
    import app.database.db as db_module

    db_module.DB_PATH = tmp_path / "test.db"
    from app.search.engine import SearchEngine

    e = SearchEngine()
    e.startup()
    return e


def _add(engine, filename, text):
    from app.documents.ingestion import ingest

    doc = ingest(filename, text.encode("utf-8"))
    return engine.add_document(doc)


def test_upload_then_search_finds_relevant_document(engine):
    _add(engine, "os.txt", "Operating systems manage memory and processes.")
    _add(engine, "cooking.txt", "How to bake bread with flour and yeast.")

    result = engine.search("memory management", "tfidf", top_k=10, page=1)
    titles = [r["title"] for r in result["results"]]
    assert "Os" in titles
    assert "Cooking" not in titles


def test_delete_document_removes_it_from_search(engine):
    row = _add(engine, "os.txt", "Operating systems manage memory and processes.")
    result = engine.search("memory", "tfidf", top_k=10, page=1)
    assert len(result["results"]) == 1

    engine.delete_document(row["id"])
    result2 = engine.search("memory", "tfidf", top_k=10, page=1)
    assert len(result2["results"]) == 0


def test_empty_query_returns_no_results(engine):
    _add(engine, "os.txt", "Operating systems manage memory.")
    result = engine.search("", "tfidf", top_k=10, page=1)
    assert result["results"] == []
    assert result["total_matches"] == 0


def test_query_with_only_unknown_terms(engine):
    _add(engine, "os.txt", "Operating systems manage memory.")
    result = engine.search("zzznonexistentxyz", "tfidf", top_k=10, page=1)
    assert result["results"] == []


def test_duplicate_documents_both_indexed(engine):
    _add(engine, "a.txt", "Duplicate content about databases.")
    _add(engine, "b.txt", "Duplicate content about databases.")
    result = engine.search("databases", "tfidf", top_k=10, page=1)
    assert len(result["results"]) == 2


def test_empty_document_does_not_crash_search(engine):
    _add(engine, "empty.txt", "")
    result = engine.search("anything", "tfidf", top_k=10, page=1)
    assert result["results"] == []


def test_stopword_only_query(engine):
    _add(engine, "os.txt", "Operating systems manage memory.")
    result = engine.search("the a an is", "tfidf", top_k=10, page=1)
    assert result["results"] == []


def test_pagination(engine):
    for i in range(5):
        _add(engine, f"doc{i}.txt", "database indexing performance tuning")
    page1 = engine.search("database", "tfidf", top_k=2, page=1)
    page2 = engine.search("database", "tfidf", top_k=2, page=2)
    assert len(page1["results"]) == 2
    assert len(page2["results"]) == 2
    assert {r["document_id"] for r in page1["results"]}.isdisjoint(
        {r["document_id"] for r in page2["results"]}
    )


def test_search_reports_latency(engine):
    _add(engine, "os.txt", "Operating systems manage memory.")
    result = engine.search("memory", "tfidf", top_k=10, page=1)
    assert result["search_time_ms"] >= 0


def test_both_ranking_strategies_available(engine):
    _add(engine, "os.txt", "Operating systems manage memory and processes efficiently.")
    tfidf_result = engine.search("memory", "tfidf", top_k=10, page=1)
    bm25_result = engine.search("memory", "bm25", top_k=10, page=1)
    assert len(tfidf_result["results"]) == 1
    assert len(bm25_result["results"]) == 1


def test_rebuild_preserves_search_correctness(engine):
    _add(engine, "os.txt", "Operating systems manage memory.")
    engine.rebuild()
    result = engine.search("memory", "tfidf", top_k=10, page=1)
    assert len(result["results"]) == 1


# --- Added during audit -----------------------------------------------------


def test_index_persists_across_restart(tmp_path):
    """upload -> index -> search -> restart -> search still finds it (explicit
    audit requirement: index persistence across an application restart)."""
    import app.database.db as db_module
    from app.search.engine import SearchEngine

    db_module.DB_PATH = tmp_path / "restart_test.db"

    engine1 = SearchEngine()
    engine1.startup()
    _add(engine1, "os.txt", "Operating systems manage memory and scheduling.")
    result1 = engine1.search("memory scheduling", "tfidf", top_k=10, page=1)
    assert len(result1["results"]) == 1

    # Simulate a full restart: brand-new SearchEngine instance, same DB file.
    engine2 = SearchEngine()
    engine2.startup()
    result2 = engine2.search("memory scheduling", "tfidf", top_k=10, page=1)
    assert len(result2["results"]) == 1
    assert result2["results"][0]["title"] == result1["results"][0]["title"]


def test_delete_then_restart_stays_deleted(tmp_path):
    import app.database.db as db_module
    from app.search.engine import SearchEngine

    db_module.DB_PATH = tmp_path / "delete_restart_test.db"

    engine1 = SearchEngine()
    engine1.startup()
    row = _add(engine1, "os.txt", "Operating systems manage memory.")
    engine1.delete_document(row["id"])

    engine2 = SearchEngine()
    engine2.startup()
    result = engine2.search("memory", "tfidf", top_k=10, page=1)
    assert len(result["results"]) == 0
    assert engine2.stats()["documents_indexed"] == 0


def test_bm25_params_pass_through_engine(engine):
    _add(engine, "a.txt", "memory " * 5 + "padding " * 5)
    _add(engine, "b.txt", "memory " * 2 + "filler " * 50)
    default = engine.search("memory", "bm25", top_k=10, page=1)
    tuned = engine.search("memory", "bm25", top_k=10, page=1, bm25_k1=1000.0, bm25_b=0.0)
    default_scores = {r["document_id"]: r["score"] for r in default["results"]}
    tuned_scores = {r["document_id"]: r["score"] for r in tuned["results"]}
    assert default_scores != tuned_scores


def test_concurrent_uploads_do_not_corrupt_index(engine):
    """Lighter-weight version of the manual concurrency stress test: enough
    threads to make interleaving likely, run inside the normal test suite."""
    import threading

    errors = []

    def upload_one(i):
        try:
            _add(engine, f"doc{i}.txt", f"concurrent document {i} about databases")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=upload_one, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert engine.index.num_docs == 15
    result = engine.search("databases", "tfidf", top_k=100, page=1)
    assert result["total_matches"] == 15
