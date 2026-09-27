"""API-level tests using FastAPI's TestClient.

NOTE: these require fastapi/httpx to be installed (`pip install -r requirements.txt`).
They were written and syntax-checked but could not be executed in the development
sandbox used to build this project, which has no network access to install packages.
Run `pytest tests/test_api.py -v` after installing requirements to verify.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.database.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_api.db")

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_and_search_flow(client):
    file_content = b"Operating systems manage memory and processes efficiently."
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("os.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert resp.status_code == 200
    doc_id = resp.json()["document"]["id"]

    resp = client.post("/api/search", json={"query": "memory management", "ranking": "tfidf", "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["document_id"] == doc_id


def test_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("archive.zip", io.BytesIO(b"PK\x03\x04"), "application/zip")},
    )
    assert resp.status_code == 415


def test_list_and_delete_document(client):
    file_content = b"Some sample content for listing."
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("sample.txt", io.BytesIO(file_content), "text/plain")},
    )
    doc_id = resp.json()["document"]["id"]

    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert any(d["id"] == doc_id for d in resp.json())

    resp = client.delete(f"/api/documents/{doc_id}")
    assert resp.status_code == 200

    resp = client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 404


def test_stats_endpoint(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents_indexed" in body
    assert "unique_terms" in body


def test_rebuild_endpoint(client):
    resp = client.post("/api/index/rebuild")
    assert resp.status_code == 200
    assert "documents_indexed" in resp.json()


def test_search_validates_ranking_field(client):
    resp = client.post("/api/search", json={"query": "test", "ranking": "invalid_ranking"})
    assert resp.status_code == 422  # pydantic pattern validation rejects it
