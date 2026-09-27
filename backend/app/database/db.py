"""SQLite connection management.

We use a single SQLite file for both document storage and the persisted inverted
index. SQLite is sufficient here: MiniSearch is a small/medium single-node corpus
(hundreds to low-thousands of documents), and SQLite gives us durability, simple
querying of the postings table, and zero operational overhead — no separate DB
server to run. See README "Design decisions" for the full justification.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "minisearch.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables if they don't already exist. Idempotent — safe to call on every startup."""
    owns_conn = conn is None
    conn = conn or get_connection()
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        if owns_conn:
            conn.close()
