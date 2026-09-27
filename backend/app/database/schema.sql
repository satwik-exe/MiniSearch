-- MiniSearch database schema
-- Two concerns are deliberately separated:
--   1. Document storage (the raw corpus + metadata)
--   2. Index storage (the inverted index, persisted so we don't rebuild on every restart)

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    title           TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    upload_ts       TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    text            TEXT NOT NULL,
    word_count      INTEGER NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0   -- tokens after preprocessing (used for BM25 length norm)
);

-- terms: one row per unique term in the vocabulary, with its document frequency (df).
-- df is maintained incrementally as documents are added/removed, avoiding a full
-- vocabulary scan on every query.
CREATE TABLE IF NOT EXISTS terms (
    term        TEXT PRIMARY KEY,
    doc_freq    INTEGER NOT NULL DEFAULT 0
);

-- postings: the inverted index itself. Each row is one (term, document) pair with the
-- term frequency in that document and the token positions (for phrase queries / snippet
-- generation). This is the structure that lets a query touch only the documents that
-- actually contain a query term, instead of scanning the whole corpus.
--
-- ON DELETE CASCADE is a safety net: app/search/engine.py's delete_document() also
-- explicitly removes this document's postings rows itself (via
-- InvertedIndex.remove_document_postings, which needs to run anyway to update the
-- affected terms' doc_freq), so in normal operation the CASCADE fires first and finds
-- nothing left to do. It exists so that ANY deletion of a documents row -- including
-- one done directly in a DB shell during debugging, not just through the application
-- code path -- can never leave orphaned postings behind.
CREATE TABLE IF NOT EXISTS postings (
    term        TEXT NOT NULL,
    doc_id      INTEGER NOT NULL,
    term_freq   INTEGER NOT NULL,
    positions   TEXT NOT NULL,  -- JSON array of token positions, e.g. "[3, 17, 42]"
    PRIMARY KEY (term, doc_id),
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id);

-- Supports `ORDER BY upload_ts DESC` in GET /api/documents (list_documents) without
-- a full-table scan + sort as the corpus grows. Added during audit -- an earlier
-- version of this schema had no index backing that query's ORDER BY.
CREATE INDEX IF NOT EXISTS idx_documents_upload_ts ON documents(upload_ts);

-- index_meta: small key/value table for corpus-level statistics (total docs, total tokens,
-- average document length, last rebuild timestamp, last index build duration).
CREATE TABLE IF NOT EXISTS index_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
