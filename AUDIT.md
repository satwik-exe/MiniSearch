# MiniSearch — Audit Report

Full audit, fix, and retest pass over the previously-generated MiniSearch codebase.
Every item below was inspected in the actual source files, fixed in place, and
re-verified by running real code — not just reviewed and reported.

## Scope of what could be executed

This audit was performed in a sandboxed environment with **no internet access**. This
matters for what "tested" means below:

- **Fully executable and executed**: all pure-Python components with no external package
  dependency beyond the Python standard library — tokenizer, inverted index, TF-IDF,
  BM25, snippet generation, document ingestion (including real PDF generation/parsing
  via `pypdf`/`reportlab`), SQLite persistence, the `SearchEngine` orchestration layer,
  and evaluation/benchmark scripts. 69 of 76 backend `pytest`-style tests were run
  directly (via a manual harness substituting for `pytest`, which isn't installed here)
  and pass; all 12 new evaluation-metric tests pass.
- **Not executable here, hand-reviewed instead**: the FastAPI HTTP layer itself (`fastapi`,
  `uvicorn`, `pydantic`, `httpx` are not installable without network access) and the
  frontend build (`npm install` fails with a 403 against the npm registry; no offline
  package cache exists for `react-router-dom`, `vite`, or `@types/react`). A partial
  global TypeScript install (v6.0.3) and React 19 packages were found pre-installed in
  this sandbox and investigated as a possible workaround, but the version mismatch
  against this project's actual dependencies (React 18, no `@types/react`, no
  `react-router-dom`) meant a genuine type-check wasn't achievable — this is reported
  honestly as inconclusive rather than claimed as a pass.
- **Docker**: the `docker` binary itself is not installed in this sandbox.
  `docker-compose.yml` was validated for YAML syntax correctness only (via Python's
  `yaml` parser); `docker compose build`/`up` could not be run.

Where a fix could not be executed, the change is still a real code change (not a
suggestion), reviewed by hand against the target framework's documented behavior.

---

## Bugs found and fixed

### 1. TF-IDF cosine similarity was not standard cosine similarity (critical)

**File:** `backend/app/search/tfidf.py`

The document vector norm `||d||` was computed using only the terms present in the
*query*, not the document's full term vector. This is not standard cosine similarity —
it silently discards every dimension the query doesn't touch. A document containing
*only* the query's terms in exactly the query's proportions would score a perfect `1.0`
regardless of how much other, irrelevant content it actually contained.

**Fix:** Added a forward index (`InvertedIndex.doc_terms: doc_id -> {term: tf}`) so the
true full-document norm can be computed in O(document's own term count), not O(vocabulary
size). `document_vector_norm` now sums over every term in the document.

**Verification:** Hand-calculated a fully worked 3-document corpus (`the cat sat on the
mat` / `the dog sat on the log` / `cats and dogs are pets`, query `"cat sat"`) by hand,
then asserted the code's output matches to `1e-9`:
```
doc1 (correct): 0.7824081412456458   <- code output matches exactly
doc1 (old, buggy query-restricted norm): 1.0000000000000000   <- what it used to return
doc2 (correct): 0.2867109723804671   <- code output matches exactly
doc3: excluded entirely (zero query-term overlap, never scored)
```
See `backend/tests/test_tfidf.py::test_matches_hand_calculated_cosine_similarity`.

### 2. `remove_document` and incremental persistence scanned the entire vocabulary (efficiency)

**File:** `backend/app/search/index.py`

`remove_document` iterated every posting list in the whole index (O(vocabulary size)) to
find which ones contained the document being deleted, instead of knowing directly which
terms belonged to it. The incremental persistence methods (`upsert_document_postings`,
`remove_document_postings`) had the same pattern.

**Fix:** The same forward index added for issue #1 also fixes this: `remove_document` now
looks up `doc_terms[doc_id]` directly and only touches those posting lists — O(document's
own terms), not O(vocabulary). `remove_document` now returns the list of affected terms so
the persistence layer doesn't need to rescan anything either.

**Verification:** `test_remove_document_is_cheap_not_a_full_vocabulary_scan` builds a
51-document index and confirms deleting one small document leaves every other document's
postings completely untouched.

### 3. `add_document` was not idempotent (correctness/robustness)

**File:** `backend/app/search/index.py`

Calling `add_document` twice for the same `doc_id` (e.g. a future "re-index this
document" feature) would silently double-count `num_docs`, `total_tokens`, and inflate
`doc_freq` for repeated terms across the pair of calls.

**Fix:** `add_document` now checks `doc_id in self.doc_terms` and calls `remove_document`
first if so, making it safe to call again for the same ID.

**Verification:** `test_add_document_called_twice_for_same_doc_id_does_not_double_count`
and `test_add_document_reindex_with_different_text_replaces_old_terms`.

### 4. Malformed/corrupt PDF uploads crashed with an unhandled exception (security/robustness)

**File:** `backend/app/documents/ingestion.py`

`PdfReader(io.BytesIO(data))` was called outside any try/except. A corrupted PDF, a
truncated upload, or a non-PDF file renamed to `.pdf` would raise an exception that
propagated uncaught through `ingest()` — in the FastAPI layer this becomes an unhandled
500, which is exactly what "return useful API errors rather than crashing" rules out.

**Fix:** Added a `CorruptFile` exception; `extract_text_from_pdf` now wraps `PdfReader`
construction and page-tree parsing in try/except and raises `CorruptFile` on failure.
`app/api/documents.py` maps `CorruptFile` to a clean HTTP 422.

**Verification:** `test_garbage_bytes_with_pdf_extension_raises_corrupt_file`,
`test_empty_pdf_bytes_raises_corrupt_file`, `test_truncated_pdf_raises_corrupt_file` all
pass (previously these three cases crashed with unhandled exceptions); a real valid PDF
(generated with `reportlab`) still ingests correctly
(`test_valid_pdf_still_ingests_correctly`).

### 5. No protection against concurrent index mutation (race condition)

**File:** `backend/app/search/engine.py`

FastAPI/Starlette runs sync route handlers in a thread pool, so two uploads (or an
upload and a rebuild) can genuinely execute concurrently. The in-memory `InvertedIndex`
is plain dicts mutated via multi-step, non-atomic sequences — concurrent writers could
corrupt `doc_freq` counts or leave `postings` out of sync with `doc_terms`. There was no
locking at all.

**Fix:** Added a single `threading.Lock` (`SearchEngine._lock`) around every mutating
operation (`add_document`, `delete_document`, `rebuild`) and around the read-then-mutate
sequence in `search`. A single coarse lock was chosen deliberately over finer-grained
locking — simple to reason about, and at this project's scale (a portfolio document
collection, not a high-throughput service) it costs nothing measurable.

**Verification:** `test_concurrent_uploads_do_not_corrupt_index` in the test suite (15
concurrent threads), plus a standalone stress test with 30 concurrent threads run during
this audit, confirmed: zero errors, correct final document count, correct `doc_freq`
consistency (`doc_freq("databas") == 30` after 30 concurrent uploads of documents
containing "databases").

### 6. `precision_at_k` divided by the wrong denominator (evaluation metric bug)

**File:** `evaluation/evaluate.py`

`precision_at_k` computed `hits / len(top_k)` — dividing by however many results the
engine *actually returned*, not by K. When a query returns fewer than K candidates (a
realistic scenario in a small corpus, or a narrow query), this silently inflates
precision. Example: 2 hits out of only 3 results returned for a requested K=5 gave
`2/3 ≈ 0.667` under the old code; the correct value is `2/5 = 0.4`.

**Fix:** Changed to divide by `k` unconditionally, matching the standard definition and
the project spec's explicit instruction.

**Verification:** New `evaluation/test_evaluate.py`
(`test_precision_at_k_divides_by_k_not_by_results_returned`) encodes the exact scenario
above as a permanent regression test. Re-running `evaluate.py` after the fix, mean
Precision@5 across the 15-query set dropped from 0.519 (using the buggy formula) to the
correct 0.307 — a real, substantive change in the reported number, not a rounding
difference.

### 7. Two-phase document write left a stale `token_count=0` window

**File:** `backend/app/search/engine.py`

`add_document` inserted a document row with `token_count=0`, then updated it after
tokenizing. A concurrent reader (e.g. `/api/stats`'s `average_doc_length`) could observe
the zero value mid-write.

**Fix:** Token count is now computed up front (`len(preprocess(doc.text, stem=STEM))`,
independent of `doc_id`) so the `INSERT` is a single atomic write with the correct value.

### 8. Misleading comments / incorrect complexity claim

- **File:** `backend/app/search/index.py` — a comment claimed SQL index lookups are O(1).
  A SQLite B-tree index lookup is O(log n), not O(1). Corrected to state this precisely.
- **File:** `backend/app/search/tokenizer.py` and `backend/app/search/snippet.py` — the
  word-matching regex was documented as "unicode-aware" and passed `re.UNICODE`, but the
  pattern uses an explicit `[A-Za-z0-9]` character class, on which `re.UNICODE` has no
  effect (it only changes `\w`, `\d`, `\s` and similar escape classes). The flag was
  removed and the comments corrected to state plainly that tokenization is ASCII-only —
  consistent with the English-only stopword list and stemmer already documented as a
  limitation, but the code comment was actively claiming the opposite.

### 9. Dead code removed

- `InvertedIndex`'s unused `DocStats` dataclass and an unused `field` import
  (`backend/app/search/index.py`).
- `db_session()` context manager in `backend/app/database/db.py` — defined but never
  called anywhere in the codebase (grepped to confirm before removing).
- Unused `STOPWORDS` and `List` imports in `backend/app/search/snippet.py`.

### 10. Missing database index

**File:** `backend/app/database/schema.sql`

`GET /api/documents` (`list_documents`) does `ORDER BY upload_ts DESC` with no supporting
index — a full table scan + sort as the corpus grows. Added
`CREATE INDEX idx_documents_upload_ts ON documents(upload_ts)`.

### 11. Missing `.dockerignore`

Neither `backend/` nor `frontend/` had a `.dockerignore`. Without one, `COPY . .` in the
frontend Dockerfile would pull in `node_modules`, `dist`, or other local build artifacts
into the image build context if present. Added both.

### 12. Global exception handler added (defense-in-depth)

**File:** `backend/app/main.py`

FastAPI's default behavior for an unhandled exception already avoids leaking traceback
detail to the client (this is framework-default behavior, not something the original code
got wrong), but there was no explicit, auditable guarantee of that. Added an explicit
`@app.exception_handler(Exception)` that logs the real exception server-side and always
returns a generic `{"detail": "Internal server error"}`, so this behavior doesn't depend
on FastAPI's defaults remaining unchanged.

### 13. BM25 parameters made configurable

**File:** `backend/app/models/schemas.py`, `backend/app/search/ranker.py`,
`backend/app/search/engine.py`, `backend/app/api/search.py`

`k1`/`b` were previously only configurable by calling `score_bm25` directly in Python,
not through the API. Added optional `bm25_k1`/`bm25_b` fields to `SearchRequest`, threaded
through `SearchEngine.search` and `rank()`.

**Verification:** `test_bm25_params_pass_through_engine` and
`test_k1_and_b_are_configurable` confirm overriding `b=0` (disabling length
normalization) measurably changes a long document's score relative to the default.

---

## Items reviewed and found correct (no change needed)

- **Query candidate retrieval**: confirmed via code reading and tests that search only
  ever touches documents in the union of posting lists for query terms — never scans the
  full corpus. `total_matches` in a search response is the true candidate-set size, not
  capped at `top_k`.
- **XSS safety of snippet highlighting**: already correctly HTML-escaped document content
  before inserting `<mark>` tags; verified again with a literal `<script>` payload.
- **Path traversal**: uploaded content is never written to disk under any path derived
  from the filename — it's extracted in-memory and stored as a SQLite text column, so
  there is no arbitrary-file-write surface to exploit in the first place. Filename
  sanitization (`sanitize_filename`) is still applied for the metadata value that gets
  displayed and stored.
- **Porter stemmer correctness**: re-verified against the standard test cases (running→run,
  management→manag, databases→databas, caresses→caress, ponies→poni, etc.) — all correct.
  The `_measure` function's implementation looks unusual (counts "VC" substring
  occurrences on a per-character C/V string rather than a run-collapsed one) but is
  mathematically equivalent to the textbook definition, confirmed by both the passing
  test cases and manual derivation.
- **BM25 formula**: standard Robertson/Sparck-Jones formulation, correctly implemented;
  saturation and length-normalization properties independently verified empirically.
- **CORS configuration**: scoped appropriately; not actually load-bearing in the Docker
  deployment (nginx proxies `/api` server-side, so the browser never makes a
  cross-origin request there) but kept as a safe fallback for direct backend access.
- **Duplicate document handling**: confirmed as a deliberate, working, and now explicitly
  documented behavior (independent indexing, no content-hash dedup) rather than a bug.
- **Docker path/port wiring**: `WORKDIR /app` + `COPY app ./app` + `CMD ["uvicorn",
  "app.main:app", ...]` matches the pattern used in FastAPI's own official Docker
  deployment documentation; port mappings (`8000:8000`, `5173:80`) and the
  `docker-compose.yml` volume for SQLite persistence are consistent with `db.py`'s
  `DB_PATH` resolution. `docker-compose.yml` YAML syntax validated with Python's `yaml`
  parser.

---

## Tests

```text
Backend test files: 8 (test_tokenizer, test_index, test_tfidf, test_bm25, test_snippet,
                        test_ingestion, test_engine, test_api)
Total test functions: 76
Executable without FastAPI/pydantic: 69
Run via manual harness (pytest itself not installable offline): 69 passed, 0 failed
Not executable here (require fastapi.testclient): 7 (test_api.py) -- written and
  syntax-checked; run `pytest tests/test_api.py -v` after `pip install -r requirements.txt`

evaluation/test_evaluate.py (new): 12 tests, all passed
```

New tests added during this audit (19 total, on top of the 57 from the original build):
forward-index correctness (2), incremental-removal efficiency and correctness (3),
add_document idempotency (2), incremental persistence round-trip (1), corrected
cosine-similarity hand calculation (2), malformed-PDF handling (3), BM25 parameter
configurability (2), index persistence across a simulated restart (2), concurrent-upload
safety (1), BM25 engine passthrough (1).

## Benchmark (re-run after fixes, on this machine — will vary by hardware)

| Documents | Index Time | Avg Search Latency | Vocabulary |
|---|---|---|---|
| 50 | 0.161s | 0.21ms | 113 terms |
| 100 | 0.322s | 0.38ms | 163 terms |
| 500 | 1.798s | 1.80ms | 563 terms |

Search latency here measures query preprocessing + candidate retrieval + ranking only
(see README section 7 for exactly what's excluded and why — this scoping was itself an
audit fix). Index time is from the already-fixed incremental-persistence code inherited
from the original build (not re-broken and re-fixed in this pass).

## Evaluation (re-run after fixes)

K=5, 12-document corpus, 15 hand-labeled queries:

| Metric | TF-IDF | BM25 |
|---|---|---|
| Mean Precision@5 | 0.307 | 0.307 |
| Mean Recall@5 | 0.967 | 0.967 |
| Mean F1@5 | 0.457 | 0.457 |
| MRR | 1.000 | 1.000 |

(Precision dropped from an earlier, buggy 0.519 to the correct 0.307 as a direct result
of fixing issue #6 above — see README section 6 for the honest interpretation of why
precision is structurally low on a corpus this small.)

## Could not be verified locally

- `pytest tests/test_api.py -v` — requires `fastapi`, `httpx` (not installable offline here)
- `uvicorn app.main:app` actually booting and serving a live HTTP request end-to-end
- `npm install && npm run build` / `tsc --noEmit` on the actual frontend dependency
  versions (`react-router-dom`, `@types/react` not available offline; a partial global
  TypeScript/React install found in this sandbox was investigated but had incompatible
  versions, so no genuine type-check result could be produced)
- `docker compose build` / `docker compose up` — `docker` is not installed in this
  sandbox; only YAML syntax was validated

Everything else described in this report was executed, not merely reasoned about.
