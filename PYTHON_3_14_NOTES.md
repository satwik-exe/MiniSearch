# Python 3.14 / Windows Compatibility Update

## What changed

### `backend/requirements.txt`

The previous file pinned exact old versions (`fastapi==0.115.0`, `pydantic==2.9.2`,
`uvicorn[standard]==0.30.6`, etc.) that predate Python 3.14's October 2025 release and
have no prebuilt wheels for it. Installing them on Python 3.14 would fall back to
building `pydantic-core` from source, which needs a Rust toolchain — not present on a
default Windows machine, and exactly what you asked to avoid.

Updated to version **floors** (not new exact pins) at each package's own first release
supporting Python 3.14, verified against each project's own release notes/changelog:

| Package | Old pin | New requirement | Why |
|---|---|---|---|
| `fastapi` | `==0.115.0` | `>=0.118.3,<1.0.0` | 0.118.3 (2025-10-10) is FastAPI's own first release supporting Python 3.14 |
| `pydantic` | `==2.9.2` | `>=2.12.0,<3.0.0` | 2.12.0 (2025-10-07) is pydantic's own first release with `cp314` `pydantic-core` wheels — this is the actual fix for the exact problem you reported |
| `uvicorn` | `[standard]==0.30.6` | `>=0.34.0,<1.0.0` (**no `[standard]` extra**) | see below |
| `python-multipart` | `==0.0.9` | `>=0.0.9` | pure Python, no C extension, no 3.14-specific issue |
| `pypdf` | `==5.9.0` | `>=5.9.0` | pure Python; Fedora already rebuilt it against Python 3.14 release candidates in 2025 with no source changes |
| `pytest` | `==8.3.3` | `>=8.3.3` | pure Python |
| `httpx` | `==0.27.2` | `>=0.27.2` | pure Python |
| `reportlab` | *(missing)* | `>=4.0.0` *(new)* | was actually used by `tests/test_ingestion.py` to generate a real PDF for the PDF-parsing tests, but was never listed in `requirements.txt` — added so `pip install -r requirements.txt` is actually sufficient to run the full test suite |

**Why `uvicorn` and not `uvicorn[standard]`:** the `[standard]` extra pulls in
`httptools` (a C extension) and `uvloop`. `uvloop` already refuses to install on
Windows via its own dependency markers, but `httptools` does not, and has historically
lagged behind new CPython releases in publishing prebuilt wheels — there's a public
uvicorn GitHub discussion (#2413) of exactly this failure mode on Windows with Python
3.13, where a missing `httptools` wheel forced a from-source build that needed the MSVC
toolchain. Since MiniSearch doesn't use WebSockets and isn't a
performance-critical/high-throughput service, plain `uvicorn` (pure-Python `h11` +
`click`, zero C extensions) sidesteps that risk entirely on Windows, regardless of
whether `httptools` has caught up to 3.14 yet. The tradeoff: without `colorama` (part of
the `[standard]` extra), uvicorn's console log colors may not render on old `cmd.exe`;
modern Windows Terminal/PowerShell handle ANSI natively regardless, so this is cosmetic
only.

### `backend/app/main.py`

Migrated `@app.on_event("startup")` to FastAPI's `lifespan` context-manager pattern.
This is **not a functional change** — `engine.startup()` still runs exactly once at
process startup, before any request is served — but `on_event` has been deprecated for
a long time and was removed entirely in later Starlette/FastAPI releases (the version
range now allowed by `requirements.txt`, `fastapi>=0.118.3,<1.0.0`, includes releases
where `on_event` no longer exists at all). Without this change, the exact
version-unpinning this update is meant to provide would risk trading one hard
incompatibility (old pydantic vs. Python 3.14) for a different one (new FastAPI vs. an
API that no longer exists).

No other application code changed. Everything else — the inverted index, TF-IDF, BM25,
ingestion, the API routes, the frontend, Docker config — is untouched, per your
instruction to preserve the existing implementation.

## What was actually verified, and how

**This sandbox has only Python 3.12.3 installed, and no internet access** (confirmed by
testing `pip install`, which returns a 403 from the network proxy, and by the complete
absence of Python 3.13 or 3.14 anywhere on this machine). Both constraints were true for
the entire MiniSearch project up to this point too.

Given that, here's exactly what could and could not be done:

- **Could verify**: that the application code itself doesn't regress under the
  `lifespan` migration. Ran the full backend test suite (69 of 76 tests executable
  without FastAPI/pydantic installed — see the main `AUDIT.md` for why) against Python
  3.12 after the `main.py` change: **69 passed, 0 failed**, identical to before the
  change. This confirms the migration doesn't break the application's own logic; it does
  **not** confirm the package version floors above actually install and run cleanly on a
  real Python 3.14 interpreter.
- **Could not verify**: literally running `pip install -r requirements.txt` on Python
  3.14, on Windows, and confirming the server boots and serves a request. This would
  require an internet-connected Python 3.14 environment, which this sandbox is not.
- **Could not verify**: whether `fastapi.testclient.TestClient` still works identically
  with the `lifespan` context-manager syntax used here (it should — `with TestClient(app)
  as client:` explicitly documented by Starlette/FastAPI to trigger `lifespan` startup
  and shutdown — but this claim rests on documented behavior, not an executed test, since
  `fastapi` itself isn't installable here).

**Please verify directly on your machine** before relying on this:

```powershell
py -3.14 -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
cd backend
pytest tests\ -v
uvicorn app.main:app --reload
```

If `pip install` or any test fails, that's real information this update could not
obtain from inside this sandbox — please share the exact error and it can be diagnosed
and fixed from there, rather than guessed at.

## A note on confidence

The specific version floors above (FastAPI 0.118.3, pydantic 2.12.0, Starlette 0.48.0)
come from each project's own dated release notes, cross-checked against multiple
independent sources (package-readiness tracking sites, Linux distribution package
changelogs that rebuild for new CPython releases, and the projects' own GitHub release
tags) found via web search — not from memory alone, and not verified by actually running
pip on Python 3.14. They should be correct, but "should be correct based on published
release notes" and "verified by installing and running" are different claims, and this
document is deliberately keeping them distinct rather than blurring them together.
