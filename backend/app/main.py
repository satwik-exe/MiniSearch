from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.dependencies import engine
from app.api import documents, search, stats

logger = logging.getLogger("minisearch")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook, using FastAPI's `lifespan` context manager rather
    than the older `@app.on_event("startup")` decorator. `on_event` has been
    deprecated for a long time and was removed entirely in the Starlette 1.0 /
    FastAPI ~0.133+ line (see requirements.txt, which allows FastAPI versions
    up to but not including 1.0.0 -- a range that includes releases where
    `on_event` no longer exists at all). `lifespan` is the form FastAPI's own
    docs have recommended since well before that removal, so this is a
    forward-compatible fix, not a functional change: engine.startup() still
    runs exactly once, at process startup, before the app serves any requests.
    """
    engine.startup()
    yield


app = FastAPI(
    title="MiniSearch",
    description="A lightweight information retrieval engine with TF-IDF and BM25 ranking.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: only the local Vite dev server / same-origin production build need access.
# Adjust origins for your deployment; avoid "*" with credentials in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch anything that escapes the route handlers (a DB error, a bug in a
    code path we didn't anticipate, etc.) and return a clean, generic JSON
    error instead of letting Starlette's default handler potentially include
    traceback detail. The real exception is logged server-side for debugging,
    never sent to the client. Expected, specific failures (bad upload, missing
    document, invalid search params) are handled by explicit HTTPException
    raises in the route modules and never reach this handler -- this is a
    defense-in-depth backstop for the unexpected case, not the primary error
    handling path.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(stats.router, prefix="/api", tags=["stats"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
