"""Shared singleton SearchEngine instance, injected into API routes via FastAPI's
Depends(). Kept in its own module so both main.py and the routers can import it
without a circular import."""
from __future__ import annotations

from app.search.engine import SearchEngine

engine = SearchEngine()


def get_engine() -> SearchEngine:
    return engine
