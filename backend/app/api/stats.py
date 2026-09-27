from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_engine
from app.models.schemas import RebuildResponse, StatsResponse
from app.search.engine import SearchEngine

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
def get_stats(engine: SearchEngine = Depends(get_engine)) -> StatsResponse:
    return StatsResponse(**engine.stats())


@router.post("/index/rebuild", response_model=RebuildResponse)
def rebuild_index(engine: SearchEngine = Depends(get_engine)) -> RebuildResponse:
    return RebuildResponse(**engine.rebuild())
