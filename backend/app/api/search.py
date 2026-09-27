from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_engine
from app.models.schemas import SearchRequest, SearchResponse
from app.search.engine import SearchEngine

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search_documents(
    req: SearchRequest, engine: SearchEngine = Depends(get_engine)
) -> SearchResponse:
    result = engine.search(
        req.query,
        req.ranking,
        req.top_k,
        req.page,
        bm25_k1=req.bm25_k1,
        bm25_b=req.bm25_b,
    )
    return SearchResponse(**result)
