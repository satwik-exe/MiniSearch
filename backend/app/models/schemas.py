from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    title: str
    file_type: str
    upload_ts: str
    size_bytes: int
    word_count: int
    token_count: int


class DocumentDetail(DocumentOut):
    text: str


class UploadResponse(BaseModel):
    document: DocumentOut
    message: str


class SearchRequest(BaseModel):
    query: str = Field(max_length=1000)
    ranking: str = Field(default="tfidf", pattern="^(tfidf|bm25)$")
    top_k: int = Field(default=10, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    # BM25 tuning parameters, exposed for experimentation (see app/search/bm25.py
    # for what each controls). Ignored when ranking="tfidf". Left unset (None) to
    # use the standard defaults (k1=1.5, b=0.75).
    bm25_k1: Optional[float] = Field(default=None, ge=0)
    bm25_b: Optional[float] = Field(default=None, ge=0, le=1)


class SearchResultItem(BaseModel):
    document_id: int
    title: str
    filename: str
    score: float
    snippet: str
    matched_terms: List[str]


class SearchResponse(BaseModel):
    query: str
    ranking: str
    results: List[SearchResultItem]
    total_matches: int
    page: int
    top_k: int
    search_time_ms: float


class StatsResponse(BaseModel):
    documents_indexed: int
    unique_terms: int
    total_tokens: int
    average_doc_length: float
    index_size_bytes: int
    last_rebuild_ts: Optional[str] = None
    last_index_build_ms: Optional[float] = None
    last_search_latency_ms: Optional[float] = None


class RebuildResponse(BaseModel):
    documents_indexed: int
    unique_terms: int
    total_tokens: int
    build_time_ms: float
