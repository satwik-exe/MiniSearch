"""Ranking dispatcher: selects TF-IDF or BM25 and reports which query terms matched."""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from app.search.index import InvertedIndex
from app.search.tfidf import score_tfidf
from app.search.bm25 import score_bm25
from app.search.tokenizer import preprocess


def rank(
    query: str,
    ranking: str,
    index: InvertedIndex,
    *,
    stem: bool = True,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
) -> Dict[int, float]:
    if ranking == "tfidf":
        return score_tfidf(query, index, stem=stem)
    if ranking == "bm25":
        kwargs = {}
        if bm25_k1 is not None:
            kwargs["k1"] = bm25_k1
        if bm25_b is not None:
            kwargs["b"] = bm25_b
        return score_bm25(query, index, stem=stem, **kwargs)
    raise ValueError(f"Unknown ranking strategy '{ranking}'. Use 'tfidf' or 'bm25'.")


def matched_terms_for_doc(query: str, doc_id: int, index: InvertedIndex, *, stem: bool = True) -> List[str]:
    """Which (preprocessed) query terms actually appear in this document."""
    query_terms = preprocess(query, stem=stem)
    seen: Set[str] = set()
    matched = []
    for term in query_terms:
        if term in seen:
            continue
        seen.add(term)
        if doc_id in index.get_posting_list(term):
            matched.append(term)
    return matched
