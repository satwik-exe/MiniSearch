"""BM25 ranking (Okapi BM25), implemented from scratch.

    BM25(q, d) = sum over query terms t of:
        IDF(t) x  ( TF(t,d) x (k1 + 1) )
                  -----------------------------------------
                  ( TF(t,d) + k1 x (1 - b + b x |d| / avgdl) )

Two properties that make BM25 an improvement over raw TF-IDF for ranking:

  1. Term frequency saturation: raw TF-IDF's score grows linearly with how many
     times a term appears, so a document that repeats "memory" 50 times scores
     10x higher than one that mentions it 5 times, even though relevance
     doesn't actually keep growing that fast. BM25's `TF x (k1+1) / (TF + k1)`
     term is bounded above by (k1+1) as TF -> infinity, so additional
     repetitions beyond a handful add diminishing returns. k1 controls how
     quickly saturation kicks in (typical: 1.2-2.0; we use 1.5).

  2. Document length normalization: a long document naturally contains more
     term occurrences just by virtue of being long, not because it's more
     relevant. BM25 penalizes documents longer than the corpus average
     (|d| / avgdl > 1) and rewards shorter-than-average ones, controlled by b
     (0 = no length normalization, 1 = full normalization; we use the
     standard b=0.75).

IDF here uses the classic BM25 formulation (can go slightly negative for
terms appearing in more than half the corpus, which is intentional — such a
term is genuinely unhelpful for ranking and should not add positive score):

    IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

We use the "+1" variant (Robertson-Sparck Jones with a floor) which keeps IDF
non-negative, avoiding the (rarer, but possible) pathological case of a
negative-weighted term flipping a ranking in a confusing way.
"""
from __future__ import annotations

import math
from typing import Dict

from app.search.index import InvertedIndex
from app.search.tokenizer import preprocess

K1 = 1.5
B = 0.75


def bm25_idf(term: str, index: InvertedIndex) -> float:
    n = max(index.num_docs, 1)
    df = index.get_document_frequency(term)
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def score_bm25(
    query: str, index: InvertedIndex, *, stem: bool = True, k1: float = K1, b: float = B
) -> Dict[int, float]:
    query_terms = set(preprocess(query, stem=stem))
    if not query_terms:
        return {}

    avgdl = index.average_doc_length() or 1e-12

    candidates = set()
    for term in query_terms:
        candidates.update(index.get_posting_list(term).keys())

    scores: Dict[int, float] = {}
    for doc_id in candidates:
        doc_len = index.doc_lengths.get(doc_id, 0)
        score = 0.0
        for term in query_terms:
            posting = index.get_posting_list(term)
            if doc_id not in posting:
                continue
            tf = posting[doc_id][0]
            idf_val = bm25_idf(term, index)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += idf_val * (numerator / denominator)
        scores[doc_id] = score

    return scores
