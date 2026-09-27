"""TF-IDF ranking, implemented from scratch (no search-engine library).

Term frequency:
    TF(t, d) = count of term t in document d      (raw count; see note below)

Inverse document frequency, with +1 smoothing to avoid division-by-zero /
log(0) for terms that (in edge cases) appear in every document, and to avoid
returning a negative weight for very common terms:

    IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1

where N is the total number of documents and df(t) is the number of documents
containing t. This is the "smooth IDF" variant used by, e.g., scikit-learn's
TfidfVectorizer — it keeps IDF strictly positive and well-defined even when
df(t) == N.

Document/query score:
    TF-IDF(t, d) = TF(t, d) x IDF(t)

We then score a query against a document via cosine similarity between their
TF-IDF vectors:

    cos(q, d) = (q . d) / (||q|| x ||d||)

Cosine similarity is preferred over a raw dot product because it normalizes
for document length: without it, a long document that happens to repeat a
query term many times would always outscore a short, highly relevant one.

IMPORTANT — document vector norm: ||d|| is computed over the document's FULL
term vector (every term in the document, via InvertedIndex.get_document_terms),
not just the terms the query happens to mention. An earlier version of this
file computed ||d|| restricted to query terms, which is not standard cosine
similarity: it silently discards every dimension the query doesn't touch,
which inflates the score of documents whose non-query content is irrelevant.
Concretely, for a document containing ONLY the query's terms in exactly the
query's proportions, the old code always returned a perfect 1.0 regardless of
how much *other*, irrelevant content the real document also contained (it
never saw that content, because it never looked outside query-term
dimensions). See backend/tests/test_tfidf.py::test_matches_hand_calculated_cosine_similarity
for a fully worked 3-document example showing the corrected score (~0.78) vs.
what the old, wrong code would have returned (1.0) for the same query/document
pair.

The dot product q . d is still safely computed by only summing over query
terms — that's mathematically exact (a document's weight on a term the query
doesn't mention contributes q_i x d_i = 0 x d_i = 0 to the dot product
regardless), so no change was needed there.
"""
from __future__ import annotations

import math
from typing import Dict

from app.search.index import InvertedIndex
from app.search.tokenizer import preprocess


def idf(term: str, index: InvertedIndex) -> float:
    n = max(index.num_docs, 1)
    df = index.get_document_frequency(term)
    return math.log((1 + n) / (1 + df)) + 1.0


def document_vector_norm(doc_id: int, index: InvertedIndex) -> float:
    """||d||, the TRUE TF-IDF vector norm of a document: every term in the
    document contributes, not just the terms a particular query happens to
    mention. Uses the forward index (InvertedIndex.get_document_terms), so
    this is O(terms in this one document), not O(vocabulary size) -- we are
    NOT rescanning the whole corpus to compute this."""
    total = 0.0
    for term, tf in index.get_document_terms(doc_id).items():
        weight = tf * idf(term, index)
        total += weight * weight
    return math.sqrt(total)


def score_tfidf(query: str, index: InvertedIndex, *, stem: bool = True) -> Dict[int, float]:
    """Return {doc_id: cosine_similarity_score} for all documents containing
    at least one query term."""
    query_terms = preprocess(query, stem=stem)
    if not query_terms:
        return {}

    # Query term frequencies (a query is treated as a tiny "document")
    query_tf: Dict[str, int] = {}
    for t in query_terms:
        query_tf[t] = query_tf.get(t, 0) + 1

    query_weights = {t: tf * idf(t, index) for t, tf in query_tf.items()}
    query_norm = math.sqrt(sum(w * w for w in query_weights.values())) or 1e-12

    # Candidate documents = union of posting lists for query terms. This is
    # the inverted-index payoff: we never touch a document that shares zero
    # terms with the query.
    candidates = set()
    for term in query_tf:
        candidates.update(index.get_posting_list(term).keys())

    scores: Dict[int, float] = {}
    for doc_id in candidates:
        dot = 0.0
        for term, q_weight in query_weights.items():
            posting = index.get_posting_list(term)
            if doc_id in posting:
                tf = posting[doc_id][0]
                d_weight = tf * idf(term, index)
                dot += q_weight * d_weight
        doc_norm = document_vector_norm(doc_id, index) or 1e-12
        scores[doc_id] = dot / (query_norm * doc_norm)

    return scores

