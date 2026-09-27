import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search.index import InvertedIndex
from app.search.bm25 import score_bm25


def test_term_frequency_saturation():
    idx = InvertedIndex()
    idx.add_document(1, " ".join(["memory"] * 5) + " padding " * 5, stem=False)
    idx.add_document(2, " ".join(["memory"] * 500) + " padding " * 5, stem=False)
    scores = score_bm25("memory", idx, stem=False)
    # 100x the raw term frequency should NOT translate into anywhere near 100x the score
    ratio = scores[2] / scores[1]
    assert ratio < 3, f"expected heavy saturation, got ratio {ratio}"


def test_length_normalization_favors_shorter_relevant_doc():
    idx = InvertedIndex()
    idx.add_document(1, "memory management memory management", stem=False)
    idx.add_document(2, "memory management memory management " + "filler " * 100, stem=False)
    idx.add_document(3, "irrelevant filler text with nothing relevant at all", stem=False)
    scores = score_bm25("memory", idx, stem=False)
    assert scores[1] > scores[2]


def test_empty_query_returns_no_results():
    idx = InvertedIndex()
    idx.add_document(1, "some content here", stem=False)
    assert score_bm25("", idx) == {}


def test_bm25_and_tfidf_agree_on_candidate_set(small_index):
    from app.search.tfidf import score_tfidf

    bm25_scores = score_bm25("memory management", small_index, stem=False)
    tfidf_scores = score_tfidf("memory management", small_index, stem=False)
    assert set(bm25_scores.keys()) == set(tfidf_scores.keys())


def test_k1_and_b_are_configurable():
    idx = InvertedIndex()
    idx.add_document(1, "memory " * 5 + "padding " * 5, stem=False)
    idx.add_document(2, "memory " * 2 + "filler " * 50, stem=False)

    default_scores = score_bm25("memory", idx, stem=False)
    # b=0 disables length normalization entirely -- the long doc2 should stop
    # being penalized for its length relative to the default b=0.75 behavior.
    no_length_norm_scores = score_bm25("memory", idx, stem=False, b=0.0)

    assert default_scores != no_length_norm_scores
    # Under b=0, doc2's score relative to doc1 should improve (length penalty removed)
    default_ratio = default_scores[2] / default_scores[1]
    no_norm_ratio = no_length_norm_scores[2] / no_length_norm_scores[1]
    assert no_norm_ratio > default_ratio
