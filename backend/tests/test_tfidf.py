import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search.index import InvertedIndex
from app.search.tfidf import idf, score_tfidf, document_vector_norm


def test_idf_matches_hand_calculation(small_index):
    n = small_index.num_docs
    df = small_index.get_document_frequency("memory")
    expected = math.log((1 + n) / (1 + df)) + 1.0
    assert abs(idf("memory", small_index) - expected) < 1e-9


def test_rarer_term_has_higher_idf(small_index):
    assert idf("cat", small_index) > idf("memory", small_index)


def test_irrelevant_document_not_a_candidate(small_index):
    scores = score_tfidf("memory management", small_index, stem=False)
    assert 3 not in scores  # doc 3 ("the cat sat on the mat") has neither term


def test_docs_with_more_query_terms_outrank_docs_with_fewer(small_index):
    scores = score_tfidf("memory management", small_index, stem=False)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_two_ids = {ranked[0][0], ranked[1][0]}
    assert top_two_ids == {1, 4}  # both contain "memory" AND "management"
    assert ranked[2][0] == 2  # doc 2 only contains "memory"


def test_empty_query_returns_no_results(small_index):
    assert score_tfidf("", small_index) == {}


def test_unknown_term_query_returns_no_results(small_index):
    assert score_tfidf("zzznonexistent", small_index, stem=False) == {}


def test_stopword_only_query_returns_no_results(small_index):
    assert score_tfidf("the a an is", small_index) == {}


def test_matches_hand_calculated_cosine_similarity():
    """Fully worked, hand-calculable 3-document corpus verifying the corrected
    (full-document, not query-restricted) TF-IDF cosine similarity end to end.

    Corpus (stem=False so terms stay literal):
        doc1: "the cat sat on the mat"   -> {cat, sat, mat}   (after stopwords)
        doc2: "the dog sat on the log"   -> {dog, sat, log}
        doc3: "cats and dogs are pets"   -> {cats, dogs, pets}  (no overlap with query)
    Query: "cat sat"

    N=3. IDF(t) = ln((1+N)/(1+df(t))) + 1.
        df(sat)=2 (doc1,doc2) -> idf_sat = ln(4/3)+1
        df(cat)=df(mat)=df(dog)=df(log)=1 -> idf = ln(4/2)+1 = ln2+1

    doc1 has ALL THREE of cat, sat, mat (each tf=1), so its true vector norm
    includes "mat" even though the query never mentions it -- that's exactly
    the property an earlier, buggy version of this code got wrong (it would
    have ignored "mat" entirely and returned a perfect 1.0 for doc1).
    """
    idx = InvertedIndex()
    idx.add_document(1, "the cat sat on the mat", stem=False)
    idx.add_document(2, "the dog sat on the log", stem=False)
    idx.add_document(3, "cats and dogs are pets", stem=False)

    n = 3
    def hand_idf(df):
        return math.log((1 + n) / (1 + df)) + 1

    idf_sat, idf_cat, idf_mat = hand_idf(2), hand_idf(1), hand_idf(1)
    idf_dog, idf_log = hand_idf(1), hand_idf(1)

    q_norm = math.sqrt(idf_cat ** 2 + idf_sat ** 2)

    doc1_norm = math.sqrt(idf_cat ** 2 + idf_sat ** 2 + idf_mat ** 2)
    expected_doc1 = (idf_cat * idf_cat + idf_sat * idf_sat) / (q_norm * doc1_norm)

    doc2_norm = math.sqrt(idf_dog ** 2 + idf_sat ** 2 + idf_log ** 2)
    expected_doc2 = (idf_sat * idf_sat) / (q_norm * doc2_norm)  # "cat" absent from doc2

    scores = score_tfidf("cat sat", idx, stem=False)

    assert 3 not in scores  # doc3 shares zero terms with the query
    assert abs(scores[1] - expected_doc1) < 1e-9
    assert abs(scores[2] - expected_doc2) < 1e-9
    # The regression this guards against: under the old query-restricted-norm
    # bug, doc1's score would have come out as a perfect 1.0 (it ignores "mat"
    # entirely). The correct score must be strictly less than 1.0, because 1/3
    # of doc1's real content is irrelevant to the query.
    assert scores[1] < 0.99


def test_document_norm_uses_full_document_not_just_query_terms():
    """A document containing many terms the query never mentions must have a
    LARGER norm than one containing only the query's terms -- proof the norm
    is computed over the whole document vector, not restricted to query terms."""
    idx = InvertedIndex()
    idx.add_document(1, "memory management", stem=False)
    idx.add_document(2, "memory management unrelated extra padding content here", stem=False)

    norm_short = document_vector_norm(1, idx)
    norm_long = document_vector_norm(2, idx)
    assert norm_long > norm_short, (
        "doc2 has strictly more terms than doc1 (same query terms plus extra "
        "content), so its true TF-IDF norm must be larger -- if norm only "
        "counted query terms, these would come out equal"
    )
