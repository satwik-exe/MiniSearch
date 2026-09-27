"""Unit tests for the evaluation metrics themselves (evaluate.py), as distinct
from testing the search engine's ranking quality. Run with `pytest
evaluation/test_evaluate.py` (if pytest is installed) or directly with
`python evaluation/test_evaluate.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import precision_at_k, recall_at_k, f1_score, reciprocal_rank


def test_precision_at_k_standard_case():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {2, 4}
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5


def test_precision_at_k_divides_by_k_not_by_results_returned():
    """Regression test for a real bug found during audit: precision_at_k used
    to divide by len(top_k) (however many results the engine actually
    returned) instead of by k itself. When a query returns fewer than K
    candidates, that silently inflates precision. Here only 3 results are
    returned for a requested K=5, with 2 hits: the buggy code would have
    returned 2/3 (~0.667); the correct value is 2/5 (0.4)."""
    retrieved_few = [1, 2, 3]  # fewer than K=5 results available
    relevant = {1, 2}
    result = precision_at_k(retrieved_few, relevant, 5)
    assert result == 2 / 5
    assert result != 2 / len(retrieved_few)


def test_precision_at_k_zero_hits():
    assert precision_at_k([1, 2, 3], {999}, 5) == 0.0


def test_precision_at_k_empty_retrieved_list():
    assert precision_at_k([], {1, 2}, 5) == 0.0


def test_recall_at_k_standard_case():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {2, 4, 10}  # 3 relevant docs exist total, 2 appear in top 5
    assert recall_at_k(retrieved, relevant, 5) == 2 / 3


def test_recall_at_k_no_relevant_docs_defined():
    assert recall_at_k([1, 2, 3], set(), 5) == 0.0


def test_recall_at_k_only_considers_top_k_window():
    retrieved = [1, 2, 3, 4, 5, 6, 7]
    relevant = {6, 7}  # both relevant docs are OUTSIDE the top 5
    assert recall_at_k(retrieved, relevant, 5) == 0.0


def test_f1_score_harmonic_mean():
    assert abs(f1_score(0.5, 0.5) - 0.5) < 1e-9
    assert abs(f1_score(1.0, 0.5) - (2 / 3)) < 1e-9


def test_f1_score_both_zero():
    assert f1_score(0.0, 0.0) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([5, 2, 1], {5}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([5, 2, 1], {1}) == 1 / 3


def test_reciprocal_rank_no_match():
    assert reciprocal_rank([1, 2, 3], {999}) == 0.0


if __name__ == "__main__":
    # Allow running without pytest, matching how the rest of this project's
    # test suite was verified in an environment without network access to
    # install pytest.
    import inspect

    tests = [
        (name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
