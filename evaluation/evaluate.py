"""Evaluate MiniSearch's ranking quality against a hand-labeled query set.

Metrics implemented from scratch:

  Precision@K = (# relevant documents in top K results) / K
      "Of what I showed you, how much was actually relevant?"

  Recall@K = (# relevant documents in top K results) / (total # relevant documents)
      "Of everything relevant that exists, how much did I surface in the top K?"

  F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
      Harmonic mean of the two -- penalizes a system that only does well on one.

  MRR (Mean Reciprocal Rank) = mean over queries of 1 / rank_of_first_relevant_result
      Rewards putting a relevant result EARLY. A relevant doc at rank 1 contributes
      1.0; at rank 4 it contributes 0.25; if no relevant doc appears at all, that
      query contributes 0.

Usage:
    python evaluate.py [--k 5] [--corpus ../sample_documents] [--dataset dataset/queries.json]

Uses a fresh, in-memory-only SearchEngine (backed by a temporary SQLite file) so
running the evaluation never touches your real minisearch.db.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app.database.db as db_module  # noqa: E402


def load_corpus(engine, corpus_dir: Path) -> dict:
    """Ingest every .txt/.md file in corpus_dir, return {title: doc_id}."""
    from app.documents.ingestion import ingest

    title_to_id = {}
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        data = path.read_bytes()
        doc = ingest(path.name, data)
        row = engine.add_document(doc)
        title_to_id[row["title"]] = row["id"]
    return title_to_id


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """Precision@K = (# relevant in top K) / K -- divided by K itself, NOT by
    however many results the engine actually returned. An earlier version of
    this function divided by len(top_k), which silently inflates precision
    whenever a query returns fewer than K candidates (e.g. a rare query
    against a small corpus): a query with 1 hit out of 2 actual results would
    have scored precision=0.5 instead of the correct 1/K. The standard
    definition treats "K slots requested, fewer than K relevant-or-irrelevant
    results available to fill them" as still measured against the full K, not
    against a shrunken denominator -- this is the convention the project spec
    explicitly calls for."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(relevant_ids)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(engine, dataset: dict, title_to_id: dict, ranking: str, k: int) -> dict:
    per_query = []
    for q in dataset["queries"]:
        relevant_ids = {title_to_id[t] for t in q["relevant_titles"] if t in title_to_id}
        missing = [t for t in q["relevant_titles"] if t not in title_to_id]
        if missing:
            print(f"  WARNING: query '{q['query']}' references unknown titles: {missing}")

        result = engine.search(q["query"], ranking, top_k=max(k, 10), page=1)
        retrieved_ids = [r["document_id"] for r in result["results"]]

        p = precision_at_k(retrieved_ids, relevant_ids, k)
        r = recall_at_k(retrieved_ids, relevant_ids, k)
        f1 = f1_score(p, r)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        per_query.append(
            {"query": q["query"], "precision": p, "recall": r, "f1": f1, "reciprocal_rank": rr}
        )

    n = len(per_query) or 1
    return {
        "ranking": ranking,
        "k": k,
        "mean_precision": sum(x["precision"] for x in per_query) / n,
        "mean_recall": sum(x["recall"] for x in per_query) / n,
        "mean_f1": sum(x["f1"] for x in per_query) / n,
        "mrr": sum(x["reciprocal_rank"] for x in per_query) / n,
        "per_query": per_query,
    }


def print_report(result: dict) -> None:
    print(f"\n=== {result['ranking'].upper()} (K={result['k']}) ===")
    print(f"  Mean Precision@{result['k']}: {result['mean_precision']:.3f}")
    print(f"  Mean Recall@{result['k']}:    {result['mean_recall']:.3f}")
    print(f"  Mean F1@{result['k']}:        {result['mean_f1']:.3f}")
    print(f"  MRR:                    {result['mrr']:.3f}")
    print(f"\n  Per-query breakdown:")
    for pq in result["per_query"]:
        print(
            f"    '{pq['query']}': P={pq['precision']:.2f} R={pq['recall']:.2f} "
            f"F1={pq['f1']:.2f} RR={pq['reciprocal_rank']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniSearch ranking quality")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "sample_documents"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset" / "queries.json"
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    # Use a temp DB so this never touches the real application database.
    with tempfile.TemporaryDirectory() as tmpdir:
        db_module.DB_PATH = Path(tmpdir) / "eval.db"
        from app.search.engine import SearchEngine

        engine = SearchEngine()
        engine.startup()

        print(f"Loading corpus from {args.corpus} ...")
        title_to_id = load_corpus(engine, args.corpus)
        print(f"Indexed {len(title_to_id)} documents.\n")

        tfidf_result = evaluate_ranking(engine, dataset, title_to_id, "tfidf", args.k)
        bm25_result = evaluate_ranking(engine, dataset, title_to_id, "bm25", args.k)

        print_report(tfidf_result)
        print_report(bm25_result)

        print(f"\n=== TF-IDF vs BM25 (K={args.k}) ===")
        print(f"  {'Metric':<12}{'TF-IDF':>10}{'BM25':>10}")
        for metric in ("mean_precision", "mean_recall", "mean_f1", "mrr"):
            label = metric.replace("mean_", "").replace("_", " ")
            print(f"  {label:<12}{tfidf_result[metric]:>10.3f}{bm25_result[metric]:>10.3f}")

        with open(args.output, "w") as f:
            json.dump({"tfidf": tfidf_result, "bm25": bm25_result}, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
