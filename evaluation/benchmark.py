"""Benchmark MiniSearch's indexing time and search latency as corpus size grows.

Runs against 50, 100, and 500 SYNTHETIC documents (clearly labeled as such --
these are generated, not real corpus text) so results are reproducible without
needing a large real dataset checked into the repo.

Usage:
    python benchmark.py [--sizes 50,100,500] [--queries 20]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app.database.db as db_module  # noqa: E402

# A fixed vocabulary drawn from a few topic clusters, so synthetic documents have
# realistic term-frequency/document-frequency distributions (some common words,
# some rare ones) rather than pure random noise.
_TOPIC_WORDS = {
    "systems": [
        "operating", "system", "memory", "process", "scheduler", "kernel", "thread",
        "virtual", "paging", "deadlock", "synchronization", "interrupt", "filesystem",
    ],
    "networks": [
        "network", "protocol", "packet", "router", "tcp", "ip", "socket", "bandwidth",
        "latency", "ethernet", "routing", "firewall", "dns",
    ],
    "databases": [
        "database", "query", "index", "table", "join", "transaction", "schema",
        "sql", "normalization", "btree", "cache", "replication",
    ],
    "algorithms": [
        "algorithm", "complexity", "sorting", "searching", "graph", "tree", "hash",
        "recursion", "dynamic", "programming", "optimal", "greedy",
    ],
    "common": [
        "the", "system", "data", "using", "performance", "efficient", "example",
        "analysis", "design", "implementation", "structure", "function",
    ],
}


def generate_synthetic_document(rng: random.Random, doc_index: int) -> tuple[str, bytes]:
    topic = rng.choice(list(_TOPIC_WORDS.keys()))
    words = _TOPIC_WORDS[topic] + _TOPIC_WORDS["common"]
    length = rng.randint(80, 250)
    body = " ".join(rng.choice(words) for _ in range(length))
    text = f"Synthetic Document {doc_index} ({topic}). {body.capitalize()}."
    filename = f"synthetic-{doc_index:04d}-{topic}.txt"
    return filename, text.encode("utf-8")


def run_benchmark(size: int, num_queries: int, seed: int = 42) -> dict:
    from app.documents.ingestion import ingest
    from app.search.engine import SearchEngine

    rng = random.Random(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_module.DB_PATH = Path(tmpdir) / "bench.db"
        engine = SearchEngine()
        engine.startup()

        index_start = time.perf_counter()
        for i in range(size):
            fname, data = generate_synthetic_document(rng, i)
            doc = ingest(fname, data)
            engine.add_document(doc)
        index_time_s = time.perf_counter() - index_start

        # Generate benchmark queries from the same vocabulary
        query_words = [w for words in _TOPIC_WORDS.values() for w in words]
        latencies_ms = []
        for _ in range(num_queries):
            q = " ".join(rng.sample(query_words, k=rng.randint(1, 3)))
            ranking = rng.choice(["tfidf", "bm25"])
            result = engine.search(q, ranking, top_k=10, page=1)
            latencies_ms.append(result["search_time_ms"])

        stats = engine.stats()

        return {
            "documents": size,
            "index_time_s": round(index_time_s, 4),
            "avg_search_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
            "min_search_ms": round(min(latencies_ms), 3),
            "max_search_ms": round(max(latencies_ms), 3),
            "vocabulary_size": stats["unique_terms"],
            "total_tokens": stats["total_tokens"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniSearch indexing and search performance")
    parser.add_argument("--sizes", type=str, default="50,100,500")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "benchmark_results.json")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    results = []

    print("Running MiniSearch benchmark (documents are SYNTHETIC, generated for this run)...\n")
    for size in sizes:
        print(f"  Benchmarking {size} documents...")
        result = run_benchmark(size, args.queries)
        results.append(result)
        print(
            f"    Index time: {result['index_time_s']}s   "
            f"Avg search: {result['avg_search_ms']}ms   "
            f"Vocab: {result['vocabulary_size']} terms"
        )

    print("\n=== Benchmark Report ===")
    print(f"{'Documents':<12}{'Index Time':<14}{'Avg Search':<14}{'Vocab Size':<12}")
    for r in results:
        print(
            f"{r['documents']:<12}{r['index_time_s']}s{'':<8}"
            f"{r['avg_search_ms']}ms{'':<8}{r['vocabulary_size']:<12}"
        )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
