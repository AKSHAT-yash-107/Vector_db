import time
from pathlib import Path

import numpy as np

from vectordb import VectorDB


# ============================================================
# CONFIG
# ============================================================

DATASET_SIZES = [1_000, 5_000, 10_000]

DIMENSION = 384
K = 10

NUM_QUERIES = 100

SEED = 42


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p):
    return float(np.percentile(values, p))


def build_dataset(n):
    rng = np.random.default_rng(SEED)

    vectors = rng.normal(
        size=(n, DIMENSION)
    ).astype(np.float32)

    documents = [
        f"benchmark document {i}"
        for i in range(n)
    ]

    metadata = [
        {
            "source": "benchmark",
            "id": i % 10
        }
        for i in range(n)
    ]

    return documents, vectors, metadata


# ============================================================
# INSERT BENCHMARK
# ============================================================

def benchmark_insert(n):

    documents, vectors, metadata = build_dataset(n)

    db = VectorDB()

    start = time.perf_counter()

    result = db.add_batch(
        documents,
        vectors,
        metadata
    )

    elapsed = time.perf_counter() - start

    throughput = result["added"] / elapsed

    return db, elapsed, throughput


# ============================================================
# SEARCH BENCHMARK
# ============================================================

def benchmark_search(db):

    rng = np.random.default_rng(SEED + 1)

    queries = rng.normal(
        size=(NUM_QUERIES, DIMENSION)
    ).astype(np.float32)

    latencies = []

    for query in queries:

        start = time.perf_counter()

        db.search(
            query,
            k=K
        )

        elapsed = time.perf_counter() - start

        latencies.append(
            elapsed * 1000
        )

    latencies = np.asarray(latencies)

    return {
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies))
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VectorDB Basic V1 Benchmark")
    print("=" * 60)

    print(f"Dimension: {DIMENSION}")
    print(f"Top-k:     {K}")
    print(f"Queries:   {NUM_QUERIES}")

    for n in DATASET_SIZES:

        print("\n" + "-" * 60)
        print(f"Dataset: {n:,} vectors")
        print("-" * 60)

        db, insert_time, throughput = benchmark_insert(n)

        print(
            f"Insert time:       {insert_time:.4f} sec"
        )

        print(
            f"Insert throughput: {throughput:.2f} vectors/sec"
        )

        print(
            f"HNSW count:        "
            f"{db.index.get_current_count()}"
        )

        search_stats = benchmark_search(db)

        print("\nSearch latency:")

        print(
            f"Average: {search_stats['avg_ms']:.4f} ms"
        )

        print(
            f"P50:     {search_stats['p50_ms']:.4f} ms"
        )

        print(
            f"P95:     {search_stats['p95_ms']:.4f} ms"
        )

        print(
            f"P99:     {search_stats['p99_ms']:.4f} ms"
        )

        print(
            f"Min:     {search_stats['min_ms']:.4f} ms"
        )

        print(
            f"Max:     {search_stats['max_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()