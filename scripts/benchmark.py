"""Latency and throughput benchmarking script for VeriRAG components.

Usage:
    python scripts/benchmark.py [--iterations 10]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import DEFAULT_CONFIG
from app.evaluation.datasets import BENCHMARK_QUERIES, get_sample_documents
from app.ingestion.chunking import SlidingWindowChunker
from app.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def calculate_percentiles(values: list[float]) -> tuple[float, float, float]:
    """Calculate p50, p95, and p99 percentiles."""
    arr = np.array(values)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark component latency and throughput.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of repetitions per query.")
    args = parser.parse_args()

    pipeline = RAGPipeline(config=DEFAULT_CONFIG)

    # Ingest documents
    sample_docs = get_sample_documents()
    chunker = SlidingWindowChunker()
    all_chunks = []
    for doc in sample_docs:
        all_chunks.extend(chunker.chunk_document(doc))
    pipeline.retriever.index(all_chunks)

    test_queries = [bq.query for bq in BENCHMARK_QUERIES]
    logger.info("Running benchmarks across %d queries x %d iterations...", len(test_queries), args.iterations)

    dense_times = []
    bm25_times = []
    pipeline_times = []

    for _ in range(args.iterations):
        for q in test_queries:
            # BM25 standalone
            t0 = time.perf_counter()
            pipeline.retriever.retrieve_bm25(q, top_k=15)
            bm25_times.append((time.perf_counter() - t0) * 1000)

            # Dense standalone
            t0 = time.perf_counter()
            pipeline.retriever.retrieve_dense(q, top_k=15)
            dense_times.append((time.perf_counter() - t0) * 1000)

            # End to end
            t0 = time.perf_counter()
            pipeline.query(q, strategy="hybrid_rerank_verify")
            pipeline_times.append((time.perf_counter() - t0) * 1000)

    p50_bm25, p95_bm25, p99_bm25 = calculate_percentiles(bm25_times)
    p50_dense, p95_dense, p99_dense = calculate_percentiles(dense_times)
    p50_pipe, p95_pipe, p99_pipe = calculate_percentiles(pipeline_times)

    print("\n" + "=" * 65)
    print(" VeriRAG Latency Benchmark Results (ms)")
    print("=" * 65)
    print(f"{'Component':<28} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9}")
    print("-" * 65)
    print(f"{'BM25 Retrieval':<28} | {p50_bm25:<9.2f} | {p95_bm25:<9.2f} | {p99_bm25:<9.2f}")
    print(f"{'Dense Vector Retrieval':<28} | {p50_dense:<9.2f} | {p95_dense:<9.2f} | {p99_dense:<9.2f}")
    print(f"{'End-to-End Pipeline':<28} | {p50_pipe:<9.2f} | {p95_pipe:<9.2f} | {p99_pipe:<9.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
