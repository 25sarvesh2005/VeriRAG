"""Evaluation runner executing systematic ablation experiments.

Per AGENTS.md Section 23:
Demonstrates experimentally why each component exists:
1. Baseline: Dense Retrieval -> LLM
2. Experiment 1: BM25 -> LLM
3. Experiment 2: Dense + BM25 (Hybrid) -> LLM
4. Experiment 3: Dense + BM25 + Reranking -> LLM
5. Final: Dense + BM25 + Reranking + Citation Verification
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import statistics
import time
from typing import Sequence

from app.evaluation.citation_metrics import (
    calculate_citation_precision,
    calculate_unsupported_citation_rate,
)
from app.evaluation.datasets import BENCHMARK_QUERIES, BenchmarkQuery
from app.evaluation.retrieval_metrics import (
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.pipeline import PipelineStrategy, RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Aggregated evaluation metrics for an experimental condition."""

    strategy_name: str
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    ndcg_at_3: float = 0.0
    citation_precision: float = 0.0
    unsupported_citation_rate: float = 0.0
    avg_latency_ms: float = 0.0


class EvaluationRunner:
    """Executes benchmark queries against RAG pipeline strategies and compiles metrics."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def run_strategy(
        self,
        strategy: PipelineStrategy,
        queries: Sequence[BenchmarkQuery] = BENCHMARK_QUERIES,
    ) -> ExperimentMetrics:
        """Run all benchmark queries through a specific pipeline strategy."""
        mrrs: list[float] = []
        recalls_1: list[float] = []
        recalls_3: list[float] = []
        precisions_1: list[float] = []
        precisions_3: list[float] = []
        ndcgs_3: list[float] = []
        latencies: list[float] = []
        all_verifications = []

        for bq in queries:
            t0 = time.perf_counter()
            response = self.pipeline.query(bq.query, strategy=strategy)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Map retrieved chunks back to their parent document IDs
            retrieved_doc_ids = [doc.document_id for doc in response.context_documents]

            recalls_1.append(calculate_recall_at_k(retrieved_doc_ids, bq.target_document_ids, k=1))
            recalls_3.append(calculate_recall_at_k(retrieved_doc_ids, bq.target_document_ids, k=3))
            precisions_1.append(calculate_precision_at_k(retrieved_doc_ids, bq.target_document_ids, k=1))
            precisions_3.append(calculate_precision_at_k(retrieved_doc_ids, bq.target_document_ids, k=3))
            mrrs.append(calculate_reciprocal_rank(retrieved_doc_ids, bq.target_document_ids))
            ndcgs_3.append(calculate_ndcg_at_k(retrieved_doc_ids, bq.target_document_ids, k=3))

            all_verifications.extend(response.verifications)

        cit_prec = calculate_citation_precision(all_verifications) if all_verifications else 0.0
        unsup_rate = calculate_unsupported_citation_rate(all_verifications) if all_verifications else 0.0

        return ExperimentMetrics(
            strategy_name=strategy,
            mrr=round(statistics.mean(mrrs), 4),
            recall_at_1=round(statistics.mean(recalls_1), 4),
            recall_at_3=round(statistics.mean(recalls_3), 4),
            precision_at_1=round(statistics.mean(precisions_1), 4),
            precision_at_3=round(statistics.mean(precisions_3), 4),
            ndcg_at_3=round(statistics.mean(ndcgs_3), 4),
            citation_precision=round(cit_prec, 4),
            unsupported_citation_rate=round(unsup_rate, 4),
            avg_latency_ms=round(statistics.mean(latencies), 2),
        )

    def run_full_benchmark(
        self,
        queries: Sequence[BenchmarkQuery] = BENCHMARK_QUERIES,
    ) -> list[ExperimentMetrics]:
        """Run all 5 baseline experiments prescribed in AGENTS.md Section 23."""
        strategies: list[PipelineStrategy] = [
            "dense_only",
            "bm25_only",
            "hybrid_only",
            "hybrid_rerank",
            "hybrid_rerank_verify",
        ]

        results = []
        for strat in strategies:
            logger.info("Executing benchmark for strategy: %s ...", strat)
            results.append(self.run_strategy(strat, queries=queries))

        return results

    @staticmethod
    def format_markdown_table(results: list[ExperimentMetrics]) -> str:
        """Format benchmark comparison results as a GitHub-flavored Markdown table."""
        header = (
            "| Configuration | MRR | Recall@1 | Recall@3 | Precision@1 | nDCG@3 | Citation Prec | Unsupported Rate | Avg Latency (ms) |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        rows = []
        labels = {
            "dense_only": "1. Baseline (Dense only)",
            "bm25_only": "2. Exp 1 (BM25 only)",
            "hybrid_only": "3. Exp 2 (Dense + BM25 Hybrid)",
            "hybrid_rerank": "4. Exp 3 (Hybrid + Cross-Encoder)",
            "hybrid_rerank_verify": "5. Final (Hybrid + Rerank + Verifier)",
        }
        for r in results:
            label = labels.get(r.strategy_name, r.strategy_name)
            cit_prec_str = f"{r.citation_precision:.1%}" if r.citation_precision > 0 else "N/A"
            unsup_str = f"{r.unsupported_citation_rate:.1%}" if r.citation_precision > 0 else "N/A"
            rows.append(
                f"| **{label}** | {r.mrr:.3f} | {r.recall_at_1:.3f} | {r.recall_at_3:.3f} | "
                f"{r.precision_at_1:.3f} | {r.ndcg_at_3:.3f} | {cit_prec_str} | {unsup_str} | {r.avg_latency_ms:.1f} |"
            )
        return header + "\n".join(rows) + "\n"
