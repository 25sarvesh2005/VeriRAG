"""Evaluation package.

Provides retrieval metrics, citation verification metrics, benchmark datasets, and runner.
"""

from app.evaluation.citation_metrics import (
    calculate_citation_coverage,
    calculate_citation_precision,
    calculate_uncertainty_rate,
    calculate_unsupported_citation_rate,
)
from app.evaluation.datasets import (
    BENCHMARK_QUERIES,
    BenchmarkQuery,
    get_sample_documents,
)
from app.evaluation.retrieval_metrics import (
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.evaluation.runner import EvaluationRunner, ExperimentMetrics

__all__ = [
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_reciprocal_rank",
    "calculate_ndcg_at_k",
    "calculate_citation_precision",
    "calculate_unsupported_citation_rate",
    "calculate_uncertainty_rate",
    "calculate_citation_coverage",
    "BENCHMARK_QUERIES",
    "BenchmarkQuery",
    "get_sample_documents",
    "EvaluationRunner",
    "ExperimentMetrics",
]
