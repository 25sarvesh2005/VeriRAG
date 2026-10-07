"""Standard quantitative retrieval evaluation metrics.

Per AGENTS.md Section 22:
Measures Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG@K.
"""

from __future__ import annotations

import math
from typing import Sequence


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K: fraction of relevant documents successfully retrieved in top-k.

    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    """
    if not ground_truth_relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k.intersection(ground_truth_relevant_ids))
    return hits / len(ground_truth_relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K: fraction of retrieved top-k documents that are relevant.

    Precision@K = |Retrieved@K ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k.intersection(ground_truth_relevant_ids))
    return hits / k


def calculate_reciprocal_rank(
    retrieved_ids: Sequence[str],
    ground_truth_relevant_ids: set[str],
) -> float:
    """Calculate Reciprocal Rank (RR): reciprocal of the 1-based rank of the FIRST relevant document.

    RR = 1 / rank_first_relevant (or 0.0 if none retrieved)
    """
    if not ground_truth_relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_relevant_ids:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (nDCG@K).

    Using binary relevance:
    DCG@K = Σ (rel_i / log2(i + 1))  for i in 1..k
    IDCG@K = Ideal DCG with all relevant documents sorted first.
    nDCG@K = DCG@K / IDCG@K
    """
    if not ground_truth_relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in ground_truth_relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Compute Ideal DCG (all true positives ranked at the top)
    ideal_hits = min(len(ground_truth_relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
