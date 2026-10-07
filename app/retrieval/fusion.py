"""Candidate fusion using Reciprocal Rank Fusion (RRF).

Mathematical Formulation:
    RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]  for i in {dense, bm25}

Why Reciprocal Rank Fusion is chosen:
1. Incompatible Score Scales: Dense cosine similarity values lie strictly in [-1.0, 1.0]
   (or [0, 1] for typical embeddings), whereas BM25 scores are unbounded positive numbers
   proportional to term frequencies and corpus length.
2. Calibration Independence: Direct linear score combination (e.g. α·dense + (1-α)·bm25)
   fails unless scores are rigorously calibrated on a large training corpus.
3. Outlier Resistance: Ranking positions are invariant to non-linear score distributions.
   The smoothing constant `k` (commonly 60) dampens the impact of high top ranks while
   ensuring documents ranked highly in both modalities get the strongest boost.
"""

from __future__ import annotations

import logging
from app.models import FusedCandidate, RetrievedDocument

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[RetrievedDocument],
    bm25_results: list[RetrievedDocument],
    k: int = 60,
    top_k: int = 20,
) -> list[FusedCandidate]:
    """Fuse candidate rankings from dense vector and lexical BM25 retrievers.

    Args:
        dense_results: Ranked list of candidates from DenseRetriever.
        bm25_results: Ranked list of candidates from BM25Retriever.
        k: Smoothing constant. Default 60 as established by Cormack et al. (2009).
        top_k: Maximum number of fused candidates to return.

    Returns:
        List of FusedCandidate instances sorted in descending order of fused RRF score.
    """
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    doc_registry: dict[str, RetrievedDocument] = {}

    # Accumulate Dense RRF scores
    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc.chunk_id
        dense_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        if chunk_id not in doc_registry:
            doc_registry[chunk_id] = doc

    # Accumulate BM25 RRF scores
    for rank, doc in enumerate(bm25_results, start=1):
        chunk_id = doc.chunk_id
        bm25_ranks[chunk_id] = rank
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        if chunk_id not in doc_registry:
            doc_registry[chunk_id] = doc

    # Sort all candidates descending by fused score
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    fused_candidates: list[FusedCandidate] = []
    for chunk_id in sorted_chunk_ids[:top_k]:
        base_doc = doc_registry[chunk_id]
        fused_candidates.append(
            FusedCandidate(
                chunk_id=chunk_id,
                document_id=base_doc.document_id,
                text=base_doc.text,
                rrf_score=scores[chunk_id],
                dense_rank=dense_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
                source=base_doc.source,
                metadata=base_doc.metadata,
            )
        )

    logger.debug(
        "RRF Fusion completed: %d dense + %d bm25 -> %d unique fused candidates",
        len(dense_results),
        len(bm25_results),
        len(fused_candidates),
    )

    return fused_candidates
