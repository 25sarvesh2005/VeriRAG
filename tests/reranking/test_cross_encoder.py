"""Tests for Cross-Encoder reranker."""

from app.models import FusedCandidate
from app.reranking.cross_encoder import CrossEncoderReranker


def test_cross_encoder_rerank_promotes_relevant_document():
    reranker = CrossEncoderReranker()

    # Create two candidates:
    # First candidate is generic/irrelevant but artificially placed at rank 1
    # Second candidate is highly specific to the query but placed at rank 2
    candidates = [
        FusedCandidate(
            chunk_id="chunk_distractor",
            document_id="doc_distractor",
            text="The weather in July is generally warm and humid with occasional rainfall in many regions.",
            rrf_score=0.035,
            dense_rank=1,
            bm25_rank=1,
            source="weather.txt",
        ),
        FusedCandidate(
            chunk_id="chunk_target",
            document_id="doc_target",
            text="In Raft consensus, followers use an election timeout between 150ms and 300ms before starting an election.",
            rrf_score=0.015,
            dense_rank=2,
            bm25_rank=2,
            source="raft.txt",
        ),
    ]

    reranked = reranker.rerank(
        query="What is the election timeout duration in Raft consensus?",
        candidates=candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    # The target chunk should be promoted to rank 1 by the cross-encoder
    assert reranked[0].chunk_id == "chunk_target"
    assert reranked[0].rerank_score > reranked[1].rerank_score
    assert reranked[0].original_rank == 2


def test_cross_encoder_empty_candidates():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("Query", []) == []
