"""Tests for evaluation metrics: retrieval, reranking, and citation verification."""

from app.evaluation.citation_metrics import (
    calculate_citation_coverage,
    calculate_citation_precision,
    calculate_unsupported_citation_rate,
)
from app.evaluation.retrieval_metrics import (
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from app.models import CitationVerification, Claim, VerificationVerdict


def test_retrieval_metrics():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    relevant = {"doc_b", "doc_d"}

    # Recall@1: doc_a is not in relevant -> 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=1) == 0.0

    # Recall@2: doc_b is in relevant (1 of 2) -> 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5

    # Precision@2: 1 relevant hit in top 2 -> 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5

    # Reciprocal Rank: first relevant doc is at rank 2 -> 1/2 = 0.5
    assert calculate_reciprocal_rank(retrieved, relevant) == 0.5

    # nDCG@2: DCG = 1 / log2(3) ≈ 0.6309, IDCG = 1 / log2(2) + 1 / log2(3) ≈ 1.6309 -> ndcg ≈ 0.3868
    ndcg = calculate_ndcg_at_k(retrieved, relevant, k=2)
    assert 0.38 <= ndcg <= 0.39


def test_citation_metrics():
    verifications = [
        CitationVerification(
            claim_id="1",
            claim_text="T1",
            citation_id="c1",
            source="s1",
            chunk_id="chk1",
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.9,
            evidence="E1",
            explanation="",
        ),
        CitationVerification(
            claim_id="2",
            claim_text="T2",
            citation_id="c2",
            source="s2",
            chunk_id="chk2",
            verdict=VerificationVerdict.UNSUPPORTED,
            confidence=0.8,
            evidence="E2",
            explanation="",
        ),
    ]

    prec = calculate_citation_precision(verifications)
    assert prec == 0.5

    unsup = calculate_unsupported_citation_rate(verifications)
    assert unsup == 0.5

    claims = [
        Claim(claim_id="1", text="C1", citation_markers=["[1]"]),
        Claim(claim_id="2", text="C2", citation_markers=[]),
    ]
    coverage = calculate_citation_coverage(claims, total_sentences=2)
    assert coverage == 0.5
