"""Tests for Reciprocal Rank Fusion (RRF)."""

from app.models import RetrievedDocument
from app.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_scoring():
    # Construct mock dense and bm25 results
    doc_a = RetrievedDocument(
        chunk_id="chunk_a",
        document_id="doc_a",
        text="Content A",
        score=0.9,
        source="src_a",
        retrieval_strategy="dense",
        rank=1,
    )
    doc_b = RetrievedDocument(
        chunk_id="chunk_b",
        document_id="doc_b",
        text="Content B",
        score=0.8,
        source="src_b",
        retrieval_strategy="dense",
        rank=2,
    )

    # In BM25, chunk_b is ranked #1 and chunk_c is ranked #2
    doc_b_bm25 = RetrievedDocument(
        chunk_id="chunk_b",
        document_id="doc_b",
        text="Content B",
        score=5.5,
        source="src_b",
        retrieval_strategy="bm25",
        rank=1,
    )
    doc_c_bm25 = RetrievedDocument(
        chunk_id="chunk_c",
        document_id="doc_c",
        text="Content C",
        score=4.2,
        source="src_c",
        retrieval_strategy="bm25",
        rank=2,
    )

    fused = reciprocal_rank_fusion(
        dense_results=[doc_a, doc_b],
        bm25_results=[doc_b_bm25, doc_c_bm25],
        k=60,
        top_k=5,
    )

    assert len(fused) == 3
    # chunk_b appeared in both (rank 2 in dense, rank 1 in bm25)
    # Score for b: (1/(60+2)) + (1/(60+1)) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.0325
    # Score for a: 1/61 = ~0.01639
    # Score for c: 1/62 = ~0.01612
    # Therefore, chunk_b should be rank 1 overall
    assert fused[0].chunk_id == "chunk_b"
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].rrf_score > fused[1].rrf_score
