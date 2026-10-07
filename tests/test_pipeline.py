"""End-to-end integration tests for the RAGPipeline across all baseline strategies."""

from app.models import DocumentChunk
from app.pipeline import RAGPipeline


def test_pipeline_hybrid_rerank_verify_end_to_end(sample_chunks: list[DocumentChunk]):
    pipeline = RAGPipeline()
    pipeline.retriever.index(sample_chunks)

    response = pipeline.query(
        "How does Raft elect a leader and what is the timeout duration?",
        strategy="hybrid_rerank_verify",
    )

    assert response.query is not None
    assert len(response.answer) > 0
    assert len(response.context_documents) > 0
    assert response.retrieval_strategy_used == "hybrid_rerank_verify"
    assert "total_pipeline_ms" in response.timing_ms

    # Verify citation extraction & verification occurred
    assert len(response.citations) > 0
    assert len(response.verifications) > 0
    summary = response.verification_summary
    assert (summary["SUPPORTED"] + summary["UNSUPPORTED"] + summary["UNCERTAIN"]) == len(response.verifications)


def test_pipeline_ablation_strategies(sample_chunks: list[DocumentChunk]):
    pipeline = RAGPipeline()
    pipeline.retriever.index(sample_chunks)

    # Dense only
    resp_dense = pipeline.query("Raft consensus leader", strategy="dense_only")
    assert resp_dense.retrieval_strategy_used == "dense_only"
    assert len(resp_dense.context_documents) > 0

    # BM25 only
    resp_bm25 = pipeline.query("MemTable SSTable compaction", strategy="bm25_only")
    assert resp_bm25.retrieval_strategy_used == "bm25_only"
    assert len(resp_bm25.context_documents) > 0

    # Hybrid only (no cross-encoder)
    resp_hybrid = pipeline.query("Dense and BM25 RRF fusion", strategy="hybrid_only")
    assert resp_hybrid.retrieval_strategy_used == "hybrid_only"
    assert len(resp_hybrid.context_documents) > 0
