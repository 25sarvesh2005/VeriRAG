"""Tests for HybridRetriever combining dense and BM25."""

from app.models import DocumentChunk
from app.retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_indexing_and_search(sample_chunks: list[DocumentChunk]):
    retriever = HybridRetriever()
    retriever.index(sample_chunks)

    fused = retriever.retrieve_hybrid("What is the election timeout in Raft?")
    assert len(fused) > 0
    assert fused[0].chunk_id == "doc_raft_c0"
    assert fused[0].dense_rank is not None or fused[0].bm25_rank is not None
