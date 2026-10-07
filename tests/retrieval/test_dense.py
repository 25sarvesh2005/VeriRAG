"""Tests for Dense vector retriever."""

from pathlib import Path
from app.models import DocumentChunk
from app.retrieval.dense import DenseRetriever


def test_dense_semantic_search(sample_chunks: list[DocumentChunk]):
    retriever = DenseRetriever()
    retriever.index(sample_chunks)

    # Query with semantic intent regarding external storage/memory enhancement
    results = retriever.search("external factual grounding for language models", top_k=2)

    assert len(results) > 0
    # The RAG chunk should score highest semantically
    assert results[0].chunk_id == "doc_rag_c0"
    assert results[0].retrieval_strategy == "dense"
    assert -1.0 <= results[0].score <= 1.0


def test_dense_persistence(sample_chunks: list[DocumentChunk], tmp_path: Path):
    retriever = DenseRetriever()
    retriever.index(sample_chunks)
    retriever.save(tmp_path)

    loaded = DenseRetriever()
    loaded.load(tmp_path)

    results = loaded.search("distributed consensus and leader heartbeats", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "doc_raft_c0"
