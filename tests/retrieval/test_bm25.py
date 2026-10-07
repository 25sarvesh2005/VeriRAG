"""Tests for BM25 Okapi lexical retriever."""

from pathlib import Path
from app.models import DocumentChunk
from app.retrieval.bm25 import BM25Retriever, tokenize_text


def test_tokenize_text():
    tokens = tokenize_text("RAG with BM25 Okapi and Reciprocal-Rank-Fusion!")
    assert "rag" in tokens
    assert "bm25" in tokens
    assert "okapi" in tokens
    assert "reciprocal-rank-fusion" in tokens or "reciprocal" in tokens


def test_bm25_retrieval(sample_chunks: list[DocumentChunk]):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)

    # Query matching Raft keywords exactly
    results = retriever.search("Raft election timeout leader", top_k=2)

    assert len(results) > 0
    assert results[0].chunk_id == "doc_raft_c0"
    assert results[0].score > 0.0
    assert results[0].retrieval_strategy == "bm25"


def test_bm25_persistence(sample_chunks: list[DocumentChunk], tmp_path: Path):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)
    retriever.save(tmp_path)

    loaded = BM25Retriever()
    loaded.load(tmp_path)

    results = loaded.search("MemTable SSTables compaction", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "doc_lsm_c0"
