"""Pytest deterministic fixtures and test utilities."""

from __future__ import annotations

import pytest

from app.config import AppConfig, IngestionConfig, RetrievalConfig
from app.generation.answer_generator import MockAnswerGenerator
from app.models import Document, DocumentChunk


@pytest.fixture
def sample_raw_documents() -> list[Document]:
    """Provide a deterministic fixture of sample documents."""
    return [
        Document(
            document_id="doc_raft",
            content=(
                "Raft is a distributed consensus algorithm. It divides consensus into leader election, "
                "log replication, and safety. Followers have an election timeout between 150ms and 300ms. "
                "A leader is elected when it receives votes from a majority of servers."
            ),
            source="raft.txt",
            title="Raft Consensus",
        ),
        Document(
            document_id="doc_lsm",
            content=(
                "Log-Structured Merge-Trees optimize write throughput. Incoming writes are appended to "
                "an in-memory MemTable and write-ahead log. When full, MemTable is flushed to SSTables. "
                "Compaction merges SSTables to remove deleted keys and reduce read amplification."
            ),
            source="lsm.txt",
            title="LSM Trees",
        ),
        Document(
            document_id="doc_rag",
            content=(
                "Retrieval-Augmented Generation enhances LLMs with external non-parametric knowledge. "
                "Dense retrieval captures semantic concepts via embeddings. BM25 performs lexical keyword search. "
                "Reciprocal Rank Fusion merges rankings without needing score calibration."
            ),
            source="rag.txt",
            title="RAG Systems",
        ),
    ]


@pytest.fixture
def sample_chunks(sample_raw_documents: list[Document]) -> list[DocumentChunk]:
    """Provide pre-chunked DocumentChunks for quick testing."""
    chunks = []
    for doc in sample_raw_documents:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{doc.document_id}_c0",
                document_id=doc.document_id,
                text=doc.content,
                source=doc.source,
                chunk_index=0,
                char_start=0,
                char_end=len(doc.content),
                metadata=doc.metadata,
            )
        )
    return chunks


@pytest.fixture
def mock_generator() -> MockAnswerGenerator:
    """Provide a deterministic mock answer generator."""
    return MockAnswerGenerator()


@pytest.fixture
def test_config() -> AppConfig:
    """Provide a test application configuration."""
    return AppConfig(
        ingestion=IngestionConfig(chunk_size_words=50, chunk_overlap_words=10),
        retrieval=RetrievalConfig(dense_top_k=3, bm25_top_k=3, fusion_top_k=3),
    )
