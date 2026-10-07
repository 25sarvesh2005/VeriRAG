"""Evaluation datasets and benchmark query definitions.

Provides a rich corpus of real-world technical documents across database
indexing, distributed consensus (Raft/Paxos), retrieval-augmented generation,
and neural architectures, along with annotated queries and ground-truth chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from app.models import Document


@dataclass(frozen=True)
class BenchmarkQuery:
    """An annotated benchmark question with ground-truth relevance annotations."""

    query_id: str
    query: str
    target_document_ids: set[str]
    expected_key_facts: list[str]


SAMPLE_CORPUS_DOCUMENTS: list[dict[str, Any]] = [
    {
        "document_id": "doc_rag_intro",
        "title": "Retrieval-Augmented Generation Architecture",
        "source": "rag_architecture.md",
        "content": (
            "Retrieval-Augmented Generation (RAG) blends parametric memory from large language models "
            "with non-parametric external knowledge retrieval. Dense retrieval utilizes continuous dense vector "
            "embeddings produced by bi-encoders like MiniLM to capture semantic relationships. Lexical search using "
            "BM25 Okapi matches exact surface forms and rare terms like acronyms and specific IDs. "
            "Reciprocal Rank Fusion (RRF) combines rankings from dense and lexical retrievers without requiring "
            "score calibration by evaluating rank reciprocals: RRF(d) = sum(1 / (k + rank(d)))."
        ),
    },
    {
        "document_id": "doc_cross_encoders",
        "title": "Cross-Encoder Reranking Mechanisms",
        "source": "reranking_overview.md",
        "content": (
            "Cross-encoders jointly feed the search query and the candidate passage into a transformer model "
            "simultaneously via cross-attention layers. Unlike bi-encoders, which compress documents into fixed-length "
            "isolated vectors, cross-encoders model direct inter-token attention between every word in the query and "
            "every word in the passage. Consequently, cross-encoders achieve significantly higher precision and MRR, "
            "making them ideal for reranking candidate pools of 20 to 50 items down to the top 5 passages."
        ),
    },
    {
        "document_id": "doc_raft_consensus",
        "title": "The Raft Distributed Consensus Algorithm",
        "source": "raft_consensus.md",
        "content": (
            "Raft is a consensus algorithm designed for understandability, dividing the consensus problem into "
            "leader election, log replication, and safety. A cluster consists of nodes in one of three states: "
            "Leader, Follower, or Candidate. If a follower receives no heartbeat within a randomized election timeout "
            "between 150ms and 300ms, it transitions to candidate state and starts an election. Log entries are committed "
            "only after a majority of cluster nodes have replicated them to durable disk."
        ),
    },
    {
        "document_id": "doc_lsm_trees",
        "title": "Log-Structured Merge-Trees in Modern Databases",
        "source": "lsm_storage.md",
        "content": (
            "Log-Structured Merge-Trees (LSM-trees) are storage engines optimized for high write throughput. "
            "Writes are initially appended sequentially to an in-memory MemTable and a write-ahead log (WAL). "
            "When the MemTable reaches its capacity, it is flushed to disk as an immutable Sorted String Table (SSTable) "
            "in Level 0. Background compaction processes merge overlapping SSTables across levels to eliminate deleted "
            "or overwritten keys and reduce read amplification. Bloom filters are used to avoid searching SSTables "
            "that do not contain the target key."
        ),
    },
    {
        "document_id": "doc_citation_verification",
        "title": "Automated Citation Verification in LLM Systems",
        "source": "citation_verification.md",
        "content": (
            "Automated citation verification acts as a quality-control barrier to eliminate hallucinated references. "
            "The verification pipeline decomposes generated text into individual factual claims, isolates cited bracket "
            "markers, and cross-references them with the source chunk. Entailment checks determine if the premise "
            "substantiates the claim, classifying outputs into SUPPORTED, UNSUPPORTED, or UNCERTAIN. This process "
            "detects numerical discrepancies, unwarranted causal claims, and polarity reversals."
        ),
    },
]

BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query_id="q01",
        query="How does Reciprocal Rank Fusion combine dense and BM25 search rankings?",
        target_document_ids={"doc_rag_intro"},
        expected_key_facts=["RRF", "reciprocal", "dense and lexical", "calibration"],
    ),
    BenchmarkQuery(
        query_id="q02",
        query="Why do cross-encoders achieve higher precision than bi-encoders for reranking?",
        target_document_ids={"doc_cross_encoders"},
        expected_key_facts=["cross-attention", "inter-token attention", "bi-encoders"],
    ),
    BenchmarkQuery(
        query_id="q03",
        query="What is the election timeout duration in Raft and when does a follower become a candidate?",
        target_document_ids={"doc_raft_consensus"},
        expected_key_facts=["150ms and 300ms", "heartbeat", "candidate"],
    ),
    BenchmarkQuery(
        query_id="q04",
        query="How do LSM-trees handle writes using MemTable and SSTables, and how is read amplification reduced?",
        target_document_ids={"doc_lsm_trees"},
        expected_key_facts=["MemTable", "SSTable", "compaction", "Bloom filters"],
    ),
    BenchmarkQuery(
        query_id="q05",
        query="What states does citation verification classify claims into and what errors does it detect?",
        target_document_ids={"doc_citation_verification"},
        expected_key_facts=["SUPPORTED", "UNSUPPORTED", "UNCERTAIN", "numerical discrepancies"],
    ),
]


def get_sample_documents() -> list[Document]:
    """Instantiate Document objects for the built-in sample corpus."""
    return [
        Document(
            document_id=item["document_id"],
            content=item["content"],
            source=item["source"],
            title=item["title"],
            metadata={"domain": "computer_science"},
        )
        for item in SAMPLE_CORPUS_DOCUMENTS
    ]
