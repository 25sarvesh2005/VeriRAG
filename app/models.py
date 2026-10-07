"""Core domain data models for VeriRAG.

Following AGENTS.md Section 11 & 12:
Important concepts have explicit typed models rather than arbitrary dictionaries.
Every stage in the pipeline accepts and emits these well-defined structures.
Zero circular dependencies: this module only depends on standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents an ingested raw document prior to chunking."""

    document_id: str
    content: str
    source: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a discrete indexed text segment derived from a parent document."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    """Represents a document chunk retrieved by either dense or lexical search."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    retrieval_strategy: str  # "dense" or "bm25"
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedCandidate:
    """Represents a candidate after reciprocal rank fusion between dense and BM25."""

    chunk_id: str
    document_id: str
    text: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedDocument:
    """Represents a candidate scored by the cross-encoder reranker."""

    chunk_id: str
    document_id: str
    text: str
    rerank_score: float
    original_rank: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieved(cls, doc: RetrievedDocument) -> RerankedDocument:
        """Construct context document directly from a single-modality retrieval result."""
        return cls(
            chunk_id=doc.chunk_id,
            document_id=doc.document_id,
            text=doc.text,
            rerank_score=doc.score,
            original_rank=doc.rank,
            source=doc.source,
            metadata=doc.metadata,
        )

    @classmethod
    def from_fused(cls, candidate: FusedCandidate, rank: int) -> RerankedDocument:
        """Construct context document directly from a fused candidate."""
        return cls(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            text=candidate.text,
            rerank_score=candidate.rrf_score,
            original_rank=rank,
            source=candidate.source,
            metadata=candidate.metadata,
        )


class VerificationVerdict(str, Enum):
    """Explicit verdict states for citation support.

    Per Section 18: We never collapse verification into a simple boolean.
    - SUPPORTED: Referenced source contains direct or logically entailed evidence.
    - UNSUPPORTED: Referenced source contradicts the claim or provides zero backing.
    - UNCERTAIN: Evidence is weakly related, ambiguous, or lacks conclusive confirmation.
    """

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class Citation:
    """Represents a cited reference marker within the generated text."""

    citation_id: str
    marker: str  # e.g., "[1]"
    document_id: str
    chunk_id: str
    source: str
    evidence_text: str
    char_offset: int = -1


@dataclass(frozen=True)
class Claim:
    """Represents an atomic factual statement extracted from generated text."""

    claim_id: str
    text: str
    citation_markers: list[str]  # e.g., ["[1]"]
    sentence_index: int = 0


@dataclass(frozen=True)
class CitationVerification:
    """Evaluation result verifying whether a cited chunk supports a factual claim."""

    claim_id: str
    claim_text: str
    citation_id: str
    source: str
    chunk_id: str
    verdict: VerificationVerdict
    confidence: float
    evidence: str
    explanation: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Represents the raw response produced by the generation layer."""

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RAGResponse:
    """Final, comprehensive pipeline response."""

    query: str
    answer: str
    citations: list[Citation]
    verifications: list[CitationVerification]
    context_documents: list[RerankedDocument]
    retrieval_strategy_used: str
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def verification_summary(self) -> dict[str, int]:
        """Convenience breakdown of verification results."""
        summary = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNCERTAIN": 0}
        for v in self.verifications:
            summary[v.verdict.value] += 1
        return summary
