"""Pydantic schemas for API response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Information regarding a cited reference marker in the answer."""

    marker: str = Field(description="Bracket marker appearing in answer text, e.g. '[1]'.")
    source: str = Field(description="Originating document file name or URL.")
    document_id: str
    chunk_id: str
    evidence_text: str = Field(description="Underlying text snippet in the source chunk.")


class VerificationItem(BaseModel):
    """Detailed verification record for an individual factual claim."""

    claim: str = Field(description="Extracted factual claim sentence.")
    citation_id: str
    source: str
    verdict: str = Field(description="Verification verdict: SUPPORTED, UNSUPPORTED, or UNCERTAIN.")
    confidence: float = Field(description="Confidence score in [0.0, 1.0].")
    evidence: str = Field(description="Exact snippet extracted from source demonstrating verdict.")
    explanation: str = Field(description="Clear reasoning explaining why verdict was reached.")


class VerificationSummary(BaseModel):
    """Aggregate counts of verification outcomes."""

    supported: int = Field(default=0)
    unsupported: int = Field(default=0)
    uncertain: int = Field(default=0)


class QueryResponse(BaseModel):
    """Standardized response from the RAG query endpoint."""

    question: str
    answer: str
    citations: list[CitationItem]
    verifications: list[VerificationItem]
    verification: VerificationSummary
    retrieval_strategy_used: str
    timing_ms: dict[str, float]


class IngestResponse(BaseModel):
    """Response returned upon successfully ingesting and indexing a document."""

    document_id: str
    title: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """System health check and index status response."""

    status: str
    version: str
    documents_indexed: int
    chunks_indexed: int
    embedding_model: str
    reranker_model: str
