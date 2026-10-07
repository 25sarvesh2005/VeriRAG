"""Query execution endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from app.api.schemas.requests import QueryRequest
from app.api.schemas.responses import (
    CitationItem,
    QueryResponse,
    VerificationItem,
    VerificationSummary,
)

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Submit a factual query to the hybrid RAG pipeline with citation verification.

    Executes:
    1. Dense semantic search + BM25 lexical search
    2. Reciprocal Rank Fusion (RRF)
    3. Cross-Encoder reranking
    4. Answer generation with citation references [1], [2]
    5. Claim-level citation verification (SUPPORTED / UNSUPPORTED / UNCERTAIN)
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Please ensure the server has completed startup.",
        )

    # Execute pipeline
    rag_response = pipeline.query(question=payload.question, strategy=payload.strategy)

    citation_items = [
        CitationItem(
            marker=c.marker,
            source=c.source,
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            evidence_text=c.evidence_text[:300],
        )
        for c in rag_response.citations
    ]

    verification_items = [
        VerificationItem(
            claim=v.claim_text,
            citation_id=v.citation_id,
            source=v.source,
            verdict=v.verdict.value,
            confidence=v.confidence,
            evidence=v.evidence,
            explanation=v.explanation,
        )
        for v in rag_response.verifications
    ]

    summary_dict = rag_response.verification_summary
    verification_summary = VerificationSummary(
        supported=summary_dict.get("SUPPORTED", 0),
        unsupported=summary_dict.get("UNSUPPORTED", 0),
        uncertain=summary_dict.get("UNCERTAIN", 0),
    )

    return QueryResponse(
        question=rag_response.query,
        answer=rag_response.answer,
        citations=citation_items,
        verifications=verification_items,
        verification=verification_summary,
        retrieval_strategy_used=rag_response.retrieval_strategy_used,
        timing_ms=rag_response.timing_ms,
    )
