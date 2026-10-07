"""Health and readiness check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from app.api.schemas.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Check system health, index status, and loaded models."""
    pipeline = getattr(request.app.state, "pipeline", None)

    chunks_count = 0
    if pipeline and pipeline.retriever and pipeline.retriever.bm25:
        chunks_count = len(pipeline.retriever.bm25.chunks)

    config = getattr(request.app.state, "config", None)
    emb_model = config.retrieval.embedding_model_name if config else "unknown"
    rerank_model = config.reranking.reranker_model_name if config else "unknown"

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        documents_indexed=len(getattr(request.app.state, "indexed_docs", [])),
        chunks_indexed=chunks_count,
        embedding_model=emb_model,
        reranker_model=rerank_model,
    )
