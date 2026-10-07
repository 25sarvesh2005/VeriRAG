"""Pydantic schemas for API request payloads."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for submitting a search question to the RAG pipeline."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user search question or factual inquiry.",
        examples=["How does Reciprocal Rank Fusion combine dense and BM25 scores?"],
    )
    strategy: Literal[
        "hybrid_rerank_verify",
        "hybrid_rerank",
        "hybrid_only",
        "dense_only",
        "bm25_only",
    ] = Field(
        default="hybrid_rerank_verify",
        description="Pipeline execution strategy (supports ablation experiments).",
    )


class IngestTextRequest(BaseModel):
    """Payload for directly ingesting raw document text into the index."""

    content: str = Field(
        ...,
        min_length=10,
        description="Raw document text to be cleaned, chunked, and indexed.",
    )
    title: str = Field(default="Untitled Document", description="Descriptive title.")
    source: str = Field(default="api_upload", description="Source identifier.")
