"""VeriRAG: Hybrid-search RAG with citation verification."""

from app.config import AppConfig, DEFAULT_CONFIG
from app.models import (
    Citation,
    CitationVerification,
    Claim,
    Document,
    DocumentChunk,
    FusedCandidate,
    RAGResponse,
    RerankedDocument,
    RetrievedDocument,
    VerificationVerdict,
)
from app.pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG",
    "Document",
    "DocumentChunk",
    "RetrievedDocument",
    "FusedCandidate",
    "RerankedDocument",
    "Claim",
    "Citation",
    "VerificationVerdict",
    "CitationVerification",
    "RAGResponse",
    "RAGPipeline",
]
