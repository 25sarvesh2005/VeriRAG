"""Centralized configuration for VeriRAG.

Every tunable parameter in the system is declared here with a clear default,
environment variable override, and explanatory docstring. No magic numbers
should appear across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load configuration values from local .env file if present
load_dotenv()


@dataclass(frozen=True)
class IngestionConfig:
    """Settings controlling document parsing, cleaning, and chunking."""

    chunk_size_words: int = int(os.getenv("VERIRAG_CHUNK_SIZE_WORDS", "180"))
    chunk_overlap_words: int = int(os.getenv("VERIRAG_CHUNK_OVERLAP_WORDS", "35"))
    min_chunk_length_chars: int = 40
    strip_extra_whitespace: bool = True
    normalize_bullet_points: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Settings controlling dense embedding retrieval, BM25 retrieval, and fusion."""

    # Dense retrieval settings
    dense_top_k: int = int(os.getenv("VERIRAG_DENSE_TOP_K", "15"))
    embedding_model_name: str = os.getenv("VERIRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_device: str = os.getenv("VERIRAG_EMBEDDING_DEVICE", "cpu")

    # BM25 retrieval settings
    bm25_top_k: int = int(os.getenv("VERIRAG_BM25_TOP_K", "15"))
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Reciprocal Rank Fusion (RRF) smoothing parameter
    # A standard constant k=60 prevents high-ranking items from dominating excessively
    rrf_k: int = int(os.getenv("VERIRAG_RRF_K", "60"))
    fusion_top_k: int = 20


@dataclass(frozen=True)
class RerankingConfig:
    """Settings controlling cross-encoder reranking."""

    reranker_model_name: str = os.getenv(
        "VERIRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_k: int = int(os.getenv("VERIRAG_RERANK_TOP_K", "5"))
    reranker_device: str = os.getenv("VERIRAG_RERANKER_DEVICE", "cpu")
    batch_size: int = 16


@dataclass(frozen=True)
class GenerationConfig:
    """Settings controlling LLM answer generation."""

    provider: Literal["mock", "openai", "gemini", "ollama"] = os.getenv(
        "VERIRAG_LLM_PROVIDER", "mock"
    )  # type: ignore
    model_name: str = os.getenv("VERIRAG_LLM_MODEL", "gpt-4o-mini")
    temperature: float = 0.1
    max_tokens: int = 512
    openai_api_key: str = os.getenv("VERIRAG_OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("VERIRAG_GEMINI_API_KEY", "")
    ollama_base_url: str = os.getenv("VERIRAG_OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass(frozen=True)
class CitationConfig:
    """Settings controlling citation extraction and verification."""

    # Minimum entailment/similarity confidence to classify a claim as SUPPORTED
    min_verification_confidence: float = float(
        os.getenv("VERIRAG_MIN_VERIFICATION_CONFIDENCE", "0.65")
    )
    # Threshold below which claim is flagged UNSUPPORTED; between the two is UNCERTAIN
    uncertainty_threshold: float = 0.40
    # Cross-encoder or NLI model for strict verification
    verifier_model_name: str = os.getenv(
        "VERIRAG_VERIFIER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    max_evidence_chars: int = 400


@dataclass(frozen=True)
class StorageConfig:
    """File and directory paths for index persistence and raw data."""

    base_dir: Path = Path(os.getenv("VERIRAG_BASE_DIR", Path(__file__).resolve().parent.parent))
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    indices_dir: Path = data_dir / "indices"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration containing all component configs."""

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    host: str = os.getenv("VERIRAG_HOST", "0.0.0.0")
    port: int = int(os.getenv("VERIRAG_PORT", "8000"))
    log_level: str = os.getenv("VERIRAG_LOG_LEVEL", "INFO")


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
