"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.config import AppConfig, DEFAULT_CONFIG
from app.evaluation.datasets import get_sample_documents
from app.ingestion.chunking import SlidingWindowChunker
from app.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration instance.

    Returns:
        Configured FastAPI application.
    """
    app_config = config or DEFAULT_CONFIG

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifecycle: Initialize models, pipeline, and seed corpus."""
        logging.basicConfig(
            level=getattr(logging, app_config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger.info("Starting VeriRAG API server...")

        # Initialize pipeline
        pipeline = RAGPipeline(config=app_config)

        # Ingest and index initial sample corpus
        sample_docs = get_sample_documents()
        chunker = SlidingWindowChunker(
            chunk_size_words=app_config.ingestion.chunk_size_words,
            chunk_overlap_words=app_config.ingestion.chunk_overlap_words,
        )

        all_chunks = []
        for doc in sample_docs:
            all_chunks.extend(chunker.chunk_document(doc))

        logger.info(
            "Seeding initial corpus: %d documents -> %d chunks...",
            len(sample_docs),
            len(all_chunks),
        )
        pipeline.retriever.index(all_chunks)

        # Attach to application state
        app.state.config = app_config
        app.state.pipeline = pipeline
        app.state.indexed_docs = sample_docs
        logger.info("VeriRAG API initialized and ready for queries.")

        yield

        logger.info("VeriRAG API shutting down.")

    app = FastAPI(
        title="VeriRAG API",
        version="0.1.0",
        description=(
            "Production-quality Hybrid RAG API with Dense Vector Retrieval, "
            "BM25 Retrieval, Cross-Encoder Reranking, and Citation Verification."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


app = create_app()
