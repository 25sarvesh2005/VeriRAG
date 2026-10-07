"""Hybrid retrieval orchestrator combining dense semantic search and lexical BM25 search.

Per AGENTS.md Section 13:
The retrieval layer makes both strategies explicit and observable:
- Dense retrieval answers: 'What content is semantically similar?'
- BM25 answers: 'What content contains important lexical terms?'
"""

from __future__ import annotations

import logging

from app.models import DocumentChunk, FusedCandidate, RetrievedDocument
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines Dense vector retrieval and BM25 lexical retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        """Initialize hybrid retriever with explicit sub-retrievers.

        Args:
            dense_retriever: Configured DenseRetriever instance.
            bm25_retriever: Configured BM25Retriever instance.
        """
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index chunks into both dense and BM25 index stores.

        Args:
            chunks: List of DocumentChunk instances.
        """
        logger.info("Indexing %d chunks into BM25 retriever...", len(chunks))
        self.bm25.index(chunks)

        logger.info("Indexing %d chunks into Dense retriever...", len(chunks))
        self.dense.index(chunks)
        logger.info("Hybrid indexing complete.")

    def retrieve_dense(self, query: str, top_k: int = 15) -> list[RetrievedDocument]:
        """Execute dense semantic search in isolation.

        Answers: 'What content is semantically similar?'
        """
        return self.dense.search(query, top_k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = 15) -> list[RetrievedDocument]:
        """Execute lexical BM25 search in isolation.

        Answers: 'What content contains important lexical terms?'
        """
        return self.bm25.search(query, top_k=top_k)

    def retrieve_hybrid(
        self,
        query: str,
        dense_top_k: int = 15,
        bm25_top_k: int = 15,
        fusion_top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[FusedCandidate]:
        """Execute hybrid search pipeline: Dense + BM25 -> Reciprocal Rank Fusion.

        Args:
            query: User search query.
            dense_top_k: Top-k candidates to fetch from dense search.
            bm25_top_k: Top-k candidates to fetch from BM25 search.
            fusion_top_k: Maximum merged candidates to return.
            rrf_k: RRF smoothing constant.

        Returns:
            Ranked list of FusedCandidate instances.
        """
        dense_results = self.retrieve_dense(query, top_k=dense_top_k)
        bm25_results = self.retrieve_bm25(query, top_k=bm25_top_k)

        logger.info(
            "Retrieved %d dense candidates and %d BM25 candidates for query: '%s'",
            len(dense_results),
            len(bm25_results),
            query,
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=rrf_k,
            top_k=fusion_top_k,
        )

        logger.info("Fused into %d distinct ranked candidates.", len(fused))
        return fused
