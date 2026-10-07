"""RAG Pipeline Orchestrator.

Orchestrates the modular components into a coherent, observable data flow:
1. Candidate Retrieval (Dense, BM25, or Hybrid)
2. Score Fusion (Reciprocal Rank Fusion)
3. Cross-Encoder Reranking
4. Prompt Assembly & Answer Generation
5. Citation & Claim Extraction
6. Citation Verification

In adherence to AGENTS.md Section 7 & 8:
This class does NOT implement retrieval, reranking, generation, or verification itself.
It serves purely as a clean orchestrator connecting single-responsibility components.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from app.citations.extractor import CitationExtractor
from app.citations.verifier import CitationVerifier
from app.config import AppConfig, DEFAULT_CONFIG
from app.generation.answer_generator import BaseAnswerGenerator, create_answer_generator
from app.models import (
    CitationVerification,
    RAGResponse,
    RerankedDocument,
)
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

PipelineStrategy = Literal[
    "dense_only",
    "bm25_only",
    "hybrid_only",
    "hybrid_rerank",
    "hybrid_rerank_verify",
]


class RAGPipeline:
    """Orchestrates modular retrieval, reranking, generation, and citation verification."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        generator: BaseAnswerGenerator | None = None,
        extractor: CitationExtractor | None = None,
        verifier: CitationVerifier | None = None,
        config: AppConfig | None = None,
    ) -> None:
        """Initialize pipeline with components.

        All components can be injected or will be initialized from config.
        """
        self.config = config or DEFAULT_CONFIG
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranking.reranker_model_name,
            device=self.config.reranking.reranker_device,
        )
        self.generator = generator or create_answer_generator(self.config.generation)
        self.extractor = extractor or CitationExtractor()
        self.verifier = verifier or CitationVerifier(config=self.config.citation)

    def query(
        self,
        question: str,
        strategy: PipelineStrategy = "hybrid_rerank_verify",
    ) -> RAGResponse:
        """Process a query through the configured pipeline stages.

        Args:
            question: User question string.
            strategy: Execution strategy (allows benchmarking ablation variants).

        Returns:
            RAGResponse containing answer, citations, verifications, and timing metrics.
        """
        timing: dict[str, float] = {}
        logger.info("=== Query Received: '%s' (strategy: %s) ===", question, strategy)

        # -------------------------------------------------------------
        # Stage 1: Retrieval & Fusion
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        context_docs: list[RerankedDocument] = []

        if strategy == "dense_only":
            dense_docs = self.retriever.retrieve_dense(
                question, top_k=self.config.reranking.rerank_top_k
            )
            timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000
            logger.info("Dense retrieval completed: %d candidates", len(dense_docs))
            context_docs = [RerankedDocument.from_retrieved(d) for d in dense_docs]

        elif strategy == "bm25_only":
            bm25_docs = self.retriever.retrieve_bm25(
                question, top_k=self.config.reranking.rerank_top_k
            )
            timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000
            logger.info("BM25 retrieval completed: %d candidates", len(bm25_docs))
            context_docs = [RerankedDocument.from_retrieved(d) for d in bm25_docs]

        else:
            # Dual-channel retrieval
            dense_candidates = self.retriever.retrieve_dense(
                question, top_k=self.config.retrieval.dense_top_k
            )
            bm25_candidates = self.retriever.retrieve_bm25(
                question, top_k=self.config.retrieval.bm25_top_k
            )

            # Reciprocal Rank Fusion
            fused_candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                k=self.config.retrieval.rrf_k,
                top_k=self.config.retrieval.fusion_top_k,
            )
            timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000
            logger.info(
                "Hybrid retrieval: %d dense + %d BM25 -> %d unique fused candidates",
                len(dense_candidates),
                len(bm25_candidates),
                len(fused_candidates),
            )

            if strategy == "hybrid_only":
                # Take top candidates directly without cross-encoder reranking
                context_docs = [
                    RerankedDocument.from_fused(fc, idx)
                    for idx, fc in enumerate(
                        fused_candidates[: self.config.reranking.rerank_top_k], start=1
                    )
                ]
            else:
                # -------------------------------------------------------------
                # Stage 2: Cross-Encoder Reranking
                # -------------------------------------------------------------
                t_rerank = time.perf_counter()
                context_docs = self.reranker.rerank(
                    query=question,
                    candidates=fused_candidates,
                    top_k=self.config.reranking.rerank_top_k,
                )
                timing["reranking_ms"] = (time.perf_counter() - t_rerank) * 1000
                logger.info("After reranking: top %d context documents selected", len(context_docs))

        # -------------------------------------------------------------
        # Stage 3: Answer Generation
        # -------------------------------------------------------------
        t_gen = time.perf_counter()
        generated_answer = self.generator.generate(
            query=question,
            context_documents=context_docs,
        )
        timing["generation_ms"] = (time.perf_counter() - t_gen) * 1000
        logger.info("Generation completed (model: %s)", generated_answer.model_name)

        # -------------------------------------------------------------
        # Stage 4: Citation & Claim Extraction
        # -------------------------------------------------------------
        t_extract = time.perf_counter()
        claims, citations = self.extractor.extract_claims_and_citations(
            answer_text=generated_answer.text,
            context_documents=context_docs,
        )
        timing["extraction_ms"] = (time.perf_counter() - t_extract) * 1000
        logger.info("Claims extracted: %d | Citations found: %d", len(claims), len(citations))

        # -------------------------------------------------------------
        # Stage 5: Citation Verification (if enabled)
        # -------------------------------------------------------------
        verifications: list[CitationVerification] = []
        if strategy == "hybrid_rerank_verify":
            t_verify = time.perf_counter()
            verifications = self.verifier.verify(claims, citations)
            timing["verification_ms"] = (time.perf_counter() - t_verify) * 1000

            supported_count = sum(1 for v in verifications if v.verdict.value == "SUPPORTED")
            unsupported_count = sum(1 for v in verifications if v.verdict.value == "UNSUPPORTED")
            uncertain_count = sum(1 for v in verifications if v.verdict.value == "UNCERTAIN")
            logger.info(
                "Verification completed -> Supported: %d, Unsupported: %d, Uncertain: %d",
                supported_count,
                unsupported_count,
                uncertain_count,
            )

        total_elapsed = sum(timing.values())
        timing["total_pipeline_ms"] = total_elapsed

        return RAGResponse(
            query=question,
            answer=generated_answer.text,
            citations=citations,
            verifications=verifications,
            context_documents=context_docs,
            retrieval_strategy_used=strategy,
            timing_ms={k: round(v, 2) for k, v in timing.items()},
        )
