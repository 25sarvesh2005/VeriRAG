"""Cross-Encoder reranking implementation.

Why Cross-Encoder Reranking is necessary:
1. Full Cross-Attention: Bi-encoders (used in dense retrieval) project queries and
   documents into single vector spaces independently. In contrast, a cross-encoder
   processes query and document text simultaneously through all transformer layers,
   enabling deep cross-attention between every query token and document token.
2. High Precision Filtering: While cross-encoders are too computationally heavy to
   score millions of documents from scratch, scoring the top-20 retrieved candidates
   takes only milliseconds and dramatically reduces irrelevant context.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.models import FusedCandidate, RerankedDocument, RetrievedDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved candidate documents using a pretrained cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace identifier for the cross-encoder model.
            device: Computation device ('cpu', 'cuda').
            batch_size: Batch size for scoring pairs.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazy load CrossEncoder to prevent startup lag when not in use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model: %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedDocument | FusedCandidate],
        top_k: int = 5,
    ) -> list[RerankedDocument]:
        """Score and rerank candidate documents for a given query.

        Args:
            query: The user search query.
            candidates: Retrieved or fused candidates.
            top_k: Maximum number of top candidates to return after reranking.

        Returns:
            List of RerankedDocument instances sorted descending by rerank score.
        """
        if not candidates:
            logger.warning("Reranking invoked with empty candidate list.")
            return []

        if not query or not query.strip():
            logger.warning("Reranking invoked with empty query.")
            return []

        model = self._get_model()
        pairs = [[query.strip(), doc.text] for doc in candidates]

        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored_candidates: list[tuple[float, int, RetrievedDocument | FusedCandidate]] = []
        for original_rank, (doc, score) in enumerate(zip(candidates, scores), start=1):
            scored_candidates.append((float(score), original_rank, doc))

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        reranked_results: list[RerankedDocument] = []
        for score, original_rank, doc in scored_candidates[:top_k]:
            reranked_results.append(
                RerankedDocument(
                    chunk_id=doc.chunk_id,
                    document_id=doc.document_id,
                    text=doc.text,
                    rerank_score=score,
                    original_rank=original_rank,
                    source=doc.source,
                    metadata=doc.metadata,
                )
            )

        logger.info(
            "Reranked %d candidates down to top %d (top score: %.4f)",
            len(candidates),
            len(reranked_results),
            reranked_results[0].rerank_score if reranked_results else 0.0,
        )

        return reranked_results
