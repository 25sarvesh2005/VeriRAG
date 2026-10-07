"""BM25 lexical retrieval implementation.

BM25 retrieves documents based on exact keyword occurrences, term frequency,
and inverse document frequency. It excels at identifying rare proper nouns,
specific model identifiers, technical terms, and exact phrases that dense
embeddings frequently compress or conflate.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

from app.models import DocumentChunk, RetrievedDocument


def tokenize_text(text: str) -> list[str]:
    """Tokenize a string into lowercased lexical terms.

    Extracts sequences of alphanumeric characters.
    """
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


class BM25Retriever:
    """Retrieves document chunks using Okapi BM25 lexical matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (standard default: 1.5).
            b: Document length normalization parameter (standard default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index over the provided document chunks.

        Args:
            chunks: List of DocumentChunk instances to index.
        """
        if not chunks:
            self.chunks = []
            self._bm25 = None
            self._tokenized_corpus = []
            return

        self.chunks = list(chunks)
        self._tokenized_corpus = [tokenize_text(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 15) -> list[RetrievedDocument]:
        """Perform lexical BM25 search.

        Args:
            query: User search query.
            top_k: Number of highest-scoring candidates to return.

        Returns:
            List of RetrievedDocument instances sorted descending by BM25 score.
        """
        if not query or not query.strip() or self._bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Filter out chunks with 0 score (no lexical overlap)
        scored_pairs = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score > 0.0
        ]

        # Sort descending by score
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

        results: list[RetrievedDocument] = []
        for rank, (idx, score) in enumerate(scored_pairs[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    source=chunk.source,
                    retrieval_strategy="bm25",
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results

    def save(self, directory_path: Path | str) -> None:
        """Persist BM25 chunks and tokenized corpus to disk."""
        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        (target_dir / "bm25_chunks.json").write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

    def load(self, directory_path: Path | str) -> None:
        """Load chunks from disk and rebuild BM25 index."""
        target_dir = Path(directory_path)
        chunks_file = target_dir / "bm25_chunks.json"

        if not chunks_file.exists():
            raise FileNotFoundError(f"BM25 index not found at '{chunks_file}'.")

        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in data]
        self.index(chunks)
