"""Dense vector retrieval implementation.

Uses dense sentence embeddings to identify semantically similar content,
capturing conceptual relevance even when exact keywords do not match.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app.models import DocumentChunk, RetrievedDocument


class DenseRetriever:
    """Retrieves document chunks using dense vector embeddings and cosine similarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize the dense retriever.

        Args:
            model_name: HuggingFace / SentenceTransformers model identifier.
            device: Computation device ('cpu', 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def _get_model(self):
        """Lazy load SentenceTransformer model to avoid overhead if dense retrieval is not invoked."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def index(self, chunks: list[DocumentChunk], batch_size: int = 32) -> None:
        """Compute and store embeddings for all provided document chunks.

        Args:
            chunks: List of document chunks to index.
            batch_size: Batch size for model inference.
        """
        if not chunks:
            self.chunks = []
            self.embeddings = None
            return

        self.chunks = list(chunks)
        texts = [chunk.text for chunk in self.chunks]

        model = self._get_model()
        # Encode with L2 normalization so dot product equals cosine similarity
        raw_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int = 15) -> list[RetrievedDocument]:
        """Perform dense semantic search against indexed chunks.

        Args:
            query: User search query.
            top_k: Number of highest-scoring candidates to return.

        Returns:
            List of RetrievedDocument instances sorted in descending order of similarity.
        """
        if not query or not query.strip():
            return []

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        model = self._get_model()
        query_vector = model.encode(
            [query.strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)[0]

        # Cosine similarity via dot product (since vectors are L2 normalized)
        scores = np.dot(self.embeddings, query_vector)

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort top indices descending by score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[RetrievedDocument] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                RetrievedDocument(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    source=chunk.source,
                    retrieval_strategy="dense",
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )

        return results

    def save(self, directory_path: Path | str) -> None:
        """Persist embeddings and chunk metadata to disk."""
        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.save(target_dir / "dense_embeddings.npy", self.embeddings)

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
        (target_dir / "dense_chunks.json").write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

    def load(self, directory_path: Path | str) -> None:
        """Load embeddings and chunk metadata from disk."""
        target_dir = Path(directory_path)
        chunks_file = target_dir / "dense_chunks.json"
        embeddings_file = target_dir / "dense_embeddings.npy"

        if not chunks_file.exists():
            raise FileNotFoundError(f"Dense chunks index not found at '{chunks_file}'.")

        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in data]

        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        else:
            self.embeddings = None
