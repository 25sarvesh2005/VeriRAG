"""Document chunking implementation with exact character offset tracking.

Per AGENTS.md Section 19:
Every citation must trace back to document ID, chunk ID, source name,
and character offsets.
"""

from __future__ import annotations

import re
from app.models import Document, DocumentChunk


class SlidingWindowChunker:
    """Chunks documents into overlapping word windows while preserving exact character offsets."""

    def __init__(
        self,
        chunk_size_words: int = 180,
        chunk_overlap_words: int = 35,
        min_chunk_length_chars: int = 30,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size_words: Maximum number of words in each chunk window.
            chunk_overlap_words: Number of overlapping words between consecutive chunks.
            min_chunk_length_chars: Minimum character length to retain a chunk.
        """
        if chunk_size_words <= 0:
            raise ValueError(f"chunk_size_words must be > 0, got {chunk_size_words}")
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError(
                f"chunk_overlap_words ({chunk_overlap_words}) must be strictly less than "
                f"chunk_size_words ({chunk_size_words})"
            )

        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.min_chunk_length_chars = min_chunk_length_chars

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Split a Document into DocumentChunks with traceable offsets.

        Args:
            document: Ingested document to chunk.

        Returns:
            List of DocumentChunk instances.
        """
        content = document.content
        if not content or len(content.strip()) < self.min_chunk_length_chars:
            return []

        # Find word tokens along with their exact start/end character offsets in content
        # Matches any non-whitespace sequence
        word_matches = list(re.finditer(r"\S+", content))
        if not word_matches:
            return []

        step = self.chunk_size_words - self.chunk_overlap_words
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for start_idx in range(0, len(word_matches), step):
            end_idx = min(start_idx + self.chunk_size_words, len(word_matches))
            first_match = word_matches[start_idx]
            last_match = word_matches[end_idx - 1]

            char_start = first_match.start()
            char_end = last_match.end()

            chunk_text = content[char_start:char_end].strip()

            if len(chunk_text) >= self.min_chunk_length_chars:
                chunk_id = f"{document.document_id}_chunk_{chunk_index:04d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=chunk_text,
                        source=document.source,
                        chunk_index=chunk_index,
                        char_start=char_start,
                        char_end=char_end,
                        metadata={
                            **document.metadata,
                            "title": document.title,
                            "word_count": end_idx - start_idx,
                        },
                    )
                )
                chunk_index += 1

            if end_idx >= len(word_matches):
                break

        return chunks
