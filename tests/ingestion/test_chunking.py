"""Tests for sliding window chunking and offset tracking."""

import pytest
from app.ingestion.chunking import SlidingWindowChunker
from app.models import Document


def test_chunker_preserves_char_offsets():
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10."
    doc = Document(document_id="doc_test", content=text, source="test.txt")

    chunker = SlidingWindowChunker(chunk_size_words=5, chunk_overlap_words=2, min_chunk_length_chars=5)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # Verify slice of original text matches chunk text exactly
        assert doc.content[chunk.char_start : chunk.char_end].strip() == chunk.text
        assert chunk.document_id == "doc_test"
        assert chunk.source == "test.txt"


def test_chunker_single_chunk_for_short_text():
    text = "A short document that fits in one chunk."
    doc = Document(document_id="doc_short", content=text, source="short.txt")

    chunker = SlidingWindowChunker(chunk_size_words=50, chunk_overlap_words=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_chunker_validates_parameters():
    with pytest.raises(ValueError, match="chunk_size_words must be > 0"):
        SlidingWindowChunker(chunk_size_words=0)

    with pytest.raises(ValueError, match="chunk_overlap_words .* strictly less"):
        SlidingWindowChunker(chunk_size_words=10, chunk_overlap_words=10)
