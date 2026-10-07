"""Tests for document loaders and error handling."""

import json
from pathlib import Path
import pytest
from app.ingestion.loaders import (
    DocumentLoadingError,
    create_document_from_text,
    load_json_documents,
    load_text_file,
)


def test_load_text_file(tmp_path: Path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Hello, this is a sample file.", encoding="utf-8")

    doc = load_text_file(sample_file)
    assert doc.document_id == "sample"
    assert "Hello, this is a sample file." in doc.content
    assert doc.source == "sample.txt"


def test_load_text_file_not_found():
    with pytest.raises(DocumentLoadingError, match="file does not exist"):
        load_text_file("non_existent_file_path.txt")


def test_load_json_documents(tmp_path: Path):
    json_file = tmp_path / "docs.json"
    data = [
        {"id": "doc1", "content": "First content text.", "title": "Doc 1"},
        {"id": "doc2", "content": "Second content text.", "title": "Doc 2"},
    ]
    json_file.write_text(json.dumps(data), encoding="utf-8")

    docs = load_json_documents(json_file)
    assert len(docs) == 2
    assert docs[0].document_id == "doc1"
    assert docs[1].content == "Second content text."


def test_create_document_from_text():
    doc = create_document_from_text("Direct text content.", title="Direct Title")
    assert doc.title == "Direct Title"
    assert doc.content == "Direct text content."
    assert doc.document_id.startswith("doc_")
