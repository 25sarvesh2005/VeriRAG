"""Document loading utilities for ingesting raw data files.

Supports text files, markdown files, JSON datasets, and directory scanning.
Provides actionable, descriptive error messages upon failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

from app.ingestion.cleaning import clean_text
from app.models import Document


class DocumentLoadingError(RuntimeError):
    """Raised when document ingestion or parsing fails."""


def load_text_file(
    file_path: Path | str,
    clean: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Load a single .txt or .md file into a Document instance.

    Args:
        file_path: Path to the target text file.
        clean: Whether to apply text cleaning.
        metadata: Optional additional metadata dictionary.

    Returns:
        Document instance.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadingError(
            f"Failed to load document: file does not exist at '{path.resolve()}'. "
            "Please check the file path and verify the file was placed in the intended directory."
        )

    if not path.is_file():
        raise DocumentLoadingError(
            f"Failed to load document: path '{path.resolve()}' is a directory, not a file. "
            "Use load_directory() if you wish to ingest all files within a folder."
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise DocumentLoadingError(
                f"Failed to decode text from '{path.resolve()}'. "
                f"Error: {exc}. Ensure the file contains valid UTF-8 text."
            ) from exc

    content = clean_text(raw_text) if clean else raw_text
    meta = metadata.copy() if metadata else {}
    meta["file_name"] = path.name
    meta["file_size_bytes"] = path.stat().st_size

    doc_id = path.stem.replace(" ", "_")
    return Document(
        document_id=doc_id,
        content=content,
        source=str(path.name),
        title=path.stem.replace("_", " ").title(),
        metadata=meta,
    )


def load_json_documents(
    file_path: Path | str,
    clean: bool = True,
) -> list[Document]:
    """Load a list of documents from a JSON file.

    Expected JSON format: list of objects with at minimum 'content' or 'text',
    and optional 'id', 'title', 'source', 'metadata'.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadingError(
            f"Unable to load JSON dataset: file does not exist at '{path.resolve()}'. "
            "Ensure the dataset path is valid."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DocumentLoadingError(
            f"Malformed JSON in '{path.resolve()}': {exc}. Verify the file is valid JSON syntax."
        ) from exc

    if not isinstance(data, list):
        raise DocumentLoadingError(
            f"JSON dataset in '{path.resolve()}' must be an array of document objects, got {type(data).__name__}."
        )

    documents: list[Document] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        raw_text = item.get("content") or item.get("text", "")
        if not raw_text:
            continue

        doc_id = str(item.get("id") or item.get("document_id") or f"doc_{idx:04d}")
        title = str(item.get("title") or f"Document {doc_id}")
        source = str(item.get("source") or path.name)
        meta = item.get("metadata", {})

        content = clean_text(raw_text) if clean else raw_text

        documents.append(
            Document(
                document_id=doc_id,
                content=content,
                source=source,
                title=title,
                metadata=meta,
            )
        )

    return documents


def load_directory(
    directory_path: Path | str,
    extensions: tuple[str, ...] = (".txt", ".md", ".json"),
    clean: bool = True,
) -> list[Document]:
    """Scan a directory and load all matching text and JSON files."""
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise DocumentLoadingError(
            f"Directory '{path.resolve()}' does not exist or is not a directory."
        )

    loaded_documents: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() in (".txt", ".md"):
            loaded_documents.append(load_text_file(file_path, clean=clean))
        elif file_path.suffix.lower() == ".json":
            loaded_documents.extend(load_json_documents(file_path, clean=clean))

    return loaded_documents


def create_document_from_text(
    content: str,
    title: str = "Untitled Document",
    source: str = "manual_input",
    clean: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Convenience factory to create a Document from a raw string."""
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    processed_text = clean_text(content) if clean else content
    return Document(
        document_id=doc_id,
        content=processed_text,
        source=source,
        title=title,
        metadata=metadata or {},
    )
