"""Document ingestion package.

Responsible exclusively for:
- Reading raw files (loaders.py)
- Sanitizing & cleaning text (cleaning.py)
- Chunking documents into indexed segments (chunking.py)
"""

from app.ingestion.chunking import SlidingWindowChunker
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import (
    DocumentLoadingError,
    create_document_from_text,
    load_directory,
    load_json_documents,
    load_text_file,
)

__all__ = [
    "clean_text",
    "SlidingWindowChunker",
    "load_text_file",
    "load_json_documents",
    "load_directory",
    "create_document_from_text",
    "DocumentLoadingError",
]
