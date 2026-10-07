"""Document ingestion and index creation script.

Usage:
    python scripts/ingest.py [--data-dir data/raw] [--indices-dir data/indices]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_CONFIG
from app.ingestion.chunking import SlidingWindowChunker
from app.ingestion.loaders import load_directory, load_json_documents
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents and build indices for VeriRAG.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CONFIG.storage.raw_data_dir,
        help="Path to folder containing raw text or JSON documents.",
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=DEFAULT_CONFIG.storage.indices_dir,
        help="Target folder for saving dense and BM25 index artifacts.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    indices_dir = args.indices_dir

    if not data_dir.exists():
        logger.error("Data directory '%s' does not exist.", data_dir)
        sys.exit(1)

    logger.info("Scanning and loading documents from: %s", data_dir)
    documents = load_directory(data_dir)
    if not documents:
        json_file = data_dir / "documents.json"
        if json_file.exists():
            documents = load_json_documents(json_file)

    if not documents:
        logger.error("No valid documents found in '%s'.", data_dir)
        sys.exit(1)

    logger.info("Loaded %d documents.", len(documents))

    # Chunking
    chunker = SlidingWindowChunker(
        chunk_size_words=DEFAULT_CONFIG.ingestion.chunk_size_words,
        chunk_overlap_words=DEFAULT_CONFIG.ingestion.chunk_overlap_words,
    )
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info("Generated %d chunks from %d documents.", len(all_chunks), len(documents))

    # Build and persist BM25 Index
    logger.info("Building BM25 index...")
    bm25 = BM25Retriever(k1=DEFAULT_CONFIG.retrieval.bm25_k1, b=DEFAULT_CONFIG.retrieval.bm25_b)
    bm25.index(all_chunks)
    bm25.save(indices_dir)
    logger.info("Saved BM25 index to: %s", indices_dir)

    # Build and persist Dense Index
    logger.info("Building Dense embeddings index using %s...", DEFAULT_CONFIG.retrieval.embedding_model_name)
    dense = DenseRetriever(
        model_name=DEFAULT_CONFIG.retrieval.embedding_model_name,
        device=DEFAULT_CONFIG.retrieval.embedding_device,
    )
    dense.index(all_chunks)
    dense.save(indices_dir)
    logger.info("Saved Dense index to: %s", indices_dir)

    logger.info("=== Ingestion complete! Total indexed chunks: %d ===", len(all_chunks))


if __name__ == "__main__":
    main()
