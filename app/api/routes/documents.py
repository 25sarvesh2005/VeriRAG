"""Document management and ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from app.api.schemas.requests import IngestTextRequest
from app.api.schemas.responses import IngestResponse
from app.ingestion.chunking import SlidingWindowChunker
from app.ingestion.loaders import create_document_from_text

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_text_document(payload: IngestTextRequest, request: Request) -> IngestResponse:
    """Ingest, clean, chunk, and index a document in both dense and BM25 stores."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")

    config = getattr(request.app.state, "config", None)
    chunker = SlidingWindowChunker(
        chunk_size_words=config.ingestion.chunk_size_words if config else 180,
        chunk_overlap_words=config.ingestion.chunk_overlap_words if config else 35,
    )

    doc = create_document_from_text(
        content=payload.content,
        title=payload.title,
        source=payload.source,
    )

    chunks = chunker.chunk_document(doc)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document content was too short or contained no valid textual words to generate chunks.",
        )

    # Append to existing chunks in retriever
    current_chunks = list(pipeline.retriever.bm25.chunks)
    new_chunk_list = current_chunks + chunks
    pipeline.retriever.index(new_chunk_list)

    indexed_docs = getattr(request.app.state, "indexed_docs", [])
    indexed_docs.append(doc)

    return IngestResponse(
        document_id=doc.document_id,
        title=doc.title,
        chunks_created=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{doc.title}'.",
    )
