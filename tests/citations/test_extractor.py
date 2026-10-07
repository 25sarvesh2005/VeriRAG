"""Tests for citation and claim extractor."""

from app.citations.extractor import CitationExtractor
from app.models import RerankedDocument


def test_extractor_identifies_claims_and_resolves_markers():
    extractor = CitationExtractor()

    context = [
        RerankedDocument(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Raft divides consensus into leader election, log replication, and safety.",
            rerank_score=0.95,
            original_rank=1,
            source="raft.md",
        ),
        RerankedDocument(
            chunk_id="chunk_2",
            document_id="doc_2",
            text="Followers transition to candidates if no heartbeat is received within election timeout.",
            rerank_score=0.85,
            original_rank=2,
            source="raft.md",
        ),
    ]

    answer = (
        "Raft divides consensus into leader election, log replication, and safety [1]. "
        "Followers transition to candidate state when timeouts expire [2]."
    )

    claims, citations = extractor.extract_claims_and_citations(answer, context)

    assert len(claims) == 2
    assert "leader election, log replication" in claims[0].text
    assert claims[0].citation_markers == ["[1]"]

    assert len(citations) == 2
    cite1 = next(c for c in citations if c.marker == "[1]")
    assert cite1.document_id == "doc_1"
    assert cite1.chunk_id == "chunk_1"

    cite2 = next(c for c in citations if c.marker == "[2]")
    assert cite2.document_id == "doc_2"
    assert cite2.chunk_id == "chunk_2"


def test_extractor_handles_empty_or_uncited_text():
    extractor = CitationExtractor()
    claims, citations = extractor.extract_claims_and_citations("Uncited general statement without brackets.", [])
    assert len(claims) == 0
    assert len(citations) == 0
