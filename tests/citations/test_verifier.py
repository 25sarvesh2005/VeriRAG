"""Tests for CitationVerifier evaluating SUPPORTED, UNSUPPORTED, and UNCERTAIN states."""

from app.citations.verifier import CitationVerifier
from app.config import CitationConfig
from app.models import Citation, Claim, VerificationVerdict


def test_verifier_supported_claim():
    verifier = CitationVerifier(config=CitationConfig(min_verification_confidence=0.6))

    claim = Claim(
        claim_id="c1",
        text="Raft divides consensus into leader election, log replication, and safety",
        citation_markers=["[1]"],
    )
    citation = Citation(
        citation_id="cite1",
        marker="[1]",
        document_id="doc1",
        chunk_id="chk1",
        source="raft.md",
        evidence_text=(
            "Raft is a consensus algorithm that divides the problem into leader election, "
            "log replication, and safety."
        ),
    )

    result = verifier.verify_single(claim, citation)
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence >= 0.6
    assert "Direct support verified" in result.explanation


def test_verifier_contradicted_unsupported_claim():
    verifier = CitationVerifier()

    # Claim makes an unsupported assertion contradictory to source (numerical or factual conflict)
    claim = Claim(
        claim_id="c2",
        text="The election timeout is exactly 9999ms in standard Raft configurations",
        citation_markers=["[1]"],
    )
    citation = Citation(
        citation_id="cite2",
        marker="[1]",
        document_id="doc1",
        chunk_id="chk1",
        source="raft.md",
        evidence_text="Followers use an election timeout randomized between 150ms and 300ms.",
    )

    result = verifier.verify_single(claim, citation)
    assert result.verdict == VerificationVerdict.UNSUPPORTED
    assert "Contradiction detected" in result.explanation or "Unsupported" in result.explanation


def test_verifier_uncertain_claim():
    verifier = CitationVerifier(config=CitationConfig(min_verification_confidence=0.75, uncertainty_threshold=0.3))

    claim = Claim(
        claim_id="c3",
        text="Modern database systems are widely preferred across commercial enterprise applications",
        citation_markers=["[1]"],
    )
    citation = Citation(
        citation_id="cite3",
        marker="[1]",
        document_id="doc1",
        chunk_id="chk1",
        source="db.md",
        evidence_text="Log-Structured Merge-Trees write to MemTable and SSTables.",
    )

    result = verifier.verify_single(claim, citation)
    # The source is about LSM-trees, not commercial enterprise market share
    assert result.verdict in (VerificationVerdict.UNSUPPORTED, VerificationVerdict.UNCERTAIN)
