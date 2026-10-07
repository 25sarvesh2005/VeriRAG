"""Quantitative citation and verification evaluation metrics.

Per AGENTS.md Section 22:
Evaluates citation precision, citation coverage, and unsupported citation rate.
Directly quantifies how citation verification acts as a quality-control barrier.
"""

from __future__ import annotations

from typing import Sequence
from app.models import CitationVerification, Claim, VerificationVerdict


def calculate_citation_precision(verifications: Sequence[CitationVerification]) -> float:
    """Fraction of made citations that are verified as SUPPORTED.

    Precision = Count(SUPPORTED) / Total Citations Verified
    """
    if not verifications:
        return 0.0
    supported = sum(1 for v in verifications if v.verdict == VerificationVerdict.SUPPORTED)
    return supported / len(verifications)


def calculate_unsupported_citation_rate(verifications: Sequence[CitationVerification]) -> float:
    """Fraction of citations that are flagged as UNSUPPORTED (hallucinations/misattributions).

    Unsupported Rate = Count(UNSUPPORTED) / Total Citations Verified
    """
    if not verifications:
        return 0.0
    unsupported = sum(1 for v in verifications if v.verdict == VerificationVerdict.UNSUPPORTED)
    return unsupported / len(verifications)


def calculate_uncertainty_rate(verifications: Sequence[CitationVerification]) -> float:
    """Fraction of citations flagged as UNCERTAIN."""
    if not verifications:
        return 0.0
    uncertain = sum(1 for v in verifications if v.verdict == VerificationVerdict.UNCERTAIN)
    return uncertain / len(verifications)


def calculate_citation_coverage(claims: Sequence[Claim], total_sentences: int) -> float:
    """Proportion of factual claims/sentences that carry citations.

    Coverage = Count(Claims with >= 1 citation) / Total Sentences
    """
    if total_sentences <= 0:
        return 0.0
    cited_claims = sum(1 for c in claims if len(c.citation_markers) > 0)
    return min(1.0, cited_claims / total_sentences)
