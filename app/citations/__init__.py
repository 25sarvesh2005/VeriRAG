"""Citations package.

Provides citation models, claim extraction, and verification against retrieved sources.
"""

from app.citations.extractor import CitationExtractor
from app.citations.models import (
    Citation,
    CitationVerification,
    Claim,
    VerificationVerdict,
)
from app.citations.verifier import CitationVerifier

__all__ = [
    "Claim",
    "Citation",
    "VerificationVerdict",
    "CitationVerification",
    "CitationExtractor",
    "CitationVerifier",
]
