"""Data models for citation extraction and verification.

Re-exports citation domain models from app.models for intuitive subpackage access.
"""

from app.models import (
    Citation,
    CitationVerification,
    Claim,
    VerificationVerdict,
)

__all__ = [
    "Claim",
    "Citation",
    "VerificationVerdict",
    "CitationVerification",
]
