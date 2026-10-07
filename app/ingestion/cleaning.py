"""Text cleaning and normalization utilities for document ingestion.

Ensures incoming text is sanitized, consistent, and stripped of noisy
artifacts before chunking and embedding.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(
    text: str,
    strip_extra_whitespace: bool = True,
    normalize_bullets: bool = True,
) -> str:
    """Sanitize and normalize raw document text.

    Args:
        text: Raw document text string.
        strip_extra_whitespace: Whether to collapse multiple spaces/newlines.
        normalize_bullets: Whether to standardize various unicode bullet symbols.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Normalize unicode to NFKC standard form
    cleaned = unicodedata.normalize("NFKC", text)

    # Remove non-printable control characters except standard whitespace
    cleaned = "".join(
        ch for ch in cleaned if ch in ("\n", "\r", "\t") or unicodedata.category(ch)[0] != "C"
    )

    if normalize_bullets:
        # Standardize various bullet representations (•, ‣, ⁃, etc.) to standard markdown dash
        cleaned = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]\s*", "- ", cleaned)

    if strip_extra_whitespace:
        # Normalize carriage returns to newlines
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespaces (spaces, tabs) into a single space
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        # Collapse 3 or more consecutive newlines into 2 (preserving paragraph structure)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

    return cleaned
