"""Citation and claim extraction from generated answer texts.

Parses generated response paragraphs into discrete factual claims and maps
their bracket citation markers (e.g. [1], [2]) back to the underlying
retrieved context documents.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.models import Citation, Claim, RerankedDocument

# Pattern matching citation markers like [1], [2], [1, 2], [1][2]
CITATION_PATTERN = re.compile(r"\[(?:(?:Doc|Document|Passage)\s*)?(\d+)\]")


class CitationExtractor:
    """Extracts factual claims and links their bracket citations to source documents."""

    def extract_claims_and_citations(
        self,
        answer_text: str,
        context_documents: Sequence[RerankedDocument],
    ) -> tuple[list[Claim], list[Citation]]:
        """Parse answer text into claims and their linked citations.

        Args:
            answer_text: Text produced by the answer generator.
            context_documents: The list of context passages provided to the generator.

        Returns:
            Tuple of (list of Claim, list of Citation).
        """
        if not answer_text or not answer_text.strip():
            return [], []

        # Split text into candidate sentences while preserving punctuation
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", answer_text.strip())
            if s.strip()
        ]

        claims: list[Claim] = []
        citations_registry: dict[str, Citation] = {}
        claim_counter = 1

        for sentence_idx, sentence in enumerate(sentences, start=1):
            # Find all passage numbers cited in this sentence
            found_markers = CITATION_PATTERN.findall(sentence)
            if not found_markers:
                continue

            # Strip citation brackets to isolate the core factual claim
            clean_claim = CITATION_PATTERN.sub("", sentence).strip()
            clean_claim = re.sub(r"\s+", " ", clean_claim).rstrip(" .,;")

            claim_id = f"claim_{claim_counter:03d}"
            marker_strings = [f"[{m}]" for m in found_markers]

            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=clean_claim,
                    citation_markers=marker_strings,
                    sentence_index=sentence_idx,
                )
            )
            claim_counter += 1

            # Resolve each marker number to its referenced context document
            for marker_num_str in set(found_markers):
                try:
                    passage_num = int(marker_num_str)
                except ValueError:
                    continue

                marker_key = f"[{passage_num}]"
                doc_idx = passage_num - 1

                if 0 <= doc_idx < len(context_documents):
                    referenced_doc = context_documents[doc_idx]
                    char_offset = sentence.find(f"[{marker_num_str}]")

                    citation_id = f"cite_{claim_id}_passage_{passage_num}"
                    if citation_id not in citations_registry:
                        citations_registry[citation_id] = Citation(
                            citation_id=citation_id,
                            marker=marker_key,
                            document_id=referenced_doc.document_id,
                            chunk_id=referenced_doc.chunk_id,
                            source=referenced_doc.source,
                            evidence_text=referenced_doc.text,
                            char_offset=char_offset,
                        )

        return claims, list(citations_registry.values())
