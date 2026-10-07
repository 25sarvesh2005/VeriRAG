"""Citation verification engine.

Per AGENTS.md Section 16 & 17:
This is the core quality-control layer of the system.
A citation is NEVER accepted merely because a citation tag exists.
The verifier evaluates:
1. What factual claim was made?
2. Which source was cited?
3. What evidence exists in that source?
4. Does the evidence support the claim?
5. Is the support direct or inferred?
6. Is the evidence contradictory?
7. What confidence does the verifier have?
"""

from __future__ import annotations

import logging
import math
import re
from typing import Sequence

from app.citations.models import (
    Citation,
    CitationVerification,
    Claim,
    VerificationVerdict,
)
from app.config import CitationConfig

logger = logging.getLogger(__name__)

# Common negation words for contradiction detection
NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "none", "cannot", "failed"}


class CitationVerifier:
    """Verifies whether cited document passages genuinely substantiate generated claims."""

    def __init__(
        self,
        config: CitationConfig | None = None,
        use_cross_encoder: bool = True,
    ) -> None:
        """Initialize citation verifier.

        Args:
            config: CitationConfig instance containing threshold parameters.
            use_cross_encoder: Whether to score claim-evidence pairs with cross-encoder.
        """
        self.config = config or CitationConfig()
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """Lazy load cross-encoder model for entailment scoring."""
        if self._cross_encoder is None and self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(
                    self.config.verifier_model_name,
                    device="cpu",
                )
            except Exception as exc:
                logger.warning("Could not load CrossEncoder for verification: %s. Using heuristic verifier.", exc)
                self._cross_encoder = None
        return self._cross_encoder

    def _extract_best_evidence_sentence(self, claim_text: str, source_text: str) -> tuple[str, float]:
        """Find the sentence in source_text that most closely addresses claim_text.

        Returns:
            Tuple of (best_evidence_snippet, lexical_overlap_score).
        """
        source_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", source_text)
            if s.strip()
        ]
        if not source_sentences:
            return source_text[: self.config.max_evidence_chars], 0.0

        claim_tokens = set(re.findall(r"\b\w{3,}\b", claim_text.lower()))
        if not claim_tokens:
            return source_sentences[0], 0.0

        best_sentence = source_sentences[0]
        best_overlap = 0.0

        for sentence in source_sentences:
            sent_tokens = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
            if not sent_tokens:
                continue

            intersection = claim_tokens.intersection(sent_tokens)
            # Jaccard overlap on informative words
            overlap = len(intersection) / len(claim_tokens.union(sent_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        return best_sentence, best_overlap

    def _check_contradiction(self, claim_text: str, evidence_text: str) -> tuple[bool, str]:
        """Detect obvious contradictions such as conflicting polarities or contradictory numbers."""
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()

        # Check polarity mismatch (one has negation, the other does not)
        claim_has_negation = bool(NEGATION_WORDS.intersection(set(re.findall(r"\b\w+\b", claim_lower))))
        evidence_has_negation = bool(NEGATION_WORDS.intersection(set(re.findall(r"\b\w+\b", evidence_lower))))

        # Extract digits/numbers/percentages (handles attached units like 9999ms, 50%, etc.)
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim_text))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence_text))

        # If claim asserts a specific number not found anywhere in evidence, flag numerical hallucination
        if claim_numbers and not claim_numbers.issubset(evidence_numbers):
            conflicting = claim_numbers - evidence_numbers
            return True, f"Claim cites numerical figures {conflicting} not present in cited evidence."

        if claim_has_negation != evidence_has_negation:
            # Polarity discordance detected
            return True, "Polarity mismatch detected: claim and evidence disagree on negation state."

        return False, ""

    def verify_single(
        self,
        claim: Claim,
        citation: Citation,
    ) -> CitationVerification:
        """Verify an individual claim against its referenced citation.

        Args:
            claim: Factual claim extracted from generated answer.
            citation: Citation linking to source document chunk.

        Returns:
            CitationVerification with verdict, confidence, evidence snippet, and explanation.
        """
        evidence_snippet, lexical_score = self._extract_best_evidence_sentence(
            claim.text, citation.evidence_text
        )

        # Check for direct contradictions
        is_contradiction, contradiction_reason = self._check_contradiction(
            claim.text, evidence_snippet
        )

        if is_contradiction:
            return CitationVerification(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                citation_id=citation.citation_id,
                source=citation.source,
                chunk_id=citation.chunk_id,
                verdict=VerificationVerdict.UNSUPPORTED,
                confidence=0.85,
                evidence=evidence_snippet,
                explanation=f"Contradiction detected: {contradiction_reason}",
            )

        # Compute semantic entailment confidence
        encoder = self._get_cross_encoder()
        if encoder is not None:
            try:
                # Cross-encoder scores relevance of (evidence, claim)
                raw_score = float(encoder.predict([(evidence_snippet, claim.text)])[0])
                # Convert logits to probability via sigmoid if needed
                confidence = 1.0 / (1.0 + math.exp(-raw_score))
            except Exception as exc:
                logger.debug("CrossEncoder scoring failed (%s); using lexical overlap.", exc)
                confidence = min(1.0, lexical_score * 2.2)
        else:
            # Deterministic heuristic based on lexical/content overlap
            confidence = min(1.0, lexical_score * 2.0)

        # Assign verdict based on calibrated thresholds
        if confidence >= self.config.min_verification_confidence:
            verdict = VerificationVerdict.SUPPORTED
            explanation = (
                f"Direct support verified with high confidence ({confidence:.2f}). "
                "Evidence contains corresponding facts without contradiction."
            )
        elif confidence < self.config.uncertainty_threshold:
            verdict = VerificationVerdict.UNSUPPORTED
            explanation = (
                f"Unsupported claim ({confidence:.2f} confidence). "
                "Cited passage lacks substantiating statements for this assertion."
            )
        else:
            verdict = VerificationVerdict.UNCERTAIN
            explanation = (
                f"Uncertain support ({confidence:.2f} confidence). "
                "Cited evidence is conceptually related but does not definitively prove the claim."
            )

        return CitationVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            citation_id=citation.citation_id,
            source=citation.source,
            chunk_id=citation.chunk_id,
            verdict=verdict,
            confidence=round(confidence, 4),
            evidence=evidence_snippet,
            explanation=explanation,
        )

    def verify(
        self,
        claims: Sequence[Claim],
        citations: Sequence[Citation],
    ) -> list[CitationVerification]:
        """Verify all claims against their cited documents.

        Args:
            claims: List of extracted claims.
            citations: List of extracted citations.

        Returns:
            List of CitationVerification records.
        """
        citations_by_id = {c.citation_id: c for c in citations}
        citations_by_marker = {c.marker: c for c in citations}

        verifications: list[CitationVerification] = []

        for claim in claims:
            # If claim has no citation markers, it is an uncited claim (unsupported by definition)
            if not claim.citation_markers:
                verifications.append(
                    CitationVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.text,
                        citation_id="none",
                        source="unattributed",
                        chunk_id="none",
                        verdict=VerificationVerdict.UNSUPPORTED,
                        confidence=1.0,
                        evidence="",
                        explanation="Claim was made without any cited reference or source attribution.",
                    )
                )
                continue

            for marker in claim.citation_markers:
                # Find matching citation
                citation = citations_by_marker.get(marker)
                if not citation:
                    # Marker references a passage not in context
                    verifications.append(
                        CitationVerification(
                            claim_id=claim.claim_id,
                            claim_text=claim.text,
                            citation_id=f"invalid_{marker}",
                            source="unknown",
                            chunk_id="unknown",
                            verdict=VerificationVerdict.UNSUPPORTED,
                            confidence=1.0,
                            evidence="",
                            explanation=f"Citation marker {marker} references an invalid or non-existent passage.",
                        )
                    )
                    continue

                verification = self.verify_single(claim, citation)
                verifications.append(verification)

        logger.info(
            "Verification completed for %d claims (%d citations). Results: %s",
            len(claims),
            len(citations),
            {
                "SUPPORTED": sum(1 for v in verifications if v.verdict == VerificationVerdict.SUPPORTED),
                "UNSUPPORTED": sum(1 for v in verifications if v.verdict == VerificationVerdict.UNSUPPORTED),
                "UNCERTAIN": sum(1 for v in verifications if v.verdict == VerificationVerdict.UNCERTAIN),
            },
        )

        return verifications
