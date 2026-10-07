"""Prompt templates for grounded answer generation with explicit citation markers.

These prompts enforce that the LLM attributes every factual claim
to one or more numbered context passages using bracket notation e.g. [1].
"""

from __future__ import annotations

from typing import Sequence
from app.models import RerankedDocument

SYSTEM_PROMPT = """You are a rigorous, truthful AI research assistant.
Your mission is to answer the user's question accurately using ONLY the provided context passages.

RULES FOR ANSWERING:
1. Ground every statement directly in the provided context.
2. Immediately after making any factual statement or claim, append its citation marker in brackets, e.g. [1] or [1][2].
3. The citation number [k] must correspond exactly to the numbered passage [k] below.
4. Do NOT make claims that cannot be verified from the passages.
5. If the context does not contain sufficient facts to answer the question, state: "The provided context does not contain sufficient information to answer this question."
"""


def build_context_block(documents: Sequence[RerankedDocument]) -> str:
    """Format reranked documents into a numbered, human-readable context block."""
    if not documents:
        return "No reference context available."

    passages = []
    for idx, doc in enumerate(documents, start=1):
        passages.append(
            f"--- [Passage {idx}] Source: {doc.source} (Document ID: {doc.document_id}, Chunk ID: {doc.chunk_id}) ---\n"
            f"{doc.text.strip()}\n"
        )
    return "\n".join(passages)


def build_generation_prompt(query: str, documents: Sequence[RerankedDocument]) -> str:
    """Construct full user prompt containing context passages and user query."""
    context_text = build_context_block(documents)
    return f"""Context Passages:
{context_text}

Question: {query.strip()}

Please provide a clear, factual answer citing the relevant passage numbers like [1], [2] after each claim:"""
