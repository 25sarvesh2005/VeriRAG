"""Answer generation layer.

Constructs prompts, communicates with LLM providers (or deterministic mock generator),
and returns generated responses containing citation brackets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Sequence

from app.config import GenerationConfig
from app.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from app.models import GeneratedAnswer, RerankedDocument

logger = logging.getLogger(__name__)


class BaseAnswerGenerator(ABC):
    """Abstract base class for answer generators."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context_documents: Sequence[RerankedDocument],
    ) -> GeneratedAnswer:
        """Generate a cited answer for a query based on context documents."""
        pass


class MockAnswerGenerator(BaseAnswerGenerator):
    """Deterministic, offline generator for tests, benchmarks, and keyless local runs.

    Synthesizes answers by selecting key factual sentences from the top retrieved passages
    and properly annotating them with citations like [1], [2].
    Can also simulate hallucinations when specifically configured.
    """

    def __init__(self, simulate_hallucination: bool = False) -> None:
        self.simulate_hallucination = simulate_hallucination

    def generate(
        self,
        query: str,
        context_documents: Sequence[RerankedDocument],
    ) -> GeneratedAnswer:
        if not context_documents:
            return GeneratedAnswer(
                text="The provided context does not contain sufficient information to answer this question.",
                model_name="mock-deterministic",
            )

        query_words = set(re.findall(r"\w+", query.lower()))
        answer_sentences: list[str] = []

        # Find the most relevant sentence from top passages
        for idx, doc in enumerate(context_documents[:3], start=1):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.text) if s.strip()]
            if not sentences:
                continue

            # Pick sentence with highest word overlap with query, or first sentence
            best_sentence = sentences[0]
            best_overlap = -1
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s

            # Ensure sentence ends with period before citation
            s_clean = best_sentence.rstrip(".")
            answer_sentences.append(f"{s_clean} [{idx}].")

        if self.simulate_hallucination and len(context_documents) > 0:
            # Deliberately inject an unsupported claim referencing passage 1 for testing
            answer_sentences.append(
                "Additionally, this system operates entirely on solar radiation during lunar eclipses [1]."
            )

        full_text = " ".join(answer_sentences)
        return GeneratedAnswer(
            text=full_text,
            model_name="mock-deterministic",
            prompt_tokens=150,
            completion_tokens=len(full_text.split()),
        )


class OpenAIAnswerGenerator(BaseAnswerGenerator):
    """Generates answers via OpenAI API."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(
        self,
        query: str,
        context_documents: Sequence[RerankedDocument],
    ) -> GeneratedAnswer:
        import openai

        client = openai.OpenAI(api_key=self.config.openai_api_key)
        prompt = build_generation_prompt(query, context_documents)

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return GeneratedAnswer(
            text=content.strip(),
            model_name=self.config.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


class GeminiAnswerGenerator(BaseAnswerGenerator):
    """Generates answers via Google Gemini API."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(
        self,
        query: str,
        context_documents: Sequence[RerankedDocument],
    ) -> GeneratedAnswer:
        from google import genai

        client = genai.Client(api_key=self.config.gemini_api_key)
        prompt = build_generation_prompt(query, context_documents)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        )

        text = response.text or ""
        return GeneratedAnswer(
            text=text.strip(),
            model_name="gemini-2.5-flash",
        )


class OllamaAnswerGenerator(BaseAnswerGenerator):
    """Generates answers via local Ollama API."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(
        self,
        query: str,
        context_documents: Sequence[RerankedDocument],
    ) -> GeneratedAnswer:
        import httpx

        prompt = build_generation_prompt(query, context_documents)
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"

        payload = {
            "model": self.config.model_name or "llama3",
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
        }

        with httpx.Client(timeout=60.0) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            data = res.json()

        return GeneratedAnswer(
            text=data.get("response", "").strip(),
            model_name=self.config.model_name,
        )


def create_answer_generator(config: GenerationConfig) -> BaseAnswerGenerator:
    """Factory creating an AnswerGenerator appropriate for the configured provider."""
    provider = config.provider.lower()

    if provider == "mock":
        return MockAnswerGenerator()
    elif provider == "openai":
        if not config.openai_api_key:
            logger.warning("No OpenAI API key found; falling back to MockAnswerGenerator.")
            return MockAnswerGenerator()
        return OpenAIAnswerGenerator(config)
    elif provider == "gemini":
        if not config.gemini_api_key:
            logger.warning("No Gemini API key found; falling back to MockAnswerGenerator.")
            return MockAnswerGenerator()
        return GeminiAnswerGenerator(config)
    elif provider == "ollama":
        return OllamaAnswerGenerator(config)
    else:
        logger.warning("Unknown LLM provider '%s'; defaulting to MockAnswerGenerator.", provider)
        return MockAnswerGenerator()
