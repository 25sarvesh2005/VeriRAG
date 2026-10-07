"""Generation package.

Provides prompt formatting and LLM answer generation with citation tagging.
"""

from app.generation.answer_generator import (
    BaseAnswerGenerator,
    GeminiAnswerGenerator,
    MockAnswerGenerator,
    OllamaAnswerGenerator,
    OpenAIAnswerGenerator,
    create_answer_generator,
)
from app.generation.prompts import SYSTEM_PROMPT, build_generation_prompt

__all__ = [
    "SYSTEM_PROMPT",
    "build_generation_prompt",
    "BaseAnswerGenerator",
    "MockAnswerGenerator",
    "OpenAIAnswerGenerator",
    "GeminiAnswerGenerator",
    "OllamaAnswerGenerator",
    "create_answer_generator",
]
