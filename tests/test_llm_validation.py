"""Unit tests for multi-model validation logic."""

from types import SimpleNamespace

import pytest

from src.models.toon_models import QuestionChoice, SATQuestion
from src.services.llm_service import LLMService


class StubModel:
    """Minimal async-compatible stub to imitate chat models."""

    def __init__(self, response: str):
        self.response = response

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.response)


def _build_sample_question() -> SATQuestion:
    return SATQuestion(
        id="sample",
        question="If 3x + 7 = 22, what is x?",
        choices=QuestionChoice(A="3", B="5", C="7", D="9"),
        correct_answer="B",
        explanation="Subtract 7 and divide by 3.",
        difficulty=50.0,
        category="algebra",
    )


@pytest.mark.asyncio
async def test_validate_question_requires_unanimous_agreement():
    """Validation should fail if any model reports INVALID."""
    service = object.__new__(LLMService)
    service.models = {
        "gpt4": StubModel("VALID: looks correct"),
        "claude": StubModel("INVALID: distractors not unique"),
    }

    result = await service.validate_question(_build_sample_question())

    assert not result.is_valid
    assert pytest.approx(result.agreement, 0.5)
    assert "INVALID" in result.feedback


@pytest.mark.asyncio
async def test_validate_question_reports_full_agreement():
    """Validation should pass with 100% agreement when all validators approve."""
    service = object.__new__(LLMService)
    service.models = {
        "gpt4": StubModel("VALID: solution matches"),
        "claude": StubModel("VALID: clear and unambiguous"),
    }

    result = await service.validate_question(_build_sample_question())

    assert result.is_valid
    assert result.agreement == 1.0
    assert "Agreement: 100.0%" in result.feedback

