"""Tests for the hybrid generation node logic."""

import pytest

from src.graph.nodes import generate_node
from src.models.toon_models import (
    GraphState,
    OfficialSATQuestion,
    QuestionAnalysis,
    QuestionChoice,
    SATQuestion,
)


class DummyLLMService:
    """Minimal service that produces deterministic synthetic questions."""

    def __init__(self):
        self.variation_calls = 0
        self.synthetic_calls = 0

    async def generate_variations(self, template: OfficialSATQuestion, num_questions: int, style_profile=None):
        self.variation_calls += 1
        return [
            SATQuestion(
                id=f"var_{template.question_id}_{idx}",
                question=f"{template.question_text} variation {idx}",
                choices=template.choices,
                correct_answer=template.correct_answer,
                explanation="Variation explanation",
                difficulty=template.difficulty,
                category=template.category,
                is_real=False,
            )
            for idx in range(num_questions)
        ]

    async def generate_questions(self, **kwargs):
        self.synthetic_calls += 1
        count = kwargs.get("num_questions", 1)
        return [
            SATQuestion(
                id=f"fallback_{idx}",
                question=f"Fallback question {idx}",
                choices=QuestionChoice(A="1", B="2", C="3", D="4"),
                correct_answer="B",
                explanation="Fallback explanation",
                difficulty=kwargs.get("difficulty", 50.0),
                category=kwargs.get("category", "algebra"),
                is_real=False,
            )
            for idx in range(count)
        ]


def _build_official_question(idx: int) -> OfficialSATQuestion:
    return OfficialSATQuestion(
        question_id=f"official_{idx}",
        source="Practice Test",
        category="algebra",
        subcategory="linear",
        difficulty=45.0,
        question_text=f"Official question {idx}",
        choices=QuestionChoice(A="1", B="2", C="3", D="4"),
        correct_answer="B",
        explanation="Official explanation",
        national_correct_rate=60.0,
        avg_time_seconds=75,
        common_wrong_answers=["A", "C"],
        tags=["official", "algebra"],
    )


@pytest.mark.asyncio
async def test_generate_node_builds_hybrid_mix():
    """Hybrid generation should set targets and produce synthetic candidates."""
    services = {"llm": DummyLLMService()}
    state = GraphState(
        description="Need algebra practice",
        num_questions=4,
        analysis=QuestionAnalysis(
            category="algebra",
            difficulty=40.0,
            style="concise",
            characteristics=["linear equations"],
            example_structure="two-step equation",
        ),
    )
    state.real_questions = [_build_official_question(i) for i in range(4)]

    updated_state = await generate_node(state, services)

    assert updated_state.hybrid_targets == {"real": 2, "generated": 2}
    assert len(updated_state.generated_candidates) > 0
    assert services["llm"].variation_calls > 0

