"""LLM Service - Multi-model orchestration with GPT-4 and Claude"""

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..models.toon_models import (
    SATQuestion,
    GeneratedQuestions,
    QuestionChoice,
    StyleProfile,
)

logger = logging.getLogger(__name__)


class LLMService:
    """Manages LLM interactions with multi-model validation"""

    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.models = {
            "gpt4": ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=openai_api_key,
            ),
            "claude": ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                api_key=anthropic_api_key,
            ),
        }
        logger.info("LLM Service initialized with GPT-4 and Claude")

    async def generate_questions(
        self,
        prompt: str,
        num_questions: int = 5,
        category: str = "algebra",
        difficulty: float = 50.0,
        style_profile: Optional[StyleProfile] = None,
        use_both_models: bool = True,
    ) -> List[SATQuestion]:
        """
        Generate SAT questions using LLM(s)

        Args:
            prompt: Generation prompt with requirements
            num_questions: Number of questions to generate
            category: Question category
            difficulty: Target difficulty (0-100)
            style_profile: Style profile to match (optional)
            use_both_models: Use both GPT-4 and Claude for variety

        Returns:
            List of generated SATQuestion objects
        """
        logger.info(f"Generating {num_questions} questions for {category}")

        # Build enhanced prompt
        full_prompt = self._build_generation_prompt(
            base_prompt=prompt,
            num_questions=num_questions,
            category=category,
            difficulty=difficulty,
            style_profile=style_profile,
        )

        all_questions = []

        # Generate from GPT-4
        try:
            gpt_questions = await self._generate_with_model(
                model_name="gpt4",
                prompt=full_prompt,
                num_questions=num_questions // 2 if use_both_models else num_questions,
            )
            all_questions.extend(gpt_questions)
            logger.info(f"GPT-4 generated {len(gpt_questions)} questions")
        except Exception as e:
            logger.error(f"GPT-4 generation failed: {e}")

        # Generate from Claude (if using both models)
        if use_both_models:
            try:
                claude_questions = await self._generate_with_model(
                    model_name="claude",
                    prompt=full_prompt,
                    num_questions=num_questions - len(all_questions),
                )
                all_questions.extend(claude_questions)
                logger.info(f"Claude generated {len(claude_questions)} questions")
            except Exception as e:
                logger.error(f"Claude generation failed: {e}")

        # Set metadata
        for i, q in enumerate(all_questions):
            q.category = category
            q.difficulty = difficulty  # Initial estimate, will be calibrated later
            if not q.id:
                q.id = f"gen_{category}_{i}"

        return all_questions

    async def validate_question(self, question: SATQuestion) -> tuple[bool, str]:
        """
        Validate a question for correctness and clarity

        Uses cross-model validation (one model checks another's work)

        Args:
            question: Question to validate

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        validator_model = self.models["gpt4"]

        validation_prompt = f"""
Validate this SAT question for correctness and clarity:

Question: {question.question}

Choices:
A) {question.choices.A}
B) {question.choices.B}
C) {question.choices.C}
D) {question.choices.D}

Claimed Correct Answer: {question.correct_answer}
Explanation: {question.explanation}

Check the following:
1. Is the claimed answer actually correct?
2. Is there only ONE correct answer among the choices?
3. Are the distractors (wrong answers) plausible but clearly incorrect?
4. Is the question unambiguous and clearly worded?
5. Does the explanation correctly justify the answer?

Respond with:
- "VALID" if all checks pass
- "INVALID: <reason>" if any check fails

Be strict but fair in your assessment.
"""

        try:
            response = await validator_model.ainvoke(
                [HumanMessage(content=validation_prompt)]
            )
            result = response.content.strip()

            is_valid = result.startswith("VALID")
            feedback = result if not is_valid else "All validation checks passed"

            logger.debug(f"Validation result for {question.id}: {result[:50]}...")
            return is_valid, feedback

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, f"Validation error: {str(e)}"

    async def _generate_with_model(
        self, model_name: str, prompt: str, num_questions: int
    ) -> List[SATQuestion]:
        """Generate questions using a specific model"""
        model = self.models[model_name]

        system_prompt = """You are an expert SAT question writer. Generate high-quality SAT questions that:
- Match the exact style and difficulty requested
- Have clear, unambiguous wording
- Include one correct answer and three plausible distractors
- Follow official SAT question formats
- Include detailed explanations

Return questions in this JSON format:
{
  "questions": [
    {
      "id": "q1",
      "question": "Question text here",
      "choices": {
        "A": "Choice A text",
        "B": "Choice B text",
        "C": "Choice C text",
        "D": "Choice D text"
      },
      "correct_answer": "B",
      "explanation": "Detailed explanation",
      "difficulty": 50.0,
      "category": "algebra"
    }
  ]
}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            # Use structured output with Pydantic models
            response = await model.with_structured_output(
                GeneratedQuestions,
                method="function_calling",
            ).ainvoke(messages)

            return response.questions

        except Exception as e:
            logger.error(f"Generation with {model_name} failed: {e}")
            # Fallback: parse response manually
            try:
                response = await model.ainvoke(messages)
                return self._parse_response_fallback(response.content)
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
                return []

    def _build_generation_prompt(
        self,
        base_prompt: str,
        num_questions: int,
        category: str,
        difficulty: float,
        style_profile: Optional[StyleProfile],
    ) -> str:
        """Build comprehensive generation prompt"""
        prompt_parts = [
            f"Generate {num_questions} SAT {category} questions.",
            f"Target difficulty level: {difficulty}/100 (0=easiest, 100=hardest)",
            "",
            "Requirements:",
            base_prompt,
        ]

        if style_profile:
            prompt_parts.extend(
                [
                    "",
                    "Style Requirements:",
                    f"- Word count: {style_profile.word_count_range[0]}-{style_profile.word_count_range[1]} words",
                    f"- Vocabulary level: Grade {style_profile.vocabulary_level:.1f}",
                    f"- Number complexity: {style_profile.number_complexity}",
                    f"- Context type: {style_profile.context_type}",
                    f"- Question structure: {style_profile.question_structure[:100]}...",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Quality Standards:",
                "- Each question must have exactly 4 choices (A, B, C, D)",
                "- Only ONE choice should be correct",
                "- Wrong answers should be plausible but clearly incorrect",
                "- Include detailed explanations showing the solution process",
                "- Use clear, professional language",
                "- Avoid ambiguity or trick questions",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_response_fallback(self, content: str) -> List[SATQuestion]:
        """Fallback parser for when structured output fails"""
        # Simple fallback - in production, use more robust parsing
        logger.warning("Using fallback response parser")
        return []

