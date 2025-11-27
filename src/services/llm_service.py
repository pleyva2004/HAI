"""LLM Service - Multi-model orchestration with GPT-4 and Claude"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..models.toon_models import (
    SATQuestion,
    GeneratedQuestions,
    QuestionChoice,
    StyleProfile,
    OfficialSATQuestion,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured result returned by multi-model validation."""

    is_valid: bool
    feedback: str
    agreement: float


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

    async def generate_variations(
        self,
        template: OfficialSATQuestion,
        num_questions: int,
        style_profile: Optional[StyleProfile] = None,
    ) -> List[SATQuestion]:
        """
        Generate template-based variations of a real SAT question.

        Args:
            template: Real SAT question to mirror
            num_questions: Number of variations requested
            style_profile: Optional style constraints

        Returns:
            List of SATQuestion variations
        """
        if num_questions <= 0:
            return []

        prompt = self._build_variation_prompt(
            template=template,
            num_questions=num_questions,
            style_profile=style_profile,
        )

        variations = await self._generate_with_model(
            model_name="gpt4",
            prompt=prompt,
            num_questions=num_questions,
        )

        for idx, variation in enumerate(variations):
            variation.category = template.category
            variation.subcategory = template.subcategory
            variation.is_real = False
            if not variation.id:
                variation.id = f"var_{template.question_id}_{idx}"

        return variations

    async def validate_question(self, question: SATQuestion) -> ValidationResult:
        """
        Validate a question for correctness and clarity

        Uses cross-model validation (each model checks the question independently)

        Args:
            question: Question to validate

        Returns:
            ValidationResult with verdict, aggregated feedback, and agreement score
        """
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

        validator_results = []

        for name, model in self.models.items():
            try:
                response = await model.ainvoke([HumanMessage(content=validation_prompt)])
                content = response.content.strip()
                is_valid = content.upper().startswith("VALID")
                validator_results.append(
                    {
                        "model": name,
                        "is_valid": is_valid,
                        "raw_feedback": content,
                    }
                )
                logger.debug(
                    "Validation result for %s via %s: %s",
                    question.id,
                    name,
                    content[:80],
                )
            except Exception as e:
                logger.error("Validation via %s failed: %s", name, e)
                validator_results.append(
                    {
                        "model": name,
                        "is_valid": False,
                        "raw_feedback": f"Validation error: {str(e)}",
                    }
                )

        if not validator_results:
            return ValidationResult(
                is_valid=False,
                feedback="Validation failed: no validator results",
                agreement=0.0,
            )

        total_validators = len(validator_results)
        valid_votes = sum(1 for result in validator_results if result["is_valid"])
        agreement = valid_votes / total_validators
        is_valid = valid_votes == total_validators

        feedback_lines = [
            f"[{result['model'].upper()}] "
            f"{'VALID' if result['is_valid'] else 'INVALID'} - {result['raw_feedback']}"
            for result in validator_results
        ]

        feedback_lines.append(
            f"Agreement: {agreement * 100:.1f}% ({valid_votes}/{total_validators} validators)"
        )

        feedback = "\n".join(feedback_lines)

        return ValidationResult(
            is_valid=is_valid,
            feedback=feedback if feedback else "Validation complete",
            agreement=agreement,
        )

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

    def _build_variation_prompt(
        self,
        template: OfficialSATQuestion,
        num_questions: int,
        style_profile: Optional[StyleProfile],
    ) -> str:
        """Build prompt for template-based variation generation."""
        choice_lines = [
            f"A) {template.choices.A}",
            f"B) {template.choices.B}",
            f"C) {template.choices.C}",
            f"D) {template.choices.D}",
        ]
        choices_block = "\n".join(choice_lines)

        style_lines: List[str] = []
        if style_profile:
            style_lines = [
                "",
                "Match these style attributes:",
                f"- Word count range: {style_profile.word_count_range[0]}-{style_profile.word_count_range[1]} words",
                f"- Vocabulary level: Grade {style_profile.vocabulary_level:.1f}",
                f"- Number complexity: {style_profile.number_complexity}",
                f"- Context type: {style_profile.context_type}",
                f"- Structure cues: {style_profile.question_structure[:120]}...",
            ]

        prompt_parts = [
            f"Generate {num_questions} NEW SAT questions by creating variations of the official question below.",
            "",
            "Template question:",
            f"Question: {template.question_text}",
            "Choices:",
            choices_block,
            f"Correct answer: {template.correct_answer}",
            "",
            "Requirements:",
            "- Test the SAME mathematical concept as the template.",
            "- Keep the same overall structure and reasoning steps.",
            "- Use new numbers, contexts, and surface wording so the question is fresh.",
            f"- Match difficulty: {template.difficulty:.1f}/100 (national correct rate {template.national_correct_rate:.1f}%).",
            "- Provide one correct option and three plausible distractors.",
            "- Include a full explanation mirroring official SAT style.",
            "- Never copy text verbatim from the template.",
        ]

        prompt_parts.extend(style_lines)

        prompt_parts.extend(
            [
                "",
                "Return the questions using the exact JSON schema from the system prompt (SATQuestion list).",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_response_fallback(self, content: str) -> List[SATQuestion]:
        """Fallback parser for when structured output fails"""
        # Simple fallback - in production, use more robust parsing
        logger.warning("Using fallback response parser")
        return []

