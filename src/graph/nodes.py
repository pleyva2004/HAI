"""LangGraph workflow nodes"""

import logging
import math
from typing import Any, Dict, List

from ..models.toon_models import GraphState, QuestionAnalysis, SATQuestion
from ..services.ocr_service import OCRService
from ..services.question_bank import QuestionBankService
from ..services.llm_service import LLMService
from ..services.style_analyzer import StyleAnalyzer, StyleMatcher
from ..services.difficulty_calibrator import DifficultyCalibrator
from ..services.duplication_detector import DuplicationDetector

logger = logging.getLogger(__name__)


def _build_generation_brief(state: GraphState) -> str:
    """Helper to build a reusable generation prompt."""
    category = state.analysis.category if state.analysis else "algebra"
    difficulty = state.analysis.difficulty if state.analysis else 50
    base_requirements = state.description or state.extracted_text or "Generate SAT questions."

    return f"""
Requirements: {base_requirements}

Generate SAT questions with these characteristics:
- Category: {category}
- Difficulty: {difficulty}/100
- Style: Match the examples provided
"""


async def ocr_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 1: Extract text from uploaded file (if provided)

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with extracted_text
    """
    logger.info("=== OCR Node ===")

    if not state.uploaded_file_path:
        logger.info("No file uploaded, skipping OCR")
        state.extracted_text = state.description
        return state

    try:
        ocr_service: OCRService = services["ocr"]
        text = ocr_service.extract_text(state.uploaded_file_path)
        state.extracted_text = text
        logger.info(f"Extracted {len(text)} characters from file")
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        state.errors.append(f"OCR error: {str(e)}")
        # Fallback to description
        state.extracted_text = state.description

    return state


async def analyze_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 2: Analyze requirements and extract style profile

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with analysis and style_profile
    """
    logger.info("=== Analysis Node ===")

    # Combine description and extracted text
    full_text = f"{state.description}\n\n{state.extracted_text}".strip()

    try:
        # Use LLM to analyze requirements
        llm_service: LLMService = services["llm"]

        analysis_prompt = f"""
Analyze these SAT question requirements:

{full_text}

Extract:
1. Category (e.g., algebra, geometry, statistics)
2. Target difficulty (0-100 scale, default 50)
3. Style characteristics
4. Key requirements

Return in this JSON format:
{{
  "category": "algebra",
  "difficulty": 50.0,
  "style": "description of style",
  "characteristics": ["characteristic1", "characteristic2"],
  "example_structure": "pattern found",
  "num_questions": {state.num_questions}
}}
"""

        # Simple parsing for now - in production use structured output
        state.analysis = QuestionAnalysis(
            category="algebra",  # Default
            difficulty=state.target_difficulty or 50.0,
            style="standard",
            characteristics=["clear", "concise"],
            example_structure="standard SAT format",
            num_questions=state.num_questions,
        )

        # Extract style profile if we have examples
        if state.extracted_text:
            style_analyzer: StyleAnalyzer = services["style_analyzer"]
            # Split into example questions (simplified)
            examples = [state.extracted_text]
            state.style_profile = style_analyzer.analyze(examples)
            logger.info(f"Style profile extracted: {state.style_profile.context_type}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        state.errors.append(f"Analysis error: {str(e)}")

    return state


async def search_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 3: Search question bank for similar real questions

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with real_questions
    """
    logger.info("=== Question Bank Search Node ===")

    try:
        qbank: QuestionBankService = services["question_bank"]

        # Build search query
        search_query = f"{state.description} {state.extracted_text}".strip()

        # Search for similar questions
        real_questions = await qbank.search_similar(
            query=search_query[:500],  # Limit query length
            category=state.analysis.category if state.analysis else None,
            difficulty_range=(
                (state.analysis.difficulty - 15, state.analysis.difficulty + 15)
                if state.analysis
                else None
            ),
            top_k=state.num_questions * 2,  # Get extra for filtering
        )

        state.real_questions = real_questions
        logger.info(f"Found {len(real_questions)} similar real questions")

    except Exception as e:
        logger.error(f"Question bank search failed: {e}")
        state.errors.append(f"Search error: {str(e)}")

    return state


async def generate_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 4: Generate new questions using LLM

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with generated_candidates
    """
    logger.info("=== Generation Node ===")

    try:
        llm_service: LLMService = services["llm"]

        total_requested = max(state.num_questions, 1)
        real_target = total_requested // 2
        synthetic_target = total_requested - real_target

        if not state.use_hybrid or not state.real_questions:
            logger.warning(
                "Hybrid generation unavailable (use_hybrid=%s, real_questions=%d). "
                "Falling back to full synthetic generation.",
                state.use_hybrid,
                len(state.real_questions),
            )
            state.hybrid_targets = {}

            fallback_generated = await llm_service.generate_questions(
                prompt=_build_generation_brief(state),
                num_questions=state.num_questions * 3,
                category=state.analysis.category if state.analysis else "algebra",
                difficulty=state.analysis.difficulty if state.analysis else 50.0,
                style_profile=state.style_profile,
                use_both_models=True,
            )

            state.generated_candidates = fallback_generated
            logger.info("Generated %d synthetic fallback questions", len(fallback_generated))
            return state

        state.hybrid_targets = {
            "real": real_target,
            "generated": synthetic_target,
        }

        max_real_candidates = max(real_target * 3, total_requested)
        state.real_questions = state.real_questions[:max_real_candidates]

        synthetic_candidates: List[SATQuestion] = []
        templates = state.real_questions[: min(len(state.real_questions), 3)]

        if synthetic_target > 0 and templates:
            variations_per_template = max(
                1, math.ceil((synthetic_target * 2) / len(templates))
            )

            for template in templates:
                variations = await llm_service.generate_variations(
                    template=template,
                    num_questions=variations_per_template,
                    style_profile=state.style_profile,
                )
                synthetic_candidates.extend(variations)
        else:
            logger.warning("Insufficient templates for variation generation; skipping hybrid mix")

        desired_pool = max(synthetic_target * 2, state.num_questions)
        if len(synthetic_candidates) < desired_pool:
            extra_needed = desired_pool - len(synthetic_candidates)
            logger.info(
                "Generating %d additional synthetic questions to reach target pool",
                extra_needed,
            )
            extra = await llm_service.generate_questions(
                prompt=_build_generation_brief(state),
                num_questions=extra_needed,
                category=state.analysis.category if state.analysis else "algebra",
                difficulty=state.analysis.difficulty if state.analysis else 50.0,
                style_profile=state.style_profile,
                use_both_models=True,
            )
            synthetic_candidates.extend(extra)

        state.generated_candidates = synthetic_candidates
        logger.info(
            "Hybrid generation ready: %d real templates, %d synthetic candidates (target real=%d, synthetic=%d)",
            len(state.real_questions),
            len(state.generated_candidates),
            real_target,
            synthetic_target,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        state.errors.append(f"Generation error: {str(e)}")

    return state


async def validate_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 5: Validate questions for correctness and clarity

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with validated_questions
    """
    logger.info("=== Validation Node ===")

    validated = []
    agreement_scores: List[float] = []
    llm_service: LLMService = services["llm"]

    # Validate generated questions
    for question in state.generated_candidates:
        try:
            result = await llm_service.validate_question(question)
            agreement_scores.append(result.agreement)
            logger.debug(
                "Validator agreement for %s: %.1f%%",
                question.id,
                result.agreement * 100,
            )

            if result.is_valid:
                validated.append(question)
            else:
                logger.debug(
                    "Question %s failed validation: %s", question.id, result.feedback
                )

        except Exception as e:
            logger.warning(f"Validation error for {question.id}: {e}")
            # Include question anyway if validation fails
            validated.append(question)

    state.validated_questions = validated
    logger.info(
        f"Validation complete: {len(validated)}/{len(state.generated_candidates)} passed"
    )

    if agreement_scores:
        avg_agreement = sum(agreement_scores) / len(agreement_scores)
        logger.info(f"Average validator agreement: {avg_agreement * 100:.1f}%")

    return state


async def filter_node(state: GraphState, services: Dict[str, Any]) -> GraphState:
    """
    Node 6: Apply quality filters (style, difficulty, duplication)

    This is the core quality control node that applies all three MVP features:
    1. Style Matching
    2. Difficulty Calibration
    3. Anti-Duplication

    Args:
        state: Current graph state
        services: Dictionary of initialized services

    Returns:
        Updated state with final_questions
    """
    logger.info("=== Quality Filtering Node ===")

    candidates = state.validated_questions.copy()

    # Get services
    style_matcher: StyleMatcher = services["style_matcher"]
    difficulty_calibrator: DifficultyCalibrator = services["difficulty_calibrator"]
    duplication_detector: DuplicationDetector = services["duplication_detector"]

    # Convert real questions to SATQuestion format for filtering
    real_as_sat = []
    for rq in state.real_questions:
        sat_q = SATQuestion(
            id=rq.question_id,
            question=rq.question_text,
            choices=rq.choices,
            correct_answer=rq.correct_answer,
            explanation=rq.explanation,
            difficulty=rq.difficulty,
            category=rq.category,
            subcategory=rq.subcategory,
            is_real=True,
        )
        real_as_sat.append(sat_q)

    # Combine real and generated questions
    all_candidates = real_as_sat + candidates

    logger.info(
        f"Starting with {len(all_candidates)} candidates "
        f"({len(real_as_sat)} real, {len(candidates)} generated)"
    )

    # Filter 1: Style Matching (if style profile available)
    if state.style_profile:
        from ..utils.config import get_settings

        settings = get_settings()
        all_candidates = style_matcher.filter_by_style(
            all_candidates, state.style_profile, threshold=settings.style_match_threshold
        )
        logger.info(f"After style matching: {len(all_candidates)} candidates")

    # Filter 2: Difficulty Calibration (if target difficulty specified)
    if state.analysis and state.analysis.difficulty:
        from ..utils.config import get_settings

        settings = get_settings()
        target_difficulty = float(state.analysis.difficulty)
        pre_calibration = all_candidates[:]
        all_candidates = difficulty_calibrator.calibrate_questions(
            pre_calibration,
            target_difficulty=target_difficulty,
            tolerance=settings.difficulty_tolerance,
        )

        # Fallback: if calibration removed everything, keep closest matches
        if not all_candidates and pre_calibration:
            logger.warning(
                "Difficulty calibration removed all candidates; "
                "falling back to closest matches"
            )

            predictions = difficulty_calibrator.predict_batch(pre_calibration)
            scored = sorted(
                zip(pre_calibration, predictions),
                key=lambda qp: abs(qp[1] - target_difficulty),
            )

            fallback_count = min(
                len(scored),
                max(state.num_questions * 2, state.num_questions),
            )

            all_candidates = []
            fallback_slice = scored[:fallback_count]
            for question, predicted in fallback_slice:
                question.difficulty = predicted
                all_candidates.append(question)

            max_deviation = (
                max(abs(pred - target_difficulty) for _, pred in fallback_slice)
                if fallback_slice
                else 0.0
            )

            logger.info(
                "Difficulty fallback kept %d questions (max deviation %.1f)",
                len(all_candidates),
                max_deviation,
            )

        logger.info(f"After difficulty calibration: {len(all_candidates)} candidates")

    # Filter 3: Anti-Duplication
    from ..utils.config import get_settings

    settings = get_settings()
    all_candidates = duplication_detector.filter_duplicates(
        all_candidates, threshold=settings.duplication_threshold
    )
    logger.info(f"After deduplication: {len(all_candidates)} candidates")

    # Rank by style match score
    if state.style_profile:
        all_candidates = style_matcher.rank_by_style(
            all_candidates, state.style_profile
        )

    mix_targets = getattr(state, "hybrid_targets", {}) or {}
    target_real = mix_targets.get("real")
    target_generated = mix_targets.get("generated")

    if (
        target_real is not None
        and target_generated is not None
        and state.num_questions > 0
    ):
        real_pool = [q for q in all_candidates if q.is_real]
        synthetic_pool = [q for q in all_candidates if not q.is_real]

        selected_real = real_pool[: target_real]
        selected_generated = synthetic_pool[: target_generated]
        combined = selected_real + selected_generated

        if len(combined) < state.num_questions:
            remaining = [q for q in all_candidates if q not in combined]
            combined.extend(remaining[: state.num_questions - len(combined)])

        state.final_questions = combined[: state.num_questions]
    else:
        final_count = min(state.num_questions, len(all_candidates))
        state.final_questions = all_candidates[:final_count]

    # Calculate metadata
    real_count = sum(1 for q in state.final_questions if q.is_real)
    generated_count = len(state.final_questions) - real_count

    state.metadata = {
        "total_questions": len(state.final_questions),
        "real_questions": real_count,
        "generated_questions": generated_count,
        "avg_difficulty": (
            sum(q.difficulty for q in state.final_questions) / len(state.final_questions)
            if state.final_questions
            else 0
        ),
        "avg_style_match": (
            sum(q.style_match_score or 0 for q in state.final_questions)
            / len(state.final_questions)
            if state.final_questions
            else 0
        ),
        "target_mix": mix_targets,
        "filters_applied": ["style_matching", "difficulty_calibration", "deduplication"],
    }

    logger.info(
        f"✅ Final selection: {len(state.final_questions)} questions "
        f"({real_count} real, {generated_count} generated)"
    )

    return state

