"""LangGraph workflow nodes"""

import logging
from typing import Any, Dict

from ..models.toon_models import GraphState, QuestionAnalysis, SATQuestion
from ..services.ocr_service import OCRService
from ..services.question_bank import QuestionBankService
from ..services.llm_service import LLMService
from ..services.style_analyzer import StyleAnalyzer, StyleMatcher
from ..services.difficulty_calibrator import DifficultyCalibrator
from ..services.duplication_detector import DuplicationDetector

logger = logging.getLogger(__name__)


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

        # Build generation prompt
        prompt = f"""
Requirements: {state.description}

Generate SAT questions with these characteristics:
- Category: {state.analysis.category if state.analysis else 'algebra'}
- Difficulty: {state.analysis.difficulty if state.analysis else 50}/100
- Style: Match the examples provided
"""

        # Generate questions
        generated = await llm_service.generate_questions(
            prompt=prompt,
            num_questions=state.num_questions * 3,  # Generate extra for filtering
            category=state.analysis.category if state.analysis else "algebra",
            difficulty=state.analysis.difficulty if state.analysis else 50.0,
            style_profile=state.style_profile,
            use_both_models=True,
        )

        state.generated_candidates = generated
        logger.info(f"Generated {len(generated)} candidate questions")

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
    llm_service: LLMService = services["llm"]

    # Validate generated questions
    for question in state.generated_candidates:
        try:
            is_valid, feedback = await llm_service.validate_question(question)

            if is_valid:
                validated.append(question)
            else:
                logger.debug(f"Question {question.id} failed validation: {feedback}")

        except Exception as e:
            logger.warning(f"Validation error for {question.id}: {e}")
            # Include question anyway if validation fails
            validated.append(question)

    state.validated_questions = validated
    logger.info(
        f"Validation complete: {len(validated)}/{len(state.generated_candidates)} passed"
    )

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

    # Rank by style match score and select top N
    if state.style_profile:
        all_candidates = style_matcher.rank_by_style(
            all_candidates, state.style_profile
        )

    # Select final questions
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
        "filters_applied": ["style_matching", "difficulty_calibration", "deduplication"],
    }

    logger.info(
        f"✅ Final selection: {len(state.final_questions)} questions "
        f"({real_count} real, {generated_count} generated)"
    )

    return state

