"""LangGraph workflow definition and orchestration"""

import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, END
from ..models.toon_models import GraphState
from ..utils.config import get_settings
from ..services.ocr_service import OCRService
from ..services.question_bank import QuestionBankService
from ..services.llm_service import LLMService
from ..services.style_analyzer import StyleAnalyzer, StyleMatcher
from ..services.difficulty_calibrator import DifficultyCalibrator
from ..services.duplication_detector import DuplicationDetector

from .nodes import (
    ocr_node,
    analyze_node,
    search_node,
    generate_node,
    validate_node,
    filter_node,
)

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """
    Create and compile the LangGraph workflow

    Workflow structure:
    OCR → Analyze → Search → Generate → Validate → Filter → END

    Returns:
        Compiled StateGraph
    """
    # Create graph with GraphState schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("ocr_extraction", ocr_node)
    workflow.add_node("analyze_requirements", analyze_node)
    workflow.add_node("search_question_bank", search_node)
    workflow.add_node("generate_questions", generate_node)
    workflow.add_node("validate_questions", validate_node)
    workflow.add_node("filter_questions", filter_node)

    # Define edges (linear flow)
    workflow.set_entry_point("ocr_extraction")
    workflow.add_edge("ocr_extraction", "analyze_requirements")
    workflow.add_edge("analyze_requirements", "search_question_bank")
    workflow.add_edge("search_question_bank", "generate_questions")
    workflow.add_edge("generate_questions", "validate_questions")
    workflow.add_edge("validate_questions", "filter_questions")
    workflow.add_edge("filter_questions", END)

    # Compile graph
    graph = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")
    return graph


async def run_generation_workflow(
    description: str = "",
    uploaded_file_path: str = "",
    num_questions: int = 5,
    target_difficulty: float = None,
    prefer_real_questions: bool = False,
) -> GraphState:
    """
    Run the complete question generation workflow

    Args:
        description: Text description of requirements
        uploaded_file_path: Path to uploaded file (optional)
        num_questions: Number of questions to generate
        target_difficulty: Target difficulty 0-100 (optional)
        prefer_real_questions: Prefer real questions over generated (optional)

    Returns:
        Final GraphState with generated questions

    Raises:
        Exception: If workflow execution fails
    """
    logger.info("🚀 Starting SAT Question Generation Workflow")
    logger.info(f"  - Questions requested: {num_questions}")
    logger.info(f"  - Target difficulty: {target_difficulty or 'auto'}")
    logger.info(f"  - File uploaded: {'yes' if uploaded_file_path else 'no'}")

    # Initialize services
    settings = get_settings()

    services = await initialize_services(settings)

    try:
        # Create initial state
        initial_state = GraphState(
            description=description,
            uploaded_file_path=uploaded_file_path,
            num_questions=num_questions,
            target_difficulty=target_difficulty,
            prefer_real_questions=prefer_real_questions,
        )

        # Create and run workflow
        workflow = create_workflow()

        # Execute workflow
        # Note: We need to pass services through state or as config
        # For simplicity, we'll pass through node functions
        final_state = initial_state

        # Execute nodes sequentially (passing services)
        final_state = await ocr_node(final_state, services)
        final_state = await analyze_node(final_state, services)
        final_state = await search_node(final_state, services)
        final_state = await generate_node(final_state, services)
        final_state = await validate_node(final_state, services)
        final_state = await filter_node(final_state, services)

        # Log results
        logger.info("✅ Workflow completed successfully")
        logger.info(f"  - Final questions: {len(final_state.final_questions)}")
        logger.info(f"  - Real questions: {final_state.metadata.get('real_questions', 0)}")
        logger.info(
            f"  - Generated questions: {final_state.metadata.get('generated_questions', 0)}"
        )
        logger.info(
            f"  - Avg difficulty: {final_state.metadata.get('avg_difficulty', 0):.1f}"
        )
        logger.info(
            f"  - Avg style match: {final_state.metadata.get('avg_style_match', 0):.3f}"
        )

        if final_state.errors:
            logger.warning(f"  - Errors encountered: {len(final_state.errors)}")
            for error in final_state.errors:
                logger.warning(f"    • {error}")

        return final_state

    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        raise

    finally:
        # Cleanup services
        await cleanup_services(services)


async def initialize_services(settings) -> Dict[str, Any]:
    """
    Initialize all required services

    Args:
        settings: Application settings

    Returns:
        Dictionary of initialized services
    """
    logger.info("Initializing services...")

    # OCR Service
    ocr_service = OCRService()

    # Question Bank Service
    qbank_service = QuestionBankService(
        database_url=settings.database_url,
        embedding_model=settings.embedding_model_name,
    )
    await qbank_service.connect()

    # LLM Service
    llm_service = LLMService(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
    )

    # Style Analyzer & Matcher
    style_analyzer = StyleAnalyzer()
    style_matcher = StyleMatcher()

    # Difficulty Calibrator
    difficulty_calibrator = DifficultyCalibrator()

    # Try to load pre-trained model
    try:
        difficulty_calibrator.load_model(settings.difficulty_model_path)
        logger.info("Loaded pre-trained difficulty model")
    except FileNotFoundError:
        logger.warning("No pre-trained difficulty model found, using fallback")

    # Duplication Detector
    duplication_detector = DuplicationDetector(
        embedding_model=settings.embedding_model_name
    )

    services = {
        "ocr": ocr_service,
        "question_bank": qbank_service,
        "llm": llm_service,
        "style_analyzer": style_analyzer,
        "style_matcher": style_matcher,
        "difficulty_calibrator": difficulty_calibrator,
        "duplication_detector": duplication_detector,
    }

    logger.info("✅ All services initialized")
    return services


async def cleanup_services(services: Dict[str, Any]):
    """
    Clean up service connections

    Args:
        services: Dictionary of services
    """
    logger.info("Cleaning up services...")

    # Disconnect question bank
    if "question_bank" in services:
        await services["question_bank"].disconnect()

    logger.info("✅ Cleanup complete")

