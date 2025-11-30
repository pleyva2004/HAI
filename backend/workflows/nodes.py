"""
Workflow node implementations for the SAT question generation pipeline.

Each node is a thin orchestration layer that calls services and updates state.
Nodes should be atomic and delegate actual work to the services layer.
"""

from backend.workflows.state import QuestionGenerationState, BaseQuestion, GeneratedQuestion, MathQuestionExtraction
from typing import Optional
from backend.services import claude, embeddings, validation, retrieval


def extract_structure(state: QuestionGenerationState) -> QuestionGenerationState:

    # Check what input we have
    extracted_features: Optional[MathQuestionExtraction] = None

    if state.user_image:
        # Extract from image using Claude Vision
        extracted_features = claude.extract_from_image(state.user_image)

    elif state.user_description:
        # Extract from text description
        extracted_features = claude.extract_from_description(state.user_description)

    if extracted_features is None:
        state.error = "No image or description provided"
    else:
        state.extracted_features = extracted_features

    return state

    # Get classification from Claude
    result = claude.classify_question(
        extracted_text=state.extracted_text,
        equation_content=state.equation_content,
        visual_description=state.visual_description
    )

    # Apply user preferences if provided, otherwise use classified values
    requested_section = state.user_options.requested_section
    if requested_section:
        state.section = requested_section
    else:
        state.section = result.get("section")

    requested_domain = state.user_options.requested_domain
    if requested_domain:
        state.predicted_domain = requested_domain
    else:
        state.predicted_domain = result.get("domain")

    requested_difficulty = state.user_options.requested_difficulty
    if requested_difficulty:
        state.predicted_difficulty = requested_difficulty
    else:
        state.predicted_difficulty = result.get("difficulty")

    state.skill = result.get("skill", [])

    return state


def retrieve_examples(state: QuestionGenerationState) -> QuestionGenerationState:
    """
    Find similar questions from the question bank.

    Delegates to embeddings service for query building and embedding generation,
    then to retrieval service for database search.
    """

    # Build query text from extracted features
    query_text = embeddings.build_query_text(
        extracted_text=state.extracted_text,
        equation_content=state.equation_content,
        visual_description=state.visual_description,
        skills=state.skill
    )

    # Generate embedding for the query
    query_embedding = embeddings.generate_embedding(query_text)

    # Retrieve similar questions from database
    questions, scores = retrieval.retrieve_similar_questions(
        embedding=query_embedding,
        section=state.section,
        domain=state.predicted_domain,
        difficulty=state.predicted_difficulty
    )

    state.similar_questions = questions
    state.retrieval_scores = scores

    return state


def generate_question(state: QuestionGenerationState) -> QuestionGenerationState:
    """
    Generate a new SAT question.

    Delegates to claude service for question generation.
    """

    # Increment attempt counter
    state.increment_generation_attempt()

    # Generate question using Claude
    result = claude.generate_question(
        user_description=state.user_description,
        section=state.section,
        domain=state.predicted_domain,
        difficulty=state.predicted_difficulty,
        skills=state.skill,
        similar_questions=state.similar_questions
    )

    # Create Question object from result
    # Use defaults if state doesn't have required values
    section = state.section if state.section else "Math"
    domain = state.predicted_domain if state.predicted_domain else "Algebra"
    difficulty = state.predicted_difficulty if state.predicted_difficulty else "Medium"
    question_text = result.get("question_text", "")

    question = GeneratedQuestion(
        section=section,
        domain=domain,
        difficulty=difficulty,
        question_text=question_text,
        extracted_features=state.extracted_features
    )

    state.generated_question = question

    return state


def validate_output(state: QuestionGenerationState) -> QuestionGenerationState:
    """
    Validate the generated question.

    Delegates to validation service for quality checks.
    """

    # Validate the question
    is_valid, errors = validation.validate_question(state.generated_question)

    # Update state with validation results
    if is_valid:
        state.mark_validation_passed()
    else:
        state.validation_passed = False
        state.validation_errors = errors

    return state
