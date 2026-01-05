"""
Workflow node implementations for the SAT question generation pipeline.

Each node is a thin orchestration layer that calls services and updates state.
Nodes should be atomic and delegate actual work to the services layer.
"""

from backend.workflows.state import HAIState, MathQuestionExtraction
from typing import Optional
from toon_format import encode
from backend.services import claude, embeddings, validation, retrieval
from backend.config import MAX_GENERATION_ATTEMPTS


def extract_structure(state: HAIState) -> HAIState:

    # Check what input we have
    extracted_features: Optional[MathQuestionExtraction] = None

    if state.user_input.image:
        # Extract from image using Claude Vision
        extracted_features = claude.extract_from_image(state.user_input.image)

    elif state.user_input.description:
        # Extract from text description
        extracted_features = claude.extract_from_description(state.user_input.description)

    if extracted_features is None:
        state.error = "No image or description provided"
    else:
        state.extracted_features = extracted_features

    return state

def classify_question(state: HAIState) -> HAIState:
    if state.extracted_features is None:
        state.error = "No Extracted features to classify"
        return state

    encoded_extracted_data = encode(state.extracted_features.model_dump())

    res = claude.classify_question(encoded_extracted_data)

    state.classified_features = res

    return state

def retrieve_examples(state: HAIState) -> HAIState:
    """
    Find similar questions from the question bank.

    Delegates to embeddings service for query building and embedding generation,
    then to retrieval service for database search.
    """

    # Build query text from extracted features
    query_text = embeddings.build_query_text(
        extracted_text=state.extracted_features.text if state.extracted_features else None,
        equation_content=state.extracted_features.equation if state.extracted_features else None,
        visual_description=state.extracted_features.visual if state.extracted_features else None,
        skills=state.classified_features.skill if state.classified_features else None
    )

    # Generate embedding for the query
    query_embedding = embeddings.generate_embedding(query_text)

    # Retrieve similar questions from database
    questions, scores = retrieval.retrieve_similar_questions(
        embedding=query_embedding,
        section=state.classified_features.section if state.classified_features else None,
        domain=state.classified_features.domain if state.classified_features else None,
        difficulty=state.classified_features.difficulty if state.classified_features else None
    )

    state.similar_questions = questions
    state.retrieval_scores = scores

    return state

def generate_question(state: HAIState) -> HAIState:
    """
    Generate a new SAT question.

    Delegates to claude service for question generation.
    """

    # Increment attempt counter
    state.increment_generation_attempt()

    # Encode features for Claude API
    encoded_extracted_data = encode(state.extracted_features.model_dump()) if state.extracted_features else ""
    encoded_classified_data = encode(state.classified_features.model_dump()) if state.classified_features else ""

    # Generate question using Claude
    result = claude.generate_question(
        extracted_features=encoded_extracted_data,
        classified_features=encoded_classified_data,
        similar_questions=state.similar_questions
    )

    # Result is already a GeneratedQuestion object
    state.generated_question = result

    return state

def validate_output(state: HAIState) -> HAIState:
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


#TODO: Implement CHATGPT Answering question and juding fairness
def is_question_fair(state: HAIState) -> HAIState:
    return state


#TODO: implemented LLM Call to create latex out of the question we generated in-order for the frotned to render tables, equations and visuals nicely.
#### HIGH PRIORITY WILL BE IMPLEMENTED NEXT
def conver_latex(state: HAIState) -> HAIState:
    return state


def should_validate(state: HAIState) -> str:

    # Always validate unless uxer declines
    # To-Do: give user option in the UI
    # skip_validation option not currently in UserInput model
    return "validate"

def validation_decision(state: HAIState) -> str:

    # If validation passed, we're done
    if state.validation_passed:
        return "success"

    # If we've tried too many times, give up
    if state.generation_attempt >= MAX_GENERATION_ATTEMPTS:
        return "failed"

    return "regenerate"
