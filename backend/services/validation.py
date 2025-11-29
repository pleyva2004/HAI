"""
Question validation service.

Validates that generated questions meet quality standards.
"""

from typing import List, Tuple, Optional

from backend.workflows.state import Question


def validate_question(question: Optional[Question]) -> Tuple[bool, List[str]]:
    """
    Validate a generated question for completeness and correctness.

    Returns:
        (is_valid, error_list)
        - is_valid: True if question passes all checks
        - error_list: List of error messages (empty if valid)
    """

    errors = []

    # Check if question exists
    if not question:
        errors.append("No question was generated")
        return False, errors

    # Check for question text
    if not question.question_text:
        errors.append("Missing question text")

    # Check for answer choices
    if question.answer_choices:
        # Verify we have exactly 4 choices
        num_choices = len(question.answer_choices)
        if num_choices != 4:
            errors.append(f"Expected 4 answer choices, got {num_choices}")

        # Verify choices are labeled A, B, C, D
        expected_labels = ["A", "B", "C", "D"]
        actual_labels = list(question.answer_choices.keys())

        for expected_label in expected_labels:
            if expected_label not in actual_labels:
                errors.append(f"Missing answer choice: {expected_label}")

        # Check that each choice has content
        for choice_label, choice_text in question.answer_choices.items():
            if not choice_text:
                errors.append(f"Answer choice {choice_label} is empty")

    # Check for correct answer
    if not question.correct_answer:
        errors.append("Missing correct answer")
    else:
        # Verify correct answer is one of the choices
        if question.answer_choices:
            valid_answers = ["A", "B", "C", "D"]
            if question.correct_answer not in valid_answers:
                errors.append(f"Invalid correct answer: {question.correct_answer}")

    # Check for explanation
    if not question.explanation:
        errors.append("Missing explanation")

    # Determine if valid
    is_valid = len(errors) == 0

    return is_valid, errors
