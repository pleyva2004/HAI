"""
State schema definition for the LangGraph workflow.

This module defines the QuestionGenerationState class, which represents
the state that flows through the entire SAT question generation workflow.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class UserOptions(BaseModel):

    requested_section: Optional[Literal["Math", "Reading and Writing"]] = "Math"
    requested_difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = None
    requested_domain: Optional[Literal[
        "Algebra",
        "Advanced Math",
        "Problem-Solving & Data Analysis",
        "Geometry & Trigonometry"
    ]] = None
    output_format: Literal["latex-hardcoded"] = "latex-hardcoded"
    provide_answer: bool = False
    file_out: bool = False

class Question(BaseModel):

    id: Optional[str] = None
    section: Literal["Math", "Reading and Writing"]
    domain: Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]
    skill: List[str] = []
    difficulty: Literal["Easy", "Medium", "Hard"]

    question_text: str = Field(..., description="The text of the question")
    equation_content: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table_data: Optional[Dict[str, Any]] = None
    visual_description: Optional[str] = None

    answer_choices: Optional[Dict[str, str]] = None
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None
    explanation: Optional[str] = None

    source: Optional[str] = None
    created_at: Optional[str] = None
    original_image_url: Optional[str] = None

class QuestionGenerationState(BaseModel):

    # USER INPUT (from user)
    user_image: Optional[str] = None
    user_description: Optional[str] = ""
    user_options: UserOptions = UserOptions()

    # EXTRACTED FEATURES (from extract_structure node)
    extracted_text: Optional[str] = ""
    equation_content: Optional[str] = ""
    table_data: Optional[Dict[str, Any]] = {}
    visual_description: Optional[str] = ""

    # CLASSIFICATION (from classify_question node)
    section: Optional[Literal["Math", "Reading and Writing"]] = "Math"
    predicted_domain: Optional[Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]] = "Algebra"
    skill: List[str] = []
    predicted_difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = "Medium"

    # RETRIEVED CONTEXT (from retrieve_examples node)
    similar_questions: List[Question] = []
    retrieval_scores: List[float] = []

    # GENERATED OUTPUT (from generate_question node)
    generated_question: Optional[Question] = None
    generation_attempt: int = 0

    # VALIDATION (from validate_output node)
    validation_passed: bool = False
    validation_errors: List[str] = []

    # WORKFLOW METADATA
    workflow_id: str = ""
    started_at: str = ""
    error: Optional[str] = None

    def increment_generation_attempt(self) -> None:
        self.generation_attempt += 1

    def add_validation_error(self, error: str) -> None:
        self.validation_errors.append(error)

    def mark_validation_passed(self) -> None:
        self.validation_passed = True
        self.validation_errors = []

    @classmethod
    def create_initial_state(
        cls,
        user_image: Optional[str] = None,
        user_description: Optional[str] = None,
        user_options: Optional[UserOptions] = None
    ) -> "QuestionGenerationState":
        return cls(
            user_image=user_image,
            user_description=user_description,
            user_options=user_options or UserOptions(),
            workflow_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat()
        )
