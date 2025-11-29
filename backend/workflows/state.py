"""
State schema definition for the LangGraph workflow.

This module defines the QuestionGenerationState class, which represents
the state that flows through the entire SAT question generation workflow.
"""

from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class UserOptions(BaseModel):
    requested_section: Optional[Literal["Math", "Reading and Writing"]] = None
    requested_difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = None
    requested_domain: Optional[Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]] = None
    output_format: Literal["latex-hardcoded"] = "latex-hardcoded"
    provide_answer: bool = False
    file_out: bool = False

class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class BaseQuestion(BaseModel):

    id: str
    section: Literal["Math", "Reading and Writing"]
    domain: Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]
    skill: List[str]
    difficulty: Literal["Easy", "Medium", "Hard"]

    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None

    answer_choices: Optional[Dict[str, str]] = None
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None
    explanation: Optional[str] = None

    source: Optional[str] = None
    created_at: Optional[str] = None
    original_image_url: Optional[str] = None

class MathQuestionExtraction(BaseModel):
    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None

class GeneratedQuestion(BaseModel):
    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None

    answer_choices: Optional[Dict[str, str]] = None # {"A": "...", "B": "...", etc.}
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None
    explanation: Optional[str] = None

class QuestionClassification(BaseModel):
    section: Literal["Math", "Reading and Writing"]
    domain: Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]
    skill: List[str]
    difficulty: Literal["Easy", "Medium", "Hard"]

class QuestionGenerationState(BaseModel):

    # USER INPUT (from user)
    user_image: str
    user_description: str
    user_options: UserOptions = UserOptions()

    # EXTRACTED FEATURES (from extract_structure node)
    extracted_features: Optional[MathQuestionExtraction]

    # CLASSIFICATION (from classify_question node)
    classified_features: Optional[QuestionClassification]

    # RETRIEVED CONTEXT (from retrieve_examples node)
    # TODO: Implement this will changed in the future based on the retrieval service implementation
    similar_questions: List[BaseQuestion]
    retrieval_scores: List[float]

    # GENERATED OUTPUT (from generate_question node)
    generated_question: Optional[GeneratedQuestion]
    generation_attempt: int

    # VALIDATION (from validate_output node)
    validation_passed: bool
    validation_errors: List[str]

    # WORKFLOW METADATA
    workflow_id: str
    started_at: str
    error: Optional[str]

    def increment_generation_attempt(self) -> None:
        self.generation_attempt += 1

    def add_validation_error(self, error: str) -> None:
        self.validation_errors.append(error)

    def mark_validation_passed(self) -> None:
        self.validation_passed = True
        self.validation_errors = []

    @classmethod
    def create_initial_state(cls, user_image: str, user_description: str, user_options: UserOptions) -> "QuestionGenerationState":
        return cls(
            user_image=user_image,
            user_description=user_description,
            user_options=user_options,
            extracted_features= None,
            classified_features= None,
            similar_questions= [],
            retrieval_scores= [],
            generated_question= None,
            generation_attempt= 0,
            validation_passed= False,
            validation_errors= [],
            workflow_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc).isoformat()
        )
