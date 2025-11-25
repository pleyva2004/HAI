"""Pydantic models for type-safe LLM responses and state management"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel


class QuestionChoice(BaseModel):
    """Answer choices for a SAT question"""

    A: str
    B: str
    C: str
    D: str


class SATQuestion(BaseModel):
    """A single SAT question (generated or from bank)"""

    id: str
    question: str
    choices: QuestionChoice
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    difficulty: float  # 0-100 scale
    category: str
    subcategory: Optional[str] = None
    predicted_correct_rate: Optional[float] = None
    style_match_score: Optional[float] = None
    is_real: bool = False  # True if from official SAT bank


class OfficialSATQuestion(BaseModel):
    """Real SAT question from the official question bank"""

    question_id: str
    source: str  # e.g., "Practice Test 7, Q14"
    category: str
    subcategory: str
    difficulty: float  # 0-100
    question_text: str
    choices: QuestionChoice
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    national_correct_rate: float  # 0-100 percentage
    avg_time_seconds: int
    common_wrong_answers: List[str]
    tags: List[str]


class GeneratedQuestions(BaseModel):
    """Collection of generated questions"""

    questions: List[SATQuestion]
    metadata: Optional[Dict[str, Any]] = None


class StyleProfile(BaseModel):
    """Extracted style characteristics from example questions"""

    word_count_range: Tuple[int, int]  # (min, max)
    vocabulary_level: float  # Flesch-Kincaid grade level
    number_complexity: str  # "fractions", "decimals", "large_integers", "small_integers"
    context_type: str  # "real_world", "abstract", "geometric"
    question_structure: str  # Template structure
    distractor_patterns: str  # How wrong answers are constructed


class QuestionAnalysis(BaseModel):
    """Analysis of user's input requirements"""

    category: str
    difficulty: float  # Target difficulty 0-100
    style: str
    characteristics: List[str]
    example_structure: str
    num_questions: int = 5


class GraphState(BaseModel):
    """LangGraph workflow state container"""

    # Input parameters
    description: str = ""
    uploaded_file_path: str = ""
    num_questions: int = 5
    target_difficulty: Optional[float] = None  # If specified
    prefer_real_questions: bool = False
    use_hybrid: bool = True

    # Intermediate state
    extracted_text: str = ""
    analysis: Optional[QuestionAnalysis] = None
    style_profile: Optional[StyleProfile] = None
    real_questions: List[OfficialSATQuestion] = []
    generated_candidates: List[SATQuestion] = []
    validated_questions: List[SATQuestion] = []

    # Output
    final_questions: List[SATQuestion] = []
    metadata: Dict[str, Any] = {}

    # Error tracking
    errors: List[str] = []

