"""API Routes - Endpoints for question generation"""

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ..graph.workflow import run_generation_workflow
from ..models.toon_models import SATQuestion
from ..services.question_bank import QuestionBankService
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["SAT Questions"])


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for question generation"""

    description: str = Field(..., description="Requirements description")
    num_questions: int = Field(5, ge=1, le=50, description="Number of questions")
    target_difficulty: Optional[float] = Field(
        None, ge=0, le=100, description="Target difficulty (0-100)"
    )
    prefer_real: bool = Field(False, description="Prefer real SAT questions")


class GenerateResponse(BaseModel):
    """Response model for question generation"""

    questions: List[SATQuestion]
    metadata: dict
    errors: List[str] = []


class QuestionSearchRequest(BaseModel):
    """Request model for question search"""

    query: str
    category: Optional[str] = None
    difficulty_min: Optional[float] = Field(None, ge=0, le=100)
    difficulty_max: Optional[float] = Field(None, ge=0, le=100)
    limit: int = Field(10, ge=1, le=100)


class ValidateRequest(BaseModel):
    """Request model for question validation"""

    question: SATQuestion


class ValidateResponse(BaseModel):
    """Response model for validation"""

    is_valid: bool
    feedback: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    services: dict


# Endpoints
@router.post("/generate", response_model=GenerateResponse)
async def generate_questions(
    description: str = Form(..., min_length=1),
    num_questions: int = Form(5, ge=1, le=50),
    target_difficulty: Optional[float] = Form(None, ge=0, le=100),
    prefer_real: bool = Form(False),
    file: Optional[UploadFile] = File(None),
):
    """
    Generate SAT questions

    Accepts either text description or file upload (PDF/image)

    Args:
        description: Requirements description
        num_questions: Number of questions to generate
        target_difficulty: Optional target difficulty (0-100)
        prefer_real: Prefer real questions from bank
        file: Optional uploaded file (PDF or image)

    Returns:
        Generated questions with metadata
    """
    logger.info(
        f"POST /api/v1/generate - {num_questions} questions, "
        f"difficulty={target_difficulty}, file={'yes' if file else 'no'}"
    )

    # Handle file upload
    file_path = ""
    if file:
        # Save file temporarily
        import tempfile
        from pathlib import Path

        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            file_path = tmp.name

        logger.info(f"File uploaded: {file.filename} → {file_path}")

    try:
        # Run workflow
        result = await run_generation_workflow(
            description=description,
            uploaded_file_path=file_path,
            num_questions=num_questions,
            target_difficulty=target_difficulty,
            prefer_real_questions=prefer_real,
        )

        return GenerateResponse(
            questions=result.final_questions,
            metadata=result.metadata,
            errors=result.errors,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}",
        )

    finally:
        # Clean up temp file
        if file_path:
            import os

            try:
                os.unlink(file_path)
            except Exception:
                pass


@router.get("/questions/{question_id}", response_model=SATQuestion)
async def get_question(question_id: str):
    """
    Get a specific question by ID

    Args:
        question_id: Question identifier

    Returns:
        Question details
    """
    settings = get_settings()
    qbank = QuestionBankService(settings.database_url)

    try:
        await qbank.connect()
        question = await qbank.get_by_id(question_id)

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found",
            )

        # Convert to SATQuestion
        return SATQuestion(
            id=question.question_id,
            question=question.question_text,
            choices=question.choices,
            correct_answer=question.correct_answer,
            explanation=question.explanation,
            difficulty=question.difficulty,
            category=question.category,
            is_real=True,
        )

    finally:
        await qbank.disconnect()


@router.post("/search", response_model=List[SATQuestion])
async def search_questions(request: QuestionSearchRequest):
    """
    Search question bank

    Args:
        request: Search parameters

    Returns:
        List of matching questions
    """
    settings = get_settings()
    qbank = QuestionBankService(settings.database_url)

    try:
        await qbank.connect()

        difficulty_range = None
        if request.difficulty_min is not None and request.difficulty_max is not None:
            difficulty_range = (request.difficulty_min, request.difficulty_max)

        questions = await qbank.search_similar(
            query=request.query,
            category=request.category,
            difficulty_range=difficulty_range,
            top_k=request.limit,
        )

        # Convert to SATQuestion
        return [
            SATQuestion(
                id=q.question_id,
                question=q.question_text,
                choices=q.choices,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                difficulty=q.difficulty,
                category=q.category,
                is_real=True,
            )
            for q in questions
        ]

    finally:
        await qbank.disconnect()


@router.post("/validate", response_model=ValidateResponse)
async def validate_question(request: ValidateRequest):
    """
    Validate a question for correctness

    Args:
        request: Question to validate

    Returns:
        Validation result
    """
    from ..services.llm_service import LLMService

    settings = get_settings()
    llm_service = LLMService(
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
    )

    try:
        is_valid, feedback = await llm_service.validate_question(request.question)

        return ValidateResponse(is_valid=is_valid, feedback=feedback)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Service health status
    """
    settings = get_settings()

    # Check database connection
    db_status = "unknown"
    qbank = QuestionBankService(settings.database_url)
    try:
        await qbank.connect()
        count = await qbank.get_count()
        db_status = f"connected ({count} questions)"
        await qbank.disconnect()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "database": db_status,
            "llm": "configured",
            "ocr": "available" if settings.enable_ocr else "disabled",
        },
    )

