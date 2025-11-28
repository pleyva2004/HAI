# SAT Question Generator - Backend Codebase

## Overview

This document describes the backend architecture for the SAT Practice Question Generator MVP. The backend handles multimodal input processing, question generation using LangGraph workflows, and database operations.

---

## Folder Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── config.py              # Configuration settings
│
├── api/
│   ├── __init__.py
│   └── routes.py          # All API endpoints
│
├── database/
│   ├── __init__.py
│   ├── connection.py      # Database connection setup
│   ├── models.py          # SQLAlchemy models (questions table)
│   └── queries.py         # Database query functions
│
├── workflows/
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow definition
│   ├── nodes.py           # All workflow nodes
│   └── state.py           # State schema definition
│
├── services/
│   ├── __init__.py
│   ├── claude.py          # Claude API calls
│   ├── embeddings.py      # Embedding generation
│   └── validation.py      # Question validation logic
│
└── scripts/
    ├── __init__.py
    └── populate_db.py     # Script to populate question bank
```

---

## Design Principles

### 1. Flat Structure
- No deep nesting
- Easy to find any file
- Maximum of 2 levels deep

### 2. Clear Separation of Concerns
- `api/` = HTTP endpoints only
- `workflows/` = LangGraph logic only  
- `services/` = Reusable functions (Claude, embeddings, validation)
- `database/` = All database code
- `scripts/` = One-time setup scripts

### 3. Simple Files
- `main.py` = Just FastAPI app setup
- `routes.py` = All endpoints in one file (MVP scope)
- `nodes.py` = All 5 workflow nodes in one file
- `queries.py` = All database queries in one file

### 4. No Over-Engineering
- No separate "schemas" folder (use Pydantic in routes.py)
- No "utils" folder (put helpers in the service they belong to)
- No "constants" folder (put them in config.py)
- No premature abstractions

---

## File Descriptions

### Root Level Files

#### `main.py`
**Purpose**: FastAPI application entry point

**Responsibilities**:
- Create FastAPI app instance
- Configure CORS for frontend
- Initialize database on startup
- Include API routes
- Run uvicorn server

**Example**:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from database.connection import init_database

app = FastAPI(title="SAT Question Generator")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
@app.on_event("startup")
async def startup():
    await init_database()

# Routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### `config.py`
**Purpose**: Centralized configuration

**Responsibilities**:
- Load environment variables
- Define all configuration constants
- Provide configuration values to other modules

**Example**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Claude
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "sat-question-generator"

# Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Workflow
MAX_GENERATION_ATTEMPTS = 3
RETRIEVAL_LIMIT = 5
SIMILARITY_THRESHOLD = 0.75
```

#### `requirements.txt`
**Purpose**: Python dependencies

**Contents**:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-dotenv==1.0.0
anthropic==0.18.0
openai==1.12.0
langgraph==0.0.20
langchain-core==0.1.23
langsmith==0.0.87
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
pgvector==0.2.4
pydantic==2.5.3
```

#### `.env.example`
**Purpose**: Template for environment variables

**Contents**:
```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sat_questions

# Claude
CLAUDE_API_KEY=your_claude_api_key_here

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith (for observability)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=sat-question-generator
LANGCHAIN_TRACING_V2=true
```

---

### `api/` Folder

#### `api/routes.py`
**Purpose**: All HTTP endpoints

**Responsibilities**:
- Define Pydantic request/response models
- Handle HTTP requests
- Call workflow graph
- Return formatted responses
- Handle errors

**Endpoints**:
1. `POST /api/generate` - Generate new question
2. `POST /api/upload-screenshot` - Handle image upload
3. `POST /api/feedback` - Submit question rating
4. `GET /api/questions/similar` - Search similar questions (for debugging)

**Example**:
```python
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Optional
import base64
from workflows.graph import workflow_app
from database.queries import store_generated_question, store_feedback

router = APIRouter()

# Request models
class GenerateRequest(BaseModel):
    image: Optional[str] = None  # base64
    description: Optional[str] = None
    options: dict = {}

class FeedbackRequest(BaseModel):
    question_id: str
    rating: int
    feedback: Optional[str] = None

# Response models
class GenerateResponse(BaseModel):
    id: str
    question: dict
    metadata: dict

@router.post("/generate", response_model=GenerateResponse)
async def generate_question(request: GenerateRequest):
    """Generate a new SAT question"""
    
    # Validate input
    if not request.image and not request.description:
        raise HTTPException(400, "Must provide image or description")
    
    # Initialize workflow state
    initial_state = {
        "user_image": request.image,
        "user_description": request.description,
        "user_options": request.options,
        "generation_attempt": 0,
    }
    
    # Run workflow
    try:
        result = await workflow_app.ainvoke(initial_state)
        
        # Check if workflow succeeded
        if result.get("error"):
            raise HTTPException(500, f"Generation failed: {result['error']}")
        
        if not result.get("validation_passed"):
            error_msg = ", ".join(result.get("validation_errors", []))
            raise HTTPException(500, f"Validation failed: {error_msg}")
        
        # Store in database
        question_id = await store_generated_question(
            question=result["generated_question"],
            source_ids=[q["id"] for q in result["similar_questions"]],
            user_prompt=request.description,
            user_image=request.image
        )
        
        # Return response
        return GenerateResponse(
            id=question_id,
            question=result["generated_question"],
            metadata={
                "workflow_id": result["workflow_id"],
                "similar_questions_used": len(result["similar_questions"]),
                "generation_attempts": result["generation_attempt"]
            }
        )
        
    except Exception as e:
        raise HTTPException(500, f"Workflow error: {str(e)}")

@router.post("/upload-screenshot")
async def upload_screenshot(file: UploadFile):
    """Convert uploaded file to base64"""
    
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    return {"image": base64_image}

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Store user feedback on generated question"""
    
    await store_feedback(
        question_id=request.question_id,
        rating=request.rating,
        feedback=request.feedback
    )
    
    return {"success": True}
```

---

### `database/` Folder

#### `database/connection.py`
**Purpose**: Database connection setup

**Responsibilities**:
- Create PostgreSQL connection
- Initialize pgvector extension
- Create database tables
- Provide connection/session management

**Example**:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from config import DATABASE_URL
from database.models import Base

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)

# Create async session factory
async_session = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def init_database():
    """Initialize database and create tables"""
    
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

async def get_session():
    """Get database session for queries"""
    
    async with async_session() as session:
        yield session
```

#### `database/models.py`
**Purpose**: SQLAlchemy models

**Responsibilities**:
- Define Questions table
- Define GeneratedQuestions table
- Define column types and constraints

**Example**:
```python
from sqlalchemy import Column, String, Integer, TIMESTAMP, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()

class Question(Base):
    """SAT question bank table"""
    
    __tablename__ = "questions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Original content
    original_image_url = Column(Text, nullable=True)
    original_text = Column(Text, nullable=True)
    
    # Classification
    question_type = Column(String(50), nullable=False)
    sat_section = Column(String(20), nullable=False)
    sat_subsection = Column(String(50), nullable=False)
    difficulty = Column(String(10), nullable=False)
    
    # Content fields
    question_text = Column(Text, nullable=False)
    equation_content = Column(Text, nullable=True)
    table_data = Column(JSONB, nullable=True)
    visual_description = Column(Text, nullable=True)
    
    # Answer data
    answer_choices = Column(JSONB, nullable=False)
    correct_answer = Column(String(1), nullable=False)
    explanation = Column(Text, nullable=False)
    
    # Embedding for RAG
    embedding = Column(Vector(1536), nullable=False)
    
    # Metadata
    source = Column(String(100), nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

class GeneratedQuestion(Base):
    """Generated questions table"""
    
    __tablename__ = "generated_questions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Source references
    source_question_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    user_prompt = Column(Text, nullable=True)
    user_image_url = Column(Text, nullable=True)
    
    # Generated content (same structure as Question)
    question_type = Column(String(50), nullable=False)
    sat_section = Column(String(20), nullable=False)
    sat_subsection = Column(String(50), nullable=False)
    difficulty = Column(String(10), nullable=False)
    question_text = Column(Text, nullable=False)
    equation_content = Column(Text, nullable=True)
    table_data = Column(JSONB, nullable=True)
    visual_description = Column(Text, nullable=True)
    answer_choices = Column(JSONB, nullable=False)
    correct_answer = Column(String(1), nullable=False)
    explanation = Column(Text, nullable=False)
    
    # Tracking
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    user_rating = Column(Integer, nullable=True)
    feedback = Column(Text, nullable=True)
```

#### `database/queries.py`
**Purpose**: Database query functions

**Responsibilities**:
- Insert questions into database
- Search similar questions using vector similarity
- Store generated questions
- Store user feedback
- All database operations

**Example**:
```python
from sqlalchemy import select, func
from database.models import Question, GeneratedQuestion
from database.connection import async_session
from typing import List, Dict
import uuid

async def insert_question(question_data: dict) -> str:
    """Insert a new question into the database"""
    
    async with async_session() as session:
        question = Question(**question_data)
        session.add(question)
        await session.commit()
        return str(question.id)

async def search_similar_questions(
    embedding: List[float],
    question_type: str,
    sat_subsection: str,
    difficulty: str,
    limit: int = 5
) -> List[Dict]:
    """Search for similar questions using vector similarity"""
    
    async with async_session() as session:
        # Build query with filters
        query = select(Question).where(
            Question.question_type == question_type,
            Question.sat_subsection == sat_subsection,
            Question.difficulty == difficulty
        ).order_by(
            Question.embedding.cosine_distance(embedding)
        ).limit(limit)
        
        result = await session.execute(query)
        questions = result.scalars().all()
        
        # Convert to dict
        return [
            {
                "id": str(q.id),
                "question_text": q.question_text,
                "equation_content": q.equation_content,
                "table_data": q.table_data,
                "visual_description": q.visual_description,
                "answer_choices": q.answer_choices,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
                "question_type": q.question_type,
                "sat_section": q.sat_section,
                "sat_subsection": q.sat_subsection,
                "difficulty": q.difficulty,
            }
            for q in questions
        ]

async def store_generated_question(
    question: dict,
    source_ids: List[str],
    user_prompt: str,
    user_image: str
) -> str:
    """Store a generated question in the database"""
    
    async with async_session() as session:
        gen_question = GeneratedQuestion(
            source_question_ids=[uuid.UUID(id) for id in source_ids],
            user_prompt=user_prompt,
            user_image_url=user_image,
            question_type=question["metadata"]["type"],
            sat_section=question["metadata"]["section"],
            sat_subsection=question["metadata"]["subsection"],
            difficulty=question["metadata"]["difficulty"],
            question_text=question["question"]["text"],
            equation_content=question["question"].get("equation_latex"),
            table_data=question["question"].get("table_data"),
            visual_description=question["question"].get("visual_description"),
            answer_choices=question["answer_choices"],
            correct_answer=question["correct_answer"],
            explanation=question["explanation"]
        )
        
        session.add(gen_question)
        await session.commit()
        return str(gen_question.id)

async def store_feedback(
    question_id: str,
    rating: int,
    feedback: str
):
    """Store user feedback on a generated question"""
    
    async with async_session() as session:
        query = select(GeneratedQuestion).where(
            GeneratedQuestion.id == uuid.UUID(question_id)
        )
        result = await session.execute(query)
        question = result.scalar_one()
        
        question.user_rating = rating
        question.feedback = feedback
        
        await session.commit()
```

---

### `workflows/` Folder

#### `workflows/state.py`
**Purpose**: State schema definition

**Responsibilities**:
- Define QuestionGenerationState TypedDict
- Document all state fields
- Provide type hints for workflow

**Example**:
```python
from typing import TypedDict, Optional, List
from datetime import datetime

class QuestionGenerationState(TypedDict):
    """State object that flows through the workflow"""
    
    # ═══════════════════════════════════════
    # INPUT (from user)
    # ═══════════════════════════════════════
    user_image: Optional[str]              # base64 encoded screenshot
    user_description: Optional[str]        # text description
    user_options: dict                     # { difficulty, topic, section }
    
    # ═══════════════════════════════════════
    # EXTRACTED FEATURES (from extract_structure)
    # ═══════════════════════════════════════
    extracted_text: Optional[str]
    equation_content: Optional[str]
    table_data: Optional[dict]
    visual_description: Optional[str]
    
    # ═══════════════════════════════════════
    # CLASSIFICATION (from classify_question)
    # ═══════════════════════════════════════
    question_type: Optional[str]
    sat_section: Optional[str]
    sat_subsection: Optional[str]
    difficulty: Optional[str]
    topics: List[str]
    
    # ═══════════════════════════════════════
    # RETRIEVED CONTEXT (from retrieve_examples)
    # ═══════════════════════════════════════
    similar_questions: List[dict]
    retrieval_scores: List[float]
    
    # ═══════════════════════════════════════
    # GENERATED OUTPUT (from generate_question)
    # ═══════════════════════════════════════
    generated_question: Optional[dict]
    generation_attempt: int
    
    # ═══════════════════════════════════════
    # VALIDATION (from validate_output)
    # ═══════════════════════════════════════
    validation_passed: bool
    validation_errors: List[str]
    
    # ═══════════════════════════════════════
    # METADATA
    # ═══════════════════════════════════════
    workflow_id: str
    started_at: str
    error: Optional[str]
```

#### `workflows/nodes.py`
**Purpose**: All workflow node implementations

**Responsibilities**:
- Implement extract_structure node
- Implement classify_question node
- Implement retrieve_examples node
- Implement generate_question node
- Implement validate_output node

**Example**:
```python
from workflows.state import QuestionGenerationState
from services.claude import extract_structure_from_image, classify_question_type, generate_sat_question
from services.embeddings import generate_embedding
from services.validation import validate_question
from database.queries import search_similar_questions
import uuid
from datetime import datetime

async def extract_structure_node(state: QuestionGenerationState) -> QuestionGenerationState:
    """Node 1: Extract structure from user input"""
    
    try:
        # If user provided an image
        if state.get("user_image"):
            result = await extract_structure_from_image(state["user_image"])
            
            state["extracted_text"] = result.get("text")
            state["equation_content"] = result.get("equation")
            state["table_data"] = result.get("table")
            state["visual_description"] = result.get("visual")
        
        # If user only provided description
        elif state.get("user_description"):
            state["extracted_text"] = state["user_description"]
        
        # Set workflow metadata
        state["workflow_id"] = str(uuid.uuid4())
        state["started_at"] = datetime.utcnow().isoformat()
        
        return state
        
    except Exception as e:
        state["error"] = f"Extraction failed: {str(e)}"
        return state

async def classify_question_node(state: QuestionGenerationState) -> QuestionGenerationState:
    """Node 2: Classify question into SAT taxonomy"""
    
    try:
        classification = await classify_question_type(
            text=state.get("extracted_text"),
            equation=state.get("equation_content"),
            visual=state.get("visual_description")
        )
        
        state["question_type"] = classification["question_type"]
        state["sat_section"] = classification["sat_section"]
        state["sat_subsection"] = classification["sat_subsection"]
        state["difficulty"] = classification["difficulty"]
        state["topics"] = classification["topics"]
        
        return state
        
    except Exception as e:
        state["error"] = f"Classification failed: {str(e)}"
        return state

async def retrieve_examples_node(state: QuestionGenerationState) -> QuestionGenerationState:
    """Node 3: Retrieve similar questions from database"""
    
    try:
        # Create query text for embedding
        query_parts = []
        if state.get("extracted_text"):
            query_parts.append(state["extracted_text"])
        if state.get("equation_content"):
            query_parts.append(f"Equation: {state['equation_content']}")
        if state.get("visual_description"):
            query_parts.append(f"Visual: {state['visual_description']}")
        if state.get("topics"):
            query_parts.append(f"Topics: {', '.join(state['topics'])}")
        
        query_text = " ".join(query_parts)
        
        # Generate embedding
        embedding = await generate_embedding(query_text)
        
        # Search database
        similar_questions = await search_similar_questions(
            embedding=embedding,
            question_type=state["question_type"],
            sat_subsection=state["sat_subsection"],
            difficulty=state["difficulty"],
            limit=5
        )
        
        state["similar_questions"] = similar_questions
        
        return state
        
    except Exception as e:
        state["error"] = f"Retrieval failed: {str(e)}"
        return state

async def generate_question_node(state: QuestionGenerationState) -> QuestionGenerationState:
    """Node 4: Generate new question using retrieved examples"""
    
    try:
        generated = await generate_sat_question(
            similar_questions=state["similar_questions"],
            user_description=state.get("user_description"),
            question_type=state["question_type"],
            difficulty=state["difficulty"],
            topics=state["topics"]
        )
        
        state["generated_question"] = generated
        state["generation_attempt"] = state.get("generation_attempt", 0) + 1
        
        return state
        
    except Exception as e:
        state["error"] = f"Generation failed: {str(e)}"
        return state

async def validate_output_node(state: QuestionGenerationState) -> QuestionGenerationState:
    """Node 5: Validate the generated question"""
    
    try:
        validation_result = await validate_question(state["generated_question"])
        
        state["validation_passed"] = validation_result["passed"]
        state["validation_errors"] = validation_result["errors"]
        
        return state
        
    except Exception as e:
        state["error"] = f"Validation failed: {str(e)}"
        state["validation_passed"] = False
        state["validation_errors"] = [str(e)]
        return state
```

#### `workflows/graph.py`
**Purpose**: LangGraph workflow definition

**Responsibilities**:
- Create StateGraph
- Add all nodes
- Define edges and conditional logic
- Compile workflow

**Example**:
```python
from langgraph.graph import StateGraph, END
from workflows.state import QuestionGenerationState
from workflows.nodes import (
    extract_structure_node,
    classify_question_node,
    retrieve_examples_node,
    generate_question_node,
    validate_output_node
)
from config import MAX_GENERATION_ATTEMPTS

# Conditional edge functions
def should_validate(state: QuestionGenerationState) -> str:
    """Decide whether to validate"""
    
    if state.get("error"):
        return "end"
    
    if state["user_options"].get("skip_validation"):
        return "end"
    
    return "validate"

def validation_decision(state: QuestionGenerationState) -> str:
    """Decide whether to regenerate or accept"""
    
    if state.get("error"):
        return "failed"
    
    if state.get("validation_passed"):
        return "success"
    
    if state.get("generation_attempt", 0) >= MAX_GENERATION_ATTEMPTS:
        return "failed"
    
    return "regenerate"

# Create graph
workflow = StateGraph(QuestionGenerationState)

# Add nodes
workflow.add_node("extract_structure", extract_structure_node)
workflow.add_node("classify_question", classify_question_node)
workflow.add_node("retrieve_examples", retrieve_examples_node)
workflow.add_node("generate_question", generate_question_node)
workflow.add_node("validate_output", validate_output_node)

# Set entry point
workflow.set_entry_point("extract_structure")

# Add sequential edges
workflow.add_edge("extract_structure", "classify_question")
workflow.add_edge("classify_question", "retrieve_examples")
workflow.add_edge("retrieve_examples", "generate_question")

# Add conditional edges
workflow.add_conditional_edges(
    "generate_question",
    should_validate,
    {
        "validate": "validate_output",
        "end": END
    }
)

workflow.add_conditional_edges(
    "validate_output",
    validation_decision,
    {
        "success": END,
        "regenerate": "generate_question",
        "failed": END
    }
)

# Compile
workflow_app = workflow.compile()
```

---

### `services/` Folder

#### `services/claude.py`
**Purpose**: Claude API interactions

**Responsibilities**:
- Call Claude Vision API for structure extraction
- Call Claude for question classification
- Call Claude for question generation
- Handle API errors

**Example**:
```python
from anthropic import AsyncAnthropic
from config import CLAUDE_API_KEY, CLAUDE_MODEL
import json

client = AsyncAnthropic(api_key=CLAUDE_API_KEY)

async def extract_structure_from_image(base64_image: str) -> dict:
    """Extract structure from SAT question image"""
    
    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Extract the following from this SAT question image:

1. Question text (the main question being asked)
2. Any equations or formulas (convert to LaTeX format)
3. Any table data (convert to JSON format)
4. Any visual elements like graphs or diagrams (describe in text)

Return your response as JSON:
{
  "text": "...",
  "equation": "..." or null,
  "table": {...} or null,
  "visual": "..." or null
}"""
                    }
                ]
            }
        ]
    )
    
    # Parse response
    content = response.content[0].text
    result = json.loads(content)
    
    return result

async def classify_question_type(
    text: str,
    equation: str = None,
    visual: str = None
) -> dict:
    """Classify question into SAT taxonomy"""
    
    prompt = f"""Classify this SAT question into the taxonomy:

Question: {text}
Equation: {equation or 'None'}
Visual: {visual or 'None'}

Return JSON:
{{
  "question_type": "algebra" | "geometry" | "data_analysis" | "reading" | "writing",
  "sat_section": "math" | "reading_writing",
  "sat_subsection": "heart_of_algebra" | "passport_to_advanced_math" | "problem_solving_data_analysis" | "additional_topics" | "standard_english_conventions" | "expression_of_ideas",
  "difficulty": "easy" | "medium" | "hard",
  "topics": ["topic1", "topic2", ...]
}}"""
    
    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.content[0].text
    result = json.loads(content)
    
    return result

async def generate_sat_question(
    similar_questions: list,
    user_description: str,
    question_type: str,
    difficulty: str,
    topics: list
) -> dict:
    """Generate new SAT question"""
    
    # Format examples
    examples_text = ""
    for i, q in enumerate(similar_questions[:3], 1):
        examples_text += f"""
EXAMPLE {i}:
Question: {q['question_text']}
Equation: {q.get('equation_content') or 'None'}
Visual: {q.get('visual_description') or 'None'}
Choices: {json.dumps(q['answer_choices'])}
Correct: {q['correct_answer']}
Explanation: {q['explanation']}
---
"""
    
    prompt = f"""You are an expert SAT question writer. Generate a new SAT {question_type} question.

EXAMPLES FROM QUESTION BANK:
{examples_text}

USER REQUEST:
{user_description or 'Generate a new question similar to the examples'}

CONSTRAINTS:
- Question type: {question_type}
- Difficulty: {difficulty}
- Topics: {', '.join(topics)}
- Must include: question text, 4 answer choices (A-D), correct answer, detailed explanation

OUTPUT FORMAT (JSON):
{{
  "question": {{
    "text": "...",
    "equation_latex": "..." or null,
    "visual_description": "..." or null,
    "table_data": {{...}} or null
  }},
  "answer_choices": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct_answer": "A" | "B" | "C" | "D",
  "explanation": "Step-by-step solution...",
  "metadata": {{
    "type": "{question_type}",
    "section": "math" | "reading_writing",
    "subsection": "...",
    "difficulty": "{difficulty}",
    "topics": {json.dumps(topics)}
  }}
}}"""
    
    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.content[0].text
    result = json.loads(content)
    
    return result
```

#### `services/embeddings.py`
**Purpose**: Generate embeddings

**Responsibilities**:
- Call OpenAI embedding API
- Handle batching if needed
- Return embedding vectors

**Example**:
```python
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL
from typing import List

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    embedding = response.data[0].embedding
    return embedding

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    return embeddings
```

#### `services/validation.py`
**Purpose**: Validate generated questions

**Responsibilities**:
- Structural validation
- Content validation
- Return validation results

**Example**:
```python
import json

async def validate_question(question: dict) -> dict:
    """Validate generated question structure and content"""
    
    errors = []
    
    # Structural validation
    if not question.get("question"):
        errors.append("Missing question field")
    elif not question["question"].get("text"):
        errors.append("Missing question text")
    
    if not question.get("answer_choices"):
        errors.append("Missing answer choices")
    else:
        choices = question["answer_choices"]
        if not all(key in choices for key in ["A", "B", "C", "D"]):
            errors.append("Answer choices must have A, B, C, D")
    
    if not question.get("correct_answer"):
        errors.append("Missing correct answer")
    elif question["correct_answer"] not in ["A", "B", "C", "D"]:
        errors.append("Correct answer must be A, B, C, or D")
    
    if not question.get("explanation"):
        errors.append("Missing explanation")
    
    if not question.get("metadata"):
        errors.append("Missing metadata")
    
    # Content validation
    if question.get("question", {}).get("equation_latex"):
        equation = question["question"]["equation_latex"]
        # Basic LaTeX syntax check
        if equation.count("{") != equation.count("}"):
            errors.append("Equation has mismatched braces")
    
    if question.get("question", {}).get("table_data"):
        table = question["question"]["table_data"]
        if not isinstance(table, dict):
            errors.append("Table data must be a JSON object")
    
    # Check answer choice lengths
    if question.get("answer_choices"):
        for key, value in question["answer_choices"].items():
            if len(value) > 200:
                errors.append(f"Answer choice {key} is too long (>200 chars)")
    
    # Check explanation references correct answer
    if question.get("explanation") and question.get("correct_answer"):
        explanation = question["explanation"].lower()
        correct = question["correct_answer"].lower()
        # This is a simple check - could be more sophisticated
        # Just checking if the letter appears in explanation
        if correct not in explanation:
            errors.append(f"Explanation may not reference correct answer ({correct.upper()})")
    
    return {
        "passed": len(errors) == 0,
        "errors": errors
    }
```

---

### `scripts/` Folder

#### `scripts/populate_db.py`
**Purpose**: Populate question bank from external source

**Responsibilities**:
- Load questions from external database/files
- Process each question through Claude for extraction
- Generate embeddings
- Insert into database

**Example**:
```python
import asyncio
import json
from database.connection import init_database
from database.queries import insert_question
from services.claude import extract_structure_from_image, classify_question_type
from services.embeddings import generate_embedding

async def populate_from_json(json_file: str):
    """Populate database from JSON file of questions"""
    
    # Initialize database
    await init_database()
    
    # Load questions
    with open(json_file, 'r') as f:
        questions = json.load(f)
    
    print(f"Loading {len(questions)} questions...")
    
    for i, q in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}...")
        
        # Extract structure if image provided
        if q.get("image_url"):
            structure = await extract_structure_from_image(q["image_url"])
        else:
            structure = {
                "text": q["text"],
                "equation": q.get("equation"),
                "table": q.get("table"),
                "visual": q.get("visual")
            }
        
        # Classify question
        classification = await classify_question_type(
            text=structure["text"],
            equation=structure.get("equation"),
            visual=structure.get("visual")
        )
        
        # Generate embedding
        embedding_text = " ".join([
            structure["text"],
            structure.get("equation") or "",
            structure.get("visual") or "",
            " ".join(classification["topics"])
        ])
        embedding = await generate_embedding(embedding_text)
        
        # Prepare question data
        question_data = {
            "original_image_url": q.get("image_url"),
            "original_text": q.get("text"),
            "question_type": classification["question_type"],
            "sat_section": classification["sat_section"],
            "sat_subsection": classification["sat_subsection"],
            "difficulty": classification["difficulty"],
            "question_text": structure["text"],
            "equation_content": structure.get("equation"),
            "table_data": structure.get("table"),
            "visual_description": structure.get("visual"),
            "answer_choices": q["answer_choices"],
            "correct_answer": q["correct_answer"],
            "explanation": q["explanation"],
            "embedding": embedding,
            "source": q.get("source", "external_db")
        }
        
        # Insert into database
        await insert_question(question_data)
        
        print(f"✓ Inserted question {i}")
    
    print("Done!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python populate_db.py <questions.json>")
        sys.exit(1)
    
    asyncio.run(populate_from_json(sys.argv[1]))
```

---

## Implementation Order

### Phase 1: Database Setup
1. Create `database/connection.py`
2. Create `database/models.py`
3. Create `database/queries.py`
4. Test database connection and table creation

### Phase 2: Core Services
5. Create `services/claude.py` - implement Claude API calls
6. Create `services/embeddings.py` - implement embedding generation
7. Create `services/validation.py` - implement validation logic
8. Test each service independently

### Phase 3: Workflow
9. Create `workflows/state.py` - define state schema
10. Create `workflows/nodes.py` - implement all 5 nodes
11. Create `workflows/graph.py` - connect nodes with LangGraph
12. Test workflow with sample inputs

### Phase 4: API
13. Create `api/routes.py` - implement endpoints
14. Create `main.py` - wire everything together
15. Test API endpoints with Postman/curl

### Phase 5: Population
16. Create `scripts/populate_db.py` - implement database population
17. Run script to load initial question bank

---

## Testing Strategy

### Unit Tests
- Test each node function independently
- Test each service function independently
- Test each database query independently

### Integration Tests
- Test complete workflow execution
- Test API endpoints end-to-end
- Test database operations with real PostgreSQL

### Manual Testing
- Upload various screenshot types
- Test with different difficulty levels
- Verify question quality
- Test error handling

---

## Deployment Considerations

### Environment Variables
All sensitive config must be in `.env` file:
- Database connection string
- API keys (Claude, OpenAI, LangSmith)
- Any other secrets

### Dependencies
Keep `requirements.txt` minimal and pinned to specific versions

### Database
- Use PostgreSQL with pgvector extension
- Consider connection pooling for production
- Set up database backups

### Monitoring
- LangSmith for workflow observability
- Application logs for errors
- Database query performance monitoring

### Scaling
- Horizontal scaling: Run multiple FastAPI instances
- Vertical scaling: Increase worker count
- Database: Connection pooling and read replicas

---

## Next Steps

1. **Set up development environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create `.env` file** (copy from `.env.example`)

3. **Start PostgreSQL** with pgvector:
   ```bash
   docker run -d \
     --name sat-postgres \
     -e POSTGRES_PASSWORD=password \
     -e POSTGRES_DB=sat_questions \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

4. **Implement in order** following the phases above

5. **Test each component** before moving to next phase

6. **Deploy** when all tests pass
