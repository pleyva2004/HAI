# LangGraph Workflow - SAT Question Generator

## Overview

This document outlines the agent workflow for the SAT practice question generator using LangGraph. The workflow provides full traceability through LangSmith while handling multimodal inputs and generating high-quality SAT questions.

---

## Current Implementation Status

> **Last Updated**: January 2025

| Node | Status | Notes |
|------|--------|-------|
| `extract_structure` | ✅ Implemented | Uses Claude Vision with structured outputs |
| `classify_question` | ✅ Implemented | Uses Claude with structured outputs |
| `retrieve_examples` | ✅ Implemented | Calls embeddings + retrieval services (returns empty until DB connected) |
| `generate_question` | ✅ Implemented | Uses Claude + `parse.py` for response parsing, saves to `generated_questions/` |
| `validate_output` | ✅ Implemented | Lenient mode - `correct_answer` and `explanation` are optional |

### Services

| Service | File | Status |
|---------|------|--------|
| Claude API | `backend/services/claude.py` | ✅ Implemented |
| Response Parser | `backend/services/parse.py` | ✅ Implemented |
| Embeddings | `backend/services/embeddings.py` | ✅ Implemented |
| Retrieval | `backend/services/retrieval.py` | ⏸️ Stub - returns empty until DB connected |
| Validation | `backend/services/validation.py` | ✅ Implemented (lenient mode) |
| Gemini API | `backend/services/gemini.py` | 🚧 Placeholder |
| GPT API | `backend/services/gpt.py` | 🚧 Placeholder |

### Graph Definition

| Component | File | Status |
|-----------|------|--------|
| Graph Definition | `backend/workflows/graph.py` | ✅ Implemented |
| Node Functions | `backend/workflows/nodes.py` | ✅ Implemented |
| State Schema | `backend/workflows/state.py` | ✅ Implemented |

### Known Limitations

1. **RAG Retrieval Returns Empty**: The `retrieve_examples` node generates embeddings but returns empty results since the database is not connected yet.
2. **Optional Fields**: `correct_answer` and `explanation` are only extracted when `provide_answer=True` is set. Validation skips these checks if not present.
3. **Output Files**: Generated questions are automatically saved to `generated_questions/` directory with timestamps.

---

## Workflow Architecture

```
                                    START
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  extract_structure    │
                          │  (Claude Vision)      │
                          └───────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  classify_question    │
                          │  (LLM Classification) │
                          └───────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  retrieve_examples    │
                          │  (Embeddings + DB)    │
                          └───────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  generate_question    │
                          │  (LLM Generation)     │
                          └───────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  validate_output      │
                          │  (Quality Checks)     │
                          └───────────────────────┘
                                      │
                             ┌────────┴────────┐
                             │                 │
                         Valid?            Invalid?
                             │                 │
                             ▼                 ▼
                           END         [Loop back to generate]
                                       (max 3 attempts)
```

---

## State Schema

The workflow maintains a shared state object (`HAIState`) that flows through each node. The state is built using Pydantic models for type safety and validation.

### UserInput Model

```python
class UserInput(BaseModel):
    image: Optional[str] = None
    description: Optional[str] = None
    requested_section: Optional[Literal["Math", "Reading and Writing"]] = None
    requested_difficulty: Optional[Literal["Easy", "Medium", "Hard"]] = None
    requested_domain: Optional[Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]] = None
    output_format: Literal["latex-hardcoded"] = "latex-hardcoded"
    provide_answer: bool = False
    file_out: bool = False
```

### HAIState Model

```python
class HAIState(BaseModel):
    # ═══════════════════════════════════════
    # USER INPUT (from user)
    # ═══════════════════════════════════════
    user_input: UserInput = UserInput()

    # ═══════════════════════════════════════
    # EXTRACTED FEATURES (from extract_structure node)
    # ═══════════════════════════════════════
    extracted_features: Optional[MathQuestionExtraction]

    # ═══════════════════════════════════════
    # CLASSIFICATION (from classify_question node)
    # ═══════════════════════════════════════
    classified_features: Optional[QuestionClassification]

    # ═══════════════════════════════════════
    # RETRIEVED CONTEXT (from retrieve_examples node)
    # ═══════════════════════════════════════
    similar_questions: List[BaseQuestion]
    retrieval_scores: List[float]

    # ═══════════════════════════════════════
    # GENERATED OUTPUT (from generate_question node)
    # ═══════════════════════════════════════
    generated_question: Optional[GeneratedQuestion]
    generation_attempt: int

    # ═══════════════════════════════════════
    # VALIDATION (from validate_output node)
    # ═══════════════════════════════════════
    validation_passed: bool
    validation_errors: List[str]

    # ═══════════════════════════════════════
    # WORKFLOW METADATA
    # ═══════════════════════════════════════
    workflow_id: str
    started_at: str
    error: Optional[str]

    # Helper methods
    def increment_generation_attempt(self) -> None: ...
    def add_validation_error(self, error: str) -> None: ...
    def mark_validation_passed(self) -> None: ...

    @classmethod
    def create_initial_state(cls, user_input: UserInput) -> HAIState: ...
```

### Supporting Models

```python
class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class AnswerChoices(BaseModel):
    choices: List[str]

class MathQuestionExtraction(BaseModel):
    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None
    answer_choices: Optional[AnswerChoices] = None

class QuestionClassification(BaseModel):
    section: Literal["Math", "Reading and Writing"]
    domain: Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]
    skill: List[str]
    difficulty: Literal["Easy", "Medium", "Hard"]

class GeneratedQuestion(BaseModel):
    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None
    answer_choices: Dict[str, str]  # {"A": "...", "B": "...", etc.}
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None
    explanation: Optional[str] = None

class BaseQuestion(BaseModel):
    id: str
    section: Literal["Math", "Reading and Writing"]
    domain: Literal["Algebra", "Advanced Math", "Problem-Solving and Data Analysis", "Geometry and Trigonometry"]
    skill: List[str]
    difficulty: Literal["Easy", "Medium", "Hard"]
    text: str
    equation: Optional[str] = None
    table: Optional[TableData] = None
    visual: Optional[str] = None
    answer_choices: Optional[Dict[str, str]] = None
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None
    explanation: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[str] = None
    original_image_url: Optional[str] = None
```

> **Note**: All models are defined in `backend/workflows/state.py`.

---

## Node Definitions

### Node 1: `extract_structure`

**Purpose**: Extract structured information from user input (image or text description)

**File**: `backend/workflows/nodes.py`

**Inputs**:
- `state.user_input.image` (optional) - base64 encoded image
- `state.user_input.description` (optional) - text description

**Processing**:
1. If `state.user_input.image` exists:
   - Call `claude.extract_from_image()` with base64 image
   - Uses Claude Vision API with structured output parsing via `client.beta.messages.parse()`
   - Returns `MathQuestionExtraction` object

2. If only `state.user_input.description` exists:
   - Call `claude.extract_from_description()` with description text
   - Uses Claude to parse and structure the description
   - Returns `MathQuestionExtraction` object

3. If neither exists, set `state.error = "No image or description provided"`

**Outputs**:
- `state.extracted_features` (MathQuestionExtraction) containing:
  - `text`: main question text
  - `equation`: LaTeX formatted equations (optional)
  - `table`: TableData object with headers and rows (optional)
  - `visual`: description of graphs/figures (optional)
  - `answer_choices`: AnswerChoices object (optional)

**Implementation**:
```python
def extract_structure(state: HAIState) -> HAIState:
    extracted_features: Optional[MathQuestionExtraction] = None

    if state.user_input.image:
        extracted_features = claude.extract_from_image(state.user_input.image)
    elif state.user_input.description:
        extracted_features = claude.extract_from_description(state.user_input.description)

    if extracted_features is None:
        state.error = "No image or description provided"
    else:
        state.extracted_features = extracted_features

    return state
```

---

### Node 2: `classify_question`

**Purpose**: Classify the question into SAT taxonomy

**File**: `backend/workflows/nodes.py`

**Inputs**:
- `state.extracted_features` (MathQuestionExtraction) - required

**Processing**:
1. Check if `state.extracted_features` exists, otherwise set error and return
2. Convert to dict with `.model_dump()` then encode using TOON format: `encode(state.extracted_features.model_dump())`
3. Call `claude.classify_question()` with encoded extracted data
4. Uses Claude with structured output parsing to return `QuestionClassification`

**Outputs**:
- `state.classified_features` (QuestionClassification) containing:
  - `section`: "Math" or "Reading and Writing"
  - `domain`: "Algebra", "Advanced Math", "Problem-Solving and Data Analysis", or "Geometry and Trigonometry"
  - `skill`: List of specific skills tested
  - `difficulty`: "Easy", "Medium", or "Hard"

**Implementation**:
```python
def classify_question(state: HAIState) -> HAIState:
    if state.extracted_features is None:
        state.error = "No Extracted features to classify"
        return state

    encoded_extracted_data = encode(state.extracted_features.model_dump())
    res = claude.classify_question(encoded_extracted_data)
    state.classified_features = res

    return state
```

---

### Node 3: `retrieve_examples`

**Purpose**: Find similar questions from the question bank using embeddings

**File**: `backend/workflows/nodes.py`

**Inputs**:
- `state.extracted_features` (MathQuestionExtraction) - for query text
- `state.classified_features` (QuestionClassification) - for filtering

**Processing**:
1. Build query text using `embeddings.build_query_text()`:
   - `extracted_text`: from `state.extracted_features.text`
   - `equation_content`: from `state.extracted_features.equation`
   - `visual_description`: from `state.extracted_features.visual`
   - `skills`: from `state.classified_features.skill`

2. Generate embedding using `embeddings.generate_embedding()` with OpenAI's `text-embedding-3-small`

3. Call `retrieval.retrieve_similar_questions()` with:
   - `embedding`: the query embedding vector
   - `section`, `domain`, `difficulty`: filters from classification
   - Returns empty until database is connected

**Current Output** (database not connected):
- `state.similar_questions`: Empty list `[]`
- `state.retrieval_scores`: Empty list `[]`

**Implementation**:
```python
def retrieve_examples(state: HAIState) -> HAIState:
    query_text = embeddings.build_query_text(
        extracted_text=state.extracted_features.text if state.extracted_features else None,
        equation_content=state.extracted_features.equation if state.extracted_features else None,
        visual_description=state.extracted_features.visual if state.extracted_features else None,
        skills=state.classified_features.skill if state.classified_features else None
    )

    query_embedding = embeddings.generate_embedding(query_text)

    questions, scores = retrieval.retrieve_similar_questions(
        embedding=query_embedding,
        section=state.classified_features.section if state.classified_features else None,
        domain=state.classified_features.domain if state.classified_features else None,
        difficulty=state.classified_features.difficulty if state.classified_features else None
    )

    state.similar_questions = questions
    state.retrieval_scores = scores

    return state
```

---

### Node 4: `generate_question`

**Purpose**: Generate a new SAT question using retrieved examples as few-shot context

**File**: `backend/workflows/nodes.py`

**Inputs**:
- `state.extracted_features` (MathQuestionExtraction) - original question template
- `state.classified_features` (QuestionClassification) - classification constraints
- `state.similar_questions` (List[BaseQuestion]) - similar examples for few-shot
- `state.user_input.provide_answer` (bool) - whether to include answer and explanation

**Processing**:
1. Increment `state.generation_attempt` counter
2. Encode `state.extracted_features` and `state.classified_features` using TOON format
3. Call `claude.generate_question()` with:
   - `user_requests`: original description
   - `extracted_features`: encoded extracted features (original question template)
   - `classified_features`: encoded classification (constraints)
   - `similar_questions`: list of similar BaseQuestion objects for few-shot examples
   - `provide_answer`: whether to generate correct answer and explanation
4. Claude generates a new question using XML-formatted output
5. Response is parsed using `parse_generated_question()` to create `GeneratedQuestion` object
6. **Question is saved to `generated_questions/question_{timestamp}.json`**

**Outputs**:
- `state.generated_question` (GeneratedQuestion) containing:
  - `text`: question text
  - `equation`: LaTeX formatted equations (optional)
  - `table`: TableData object (optional)
  - `visual`: description of graphs/figures (optional)
  - `answer_choices`: Dict[str, str] with choices A-D
  - `correct_answer`: "A", "B", "C", or "D" (only if `provide_answer=True`)
  - `explanation`: step-by-step solution (only if `provide_answer=True`)
- `state.generation_attempt`: incremented counter

**Implementation**:
```python
def generate_question(state: HAIState) -> HAIState:
    state.increment_generation_attempt()

    encoded_extracted_data = encode(state.extracted_features.model_dump()) if state.extracted_features else ""
    encoded_classified_data = encode(state.classified_features.model_dump()) if state.classified_features else ""

    result = claude.generate_question(
        user_requests=state.user_input.description or "",
        extracted_features=encoded_extracted_data,
        classified_features=encoded_classified_data,
        similar_questions=state.similar_questions,
        provide_answer=state.user_input.provide_answer
    )

    # Save to generated_questions directory
    output_dir = "generated_questions"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/question_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    state.generated_question = result
    return state
```

---

### Node 5: `validate_output`

**Purpose**: Validate the generated question meets quality standards

**File**: `backend/workflows/nodes.py`

**Inputs**:
- `state.generated_question` (GeneratedQuestion)

**Processing**:
1. Call `validation.validate_question(state.generated_question)`
2. Validation service performs checks (lenient mode):
   - ✅ Question exists
   - ✅ Has question text
   - ✅ Has exactly 4 answer choices (A, B, C, D)
   - ✅ Each answer choice has content
   - ⏸️ `correct_answer` - only validated if present (must be A, B, C, or D)
   - ⏸️ `explanation` - skipped (optional)
3. Returns tuple: `(is_valid: bool, errors: List[str])`
4. Update state:
   - If valid: call `state.mark_validation_passed()`
   - If invalid: set `state.validation_passed = False` and `state.validation_errors = errors`

**Outputs**:
- `state.validation_passed` (boolean)
- `state.validation_errors` (list of issues found)

**Implementation**:
```python
def validate_output(state: HAIState) -> HAIState:
    is_valid, errors = validation.validate_question(state.generated_question)

    if is_valid:
        state.mark_validation_passed()
    else:
        state.validation_passed = False
        state.validation_errors = errors

    return state
```

---

## Conditional Edges

### Edge 1: `should_validate`

After `generate_question`, decide whether to validate:

```python
def should_validate(state: HAIState) -> str:
    # Always validate unless user declines
    # TODO: give user option in the UI
    # skip_validation option not currently in UserInput model
    return "validate"
```

**Routes**:
- `"validate"` → go to `validate_output` node
- `"end"` → skip validation, go to END

---

### Edge 2: `validation_decision`

After `validate_output`, decide whether to accept or regenerate:

```python
def validation_decision(state: HAIState) -> str:
    # If validation passed, we're done
    if state.validation_passed:
        return "success"

    # If we've tried too many times, give up
    if state.generation_attempt >= MAX_GENERATION_ATTEMPTS:
        return "failed"

    return "regenerate"
```

**Routes**:
- `"success"` → go to END (workflow complete)
- `"regenerate"` → go back to `generate_question` node
- `"failed"` → go to END (with error state)

---

## Complete Graph Definition

**File**: `backend/workflows/graph.py`

```python
from langgraph.graph import StateGraph, END
from backend.workflows.state import HAIState
from backend.workflows.nodes import (
    extract_structure,
    classify_question,
    retrieve_examples,
    generate_question,
    validate_output,
    should_validate,
    validation_decision
)

# Initialize Graph
workflow = StateGraph(HAIState)

# Add nodes
workflow.add_node("extract_structure", extract_structure)
workflow.add_node("classify_question", classify_question)
workflow.add_node("retrieve_examples", retrieve_examples)
workflow.add_node("generate_question", generate_question)
workflow.add_node("validate_output", validate_output)

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
agent = workflow.compile()
```

---

## FastAPI Integration

**File**: `backend/api/routes.py`

```python
from fastapi import APIRouter, UploadFile, HTTPException
from backend.workflows.state import HAIState, UserInput
from backend.workflows.graph import agent
import base64

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@router.post("/generate")
async def generate_question(request: UserInput):
    """Main endpoint for question generation"""

    # Validate input
    if not request.image and not request.description:
        raise HTTPException(400, "Must provide either image or description")

    # Initialize workflow state using factory method
    initial_state = HAIState.create_initial_state(request)

    # Run the workflow
    try:
        # LangGraph returns dictionary that needs to be processed into pydantic model
        result_unprocess = await agent.ainvoke(initial_state)
        result = HAIState.model_validate(result_unprocess)

        # Check for errors first (early failures)
        if result.error:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow error: {result.error}"
            )

        # Check if workflow succeeded (validation)
        if not result.validation_passed and result.validation_errors:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed after {result.generation_attempt} attempts: {result.validation_errors}"
            )

        # Check if we failed to generate question
        if not result.generated_question:
            raise HTTPException(
                status_code=500,
                detail="No question was generated"
            )

        # TODO: Store in database when implemented

        # Return result
        return {
            "question": result.generated_question.model_dump(),
            "metadata": {
                "workflow_id": result.workflow_id,
                "number_similar_questions_used": len(result.similar_questions),
                "similar_questions_used": result.similar_questions,
                "generate_attempts": result.generation_attempt,
                "validation_passed": result.validation_passed
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        # LangSmith will capture full trace even on error
        raise HTTPException(
            status_code=500,
            detail=f"Workflow error: {str(e)}"
        )


@router.post("/upload-screenshot")
async def upload_screenshot(file: UploadFile):
    """Handle file upload and convert to base64"""

    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')

    return {"image": base64_image}
```

**Note**: In `main.py`, include the router with the `/api` prefix:
```python
from backend.api.routes import router
app.include_router(router, prefix="/api")
```

---

## LangSmith Observability

### What Gets Traced

Every workflow execution creates a trace in LangSmith containing:

1. **Overall workflow metrics**:
   - Total duration
   - Number of LLM calls
   - Total tokens used
   - Cost estimate

2. **Per-node metrics**:
   - Node name
   - Input state
   - Output state
   - LLM calls (if any)
   - Duration
   - Errors (if any)

3. **LLM call details**:
   - Model used
   - Full prompt
   - Full completion
   - Token counts
   - Temperature/settings

### Example Trace View

```
Workflow: generate_question_workflow
Duration: 8.2s
Total Tokens: 5,234
Cost: $0.08

├─ extract_structure (2.3s)
│  └─ claude-sonnet-4-5-20250929 (2.1s, 1,557 tokens)
│     Input: [base64 image]
│     Output: { extracted_text: "...", equation_content: "..." }
│
├─ classify_question (0.8s)
│  └─ claude-sonnet-4-5-20250929 (0.7s, 545 tokens)
│     Input: "Classify this question: A ball is thrown..."
│     Output: { question_type: "algebra", difficulty: "medium" }
│
├─ retrieve_examples (0.3s)
│  └─ OpenAI embedding (0.2s)
│     Results: 0 questions (DB not connected)
│
├─ generate_question (3.1s)
│  └─ claude-sonnet-4-5-20250929 (2.9s, 3,968 tokens)
│     Input: [few-shot examples + user request]
│     Output: { generated_question: {...} }
│
└─ validate_output (0.1s)
   ✓ All checks passed
```

---

## Configuration

**File**: `backend/config.py`

```python
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Claude
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "sat-question-generator"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Workflow
MAX_GENERATION_ATTEMPTS = 3
RETRIEVAL_LIMIT = 5
SIMILARITY_THRESHOLD = 0.75

# Claude API Token Limits
CLAUDE_MAX_TOKENS_DESCRIPTION_EXTRACTION = 1000
CLAUDE_MAX_TOKENS_IMAGE_EXTRACTION = 2000
CLAUDE_MAX_TOKENS_CLASSIFICATION = 500
CLAUDE_MAX_TOKENS_GENERATION = 2500
```

### Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
CLAUDE_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
DATABASE_URL=postgresql://...
```

---

## Error Handling

```python
async def extract_structure_node(state: HAIState):
    try:
        # ... extraction logic ...
        return updated_state
    except Exception as e:
        # LangSmith captures the error automatically
        state.error = f"Extraction failed: {str(e)}"
        return state

# In FastAPI
result = await agent.ainvoke(initial_state)
if result.error:
    raise HTTPException(500, result.error)
```

---

## Planned Features (TODO)

The following nodes are defined but not yet fully implemented:

### `is_question_fair`
```python
#TODO: Implement CHATGPT Answering question and judging fairness
def is_question_fair(state: HAIState) -> HAIState:
    return state
```

### `convert_latex`
```python
#TODO: Implement LLM Call to create latex out of the question we generated
# in-order for the frontend to render tables, equations and visuals nicely.
#### HIGH PRIORITY WILL BE IMPLEMENTED NEXT
def convert_latex(state: HAIState) -> HAIState:
    return state
```

---

## Benefits of This Architecture

### 1. **Full Traceability**
Every question generation is fully traceable:
- Which example questions were retrieved?
- What was the exact prompt sent to Claude?
- How many attempts did validation take?
- Where did errors occur?

### 2. **Quality Debugging**
When a tutor rates a question poorly:
- Pull up the LangSmith trace
- See which similar questions influenced it
- Review the generation prompt
- Identify if retrieval or generation was the issue

### 3. **Iterative Improvement**
- A/B test different prompts for generation node
- Compare retrieval strategies (different embedding models, similarity thresholds)
- Track which question types have higher validation failure rates
- Measure impact of adding more examples to few-shot context

### 4. **Cost Monitoring**
- Track token usage per question type
- Identify expensive workflows (e.g., multiple regeneration attempts)
- Optimize nodes that use excessive tokens

### 5. **Modular Development**
Each node can be:
- Developed independently
- Unit tested in isolation
- Swapped out for different implementations
- Parallelized (future: generate 3 variations simultaneously)

---

## Human-in-the-Loop Extension (Future)

LangGraph supports checkpointing for human feedback:

```python
# Add interrupt before validation
workflow.add_node("tutor_review", tutor_review_node)

workflow.add_conditional_edges(
    "generate_question",
    should_get_human_feedback,
    {
        "review": "tutor_review",
        "auto_validate": "validate_output"
    }
)

# Tutor can approve, reject, or provide feedback
async def tutor_review_node(state: QuestionGenerationState):
    # Workflow pauses here until tutor responds
    feedback = await wait_for_tutor_input(state["workflow_id"])

    if feedback["action"] == "approve":
        state["validation_passed"] = True
        return state
    elif feedback["action"] == "modify":
        state["user_description"] += f"\n\nTutor feedback: {feedback['comments']}"
        return state  # Loop back to generation
    else:
        state["error"] = "Rejected by tutor"
        return state
```

---

## Summary

This LangGraph workflow provides:
- ✅ Clear separation of concerns (extraction, classification, retrieval, generation, validation)
- ✅ Full observability through LangSmith
- ✅ Conditional logic for validation and regeneration
- ✅ Easy to test, debug, and iterate
- ✅ Scales to handle batch processing
- ✅ Ready for human-in-the-loop workflows
- ✅ Automatic file output for generated questions

Ready to build the MVP with full visibility into every question generated! 🚀
