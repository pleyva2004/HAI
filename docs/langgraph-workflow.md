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
| `retrieve_examples` | ⏸️ Stub | Returns empty list - RAG disabled until question bank data is collected |
| `generate_question` | ✅ Implemented | Uses Claude + `parse.py` for response parsing |
| `validate_output` | ✅ Implemented | Lenient mode - `correct_answer` and `explanation` are optional |

### Services

| Service | File | Status |
|---------|------|--------|
| Claude API | `backend/services/claude.py` | ✅ Implemented |
| Response Parser | `backend/services/parse.py` | ✅ Implemented |
| Embeddings | `backend/services/embeddings.py` | ✅ Implemented |
| Retrieval | `backend/services/retrieval.py` | ⏸️ Stub - pending question bank |
| Validation | `backend/services/validation.py` | ✅ Implemented (lenient mode) |

### Known Limitations

1. **RAG Retrieval Disabled**: The `retrieve_examples` node currently returns empty results. Question bank data collection is pending.
2. **Optional Fields**: `correct_answer` and `explanation` are not extracted from Claude's response. Validation skips these checks.
3. **Graph Definition**: `backend/workflows/graph.py` needs to be implemented with LangGraph.

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
                          │  (RAG + pgvector)     │
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

The workflow maintains a shared state object (`HAIState`) that flows through each node. The state is built using Pydantic models for type safety and validation:

```python
class HAIState(BaseModel):
    # ═══════════════════════════════════════
    # USER INPUT (from user)
    # ═══════════════════════════════════════
    user_input: UserInput

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
```

> **Note**: The state uses supporting Pydantic models (`UserInput`, `MathQuestionExtraction`, `QuestionClassification`, `GeneratedQuestion`, `BaseQuestion`, etc.) which are defined in `backend/workflows/state.py`.

---

## Node Definitions

### Node 1: `extract_structure`

**Purpose**: Extract structured information from user input (image or text description)

**Inputs**:
- `state.user_input.image` (optional) - base64 encoded image
- `state.user_input.description` (optional) - text description

**Processing**:
1. If `state.user_input.image` exists:
   - Call `claude.extract_from_image()` with base64 image
   - Uses Claude Vision API with structured output parsing
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

**Example Trace in LangSmith**:
```
Node: extract_structure
Input: { user_input: { image: "base64...", description: null } }
LLM Call: Claude Vision (structured output)
  Prompt tokens: 1245
  Completion tokens: 312
  Duration: 2.3s
Output: {
  extracted_features: {
    text: "A ball is thrown upward...",
    equation: "h = -16t^2 + 64t",
    table: null,
    visual: null,
    answer_choices: { choices: ["A. ...", "B. ...", ...] }
  }
}
```

---

### Node 2: `classify_question`

**Purpose**: Classify the question into SAT taxonomy

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

**Example Trace in LangSmith**:
```
Node: classify_question
Input: { extracted_features: { text: "A ball is thrown upward...", equation: "h = -16t^2 + 64t" } }
LLM Call: Claude (structured output)
  Prompt tokens: 456
  Completion tokens: 89
  Duration: 0.8s
Output: {
  classified_features: {
    section: "Math",
    domain: "Advanced Math",
    skill: ["quadratic equations", "projectile motion", "factoring"],
    difficulty: "Medium"
  }
}
```

---

### Node 3: `retrieve_examples`

> ⚠️ **Currently Stub**: This node returns empty results. RAG retrieval is disabled until question bank data is collected.

**Purpose**: Find similar questions from the question bank using RAG

**Inputs**:
- `state.extracted_features` (MathQuestionExtraction) - for query text
- `state.classified_features` (QuestionClassification) - for filtering

**Processing** (when implemented):
1. Build query text using `embeddings.build_query_text()`:
   - `extracted_text`: from `state.extracted_features.text`
   - `equation_content`: from `state.extracted_features.equation`
   - `visual_description`: from `state.extracted_features.visual`
   - `skills`: from `state.classified_features.skill`

2. Generate embedding using `embeddings.generate_embedding()` with OpenAI

3. Query database using `retrieval.retrieve_similar_questions()`:
   - Filter by `section`, `domain`, and `difficulty` from `state.classified_features`
   - Use vector similarity search with pgvector
   - Return top N most similar questions

**Current Output** (stub):
- `state.similar_questions`: Empty list `[]`
- `state.retrieval_scores`: Empty list `[]`

**Outputs** (when implemented):
- `state.similar_questions`: List[BaseQuestion] - similar questions from bank
- `state.retrieval_scores`: List[float] - similarity scores for each question

**Example Trace in LangSmith**:
```
Node: retrieve_examples
Input: {
  query_embedding: [0.123, -0.456, ...],
  filters: { question_type: "algebra", difficulty: "medium" }
}
Database Query:
  Vector search in questions table
  Results: 5 questions
  Avg similarity: 0.87
  Duration: 0.3s
Output: {
  similar_questions: [
    { id: "uuid1", question_text: "...", similarity: 0.91 },
    { id: "uuid2", question_text: "...", similarity: 0.89 },
    ...
  ]
}
```

---

### Node 4: `generate_question`

**Purpose**: Generate a new SAT question using retrieved examples as few-shot context

**Inputs**:
- `state.extracted_features` (MathQuestionExtraction) - original question template
- `state.classified_features` (QuestionClassification) - classification constraints
- `state.similar_questions` (List[BaseQuestion]) - similar examples for few-shot

**Processing**:
1. Increment `state.generation_attempt` counter
2. Encode `state.extracted_features` and `state.classified_features` using TOON format
3. Call `claude.generate_question()` with:
   - `extracted_features`: encoded extracted features (original question template)
   - `classified_features`: encoded classification (constraints)
   - `similar_questions`: list of similar BaseQuestion objects for few-shot examples
4. Claude generates a new question following the template and constraints
5. Response is parsed using `parse_generated_question()` to create `GeneratedQuestion` object

**Outputs**:
- `state.generated_question` (GeneratedQuestion) containing:
  - `text`: question text
  - `equation`: LaTeX formatted equations (optional)
  - `table`: TableData object (optional)
  - `visual`: description of graphs/figures (optional)
  - `answer_choices`: Dict[str, str] with choices A-D
  - `correct_answer`: "A", "B", "C", or "D" (optional)
  - `explanation`: step-by-step solution (optional)
- `state.generation_attempt`: incremented counter

**Example Trace in LangSmith**:
```
Node: generate_question
Input: {
  extracted_features: { text: "...", equation: "..." },
  classified_features: { section: "Math", domain: "Algebra", ... },
  similar_questions: [BaseQuestion(...), ...]
}
LLM Call: Claude (non-streaming)
  Prompt tokens: 3456
  Completion tokens: 512
  Duration: 3.1s
Output: {
  generated_question: {
    text: "A ball is thrown...",
    equation: "h = -16t^2 + 64t",
    answer_choices: {"A": "...", "B": "...", "C": "...", "D": "..."},
    correct_answer: "B",
    explanation: "..."
  },
  generation_attempt: 1
}
```

---

### Node 5: `validate_output`

**Purpose**: Validate the generated question meets quality standards

**Inputs**:
- `state.generated_question` (GeneratedQuestion)

**Processing**:
1. Call `validation.validate_question(state.generated_question)`
2. Validation service performs checks (lenient mode):
   - ✅ Question exists
   - ✅ Has question text
   - ✅ Has exactly 4 answer choices (A, B, C, D)
   - ✅ Each answer choice has content
   - ⏸️ `correct_answer` - only validated if present (optional)
   - ⏸️ `explanation` - skipped (optional)
3. Returns tuple: `(is_valid: bool, errors: List[str])`
4. Update state:
   - If valid: call `state.mark_validation_passed()`
   - If invalid: set `state.validation_passed = False` and `state.validation_errors = errors`

> **Note**: Validation runs in lenient mode. `correct_answer` and `explanation` are optional fields since they are not currently extracted from Claude's response.

**Outputs**:
- `state.validation_passed` (boolean)
- `state.validation_errors` (list of issues found)

**Conditional Logic**:
```python
if validation_passed:
    return "success" → END
elif generation_attempt < 3:
    return "regenerate" → go back to generate_question
else:
    return "failed" → END (with error state)
```

**Example Trace in LangSmith**:
```
Node: validate_output
Input: { generated_question: GeneratedQuestion(...) }
Validation Service Call (lenient mode):
  ✓ Question exists
  ✓ Has question text
  ✓ Has 4 answer choices (A, B, C, D)
  ✓ Each choice has content
  ⏸ correct_answer: skipped (optional, not present)
  ⏸ explanation: skipped (optional)
Output: {
  validation_passed: true,
  validation_errors: []
}
Decision: success → END
```

---

## Conditional Edges

### Edge 1: `should_validate`

After `generate_question`, decide whether to validate:

```python
def should_validate(state: HAIState) -> str:
    """Decide whether to run validation or skip to end"""

    # Always validate unless user explicitly disabled it
    # Note: skip_validation option not currently in UserInput model
    # Could be added if needed
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
    """Decide whether to regenerate or accept output"""

    # If validation passed, we're done
    if state.validation_passed:
        return "success"

    # If we've tried too many times, give up
    if state.generation_attempt >= 3:
        return "failed"

    # Otherwise, try again
    return "regenerate"
```

**Routes**:
- `"success"` → go to END (workflow complete)
- `"regenerate"` → go back to `generate_question` node
- `"failed"` → go to END (with error state)

---

## Complete Graph Definition

```python
from langgraph.graph import StateGraph, END
from backend.workflows.state import HAIState

# Initialize graph
workflow = StateGraph(HAIState)

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
app = workflow.compile()
```

---

## FastAPI Integration

```python
from fastapi import APIRouter, UploadFile, HTTPException
from backend.workflows.state import HAIState, UserInput
from backend.workflows.graph import agent
import base64

router = APIRouter()

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
        # question_id = await store_generated_question(
        #     question=result.generated_question,
        #     source_ids=[q.id for q in result.similar_questions],
        #     user_prompt=request.description,
        #     user_image=request.image
        # )

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
│  └─ claude-sonnet-4-5 (2.1s, 1,557 tokens)
│     Input: [base64 image]
│     Output: { extracted_text: "...", equation_content: "..." }
│
├─ classify_question (0.8s)
│  └─ claude-sonnet-4-5 (0.7s, 545 tokens)
│     Input: "Classify this question: A ball is thrown..."
│     Output: { question_type: "algebra", difficulty: "medium" }
│
├─ retrieve_examples (0.3s)
│  └─ pgvector query (0.2s)
│     Results: 5 questions, avg similarity: 0.87
│
├─ generate_question (3.1s)
│  └─ claude-sonnet-4-5 (2.9s, 3,968 tokens)
│     Input: [few-shot examples + user request]
│     Output: { generated_question: {...} }
│
└─ validate_output (0.1s)
   ✓ All checks passed
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

## Next Steps

1. **Set up LangSmith**:
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=your_api_key
   export LANGCHAIN_PROJECT=sat-question-generator
   ```

2. **Implement nodes**:
   - Start with `extract_structure` (Claude vision)
   - Add `classify_question` (simple LLM call)
   - Build `retrieve_examples` (connect to PostgreSQL)
   - Implement `generate_question` (prompt engineering)
   - Add `validate_output` (structural checks)

3. **Test with sample questions**:
   - Run workflow with different input types
   - Review traces in LangSmith
   - Iterate on prompts based on results

4. **Deploy**:
   - Wrap in FastAPI endpoint
   - Add authentication
   - Set up monitoring/alerts
   - Deploy to cloud (Railway, Render, AWS, etc.)

---

## Configuration

```python
# config.py

LANGGRAPH_CONFIG = {
    "max_generation_attempts": 3,
    "retrieval_limit": 5,
    "similarity_threshold": 0.75,
    "validation_strict_mode": True,
    "enable_human_review": False,
}

LANGSMITH_CONFIG = {
    "project": "sat-question-generator",
    "tags": ["production"],
}

CLAUDE_CONFIG = {
    "model": "claude-sonnet-4-5-20250929",
    "temperature": 0.7,
    "max_tokens": 2000,
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-3-small",
    "dimensions": 1536,
}
```

---

## Error Handling

```python
async def extract_structure_node(state: QuestionGenerationState):
    try:
        # ... extraction logic ...
        return updated_state
    except Exception as e:
        # LangSmith captures the error automatically
        return {
            **state,
            "error": f"Extraction failed: {str(e)}"
        }

# In FastAPI
result = await app.ainvoke(initial_state)
if result.get("error"):
    raise HTTPException(500, result["error"])
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

Ready to build the MVP with full visibility into every question generated! 🚀
