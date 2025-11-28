# LangGraph Workflow - SAT Question Generator

## Overview

This document outlines the agent workflow for the SAT practice question generator using LangGraph. The workflow provides full traceability through LangSmith while handling multimodal inputs and generating high-quality SAT questions.

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

The workflow maintains a shared state object that flows through each node:

```python
class QuestionGenerationState(TypedDict):
    # ═══════════════════════════════════════
    # INPUT (from user)
    # ═══════════════════════════════════════
    user_image: Optional[str]              # base64 encoded screenshot
    user_description: Optional[str]        # text description of desired question
    user_options: dict                     # { difficulty, topic, section }
    
    # ═══════════════════════════════════════
    # EXTRACTED FEATURES (from extract_structure)
    # ═══════════════════════════════════════
    extracted_text: Optional[str]          # main question text
    equation_content: Optional[str]        # LaTeX formatted equations
    table_data: Optional[dict]             # structured table JSON
    visual_description: Optional[str]      # description of graphs/figures
    
    # ═══════════════════════════════════════
    # CLASSIFICATION (from classify_question)
    # ═══════════════════════════════════════
    question_type: Optional[str]           # 'algebra', 'geometry', etc.
    sat_section: Optional[str]             # 'math' or 'reading_writing'
    sat_subsection: Optional[str]          # 'heart_of_algebra', etc.
    difficulty: Optional[str]              # 'easy', 'medium', 'hard'
    topics: List[str]                      # ['quadratic equations', 'factoring']
    
    # ═══════════════════════════════════════
    # RETRIEVED CONTEXT (from retrieve_examples)
    # ═══════════════════════════════════════
    similar_questions: List[dict]          # 3-5 similar questions from bank
    retrieval_scores: List[float]          # similarity scores
    
    # ═══════════════════════════════════════
    # GENERATED OUTPUT (from generate_question)
    # ═══════════════════════════════════════
    generated_question: Optional[dict]     # complete question object
    generation_attempt: int                # retry counter
    
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

---

## Node Definitions

### Node 1: `extract_structure`

**Purpose**: Extract structured information from user input (image or text description)

**Inputs**:
- `user_image` (optional)
- `user_description` (optional)

**Processing**:
1. If `user_image` exists:
   - Send to Claude Vision API
   - Prompt: "Extract the following from this SAT question: question text, any equations (as LaTeX), table data (as JSON), and describe any visual elements"
   - Parse structured response

2. If only `user_description` exists:
   - Use Claude to parse and structure the description
   - Infer likely question components

**Outputs**:
- `extracted_text`
- `equation_content`
- `table_data`
- `visual_description`

**Example Trace in LangSmith**:
```
Node: extract_structure
Input: { user_image: "base64...", user_description: null }
LLM Call: Claude Vision (claude-sonnet-4-5)
  Prompt tokens: 1245
  Completion tokens: 312
  Duration: 2.3s
Output: {
  extracted_text: "A ball is thrown upward...",
  equation_content: "h = -16t^2 + 64t",
  table_data: null,
  visual_description: null
}
```

---

### Node 2: `classify_question`

**Purpose**: Classify the question into SAT taxonomy

**Inputs**:
- `extracted_text`
- `equation_content`
- `visual_description`
- `user_options` (may include explicit difficulty/topic)

**Processing**:
1. Build classification prompt with extracted features
2. Call Claude with few-shot examples of SAT taxonomy
3. Parse structured classification response

**Outputs**:
- `question_type`
- `sat_section`
- `sat_subsection`
- `difficulty`
- `topics`

**System Prompt**:
```
You are an SAT question classifier. Based on the provided question content,
classify it into the SAT taxonomy:

Question Types: algebra, geometry, data_analysis, reading, writing
Sections: math, reading_writing
Subsections (Math): heart_of_algebra, passport_to_advanced_math, 
                    problem_solving_data_analysis, additional_topics
Difficulty: easy, medium, hard

Return JSON format:
{
  "question_type": "...",
  "sat_section": "...",
  "sat_subsection": "...",
  "difficulty": "...",
  "topics": ["topic1", "topic2"]
}
```

**Example Trace in LangSmith**:
```
Node: classify_question
Input: { extracted_text: "A ball is thrown upward...", equation_content: "h = -16t^2 + 64t" }
LLM Call: Claude (claude-sonnet-4-5)
  Prompt tokens: 456
  Completion tokens: 89
  Duration: 0.8s
Output: {
  question_type: "algebra",
  sat_section: "math",
  sat_subsection: "passport_to_advanced_math",
  difficulty: "medium",
  topics: ["quadratic equations", "projectile motion", "factoring"]
}
```

---

### Node 3: `retrieve_examples`

**Purpose**: Find similar questions from the question bank using RAG

**Inputs**:
- `extracted_text`
- `equation_content`
- `visual_description`
- `question_type`
- `sat_subsection`
- `difficulty`
- `user_options` (filters)

**Processing**:
1. Create embedding query:
   ```python
   query_text = f"""
   Question: {extracted_text}
   Equations: {equation_content}
   Visual: {visual_description}
   Topics: {', '.join(topics)}
   """
   ```

2. Generate embedding using `text-embedding-3-small`

3. Query PostgreSQL with pgvector:
   ```sql
   SELECT * FROM questions
   WHERE question_type = ?
     AND sat_subsection = ?
     AND difficulty = ?
   ORDER BY embedding <=> ?
   LIMIT 5;
   ```

4. Return top 3-5 most similar questions

**Outputs**:
- `similar_questions` (list of complete question objects)
- `retrieval_scores` (cosine similarity scores)

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
- `similar_questions`
- `extracted_text` (if from image)
- `user_description` (if provided)
- `question_type`
- `difficulty`
- `topics`

**Processing**:
1. Build generation prompt with:
   - System instructions for SAT question format
   - 3-5 retrieved examples as few-shot demonstrations
   - User's specific request/modifications
   - Structural constraints (answer choices, explanation, etc.)

2. Call Claude for generation

3. Parse response into structured format

**System Prompt Template**:
```
You are an expert SAT question writer. Generate a new SAT {question_type} question
that follows these examples:

EXAMPLE 1:
{similar_question_1}

EXAMPLE 2:
{similar_question_2}

EXAMPLE 3:
{similar_question_3}

USER REQUEST:
{user_description}

CONSTRAINTS:
- Question type: {question_type}
- Difficulty: {difficulty}
- Topics: {topics}
- Must include: question text, 4 answer choices (A-D), correct answer, detailed explanation

OUTPUT FORMAT:
{
  "question": {
    "text": "...",
    "equation_latex": "..." (if applicable),
    "visual_description": "..." (if applicable),
    "table_data": {...} (if applicable)
  },
  "answer_choices": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct_answer": "A/B/C/D",
  "explanation": "Step-by-step solution...",
  "metadata": {
    "type": "...",
    "section": "...",
    "subsection": "...",
    "difficulty": "...",
    "topics": [...]
  }
}
```

**Outputs**:
- `generated_question` (complete question object)
- `generation_attempt` (incremented)

**Example Trace in LangSmith**:
```
Node: generate_question
Input: {
  similar_questions: [...],
  user_description: "Create a quadratic word problem about projectile motion",
  difficulty: "medium"
}
LLM Call: Claude (claude-sonnet-4-5)
  Prompt tokens: 3456
  Completion tokens: 512
  Duration: 3.1s
Output: {
  generated_question: {
    question: { text: "A ball is thrown...", equation_latex: "..." },
    answer_choices: {...},
    correct_answer: "B",
    explanation: "...",
    metadata: {...}
  }
}
```

---

### Node 5: `validate_output`

**Purpose**: Validate the generated question meets quality standards

**Inputs**:
- `generated_question`

**Processing**:
1. **Structural validation**:
   - Has question text?
   - Has 4 answer choices labeled A-D?
   - Has correct answer specified?
   - Has explanation?

2. **Content validation**:
   - Equation syntax valid (if present)?
   - Table data well-formed (if present)?
   - Answer choice lengths reasonable?
   - Explanation references the correct answer?

3. **Logical validation** (optional, slower):
   - Use Claude to verify: "Does the explanation correctly solve for the stated correct answer?"
   - Check for mathematical consistency

**Outputs**:
- `validation_passed` (boolean)
- `validation_errors` (list of issues found)

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
Input: { generated_question: {...} }
Validation Checks:
  ✓ Has question text
  ✓ Has 4 answer choices
  ✓ Has correct answer
  ✓ Has explanation
  ✓ Equation syntax valid
  ✗ Explanation doesn't reference correct answer
Output: {
  validation_passed: false,
  validation_errors: ["Explanation references answer C but correct answer is B"]
}
Decision: regenerate (attempt 2/3)
```

---

## Conditional Edges

### Edge 1: `should_validate`

After `generate_question`, decide whether to validate:

```python
def should_validate(state: QuestionGenerationState) -> str:
    """Decide whether to run validation or skip to end"""
    
    # Always validate unless user explicitly disabled it
    if state["user_options"].get("skip_validation"):
        return "end"
    
    return "validate"
```

**Routes**:
- `"validate"` → go to `validate_output` node
- `"end"` → skip validation, go to END

---

### Edge 2: `validation_decision`

After `validate_output`, decide whether to accept or regenerate:

```python
def validation_decision(state: QuestionGenerationState) -> str:
    """Decide whether to regenerate or accept output"""
    
    # If validation passed, we're done
    if state["validation_passed"]:
        return "success"
    
    # If we've tried too many times, give up
    if state["generation_attempt"] >= 3:
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

# Initialize graph
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
app = workflow.compile()
```

---

## FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import base64

app = FastAPI()

class GenerateRequest(BaseModel):
    image: Optional[str] = None  # base64
    description: Optional[str] = None
    options: dict = {}

@app.post("/api/generate")
async def generate_question(request: GenerateRequest):
    """Main endpoint for question generation"""
    
    # Validate input
    if not request.image and not request.description:
        raise HTTPException(400, "Must provide either image or description")
    
    # Initialize workflow state
    initial_state = {
        "user_image": request.image,
        "user_description": request.description,
        "user_options": request.options,
        "generation_attempt": 0,
        "workflow_id": generate_uuid(),
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Run the workflow
    try:
        result = await app.ainvoke(initial_state)
        
        # Check if workflow succeeded
        if not result.get("validation_passed") and result.get("validation_errors"):
            raise HTTPException(500, f"Generation failed: {result['validation_errors']}")
        
        # Store in database
        question_id = await store_generated_question(
            question=result["generated_question"],
            source_ids=[q["id"] for q in result["similar_questions"]],
            user_prompt=request.description,
            user_image=request.image
        )
        
        # Return result
        return {
            "id": question_id,
            "question": result["generated_question"],
            "metadata": {
                "workflow_id": result["workflow_id"],
                "similar_questions_used": len(result["similar_questions"]),
                "generation_attempts": result["generation_attempt"]
            }
        }
        
    except Exception as e:
        # LangSmith will capture full trace even on error
        raise HTTPException(500, f"Workflow error: {str(e)}")


@app.post("/api/upload-screenshot")
async def upload_screenshot(file: UploadFile):
    """Handle file upload and convert to base64"""
    
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    return {"image": base64_image}
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
