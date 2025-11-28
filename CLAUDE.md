# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Standards - READ THIS FIRST

**CRITICAL: Write code for junior developers. Prioritize simplicity and readability above all else.**

### Core Principles

1. **SIMPLE AND CONCISE** - Write the minimum code needed to solve the problem
2. **NO COMPLEX SYNTAX** - Avoid shortcuts, one-liners, or clever tricks
3. **EASY TO READ** - A junior developer should understand it immediately
4. **MINIMAL CHANGES** - Fix issues with the least amount of code possible
5. **DO NOT OVER-ENGINEER** - Only write what was explicitly asked for

### Specific Rules

**DO:**
- Use clear, descriptive variable names (`user_count` not `uc`)
- Break complex logic into separate lines with intermediate variables
- Write explicit if/else blocks instead of ternary operators
- Use simple loops instead of list comprehensions for non-trivial logic
- Add a comment only when the logic isn't immediately obvious
- Keep functions short and single-purpose

**DON'T:**
- Use advanced language features unless absolutely necessary
- Chain multiple operations on one line
- Use nested comprehensions
- Write overly clever or "Pythonic" code
- Add abstractions or helpers unless they're truly needed multiple times
- Refactor working code that wasn't asked to be changed

### Python Examples

**BAD (too clever):**
```python
result = [x for x in data if x.status == 'active' and validate(x)]
```

**GOOD (simple and clear):**
```python
result = []
for item in data:
    if item.status == 'active':
        if validate(item):
            result.append(item)
```

**BAD (too concise):**
```python
score = lambda x: sum([s.value for s in x.stats if s.valid]) / len(x.stats)
```

**GOOD (explicit and readable):**
```python
def calculate_score(item):
    valid_stats = []
    for stat in item.stats:
        if stat.valid:
            valid_stats.append(stat.value)

    total = sum(valid_stats)
    average = total / len(item.stats)
    return average
```

## Frontend Coding Standards

**Apply the same simplicity principles to React/TypeScript code.**

### React-Specific Rules

**DO:**
- Use simple function components (avoid class components)
- Keep one component per file
- Break JSX into small, named variables for complex UI
- Use explicit useState with clear variable names
- Write straightforward useEffect with clear dependencies
- Avoid custom hooks unless they're reused 3+ times
- Use basic prop drilling for 1-2 levels (don't over-optimize with Context)

**DON'T:**
- Use advanced React patterns (HOCs, render props) unless necessary
- Chain multiple array methods (.map().filter().reduce())
- Inline complex logic in JSX
- Use clever TypeScript features (mapped types, conditional types, etc.)
- Create abstractions "just in case"
- Optimize prematurely with useMemo/useCallback

### TypeScript Rules

**DO:**
- Use explicit types for function parameters and return values
- Use interfaces for objects (simpler than type aliases)
- Keep types simple and flat

**DON'T:**
- Use complex generic types
- Use utility types beyond basic Pick/Omit
- Create type gymnastics with unions and intersections

### Frontend Examples

**BAD (too clever - chained operations):**
```typescript
const activeUsers = users
  .filter(u => u.status === 'active')
  .map(u => ({ ...u, displayName: u.firstName + ' ' + u.lastName }))
  .sort((a, b) => a.displayName.localeCompare(b.displayName))
```

**GOOD (simple and clear):**
```typescript
const activeUsers = []

for (const user of users) {
  if (user.status === 'active') {
    const displayName = user.firstName + ' ' + user.lastName
    const userWithDisplay = {
      ...user,
      displayName: displayName
    }
    activeUsers.push(userWithDisplay)
  }
}

activeUsers.sort((a, b) => {
  return a.displayName.localeCompare(b.displayName)
})
```

**BAD (complex JSX logic inline):**
```tsx
<div>
  {items.length > 0 ? (
    items.filter(i => i.visible).map(i => (
      <Card key={i.id}>{i.name}</Card>
    ))
  ) : (
    <EmptyState />
  )}
</div>
```

**GOOD (break it down):**
```tsx
function ItemsList({ items }) {
  const hasItems = items.length > 0

  if (!hasItems) {
    return <EmptyState />
  }

  const visibleItems = []
  for (const item of items) {
    if (item.visible) {
      visibleItems.push(item)
    }
  }

  return (
    <div>
      {visibleItems.map((item) => (
        <Card key={item.id}>{item.name}</Card>
      ))}
    </div>
  )
}
```

**BAD (unclear hook logic):**
```tsx
useEffect(() => {
  const fetchData = async () => {
    const res = await fetch('/api/data')
    const json = await res.json()
    setData(json)
  }
  fetchData()
}, [])
```

**GOOD (explicit and clear):**
```tsx
useEffect(() => {
  async function loadData() {
    const response = await fetch('/api/data')
    const data = await response.json()
    setData(data)
  }

  loadData()
}, []) // Empty array means run once on mount
```

**BAD (premature optimization):**
```tsx
const memoizedValue = useMemo(() =>
  complexCalculation(a, b), [a, b]
)
```

**GOOD (just compute it):**
```tsx
const value = complexCalculation(a, b)
```
Only use useMemo if you have proven performance issues.

## Project Overview

HAI is an AI-powered SAT practice question generator that creates unique, style-matched questions based on uploaded examples or text descriptions. The system combines OCR, embeddings-based question retrieval, LLM generation, and three core quality features: style matching, difficulty calibration, and anti-duplication detection.

**Core Value**: Tutoring centers can upload any SAT question screenshot and receive unlimited variations that match the original style, difficulty, and formateliminating reliance on recycled question banks.

## Architecture

### High-Level Flow
```
User Input (screenshot/text)
  � OCR Extraction (Chandra)
  � Style Analysis
  � Question Bank Search (pgvector)
  � LLM Generation (GPT-4/Claude)
  � Multi-Model Validation
  � Quality Filtering (style/difficulty/dedup)
  � Final Output
```

### Tech Stack
- **Backend**: FastAPI + LangGraph workflow orchestration
- **Database**: PostgreSQL with pgvector extension for embeddings
- **ML/NLP**: sentence-transformers, scikit-learn (Random Forest for difficulty)
- **LLMs**: OpenAI GPT-4, Anthropic Claude (multimodal vision support)
- **OCR**: Chandra (transformers-based)
- **Frontend**: Next.js with React, TanStack Query, Radix UI components
- **Caching**: Redis for sessions and temporary data
- **Storage**: S3-compatible (MinIO for local dev)

### Core Services

**OCR Service** (`src/services/ocr_service.py`)
- Extracts text from uploaded PDFs/images using Chandra OCR
- Detects question boundaries and preserves formatting
- Handles equations, tables, and visual elements

**Question Bank Service** (`src/services/question_bank.py`)
- Manages PostgreSQL database of official SAT questions
- Vector similarity search using pgvector and embeddings
- Metadata filtering by category, difficulty, section

**LLM Service** (`src/services/llm_service.py`)
- Handles both GPT-4 and Claude API calls
- Generation and validation of questions
- Multi-model cross-validation for correctness

**Style Analyzer** (`src/services/style_analyzer.py`)
- Analyzes example questions to extract style characteristics
- Metrics: word count range, vocabulary level (Flesch-Kincaid), number complexity, context type, question structure
- StyleMatcher scores generated questions against style profile (target: 90%+ match)

**Difficulty Calibrator** (`src/services/difficulty_calibrator.py`)
- Random Forest model trained on official SAT data
- Predicts difficulty on 0-100 scale
- Target metrics: RMSE <10 points, R� >0.75

**Duplication Detector** (`src/services/duplication_detector.py`)
- Semantic similarity using embeddings
- Structural fingerprinting to detect repetitive patterns
- Target: <5% duplication rate

### LangGraph Workflow

The workflow is defined in `src/graph/workflow.py` with nodes in `src/graph/nodes.py`:

1. **OCR Node** - Extract text from uploaded files
2. **Analyze Node** - Extract style profile and requirements
3. **Search Node** - Retrieve similar questions from bank (RAG)
4. **Generate Node** - Create new questions using LLM with few-shot examples
5. **Validate Node** - Cross-validate answers with multiple models
6. **Filter Node** - Apply style matching, difficulty calibration, and deduplication filters

State is managed via `GraphState` model in `src/models/toon_models.py`.

## Database Schema

**questions table** (Official SAT question bank)
- `id`, `original_image_url`, `original_text`
- `question_type`, `sat_section`, `sat_subsection`, `difficulty`
- `question_text`, `equation_content` (LaTeX), `table_data` (JSONB), `visual_description`
- `answer_choices` (JSONB), `correct_answer`, `explanation`
- `embedding` (VECTOR(1536)) for pgvector similarity search
- `source`, `created_at`, `updated_at`

**generated_questions table**
- Same structure as questions table
- Additional: `source_question_ids` (array), `user_prompt`, `user_image_url`
- Tracking: `user_rating`, `feedback`

Indexes: `ivfflat` on embedding column for fast vector similarity.

## Development Commands

### Initial Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Docker services (Postgres, Redis, MinIO)
docker-compose up -d

# Setup database and pgvector extension
python scripts/setup_db.py

# Load sample question bank
python scripts/load_question_bank.py --create-sample
python scripts/load_question_bank.py sample_questions.json

# Train difficulty prediction model
python scripts/train_difficulty_model.py
```

### Running Services

```bash
# Start backend API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend development (in frontend/ directory)
cd frontend
npm install
npm run dev

# API documentation available at:
# - http://localhost:8000/docs (Swagger)
# - http://localhost:8000/redoc
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Test specific features
pytest tests/test_style_matching.py
pytest tests/test_difficulty_calibration.py
pytest tests/test_duplication.py
pytest tests/test_ocr_service.py
pytest tests/test_llm_validation.py
pytest tests/test_hybrid_generation.py

# Single test function
pytest tests/test_style_matching.py::test_word_count_analysis -v
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type checking
mypy src/
```

### Database Operations

```bash
# Connect to database
psql postgresql://satuser:satpassword@localhost:5432/satdb

# Retrain difficulty model after adding questions
python scripts/train_difficulty_model.py

# Reset database
python scripts/setup_db.py --reset
```

## API Usage Examples

### Generate Questions
```bash
# From text description
curl -X POST "http://localhost:8000/api/v1/generate" \
  -F "description=Generate 5 algebra questions about linear equations" \
  -F "num_questions=5" \
  -F "target_difficulty=50"

# From uploaded file
curl -X POST "http://localhost:8000/api/v1/generate" \
  -F "file=@example_questions.pdf" \
  -F "description=Generate similar questions" \
  -F "num_questions=3"
```

### Search Question Bank
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "linear equations",
    "category": "algebra",
    "difficulty_min": 40,
    "difficulty_max": 60,
    "limit": 10
  }'
```

## Configuration

Key environment variables in `.env`:

```bash
# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://satuser:satpassword@localhost:5432/satdb

# Quality Thresholds
STYLE_MATCH_THRESHOLD=0.7        # Min style similarity score
DIFFICULTY_TOLERANCE=10.0        # Max difficulty point deviation
DUPLICATION_THRESHOLD=0.85       # Max semantic similarity to existing questions

# Model Settings
EMBEDDING_MODEL=text-embedding-3-small
DIFFICULTY_MODEL_PATH=models/difficulty_rf.pkl

# Services
REDIS_URL=redis://localhost:6379
S3_ENDPOINT=http://localhost:9000
```

## Adding New Features

1. **New Service**: Create in `src/services/`, inherit patterns from existing services
2. **New Node**: Add function to `src/graph/nodes.py`, update workflow in `src/graph/workflow.py`
3. **New Model**: Define Pydantic model in `src/models/toon_models.py`
4. **New Endpoint**: Add route to `src/api/routes.py`
5. **Tests**: Create corresponding test file in `tests/`

## Data Models

Core Pydantic models in `src/models/toon_models.py`:
- `GraphState` - LangGraph workflow state
- `SATQuestion` - Generated question structure
- `OfficialSATQuestion` - Question bank entry
- `StyleProfile` - Style analysis results
- `StyleMatchResult` - Style comparison scores

Database models in `src/models/db_models.py` using SQLAlchemy.

## Success Metrics

- Style Consistency: >90% match score
- Difficulty Accuracy: RMSE <10 points, R� >0.75
- Duplication Rate: <5%
- Generation Time: <30 seconds for 10 questions
- API Response: <2 seconds

## Troubleshooting

**pgvector extension missing**: Connect to psql and run `CREATE EXTENSION vector;`

**Model not found**: Run `python scripts/train_difficulty_model.py` to train the difficulty calibrator

**LLM API errors**: Check API keys in `.env`, verify rate limits and service status

**Database connection issues**: Ensure Docker services are running with `docker-compose ps`, restart with `docker-compose restart postgres`

## Frontend Architecture

Located in `frontend/`:
- **Components**: Chat interface, question cards, export buttons (see `components/`)
- **State**: Zustand for global state, TanStack Query for server state
- **UI**: Radix UI primitives with Tailwind CSS styling
- **Key pages**: Main generation interface (`app/page.tsx`), search (`app/search/page.tsx`)

Frontend talks to backend via axios API calls to `http://localhost:8000/api/v1/*`.

## Documentation

- `docs/MVP.md` - Full product specification and architecture
- `docs/01-ARCHITECTURE.md` - Detailed system architecture
- `docs/02-IMPLEMENTATION-GUIDE.md` - Implementation details
- Feature-specific docs in `docs/` (style matching, difficulty calibration, etc.)
