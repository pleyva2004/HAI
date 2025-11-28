# SAT Practice Question Generator - MVP Specification

## Overview

A multimodal AI system that generates SAT practice questions based on user-provided descriptions or screenshot examples. Built for tutoring centers to create unlimited, customized practice material.

## Core Value Proposition

Tutors can upload a screenshot of any SAT question type (or describe it) and receive new, unique questions following the same pattern—eliminating dependency on recycled question banks.

---

## Architecture

```
User Input (screenshot + optional description)
         │
         ▼
┌─────────────────────────────────────┐
│  Multimodal Preprocessing Layer     │
│  - Claude vision extracts structure │
│  - Classifies question type         │
│  - Extracts: text, equations,       │
│    table data, visual description   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  RAG Retrieval                      │
│  - Embed extracted features         │
│  - Find similar questions from bank │
│  - Pull 3-5 examples as context     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Generation Layer                   │
│  - System prompt with SAT patterns  │
│  - Retrieved examples as few-shot   │
│  - User's specific request          │
│  - Output: new question + metadata  │
└─────────────────────────────────────┘
```

---

## Input Types

The system handles multiple SAT question formats:

| Type | Description | Extraction Approach |
|------|-------------|---------------------|
| Pure text | Word problems, reading comprehension, grammar | Direct text extraction |
| Equations/formulas | Typed or handwritten mathematical expressions | Vision → LaTeX normalization |
| Tables/data | Data analysis questions with tabular information | Vision → structured JSON |
| Graphs/figures | Coordinate planes, geometric diagrams | Vision → textual description of visual elements |

---

## Data Schema

### Question Bank Table

```sql
CREATE TABLE questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Original content
    original_image_url TEXT,
    original_text TEXT,
    
    -- Structured extraction
    question_type VARCHAR(50),        -- 'algebra', 'geometry', 'data_analysis', 'reading', 'writing'
    sat_section VARCHAR(20),          -- 'math', 'reading_writing'
    sat_subsection VARCHAR(50),       -- 'heart_of_algebra', 'passport_to_advanced_math', etc.
    difficulty VARCHAR(10),           -- 'easy', 'medium', 'hard'
    
    -- Content fields
    question_text TEXT NOT NULL,
    equation_content TEXT,            -- LaTeX format
    table_data JSONB,                 -- Structured table representation
    visual_description TEXT,          -- Description of any graphs/figures
    
    -- Answer data
    answer_choices JSONB,             -- {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer CHAR(1),
    explanation TEXT,
    
    -- Embedding for RAG
    embedding VECTOR(1536),           -- For pgvector similarity search
    
    -- Metadata
    source VARCHAR(100),              -- Where this question came from
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for vector similarity search
CREATE INDEX ON questions USING ivfflat (embedding vector_cosine_ops);
```

### Generated Questions Table

```sql
CREATE TABLE generated_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reference to source examples
    source_question_ids UUID[],       -- Questions used as few-shot examples
    user_prompt TEXT,                 -- Original user request/description
    user_image_url TEXT,              -- If they uploaded a screenshot
    
    -- Generated content (same structure as questions table)
    question_type VARCHAR(50),
    sat_section VARCHAR(20),
    sat_subsection VARCHAR(50),
    difficulty VARCHAR(10),
    question_text TEXT NOT NULL,
    equation_content TEXT,
    table_data JSONB,
    visual_description TEXT,
    answer_choices JSONB,
    correct_answer CHAR(1),
    explanation TEXT,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT NOW(),
    user_rating INTEGER,              -- 1-5 quality rating from tutor
    feedback TEXT
);
```

---

## Generated Output Format

For MVP, all output is text-based. Visual elements are described, not rendered.

```json
{
  "question": {
    "text": "A ball is thrown upward from ground level with an initial velocity of 64 feet per second. The height h, in feet, of the ball after t seconds is given by the equation h = -16t² + 64t. After how many seconds will the ball return to ground level?",
    "equation_latex": "h = -16t^2 + 64t",
    "visual_description": null,
    "table_data": null
  },
  "answer_choices": {
    "A": "2 seconds",
    "B": "4 seconds",
    "C": "8 seconds",
    "D": "16 seconds"
  },
  "correct_answer": "B",
  "explanation": "To find when the ball returns to ground level, set h = 0: -16t² + 64t = 0. Factor out -16t: -16t(t - 4) = 0. This gives t = 0 (initial throw) or t = 4 (return to ground).",
  "metadata": {
    "type": "algebra",
    "section": "math",
    "subsection": "passport_to_advanced_math",
    "difficulty": "medium",
    "topics": ["quadratic equations", "projectile motion", "factoring"]
  }
}
```

### Example with Visual Description

```json
{
  "question": {
    "text": "In the xy-plane above, line l passes through the origin and is perpendicular to the line 4x + y = k, where k is a constant. If the two lines intersect at the point (t, t + 1), what is the value of t?",
    "equation_latex": "4x + y = k",
    "visual_description": "A coordinate plane showing two intersecting lines. Line l passes through the origin with positive slope. A second line with negative slope intersects line l at a point in the first quadrant. The intersection point is labeled (t, t+1).",
    "table_data": null
  },
  "answer_choices": {
    "A": "-4",
    "B": "-1/4", 
    "C": "4/17",
    "D": "1/4"
  },
  "correct_answer": "C",
  "explanation": "The line 4x + y = k has slope -4. A perpendicular line has slope 1/4. Line l through origin: y = (1/4)x. The intersection point (t, t+1) lies on line l, so t + 1 = (1/4)t. Solving: 4t + 4 = t, giving 3t = -4... [continue]",
  "metadata": {
    "type": "geometry",
    "section": "math", 
    "subsection": "heart_of_algebra",
    "difficulty": "hard",
    "topics": ["coordinate geometry", "perpendicular lines", "slope"]
  }
}
```

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Database | PostgreSQL + pgvector | Structured data + vector similarity search in one place |
| Embeddings | text-embedding-3-small | Cost-effective, good quality for semantic search |
| LLM | Claude API (with vision) | Multimodal input handling, strong reasoning |
| Backend | FastAPI | Familiar, async-friendly, easy to deploy |
| Frontend | Next.js | Clean UI, familiar stack |

---

## MVP User Flow

1. **Tutor opens web app** → clean interface with upload zone and text input

2. **Tutor provides input** → one of:
   - Uploads screenshot of example question
   - Types description ("quadratic word problem about projectile motion, medium difficulty")
   - Both (screenshot + modification request)

3. **System processes** →
   - Extracts structure from input via Claude vision
   - Embeds and retrieves similar questions from bank
   - Generates new question with retrieved examples as context

4. **Tutor receives output** →
   - Formatted question with choices
   - Correct answer + explanation
   - Visual description (if applicable) for tutor to sketch
   - Topic/difficulty tags

5. **Tutor can** →
   - Copy question to clipboard
   - Export as formatted text
   - Request variations ("make it harder", "change the context")
   - Rate quality (feeds back into system improvement)

---

## Question Bank Population

### Source
External SAT question database (digital format, already organized)

### Ingestion Pipeline

1. Scrape/export questions from source database
2. For each question:
   - Store original content (text/image)
   - Run through Claude vision for structured extraction
   - Classify by type, section, subsection, difficulty
   - Extract equations → normalize to LaTeX
   - Extract tables → convert to structured JSON
   - Generate embedding from combined text representation
3. Store in PostgreSQL with pgvector

### Embedding Strategy
- One question = one document (chunk at question level)
- Embed concatenation of: question_text + equation_content + visual_description + topic tags
- Preserves existing metadata (difficulty, topic) for filtered retrieval

---

## API Endpoints

```
POST /api/generate
  - Input: { image?: base64, description?: string, options?: { difficulty, topic } }
  - Output: GeneratedQuestion object

GET /api/questions/similar
  - Input: { query: string, limit: number, filters?: { section, difficulty } }
  - Output: Question[] (for debugging/exploration)

POST /api/feedback
  - Input: { question_id: uuid, rating: 1-5, feedback?: string }
  - Output: { success: boolean }
```

---

## Success Metrics

- **Quality**: Tutor ratings average ≥ 4/5 on generated questions
- **Accuracy**: Generated questions have correct answers and valid explanations
- **Coverage**: Can handle all major SAT question types
- **Speed**: Generation completes in < 10 seconds

---

## Future Enhancements (v2+)

- **Visual rendering**: Programmatically generate graphs, coordinate planes, geometric figures
- **Worksheet builder**: Batch generate and export as formatted PDF
- **Difficulty calibration**: Fine-tune difficulty based on actual student performance data
- **Custom branding**: White-label for tutoring center's materials
- **Analytics**: Track which question types are generated most, identify gaps

---

## Open Questions

- [ ] Confirm access method for SAT question database (API? scraping? export?)
- [ ] Determine hosting requirements (self-hosted vs cloud)
- [ ] Establish question volume expectations (questions/day? batch generation?)
- [ ] Clarify export format preferences (plain text? formatted doc? specific LMS integration?)
