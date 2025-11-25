# Feature: Question Bank Foundation

## Overview
PostgreSQL database with pgvector extension for storing 10,000+ real SAT questions with semantic search capability.

**Priority:** Phase 1 (Foundation)  
**Complexity:** Medium  
**Time:** 3-4 days

## Database Schema
```sql
CREATE TABLE sat_questions (
    question_id VARCHAR(50) PRIMARY KEY,
    source VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    difficulty DECIMAL(5,2) NOT NULL,
    question_text TEXT NOT NULL,
    choice_a TEXT NOT NULL,
    choice_b TEXT NOT NULL,
    choice_c TEXT NOT NULL,
    choice_d TEXT NOT NULL,
    correct_answer CHAR(1) NOT NULL,
    explanation TEXT,
    national_correct_rate DECIMAL(5,2),
    avg_time_seconds INTEGER,
    common_wrong_answers TEXT[],
    tags TEXT[],
    embedding vector(384),  -- pgvector
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_embedding ON sat_questions 
    USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_category ON sat_questions(category);
CREATE INDEX idx_difficulty ON sat_questions(difficulty);
```

## Vector Search
```python
async def search_similar(self, query: str, top_k: int = 10):
    """Semantic similarity search"""
    
    # Generate embedding
    query_embedding = self.embedder.encode(query)
    
    # Vector similarity query
    results = await self.db.fetch("""
        SELECT *, 
               1 - (embedding <=> $1::vector) as similarity
        FROM sat_questions
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """, query_embedding, top_k)
    
    return [self.parse_row(r) for r in results]
```

## Data Loading
```python
def load_question_bank(csv_path: str):
    """Load questions from CSV into database"""
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # Generate embedding
            embedding = embedder.encode(row['question_text'])
            
            # Insert into database
            await db.execute("""
                INSERT INTO sat_questions (...)
                VALUES (...)
            """, ...)
```

## Metrics
- **Search speed**: <100ms
- **Recall@10**: >0.85
- **Database size**: ~500MB

**Dependencies:** PostgreSQL, pgvector, sentence-transformers
