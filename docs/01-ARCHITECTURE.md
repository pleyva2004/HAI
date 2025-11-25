# System Architecture

## **High-Level Architecture**

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │   Web UI       │  │   Mobile App   │  │   API Client   │    │
│  │  (Next.js)     │  │   (React Native)│  │   (Python SDK) │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS/REST
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              FastAPI + Pydantic + Redis                     │ │
│  │  • Authentication (JWT)                                     │ │
│  │  • Rate Limiting                                            │ │
│  │  • Request Validation                                       │ │
│  │  • Response Caching                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    LangGraph Workflow                       │ │
│  │                                                             │ │
│  │  [OCR] → [Analyze] → [Search] → [Generate] → [Validate]   │ │
│  │                          ↓                                  │ │
│  │                    [Filter & Select]                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  SERVICE LAYER   │  │  SERVICE LAYER   │  │  SERVICE LAYER   │
│                  │  │                  │  │                  │
│  OCR Service     │  │  Question Bank   │  │  LLM Service     │
│  (Chandra)       │  │  (Postgres)      │  │  (GPT-4/Claude)  │
│                  │  │                  │  │                  │
│  • Extract text  │  │  • Vector search │  │  • Generation    │
│  • Detect bounds │  │  • Similarity    │  │  • Validation    │
│  • Preserve fmt  │  │  • Metadata      │  │  • Analysis      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │   Redis     │  │     S3      │             │
│  │ + pgvector  │  │             │  │             │             │
│  │             │  │  • Cache    │  │  • Uploads  │             │
│  │  • Questions│  │  • Sessions │  │  • Assets   │             │
│  │  • Metadata │  │  • Temp data│  │  • Backups  │             │
│  │  • Vectors  │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## **Component Details**

### **1. API Gateway (FastAPI)**

**Responsibilities:**
- REST API endpoints
- Authentication & authorization
- Request validation
- Rate limiting
- Response caching
- Error handling

**Key Endpoints:**
```
POST /api/v1/generate
  - Body: {description, file, num_questions, options}
  - Returns: {questions[], metadata}

GET /api/v1/questions/{id}
  - Returns: Single question details

POST /api/v1/validate
  - Body: {question}
  - Returns: {is_valid, errors[], suggestions[]}

GET /api/v1/search
  - Query: {query, category, difficulty}
  - Returns: {questions[], count}
```

**Tech Stack:**
```python
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import jwt
```

---

### **2. LangGraph Workflow Orchestrator**

**Graph Structure:**
```python
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("ocr_extraction", ocr_node)
workflow.add_node("style_analysis", analyze_node)
workflow.add_node("question_search", search_node)
workflow.add_node("hybrid_generation", generate_node)
workflow.add_node("multi_model_validation", validate_node)
workflow.add_node("quality_filtering", filter_node)

# Edges
workflow.set_entry_point("ocr_extraction")
workflow.add_edge("ocr_extraction", "style_analysis")
workflow.add_edge("style_analysis", "question_search")
workflow.add_edge("question_search", "hybrid_generation")
workflow.add_edge("hybrid_generation", "multi_model_validation")
workflow.add_edge("multi_model_validation", "quality_filtering")
workflow.add_edge("quality_filtering", END)

graph = workflow.compile()
```

**State Management:**
```python
class GraphState(TypedDict):
    # Input
    description: str
    uploaded_file_path: str
    num_questions: int
    options: Dict[str, Any]
    
    # Intermediate
    extracted_text: str
    style_analysis: StyleAnalysis
    real_questions: List[OfficialSATQuestion]
    generated_candidates: List[SATQuestion]
    validated_questions: List[SATQuestion]
    
    # Output
    final_questions: List[SATQuestion]
    metadata: Dict[str, Any]
```

---

### **3. OCR Service (Chandra)**

**Purpose:** Extract text and structure from uploaded PDFs/images

**Architecture:**
```python
class OCRService:
    def __init__(self):
        self.chandra = Chandra()
        self.layout_model = LayoutLMv3()
    
    def extract_with_structure(self, file_path: str) -> ExtractedContent:
        # Step 1: Basic OCR
        raw_text = self.chandra.process_document(file_path)
        
        # Step 2: Detect question boundaries
        regions = self.detect_question_regions(file_path)
        
        # Step 3: Extract structured data
        questions = []
        for region in regions:
            q = self.extract_question_from_region(region)
            questions.append(q)
        
        return ExtractedContent(
            raw_text=raw_text,
            questions=questions,
            has_tables=self.detect_tables(file_path),
            has_graphs=self.detect_graphs(file_path)
        )
```

**Features:**
- PDF text extraction
- Image OCR
- Question boundary detection
- Table/graph preservation
- Multiple question separation

---

### **4. Question Bank Service**

**Database Schema:**
```sql
-- Main question table
CREATE TABLE sat_questions (
    question_id VARCHAR(50) PRIMARY KEY,
    source VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    difficulty DECIMAL(5,2) NOT NULL,  -- 0-100
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
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_category ON sat_questions(category);
CREATE INDEX idx_difficulty ON sat_questions(difficulty);
CREATE INDEX idx_tags ON sat_questions USING GIN(tags);
CREATE INDEX idx_embedding ON sat_questions USING ivfflat(embedding vector_cosine_ops);
```

**Service Interface:**
```python
class QuestionBankService:
    def __init__(self, db_connection, embedder):
        self.db = db_connection
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def search_similar(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty_range: Optional[Tuple[float, float]] = None,
        top_k: int = 10
    ) -> List[OfficialSATQuestion]:
        # Generate embedding
        query_embedding = self.embedder.encode(query)
        
        # Vector similarity search
        results = await self.db.fetch("""
            SELECT *, 
                   1 - (embedding <=> $1::vector) as similarity
            FROM sat_questions
            WHERE ($2::text IS NULL OR category = $2)
              AND ($3::decimal IS NULL OR difficulty >= $3)
              AND ($4::decimal IS NULL OR difficulty <= $4)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
        """, query_embedding, category, 
            difficulty_range[0] if difficulty_range else None,
            difficulty_range[1] if difficulty_range else None,
            top_k)
        
        return [self.parse_question(r) for r in results]
    
    async def get_by_id(self, question_id: str) -> OfficialSATQuestion:
        pass
    
    async def batch_insert(self, questions: List[OfficialSATQuestion]):
        pass
```

---

### **5. LLM Service Layer**

**Multi-Model Architecture:**
```python
class LLMService:
    def __init__(self):
        self.models = {
            'gpt4': ChatOpenAI(model="gpt-4o", temperature=0.7),
            'claude': ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
        }
        
    async def generate_with_validation(
        self,
        prompt: str,
        use_models: List[str] = ['gpt4', 'claude']
    ) -> List[SATQuestion]:
        # Generate from each model
        all_candidates = []
        
        for model_name in use_models:
            model = self.models[model_name]
            result = await model.with_structured_output(GeneratedQuestions).ainvoke(prompt)
            all_candidates.extend(result.questions)
        
        # Cross-validate
        validated = []
        for q in all_candidates:
            if await self.validate_question(q):
                validated.append(q)
        
        return validated
    
    async def validate_question(self, question: SATQuestion) -> bool:
        # Use one model to validate another's output
        validator = self.models['gpt4']
        
        prompt = f"""
        Validate this SAT question:
        Question: {question.question}
        Choices: A) {question.choices.A}, B) {question.choices.B}, 
                 C) {question.choices.C}, D) {question.choices.D}
        Claimed answer: {question.correct_answer}
        
        Check:
        1. Is the answer actually correct?
        2. Is there only one correct answer?
        3. Are distractors plausible but wrong?
        4. Is the question unambiguous?
        
        Respond: VALID or INVALID with reason.
        """
        
        result = await validator.ainvoke(prompt)
        return "VALID" in result.content
```

---

### **6. Style Matching System**

**Architecture:**
```python
class StyleAnalyzer:
    def analyze(self, examples: List[str]) -> StyleProfile:
        return StyleProfile(
            word_count_range=self.analyze_word_counts(examples),
            vocabulary_level=self.analyze_vocabulary(examples),
            number_complexity=self.analyze_numbers(examples),
            context_type=self.analyze_context(examples),
            question_structure=self.extract_structure(examples),
            distractor_patterns=self.analyze_distractors(examples)
        )

class StyleMatcher:
    def score_match(self, question: SATQuestion, profile: StyleProfile) -> float:
        scores = []
        
        # Word count match (20%)
        wc = len(question.question.split())
        if profile.word_count_range[0] <= wc <= profile.word_count_range[1]:
            scores.append(0.2)
        
        # Vocabulary match (20%)
        vocab_score = self.compare_vocabulary(question, profile.vocabulary_level)
        scores.append(vocab_score * 0.2)
        
        # Number complexity (20%)
        if self.classify_numbers(question) == profile.number_complexity:
            scores.append(0.2)
        
        # Context match (20%)
        if self.classify_context(question) == profile.context_type:
            scores.append(0.2)
        
        # Structure match (20%)
        structure_score = self.compare_structure(question, profile.question_structure)
        scores.append(structure_score * 0.2)
        
        return sum(scores)
```

---

### **7. Difficulty Calibration**

**ML Model Architecture:**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class DifficultyCalibrator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, question_bank: List[OfficialSATQuestion]):
        # Extract features
        X = []
        y = []
        
        for q in question_bank:
            features = self.extract_features(q)
            X.append(features)
            y.append(100 - q.national_correct_rate)  # Convert to difficulty
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
    def extract_features(self, question: Union[OfficialSATQuestion, SATQuestion]) -> np.ndarray:
        return np.array([
            len(question.question.split()),  # Word count
            len(question.question),  # Character count
            self.count_numbers(question.question),
            self.count_variables(question.question),
            textstat.flesch_kincaid_grade(question.question),
            self.estimate_steps_to_solve(question),
            len(question.category.split('_'))
        ])
    
    def predict(self, question: SATQuestion) -> float:
        features = self.extract_features(question)
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]
```

---

### **8. Anti-Duplication System**

**Architecture:**
```python
class DuplicationDetector:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_database = []
        
    def get_fingerprint(self, question: SATQuestion) -> QuestionFingerprint:
        # Structure hash (remove numbers)
        structure = re.sub(r'\d+', 'N', question.question)
        structure_hash = hashlib.md5(structure.encode()).hexdigest()
        
        # Concept extraction
        concepts = self.extract_concepts(question)
        
        # Context classification
        context = self.classify_context(question.question)
        
        return QuestionFingerprint(
            structure_hash=structure_hash,
            concept_pattern=" -> ".join(concepts),
            context_type=context
        )
    
    def is_duplicate(self, 
                     new_question: SATQuestion, 
                     threshold: float = 0.85) -> bool:
        # Semantic similarity
        new_emb = self.embedder.encode(new_question.question)
        
        for existing in self.question_database:
            existing_emb = self.embedder.encode(existing.question)
            similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
            
            if similarity > threshold:
                return True
        
        # Structural similarity
        new_fp = self.get_fingerprint(new_question)
        
        for existing in self.question_database:
            existing_fp = self.get_fingerprint(existing)
            
            if new_fp.structure_hash == existing_fp.structure_hash:
                if new_fp.context_type == existing_fp.context_type:
                    return True
        
        return False
```

---

## **Deployment Architecture**

### **Development Environment**
```
Local:
├── Docker Compose
│   ├── API (FastAPI)
│   ├── PostgreSQL + pgvector
│   ├── Redis
│   └── MinIO (S3-compatible)
├── LLM APIs (cloud)
└── Dev database (sample questions)
```

### **Production Environment**
```
Cloud (AWS/GCP):
├── API Gateway (Load Balancer)
├── API Servers (EC2/Cloud Run)
│   ├── Auto-scaling (2-10 instances)
│   └── Health checks
├── Database (RDS PostgreSQL)
│   ├── Read replicas (2)
│   └── Automated backups
├── Cache (ElastiCache Redis)
├── Storage (S3)
├── Monitoring (CloudWatch/Datadog)
└── CDN (CloudFront)
```

---

## **Security Architecture**

### **Authentication Flow**
```
User → Login → JWT Token → API Request → Token Validation → Authorized
                    ↓
               Redis (session store)
```

### **Data Protection**
- **In Transit:** TLS 1.3
- **At Rest:** AES-256 encryption (S3, RDS)
- **Secrets:** AWS Secrets Manager / Vault
- **API Keys:** Encrypted in database

### **Rate Limiting**
```python
# Redis-based rate limiter
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = get_client_id(request)
    
    # Check rate limit
    current = await redis.incr(f"rate:{client_id}")
    if current == 1:
        await redis.expire(f"rate:{client_id}", 60)  # 1 minute window
    
    if current > 100:  # 100 requests per minute
        raise HTTPException(429, "Rate limit exceeded")
    
    return await call_next(request)
```

---

## **Scalability Considerations**

### **Horizontal Scaling**
- Stateless API servers (scale via load balancer)
- Database read replicas
- Redis cluster for caching
- Async task queue (Celery) for long-running operations

### **Performance Optimization**
```python
# Caching strategy
@cache(ttl=3600)  # Cache for 1 hour
async def get_question_by_id(question_id: str):
    return await db.fetch_question(question_id)

# Batch processing
async def generate_questions_batch(requests: List[GenerateRequest]):
    # Process multiple requests in parallel
    tasks = [generate_questions(req) for req in requests]
    return await asyncio.gather(*tasks)

# Connection pooling
db_pool = await asyncpg.create_pool(
    dsn=DATABASE_URL,
    min_size=10,
    max_size=50
)
```

### **Monitoring & Observability**
```python
from prometheus_client import Counter, Histogram

# Metrics
requests_total = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
questions_generated = Counter('questions_generated_total', 'Questions generated')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    requests_total.inc()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_duration.observe(duration)
    
    return response
```

---

## **Disaster Recovery**

### **Backup Strategy**
- Database: Automated daily backups (7-day retention)
- Files: S3 versioning enabled
- Question bank: Weekly exports to cold storage

### **Failover Plan**
1. Database failover: Automatic promotion of read replica
2. Cache failure: Graceful degradation (bypass cache)
3. LLM API failure: Fallback to alternate provider

---

## **Development Workflow**

```
Feature Branch → PR → Code Review → CI/CD → Staging → Production
                  ↓
              [Tests]
              - Unit tests
              - Integration tests
              - E2E tests
              - Load tests
```

**CI/CD Pipeline:**
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: ./deploy.sh
```

---

## **Technology Decisions**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Workflow | LangGraph | Best for complex multi-step LLM workflows |
| Database | PostgreSQL + pgvector | Vector search + relational data in one system |
| API | FastAPI | High performance, async, type hints |
| LLMs | GPT-4o + Claude | Best quality, cross-validation |
| OCR | Chandra | Modern, accurate, handles structure |
| Embeddings | SentenceTransformers | Fast, good quality, open-source |
| Cache | Redis | Industry standard, fast |
| Storage | S3 | Scalable, reliable, cheap |

---

**Last Updated:** November 24, 2024  
**Version:** 1.0.0
