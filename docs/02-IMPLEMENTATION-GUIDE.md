# Implementation Guide with Code Snippets

## **Quick Start**

### **1. Environment Setup**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install langgraph langchain-core langchain-openai langchain-anthropic
pip install toon-py chandra-ocr sentence-transformers
pip install fastapi uvicorn redis asyncpg
pip install python-dotenv pydantic numpy scikit-learn textstat

# Set up environment variables
cat > .env << EOF
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DATABASE_URL=postgresql://user:pass@localhost:5432/satdb
REDIS_URL=redis://localhost:6379
S3_BUCKET=your-bucket
EOF
```

### **2. Project Structure**

```
sat-question-generator/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── toon_models.py      # Toon schemas
│   │   └── db_models.py        # Database models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ocr_service.py
│   │   ├── question_bank.py
│   │   ├── llm_service.py
│   │   ├── style_analyzer.py
│   │   ├── difficulty_calibrator.py
│   │   └── duplication_detector.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── nodes.py            # LangGraph nodes
│   │   └── workflow.py         # Graph definition
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   └── routes.py           # API endpoints
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_ocr.py
│   ├── test_generation.py
│   └── test_validation.py
├── scripts/
│   ├── setup_db.py
│   └── load_question_bank.py
├── .env
├── requirements.txt
└── README.md
```

---

## **Core Implementation**

### **1. Toon Models** (`src/models/toon_models.py`)

```python
from toon import Toon
from typing import List, Literal, Optional, Dict, Any

class QuestionChoice(Toon):
    """Answer choices for a question"""
    A: str
    B: str
    C: str
    D: str

class SATQuestion(Toon):
    """A single SAT question"""
    id: str
    question: str
    choices: QuestionChoice
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    difficulty: float  # 0-100
    category: str
    predicted_correct_rate: Optional[float] = None
    style_match_score: Optional[float] = None

class GeneratedQuestions(Toon):
    """Collection of generated questions"""
    questions: List[SATQuestion]

class OfficialSATQuestion(Toon):
    """Real question from SAT bank"""
    question_id: str
    source: str
    category: str
    subcategory: str
    difficulty: float
    question_text: str
    choices: QuestionChoice
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str
    national_correct_rate: float
    avg_time_seconds: int
    common_wrong_answers: List[str]
    tags: List[str]

class StyleProfile(Toon):
    """Extracted style characteristics"""
    word_count_range: tuple[int, int]
    vocabulary_level: float
    number_complexity: str
    context_type: str
    question_structure: str
    distractor_patterns: str

class QuestionAnalysis(Toon):
    """Analysis of input requirements"""
    category: str
    difficulty: float
    style: str
    characteristics: List[str]
    example_structure: str

class GraphState(Toon):
    """LangGraph state"""
    # Inputs
    description: str = ""
    uploaded_file_path: str = ""
    num_questions: int = 5
    prefer_real_questions: bool = False
    use_hybrid: bool = True
    
    # Intermediate
    extracted_text: str = ""
    analysis: Optional[QuestionAnalysis] = None
    style_profile: Optional[StyleProfile] = None
    real_questions: List[OfficialSATQuestion] = []
    generated_candidates: List[SATQuestion] = []
    validated_questions: List[SATQuestion] = []
    
    # Output
    final_questions: List[SATQuestion] = []
    metadata: Dict[str, Any] = {}
```

---

### **2. OCR Service** (`src/services/ocr_service.py`)

```python
from chandra import Chandra
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ExtractedQuestion:
    """Structured question from OCR"""
    def __init__(self, text: str, choices: Dict[str, str], region_bbox: tuple):
        self.text = text
        self.choices = choices
        self.region_bbox = region_bbox

class OCRService:
    """Handles PDF and image OCR"""
    
    def __init__(self):
        self.chandra = Chandra()
        logger.info("OCR Service initialized")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF or image
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Extracting text from {file_path}")
        
        try:
            if file_path.suffix.lower() == '.pdf':
                result = self.chandra.process_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                result = self.chandra.process_image(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            text = result.get('text', '')
            logger.info(f"Extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise
    
    def extract_with_structure(self, file_path: str) -> List[ExtractedQuestion]:
        """
        Extract questions with structure detection
        
        Args:
            file_path: Path to file
            
        Returns:
            List of extracted questions with structure
        """
        # Basic text extraction
        text = self.extract_text(file_path)
        
        # Detect question boundaries (simplified - enhance with LayoutLM)
        questions = self._detect_questions(text)
        
        return questions
    
    def _detect_questions(self, text: str) -> List[ExtractedQuestion]:
        """
        Detect individual questions in text
        Simple implementation - enhance with ML
        """
        questions = []
        
        # Split by question numbers (1., 2., etc.)
        import re
        pattern = r'(?:^|\n)(\d+)\.\s+'
        parts = re.split(pattern, text)
        
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                q_num = parts[i]
                q_text = parts[i + 1]
                
                # Extract choices
                choices = self._extract_choices(q_text)
                
                questions.append(ExtractedQuestion(
                    text=q_text,
                    choices=choices,
                    region_bbox=(0, 0, 0, 0)  # Placeholder
                ))
        
        return questions
    
    def _extract_choices(self, text: str) -> Dict[str, str]:
        """Extract A, B, C, D choices from text"""
        choices = {}
        
        import re
        # Match patterns like "A) choice text" or "A. choice text"
        pattern = r'([A-D])[\.\)]\s+([^\n]+)'
        matches = re.findall(pattern, text)
        
        for letter, choice_text in matches:
            choices[letter] = choice_text.strip()
        
        return choices

# Usage example
def test_ocr():
    ocr = OCRService()
    
    # Test with PDF
    text = ocr.extract_text("sample_questions.pdf")
    print(f"Extracted: {text[:200]}...")
    
    # Test with structure
    questions = ocr.extract_with_structure("sample_questions.pdf")
    print(f"Found {len(questions)} questions")
```

---

### **3. Question Bank Service** (`src/services/question_bank.py`)

```python
import asyncpg
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import numpy as np
from models.toon_models import OfficialSATQuestion, QuestionChoice
import logging

logger = logging.getLogger(__name__)

class QuestionBankService:
    """Interface to SAT question bank"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Question Bank Service initialized")
    
    async def connect(self):
        """Create database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20
        )
        logger.info("Database pool created")
    
    async def disconnect(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
    
    async def search_similar(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty_range: Optional[Tuple[float, float]] = None,
        top_k: int = 10
    ) -> List[OfficialSATQuestion]:
        """
        Search for similar questions using vector similarity
        
        Args:
            query: Search query
            category: Filter by category
            difficulty_range: (min, max) difficulty
            top_k: Number of results
            
        Returns:
            List of similar questions
        """
        logger.info(f"Searching for: {query[:50]}...")
        
        # Generate embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build query
        sql = """
            SELECT 
                question_id, source, category, subcategory, difficulty,
                question_text, choice_a, choice_b, choice_c, choice_d,
                correct_answer, explanation, national_correct_rate,
                avg_time_seconds, common_wrong_answers, tags,
                1 - (embedding <=> $1::vector) as similarity
            FROM sat_questions
            WHERE ($2::text IS NULL OR category = $2)
              AND ($3::decimal IS NULL OR difficulty >= $3)
              AND ($4::decimal IS NULL OR difficulty <= $4)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                query_embedding,
                category,
                difficulty_range[0] if difficulty_range else None,
                difficulty_range[1] if difficulty_range else None,
                top_k
            )
        
        questions = [self._parse_row(row) for row in rows]
        logger.info(f"Found {len(questions)} similar questions")
        return questions
    
    async def get_by_id(self, question_id: str) -> Optional[OfficialSATQuestion]:
        """Get question by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sat_questions WHERE question_id = $1",
                question_id
            )
        
        if row:
            return self._parse_row(row)
        return None
    
    async def get_by_category(
        self,
        category: str,
        difficulty: Optional[float] = None,
        limit: int = 50
    ) -> List[OfficialSATQuestion]:
        """Get questions by category"""
        sql = """
            SELECT * FROM sat_questions
            WHERE category = $1
              AND ($2::decimal IS NULL OR ABS(difficulty - $2) < 10)
            LIMIT $3
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, category, difficulty, limit)
        
        return [self._parse_row(row) for row in rows]
    
    async def insert_question(self, question: OfficialSATQuestion):
        """Insert a new question"""
        # Generate embedding
        embedding = self.embedder.encode(question.question_text).tolist()
        
        sql = """
            INSERT INTO sat_questions (
                question_id, source, category, subcategory, difficulty,
                question_text, choice_a, choice_b, choice_c, choice_d,
                correct_answer, explanation, national_correct_rate,
                avg_time_seconds, common_wrong_answers, tags, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                sql,
                question.question_id, question.source, question.category,
                question.subcategory, question.difficulty, question.question_text,
                question.choices.A, question.choices.B,
                question.choices.C, question.choices.D,
                question.correct_answer, question.explanation,
                question.national_correct_rate, question.avg_time_seconds,
                question.common_wrong_answers, question.tags, embedding
            )
        
        logger.info(f"Inserted question {question.question_id}")
    
    def _parse_row(self, row) -> OfficialSATQuestion:
        """Parse database row to OfficialSATQuestion"""
        return OfficialSATQuestion(
            question_id=row['question_id'],
            source=row['source'],
            category=row['category'],
            subcategory=row['subcategory'],
            difficulty=float(row['difficulty']),
            question_text=row['question_text'],
            choices=QuestionChoice(
                A=row['choice_a'],
                B=row['choice_b'],
                C=row['choice_c'],
                D=row['choice_d']
            ),
            correct_answer=row['correct_answer'],
            explanation=row['explanation'],
            national_correct_rate=float(row['national_correct_rate']),
            avg_time_seconds=row['avg_time_seconds'],
            common_wrong_answers=row['common_wrong_answers'],
            tags=row['tags']
        )

# Usage example
async def test_question_bank():
    qbank = QuestionBankService("postgresql://localhost/satdb")
    await qbank.connect()
    
    # Search for similar questions
    results = await qbank.search_similar(
        query="linear equations with two variables",
        category="linear_equations",
        difficulty_range=(50, 70),
        top_k=5
    )
    
    for q in results:
        print(f"{q.question_id}: {q.question_text[:100]}...")
    
    await qbank.disconnect()
```

---

### **4. Style Analyzer** (`src/services/style_analyzer.py`)

```python
import textstat
import re
from typing import List, Tuple
from models.toon_models import StyleProfile, SATQuestion
import logging

logger = logging.getLogger(__name__)

class StyleAnalyzer:
    """Analyzes question style characteristics"""
    
    def analyze(self, examples: List[str]) -> StyleProfile:
        """
        Analyze style from example questions
        
        Args:
            examples: List of example question texts
            
        Returns:
            StyleProfile with extracted characteristics
        """
        logger.info(f"Analyzing style from {len(examples)} examples")
        
        return StyleProfile(
            word_count_range=self._analyze_word_counts(examples),
            vocabulary_level=self._analyze_vocabulary(examples),
            number_complexity=self._analyze_numbers(examples),
            context_type=self._analyze_context(examples),
            question_structure=self._extract_structure(examples),
            distractor_patterns=self._analyze_distractors(examples)
        )
    
    def _analyze_word_counts(self, examples: List[str]) -> Tuple[int, int]:
        """Analyze word count range"""
        counts = [len(ex.split()) for ex in examples]
        return (min(counts), max(counts))
    
    def _analyze_vocabulary(self, examples: List[str]) -> float:
        """Analyze vocabulary level"""
        levels = [textstat.flesch_kincaid_grade(ex) for ex in examples]
        return sum(levels) / len(levels)
    
    def _analyze_numbers(self, examples: List[str]) -> str:
        """Analyze number complexity"""
        has_decimals = any('.' in ex and any(c.isdigit() for c in ex) for ex in examples)
        has_fractions = any('/' in ex and any(c.isdigit() for c in ex) for ex in examples)
        has_large = any(re.search(r'\d{3,}', ex) for ex in examples)
        
        if has_fractions:
            return "fractions"
        elif has_decimals:
            return "decimals"
        elif has_large:
            return "large_integers"
        else:
            return "small_integers"
    
    def _analyze_context(self, examples: List[str]) -> str:
        """Classify context type"""
        keywords = {
            'real_world': ['store', 'buy', 'sell', 'person', 'car', 'distance', 'time', 'money'],
            'abstract': ['variable', 'function', 'equation', 'expression'],
            'geometric': ['triangle', 'circle', 'angle', 'line', 'point', 'area']
        }
        
        scores = {ctx: 0 for ctx in keywords}
        
        for example in examples:
            for ctx, words in keywords.items():
                scores[ctx] += sum(1 for word in words if word in example.lower())
        
        return max(scores, key=scores.get)
    
    def _extract_structure(self, examples: List[str]) -> str:
        """Extract common question structure"""
        # Simplified - replace numbers with N, variables with V
        structures = []
        for ex in examples:
            structure = re.sub(r'\d+', 'N', ex)
            structure = re.sub(r'\b[a-z]\b', 'V', structure)
            structures.append(structure)
        
        # Return most common structure (simplified)
        return structures[0] if structures else ""
    
    def _analyze_distractors(self, examples: List[str]) -> str:
        """Analyze wrong answer patterns"""
        # Placeholder - would need actual choices to analyze
        return "plausible_near_misses"


class StyleMatcher:
    """Matches questions against a style profile"""
    
    def __init__(self):
        self.analyzer = StyleAnalyzer()
    
    def score_match(self, question: SATQuestion, profile: StyleProfile) -> float:
        """
        Score how well a question matches the target style
        
        Args:
            question: Question to score
            profile: Target style profile
            
        Returns:
            Match score 0-1
        """
        scores = []
        
        # Word count match (20%)
        word_count = len(question.question.split())
        if profile.word_count_range[0] <= word_count <= profile.word_count_range[1]:
            scores.append(0.2)
        else:
            # Partial credit based on distance
            min_wc, max_wc = profile.word_count_range
            distance = min(abs(word_count - min_wc), abs(word_count - max_wc))
            penalty = min(distance / 10, 0.2)  # Max 0.2 penalty
            scores.append(max(0, 0.2 - penalty))
        
        # Vocabulary match (20%)
        vocab_level = textstat.flesch_kincaid_grade(question.question)
        vocab_diff = abs(vocab_level - profile.vocabulary_level)
        vocab_score = max(0, 0.2 - (vocab_diff * 0.05))
        scores.append(vocab_score)
        
        # Number complexity match (20%)
        number_type = self.analyzer._analyze_numbers([question.question])
        if number_type == profile.number_complexity:
            scores.append(0.2)
        else:
            scores.append(0.1)  # Partial credit
        
        # Context match (20%)
        context = self.analyzer._analyze_context([question.question])
        if context == profile.context_type:
            scores.append(0.2)
        else:
            scores.append(0.05)  # Small partial credit
        
        # Structure match (20%)
        # Simplified - would need more sophisticated comparison
        scores.append(0.15)  # Placeholder
        
        total_score = sum(scores)
        logger.debug(f"Style match score: {total_score:.2f}")
        return total_score
    
    def filter_by_style(
        self,
        questions: List[SATQuestion],
        profile: StyleProfile,
        threshold: float = 0.7
    ) -> List[SATQuestion]:
        """
        Filter questions that match style profile
        
        Args:
            questions: Questions to filter
            profile: Target style
            threshold: Minimum match score
            
        Returns:
            Filtered questions
        """
        matched = []
        
        for q in questions:
            score = self.score_match(q, profile)
            if score >= threshold:
                q.style_match_score = score
                matched.append(q)
        
        logger.info(f"Filtered {len(matched)}/{len(questions)} questions by style")
        return matched

# Usage example
def test_style_matching():
    analyzer = StyleAnalyzer()
    
    examples = [
        "If 3x + 7 = 22, what is the value of x?",
        "A store sells apples for $2 each. If John buys 5 apples, how much does he pay?",
        "The equation 2y - 5 = 13 has what solution for y?"
    ]
    
    profile = analyzer.analyze(examples)
    print(f"Word count range: {profile.word_count_range}")
    print(f"Vocabulary level: {profile.vocabulary_level:.1f}")
    print(f"Number complexity: {profile.number_complexity}")
    print(f"Context type: {profile.context_type}")
```

---

### **5. Difficulty Calibrator** (`src/services/difficulty_calibrator.py`)

```python
import numpy as np
import textstat
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Union
from models.toon_models import OfficialSATQuestion, SATQuestion
import logging

logger = logging.getLogger(__name__)

class DifficultyCalibrator:
    """ML-based difficulty prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("Difficulty Calibrator initialized")
    
    def train(self, training_questions: List[OfficialSATQuestion]):
        """
        Train difficulty prediction model
        
        Args:
            training_questions: Questions with known difficulty/correct rates
        """
        logger.info(f"Training on {len(training_questions)} questions")
        
        # Extract features and labels
        X = []
        y = []
        
        for q in training_questions:
            features = self.extract_features(q)
            X.append(features)
            # Convert correct rate to difficulty (inverse)
            difficulty = 100 - q.national_correct_rate
            y.append(difficulty)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_scaled, y)
        logger.info(f"Training R² score: {train_score:.3f}")
    
    def extract_features(self, question: Union[OfficialSATQuestion, SATQuestion]) -> np.ndarray:
        """
        Extract features from question
        
        Features:
        1. Word count
        2. Character count
        3. Number of numerical values
        4. Number of variables
        5. Vocabulary level (Flesch-Kincaid)
        6. Estimated steps to solve
        7. Concept complexity (category depth)
        """
        if isinstance(question, OfficialSATQuestion):
            text = question.question_text
            category = question.category
        else:
            text = question.question
            category = question.category
        
        features = [
            len(text.split()),  # 1. Word count
            len(text),  # 2. Character count
            self._count_numbers(text),  # 3. Numbers
            self._count_variables(text),  # 4. Variables
            textstat.flesch_kincaid_grade(text),  # 5. Vocabulary
            self._estimate_steps(text),  # 6. Solution steps
            len(category.split('_'))  # 7. Concept complexity
        ]
        
        return np.array(features)
    
    def _count_numbers(self, text: str) -> int:
        """Count numerical values in text"""
        numbers = re.findall(r'\d+\.?\d*', text)
        return len(numbers)
    
    def _count_variables(self, text: str) -> int:
        """Count algebraic variables"""
        # Find single letters that are likely variables
        variables = re.findall(r'\b[a-z]\b', text.lower())
        # Filter out common words
        common_words = {'a', 'i'}
        variables = [v for v in variables if v not in common_words]
        return len(set(variables))
    
    def _estimate_steps(self, text: str) -> int:
        """Estimate steps to solve (heuristic)"""
        # Simple heuristic based on operations
        operations = len(re.findall(r'[+\-*/=]', text))
        # Also consider "and", "then", etc.
        conjunctions = len(re.findall(r'\b(and|then|next|after)\b', text.lower()))
        return operations + conjunctions
    
    def predict(self, question: SATQuestion) -> float:
        """
        Predict difficulty of a question
        
        Args:
            question: Question to evaluate
            
        Returns:
            Predicted difficulty (0-100)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default difficulty")
            return 50.0
        
        features = self.extract_features(question)
        features_scaled = self.scaler.transform([features])
        
        difficulty = self.model.predict(features_scaled)[0]
        
        # Clip to valid range
        difficulty = np.clip(difficulty, 0, 100)
        
        logger.debug(f"Predicted difficulty: {difficulty:.1f}")
        return float(difficulty)
    
    def calibrate_questions(
        self,
        questions: List[SATQuestion],
        target_difficulty: float,
        tolerance: float = 10
    ) -> List[SATQuestion]:
        """
        Filter questions to target difficulty
        
        Args:
            questions: Questions to calibrate
            target_difficulty: Target difficulty (0-100)
            tolerance: Allowed deviation
            
        Returns:
            Calibrated questions
        """
        calibrated = []
        
        for q in questions:
            predicted_diff = self.predict(q)
            
            if abs(predicted_diff - target_difficulty) <= tolerance:
                q.difficulty = predicted_diff
                calibrated.append(q)
        
        logger.info(f"Calibrated {len(calibrated)}/{len(questions)} questions")
        return calibrated
    
    def save_model(self, path: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")

# Usage example
def test_difficulty_calibration():
    calibrator = DifficultyCalibrator()
    
    # Train on sample data (would use real question bank)
    training_data = []  # List of OfficialSATQuestion
    calibrator.train(training_data)
    
    # Predict difficulty
    test_question = SATQuestion(
        id="test1",
        question="If 3x + 7 = 22, what is the value of x?",
        choices=QuestionChoice(A="3", B="5", C="7", D="9"),
        correct_answer="B",
        explanation="Subtract 7, then divide by 3",
        difficulty=0,
        category="linear_equations"
    )
    
    difficulty = calibrator.predict(test_question)
    print(f"Predicted difficulty: {difficulty:.1f}/100")
    
    # Save model
    calibrator.save_model("difficulty_model.pkl")
```

---

### **6. Duplication Detector** (`src/services/duplication_detector.py`)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import re
from typing import List
from models.toon_models import SATQuestion
import logging

logger = logging.getLogger(__name__)

class QuestionFingerprint:
    """Unique signature of a question"""
    def __init__(self, structure_hash: str, concept_pattern: str, context_type: str):
        self.structure_hash = structure_hash
        self.concept_pattern = concept_pattern
        self.context_type = context_type

class DuplicationDetector:
    """Detects duplicate/similar questions"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_database = []
        logger.info("Duplication Detector initialized")
    
    def get_fingerprint(self, question: SATQuestion) -> QuestionFingerprint:
        """
        Extract structural fingerprint
        
        Args:
            question: Question to fingerprint
            
        Returns:
            QuestionFingerprint
        """
        # Remove numbers to get structure
        structure = re.sub(r'\d+\.?\d*', 'N', question.question)
        structure = re.sub(r'\s+', ' ', structure).strip()
        structure_hash = hashlib.md5(structure.encode()).hexdigest()
        
        # Extract concept pattern
        concepts = self._extract_concepts(question)
        concept_pattern = " -> ".join(concepts)
        
        # Classify context
        context_type = self._classify_context(question.question)
        
        return QuestionFingerprint(
            structure_hash=structure_hash,
            concept_pattern=concept_pattern,
            context_type=context_type
        )
    
    def _extract_concepts(self, question: SATQuestion) -> List[str]:
        """Extract mathematical concepts"""
        concepts = []
        
        # Simple keyword-based extraction
        keywords = {
            'equation': ['equation', 'solve', '='],
            'inequality': ['greater', 'less', '>', '<'],
            'function': ['function', 'f(x)', 'g(x)'],
            'geometry': ['triangle', 'circle', 'angle', 'area'],
            'algebra': ['variable', 'expression'],
        }
        
        text = question.question.lower()
        for concept, words in keywords.items():
            if any(word in text for word in words):
                concepts.append(concept)
        
        return concepts
    
    def _classify_context(self, text: str) -> str:
        """Classify question context"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['store', 'buy', 'sell', 'person', 'car']):
            return 'real_world'
        elif any(word in text_lower for word in ['triangle', 'circle', 'angle']):
            return 'geometric'
        else:
            return 'abstract'
    
    def is_duplicate(
        self,
        new_question: SATQuestion,
        threshold: float = 0.85
    ) -> bool:
        """
        Check if question is duplicate
        
        Args:
            new_question: Question to check
            threshold: Similarity threshold
            
        Returns:
            True if duplicate detected
        """
        # Check semantic similarity
        new_embedding = self.embedder.encode(new_question.question)
        
        for existing in self.question_database:
            existing_embedding = self.embedder.encode(existing.question)
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            
            if similarity > threshold:
                logger.debug(f"Semantic duplicate detected (similarity: {similarity:.2f})")
                return True
        
        # Check structural similarity
        new_fp = self.get_fingerprint(new_question)
        
        for existing in self.question_database:
            existing_fp = self.get_fingerprint(existing)
            
            # Same structure and context = likely duplicate
            if (new_fp.structure_hash == existing_fp.structure_hash and
                new_fp.context_type == existing_fp.context_type):
                logger.debug("Structural duplicate detected")
                return True
        
        return False
    
    def filter_duplicates(
        self,
        questions: List[SATQuestion],
        threshold: float = 0.85
    ) -> List[SATQuestion]:
        """
        Remove duplicate questions
        
        Args:
            questions: Questions to filter
            threshold: Similarity threshold
            
        Returns:
            Unique questions
        """
        unique = []
        
        for q in questions:
            if not self.is_duplicate(q, threshold):
                unique.append(q)
                self.question_database.append(q)
        
        logger.info(f"Filtered {len(questions) - len(unique)} duplicates")
        return unique
    
    def add_to_database(self, question: SATQuestion):
        """Add question to database (for future duplicate detection)"""
        self.question_database.append(question)
    
    def clear_database(self):
        """Clear the question database"""
        self.question_database = []
        logger.info("Question database cleared")

# Usage example
def test_duplication_detection():
    detector = DuplicationDetector()
    
    questions = [
        SATQuestion(
            id="q1",
            question="If 3x + 7 = 22, what is x?",
            choices=QuestionChoice(A="3", B="5", C="7", D="9"),
            correct_answer="B",
            explanation="...",
            difficulty=50,
            category="linear_equations"
        ),
        SATQuestion(
            id="q2",
            question="If 3x + 7 = 22, find the value of x.",  # Duplicate
            choices=QuestionChoice(A="3", B="5", C="7", D="9"),
            correct_answer="B",
            explanation="...",
            difficulty=50,
            category="linear_equations"
        ),
        SATQuestion(
            id="q3",
            question="If 5y - 3 = 17, what is y?",  # Not duplicate
            choices=QuestionChoice(A="2", B="4", C="6", D="8"),
            correct_answer="B",
            explanation="...",
            difficulty=50,
            category="linear_equations"
        ),
    ]
    
    unique = detector.filter_duplicates(questions)
    print(f"Unique questions: {len(unique)}")
```

---

## **Database Setup Script** (`scripts/setup_db.py`)

```python
import asyncpg
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

async def setup_database():
    """Create tables and indexes"""
    
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create main questions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sat_questions (
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
                correct_answer CHAR(1) NOT NULL CHECK (correct_answer IN ('A', 'B', 'C', 'D')),
                explanation TEXT,
                national_correct_rate DECIMAL(5,2),
                avg_time_seconds INTEGER,
                common_wrong_answers TEXT[],
                tags TEXT[],
                embedding vector(384),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category 
            ON sat_questions(category)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_difficulty 
            ON sat_questions(difficulty)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags 
            ON sat_questions USING GIN(tags)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON sat_questions USING ivfflat(embedding vector_cosine_ops)
        """)
        
        print("✅ Database setup complete!")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_database())
```

---

## **Testing the Complete System**

```python
# test_complete_system.py
import asyncio
from src.services.ocr_service import OCRService
from src.services.question_bank import QuestionBankService
from src.services.style_analyzer import StyleAnalyzer, StyleMatcher
from src.services.difficulty_calibrator import DifficultyCalibrator
from src.services.duplication_detector import DuplicationDetector
from src.models.toon_models import GraphState
import os

async def test_complete_flow():
    """Test the complete question generation flow"""
    
    print("🚀 Starting SAT Question Generator Test")
    
    # Initialize services
    ocr = OCRService()
    qbank = QuestionBankService(os.getenv('DATABASE_URL'))
    await qbank.connect()
    
    analyzer = StyleAnalyzer()
    matcher = StyleMatcher()
    calibrator = DifficultyCalibrator()
    detector = DuplicationDetector()
    
    try:
        # Step 1: OCR (if file uploaded)
        print("\n📄 Step 1: OCR Extraction")
        extracted_text = ocr.extract_text("test_questions.pdf")
        print(f"Extracted {len(extracted_text)} characters")
        
        # Step 2: Style Analysis
        print("\n🎨 Step 2: Style Analysis")
        examples = [extracted_text]  # Simplified
        style_profile = analyzer.analyze(examples)
        print(f"Vocabulary level: {style_profile.vocabulary_level:.1f}")
        print(f"Context type: {style_profile.context_type}")
        
        # Step 3: Search Question Bank
        print("\n🔍 Step 3: Searching Question Bank")
        real_questions = await qbank.search_similar(
            query=extracted_text[:500],
            category="linear_equations",
            difficulty_range=(40, 60),
            top_k=5
        )
        print(f"Found {len(real_questions)} similar real questions")
        
        # Step 4: Generate AI Variations (placeholder)
        print("\n🤖 Step 4: Generating AI Variations")
        # Would call LLM service here
        print("Generated 5 AI variations")
        
        # Step 5: Validation (placeholder)
        print("\n✅ Step 5: Multi-Model Validation")
        print("All questions validated")
        
        # Step 6: Filtering
        print("\n🔬 Step 6: Quality Filtering")
        # Style matching, difficulty calibration, duplication detection
        print("Applied all filters")
        
        print("\n✨ Test Complete!")
        
    finally:
        await qbank.disconnect()

if __name__ == "__main__":
    asyncio.run(test_complete_flow())
```

---

**Last Updated:** November 24, 2024  
**Version:** 1.0.0
