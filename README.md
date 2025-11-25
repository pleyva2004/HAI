# SAT Question Generator - MVP Backend

AI-powered SAT question generation with **Style Matching**, **Difficulty Calibration**, and **Anti-Duplication**.

## 🎯 Features

### Priority 1: Core Differentiation (MVP)

1. **Style Matching System** - Ensures generated questions match the style of uploaded examples
   - Multi-factor analysis (word count, vocabulary, number complexity, context, structure)
   - 90%+ style consistency target

2. **Difficulty Calibration** - ML-based difficulty prediction (0-100 scale)
   - Random Forest model trained on real SAT data
   - RMSE <10 points, R² >0.75 target

3. **Anti-Duplication** - Prevents repetitive questions
   - Semantic similarity (embeddings)
   - Structural fingerprinting
   - <5% duplication rate target

## 🏗️ Architecture

```
FastAPI API Layer
    ↓
LangGraph Workflow Orchestration
    ↓
┌─────────────┬──────────────┬────────────────┐
│ OCR Service │ Question Bank│  LLM Service   │
│  (Chandra)  │ (PostgreSQL) │ (GPT-4/Claude) │
└─────────────┴──────────────┴────────────────┘
    ↓
Quality Filters (Style + Difficulty + Dedup)
    ↓
Final Questions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis
- OpenAI API key
- Anthropic API key

### Installation

1. **Clone and setup environment**

```bash
cd /Users/pabloleyva/Code/levrok/HAI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your API keys and database URLs
```

3. **Start services with Docker Compose**

```bash
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector (port 5432)
- Redis (port 6379)
- MinIO for S3 storage (ports 9000, 9001)

4. **Setup database**

```bash
python scripts/setup_db.py
```

5. **Load sample questions**

```bash
# Create sample data
python scripts/load_question_bank.py --create-sample

# Load sample data
python scripts/load_question_bank.py sample_questions.json
```

6. **Train difficulty model**

```bash
python scripts/train_difficulty_model.py
```

7. **Run API server**

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at http://localhost:8000

## 📚 API Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Generate Questions

```bash
POST /api/v1/generate
```

Example:
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -F "description=Generate 5 algebra questions about linear equations" \
  -F "num_questions=5" \
  -F "target_difficulty=50"
```

With file upload:
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -F "description=Generate similar questions" \
  -F "file=@example_questions.pdf" \
  -F "num_questions=5"
```

#### Search Questions

```bash
POST /api/v1/search
```

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "linear equations with one variable",
    "category": "algebra",
    "difficulty_min": 40,
    "difficulty_max": 60,
    "limit": 10
  }'
```

#### Validate Question

```bash
POST /api/v1/validate
```

#### Health Check

```bash
GET /api/v1/health
```

## 🧪 Testing

Run tests:

```bash
# All tests
pytest

# Specific feature
pytest tests/test_style_matching.py
pytest tests/test_difficulty_calibration.py
pytest tests/test_duplication.py

# With coverage
pytest --cov=src tests/
```

## 📁 Project Structure

```
HAI/
├── src/
│   ├── models/          # Toon data models
│   ├── services/        # Core services (OCR, LLM, features)
│   ├── graph/          # LangGraph workflow
│   ├── api/            # FastAPI app
│   └── utils/          # Config and helpers
├── scripts/            # Setup and training scripts
├── tests/              # Test suite
├── docs/               # Documentation
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

Key settings in `.env`:

```bash
# LLM APIs
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Database
DATABASE_URL=postgresql://satuser:satpassword@localhost:5432/satdb

# Feature Settings
STYLE_MATCH_THRESHOLD=0.7
DIFFICULTY_TOLERANCE=10.0
DUPLICATION_THRESHOLD=0.85
```

## 📊 Success Metrics

- **Style Consistency**: >90% match score
- **Difficulty Accuracy**: RMSE <10 points, R² >0.75
- **Duplication Rate**: <5%
- **Generation Time**: <30 seconds for 10 questions
- **API Response**: <2 seconds

## 🔄 Workflow

1. **OCR Extraction** - Extract text from uploaded files
2. **Style Analysis** - Analyze requirements and style profile
3. **Question Bank Search** - Find similar real SAT questions
4. **LLM Generation** - Generate new questions (GPT-4 + Claude)
5. **Multi-Model Validation** - Cross-validate correctness
6. **Quality Filtering** - Apply style, difficulty, and deduplication filters
7. **Final Selection** - Return top N questions

## 🛠️ Development

### Adding New Features

1. Create service in `src/services/`
2. Add node to `src/graph/nodes.py`
3. Update workflow in `src/graph/workflow.py`
4. Add tests in `tests/`

### Database Migrations

```bash
# Add new columns/tables in scripts/setup_db.py
python scripts/setup_db.py
```

### Training Difficulty Model

```bash
# After adding more questions to the bank
python scripts/train_difficulty_model.py
```

## 🐳 Docker Deployment

Build and run:

```bash
# Build image
docker build -t sat-generator:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f api
```

## 📈 Next Steps (Future Phases)

- **Priority 2**: Multi-model validation, Smart OCR with boundaries
- **Phase 1**: Question bank expansion (10k+ questions)
- **Phase 2**: Hybrid generation (50/50 real + AI mix)
- **Phase 3**: Style transfer, performance-based selection

## 🤝 Contributing

1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## 📝 License

See LICENSE file for details.

## 🔗 Documentation

- Full documentation: `/docs`
- API docs: http://localhost:8000/docs
- Architecture: `/docs/01-ARCHITECTURE.md`
- Implementation guide: `/docs/02-IMPLEMENTATION-GUIDE.md`

## 🆘 Troubleshooting

### Database Connection Error

```bash
# Check PostgreSQL is running
docker-compose ps

# Restart services
docker-compose restart postgres
```

### Missing pgvector Extension

```bash
# Connect to database
psql postgresql://satuser:satpassword@localhost:5432/satdb

# Create extension
CREATE EXTENSION vector;
```

### LLM API Errors

- Check API keys in `.env`
- Verify rate limits
- Check API service status

### Model Not Found Error

```bash
# Train the difficulty model
python scripts/train_difficulty_model.py
```

## 📞 Support

For issues and questions, see `/docs` folder or create an issue in the repository.

---

**Version**: 0.1.0
**Last Updated**: November 2024
**Status**: MVP Complete
