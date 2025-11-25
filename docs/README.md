# SAT Question Generator Documentation

Complete documentation for the SAT Question Generator project.

## 📚 Documentation Structure

### Core Documents
1. **[00-PRODUCT-SPEC.md](./00-PRODUCT-SPEC.md)** - Complete product specification
   - Overview & value proposition
   - User flow
   - Feature roadmap
   - Technical architecture
   - Success metrics

2. **[01-ARCHITECTURE.md](./01-ARCHITECTURE.md)** - System architecture
   - High-level architecture diagram
   - Component details
   - Database schema
   - Deployment architecture
   - Security & scalability

3. **[02-IMPLEMENTATION-GUIDE.md](./02-IMPLEMENTATION-GUIDE.md)** - Implementation guide with code
   - Quick start setup
   - Core implementations with code snippets
   - Service implementations
   - Testing examples

### Feature Documentation (To be created)
Located in `features/` directory:

#### Priority 1: Core Differentiation
- `01-style-matching.md` - Style matching system
- `02-difficulty-calibration.md` - ML-based difficulty prediction
- `03-anti-duplication.md` - Duplication detection system

#### Priority 2: Polish
- `04-multi-model-validation.md` - Multi-model validation
- `05-smart-ocr.md` - OCR with question boundary detection

#### Phase 1: Question Bank
- `06-question-bank.md` - Database setup and vector search

#### Phase 2: Hybrid Generation
- `07-hybrid-generation.md` - Real + AI question mixing

#### Phase 3: Advanced Features
- `08-advanced-features.md` - Style transfer, performance-based selection

## 🚀 Quick Start

### 1. Read the Documentation
Start with `00-PRODUCT-SPEC.md` for overview, then proceed to:
- Architecture → `01-ARCHITECTURE.md`
- Implementation → `02-IMPLEMENTATION-GUIDE.md`

### 2. Set Up Development Environment

```bash
# Clone repository
git clone <your-repo>
cd sat-question-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Set up database
python scripts/setup_db.py

# Load question bank
python scripts/load_question_bank.py
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_generation.py -v

# Run with coverage
pytest --cov=src tests/
```

### 4. Start Development Server

```bash
# Run API server
uvicorn src.api.main:app --reload --port 8000

# API will be available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 📦 Project Structure

```
sat-question-generator/
├── docs/                       # This documentation
│   ├── 00-PRODUCT-SPEC.md
│   ├── 01-ARCHITECTURE.md
│   ├── 02-IMPLEMENTATION-GUIDE.md
│   └── features/
│       ├── 01-style-matching.md
│       ├── 02-difficulty-calibration.md
│       └── ...
├── src/
│   ├── models/                 # Data models (Toon schemas)
│   ├── services/               # Core services
│   ├── graph/                  # LangGraph workflow
│   ├── api/                    # FastAPI endpoints
│   └── utils/                  # Utilities
├── tests/                      # Test suite
├── scripts/                    # Setup scripts
├── requirements.txt
└── README.md
```

## 🎯 Development Roadmap

### Week 1-2: Priority 1 Features
- [ ] Style matching system
- [ ] Difficulty calibration
- [ ] Anti-duplication

### Week 3: Priority 2 Features
- [ ] Multi-model validation
- [ ] Smart OCR with boundaries

### Week 4-5: Question Bank
- [ ] Database setup
- [ ] Embedding index
- [ ] Similarity search

### Week 6-7: Hybrid Generation
- [ ] Real question recommendations
- [ ] Variation generator
- [ ] 50/50 mixing

### Week 8: Advanced Features
- [ ] Difficulty calibration model
- [ ] Style transfer system
- [ ] Performance-based selection

## 📝 Key Technologies

- **Workflow:** LangGraph
- **LLMs:** OpenAI GPT-4o, Anthropic Claude Sonnet 4
- **Structured Output:** Toon
- **OCR:** Chandra
- **Database:** PostgreSQL + pgvector
- **Embeddings:** SentenceTransformers
- **ML:** scikit-learn
- **API:** FastAPI
- **Caching:** Redis

## 🤝 Contributing

1. Read the product spec and architecture docs
2. Check the feature documentation for implementation details
3. Follow the code style in implementation guide
4. Write tests for new features
5. Submit PR with clear description

## 📧 Contact

- **Issues:** [GitHub Issues]
- **Discussions:** [GitHub Discussions]
- **Email:** team@satquestiongen.com

## 📄 License

[Your License Here]

---

**Last Updated:** November 24, 2024  
**Version:** 1.0.0
