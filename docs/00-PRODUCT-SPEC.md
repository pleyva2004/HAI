# SAT Question Generator - Product Specification

## **Overview**

An AI-powered SAT question generator that combines real SAT questions from an official question bank with AI-generated variations to provide tutors with high-quality, style-consistent, difficulty-calibrated practice questions.

## **Core Value Proposition**

### **vs. ChatGPT:**
| Feature | ChatGPT | Our System |
|---------|---------|------------|
| Question Source | AI-generated only | 50% real SAT + 50% AI variations |
| Style Consistency | Variable | 90%+ match to examples |
| Difficulty | Vague ("easy/medium/hard") | Calibrated 0-100 scale |
| Validation | None | Multi-model cross-validation |
| Performance Data | None | National correct rates, avg time |
| Duplication Prevention | No | Structural + semantic detection |
| Question Bank Access | No | 10k+ official SAT questions |

---

## **User Flow**

```
TUTOR INPUT
├─ Upload PDF/PNG of example questions
├─ OR describe question requirements in text
└─ Specify number of questions (default: 5)
    │
    ▼
BACKEND PROCESSING
├─ 1. Smart OCR (extract text + detect boundaries)
├─ 2. Style Analysis (extract patterns)
├─ 3. Question Bank Search (find similar real questions)
├─ 4. Hybrid Generation (50% real + 50% AI)
├─ 5. Quality Pipeline
│   ├─ Multi-model validation
│   ├─ Difficulty calibration
│   ├─ Style matching
│   └─ Anti-duplication
└─ 6. Final Selection (top N passing all filters)
    │
    ▼
OUTPUT TO TUTOR
├─ Real SAT questions with metadata
│   ├─ Source (e.g., "Practice Test 7, Q14")
│   ├─ Difficulty score (0-100)
│   ├─ National correct rate
│   └─ Common wrong answers
│
└─ AI-generated variations
    ├─ Based on real question template
    ├─ Predicted difficulty
    ├─ Style match score
    └─ Validation status
```

---

## **Target Users**

### **Primary: SAT Tutors**
- Need high-quality practice questions
- Want questions matching specific styles/difficulties
- Require variety to avoid repetition
- Value authenticity and SAT-alignment

### **Secondary: Students (Future)**
- Self-study practice
- Targeted weakness improvement
- Progress tracking

---

## **Feature Roadmap**

### **Priority 1: Core Differentiation** (Weeks 1-2)
1. ✅ Style Matching System
2. ✅ Difficulty Calibration
3. ✅ Anti-Duplication

### **Priority 2: Polish** (Week 3)
4. ✅ Multi-Model Validation
5. ✅ Smart OCR with Boundaries

### **Phase 1: Question Bank Foundation** (Weeks 4-5)
1. ✅ Database setup
2. ✅ Embedding index
3. ✅ Similarity search

### **Phase 2: Hybrid Generation** (Weeks 6-7)
4. ✅ Real question recommendations
5. ✅ Variation generator
6. ✅ 50/50 mixing

### **Phase 3: Advanced Features** (Week 8)
7. ✅ Difficulty calibration model
8. ✅ Style transfer system
9. ✅ Performance-based selection

---

## **Technical Architecture**

### **Tech Stack**
- **Workflow:** LangGraph
- **LLMs:** OpenAI GPT-4o + Anthropic Claude Sonnet 4
- **Structured Output:** Toon
- **OCR:** Chandra
- **Database:** PostgreSQL + pgvector
- **Embeddings:** SentenceTransformers
- **ML:** scikit-learn (difficulty calibration)
- **API:** FastAPI
- **Caching:** Redis
- **Storage:** S3

### **Key Components**
```
┌─────────────────────────────────────────┐
│         Frontend (Future)               │
│   Next.js + React + TailwindCSS        │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│            API Layer                    │
│          FastAPI + Redis                │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        LangGraph Workflow               │
│  ┌─────────────────────────────────┐  │
│  │ OCR → Analysis → Search →       │  │
│  │ Generate → Validate → Select    │  │
│  └─────────────────────────────────┘  │
└─────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Question Bank│    │  LLM Services│
│  PostgreSQL  │    │ GPT-4 + Claude│
│  + pgvector  │    │              │
└──────────────┘    └──────────────┘
```

---

## **Success Metrics**

### **Quality Metrics**
- Style consistency: >90% match to uploaded examples
- Difficulty accuracy: ±10 points from target
- Duplication rate: <5% structural similarity
- Validation pass rate: >95% questions correct

### **Performance Metrics**
- Generation time: <30 seconds for 10 questions
- Search latency: <100ms
- API response time: <2 seconds

### **Business Metrics**
- Tutor satisfaction: >4.5/5
- Question reuse rate: >60%
- Time saved per tutor: 10+ hours/week

---

## **Data Requirements**

### **Question Bank**
- **Size:** 10,000+ official SAT questions
- **Sources:** 
  - Official SAT Practice Tests
  - Khan Academy SAT prep
  - College Board released questions
- **Metadata:** Category, difficulty, correct rate, timing

### **Storage Estimates**
- Question bank: ~500MB (text + embeddings)
- User uploads: ~10GB/year (PDFs/images)
- Generated questions: ~1GB/year

---

## **Security & Privacy**

- User uploads stored encrypted (S3 server-side encryption)
- Question bank access controlled (API keys)
- No student PII collected (tutor-focused MVP)
- Rate limiting on API endpoints
- Input validation on all uploads

---

## **Future Enhancements**

### **Phase 4: Student Features** (Future)
- Student progress tracking
- Adaptive question selection
- Performance analytics dashboard
- Predicted SAT score

### **Phase 5: Collaboration** (Future)
- Shared question banks between tutors
- Question rating system
- Community contributions
- Best practices library

### **Phase 6: Integration** (Future)
- Google Classroom integration
- Canvas LMS integration
- Printable worksheets
- Mobile app

---

## **Competitive Analysis**

### **Direct Competitors**
- **Khan Academy:** Free, limited customization
- **PrepScholar:** Expensive, closed system
- **Magoosh:** Good content, no customization
- **College Board:** Official but limited supply

### **Our Advantages**
1. **Unlimited supply** (AI generation)
2. **High quality** (real question baseline)
3. **Customization** (style matching)
4. **Transparency** (difficulty scores, sources)
5. **Cost-effective** (vs. PrepScholar/Magoosh)

---

## **Pricing Strategy (Future)**

### **Free Tier**
- 50 questions/month
- AI-generated only
- Basic difficulty levels

### **Pro Tier ($29/month)**
- Unlimited questions
- 50/50 real + AI mix
- Difficulty calibration
- Style matching
- API access

### **Enterprise ($99/month)**
- Everything in Pro
- Custom question banks
- Batch generation
- White-label option
- Priority support

---

## **Development Milestones**

### **MVP (8 weeks)**
- ✅ Core generation pipeline
- ✅ Question bank integration
- ✅ Quality filters (style, difficulty, duplication)
- ✅ Basic API

### **Beta (12 weeks)**
- ✅ Multi-model validation
- ✅ Smart OCR
- ✅ Performance analytics
- ✅ Web interface

### **V1.0 (16 weeks)**
- ✅ Student progress tracking
- ✅ Advanced features
- ✅ Mobile-responsive UI
- ✅ Production deployment

---

## **Risk Mitigation**

### **Technical Risks**
| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinations | High | Multi-model validation + real question baseline |
| Question bank access | High | Backup sources, caching, rate limiting |
| OCR accuracy | Medium | Chandra + manual review fallback |
| Scalability | Medium | Redis caching, async processing |

### **Business Risks**
| Risk | Impact | Mitigation |
|------|--------|------------|
| Copyright issues | High | Use only licensed/public domain questions |
| Low adoption | Medium | Tutor beta program, referral incentives |
| Competition | Medium | Focus on quality + customization |

---

## **Documentation Structure**

```
docs/
├── 00-PRODUCT-SPEC.md (this file)
├── 01-ARCHITECTURE.md
├── 02-IMPLEMENTATION-GUIDE.md
├── features/
│   ├── 01-style-matching.md
│   ├── 02-difficulty-calibration.md
│   ├── 03-anti-duplication.md
│   ├── 04-multi-model-validation.md
│   ├── 05-smart-ocr.md
│   ├── 06-question-bank.md
│   ├── 07-hybrid-generation.md
│   └── 08-advanced-features.md
└── api/
    └── endpoints.md (future)
```

---

## **Contact & Feedback**

- **Repository:** [GitHub link]
- **Issues:** [GitHub Issues]
- **Discussions:** [GitHub Discussions]
- **Email:** team@satquestiongen.com

---

**Last Updated:** November 24, 2024  
**Version:** 1.0.0  
**Status:** In Development
