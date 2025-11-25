# Feature: Anti-Duplication System

## Overview

Prevents repetitive question generation using semantic embeddings and structural fingerprinting to ensure maximum variety and uniqueness across generated questions.

**Priority:** Priority 1 (Core Differentiation)  
**Status:** To Be Implemented  
**Complexity:** Medium  
**Estimated Time:** 3-4 days

---

## Problem Statement

### Issue with Pure LLM Generation
- LLMs often repeat patterns: "If 3x + 7 = 22..." then "If 4x + 8 = 24..." (same structure, different numbers)
- Students get bored with repetitive formats
- Reduces educational value (predictability)

### Our Solution
Two-layer detection system:
1. **Semantic similarity** - Embeddings to detect meaning similarity
2. **Structural fingerprinting** - Hash-based detection of identical patterns

---

## Technical Approach

### Layer 1: Semantic Similarity

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticDuplicateDetector:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_database = []
    
    def is_semantic_duplicate(self, new_question: str, threshold: float = 0.85) -> bool:
        """Check if semantically too similar to existing questions"""
        
        new_embedding = self.embedder.encode(new_question)
        
        for existing in self.question_database:
            existing_embedding = self.embedder.encode(existing)
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]
            
            if similarity > threshold:
                return True
        
        return False
```

### Layer 2: Structural Fingerprinting

```python
import hashlib
import re

class StructuralDuplicateDetector:
    def __init__(self):
        self.fingerprints = set()
    
    def get_fingerprint(self, question: str) -> str:
        """Extract structural signature"""
        
        # Remove numbers
        structure = re.sub(r'\d+\.?\d*', 'N', question)
        
        # Remove single-letter variables
        structure = re.sub(r'\b[a-z]\b', 'V', structure)
        
        # Normalize whitespace
        structure = re.sub(r'\s+', ' ', structure).strip()
        
        # Hash the structure
        fingerprint = hashlib.md5(structure.encode()).hexdigest()
        
        return fingerprint
    
    def is_structural_duplicate(self, question: str) -> bool:
        """Check if structure already seen"""
        
        fingerprint = self.get_fingerprint(question)
        
        if fingerprint in self.fingerprints:
            return True
        
        self.fingerprints.add(fingerprint)
        return False
```

### Combined Detection

```python
class DuplicationDetector:
    def __init__(self):
        self.semantic_detector = SemanticDuplicateDetector()
        self.structural_detector = StructuralDuplicateDetector()
    
    def is_duplicate(self, question: SATQuestion, threshold: float = 0.85) -> bool:
        """Check both semantic and structural duplication"""
        
        # Check semantic similarity
        if self.semantic_detector.is_semantic_duplicate(question.question, threshold):
            logger.debug(f"Semantic duplicate detected: {question.id}")
            return True
        
        # Check structural similarity
        if self.structural_detector.is_structural_duplicate(question.question):
            logger.debug(f"Structural duplicate detected: {question.id}")
            return True
        
        # Not a duplicate - add to database
        self.semantic_detector.question_database.append(question.question)
        
        return False
    
    def filter_duplicates(
        self,
        questions: List[SATQuestion],
        threshold: float = 0.85
    ) -> List[SATQuestion]:
        """Remove duplicate questions from list"""
        
        unique = []
        
        for q in questions:
            if not self.is_duplicate(q, threshold):
                unique.append(q)
        
        logger.info(f"Filtered {len(questions) - len(unique)} duplicates, "
                   f"{len(unique)} unique questions remain")
        
        return unique
```

---

## Integration with LangGraph

```python
def anti_duplication_node(state: GraphState) -> GraphState:
    """Filter duplicate questions"""
    
    detector = DuplicationDetector()
    
    # Filter duplicates
    unique_questions = detector.filter_duplicates(
        questions=state.validated_questions,
        threshold=0.85
    )
    
    state.filtered_questions = unique_questions
    
    # Store metadata
    state.metadata['duplication_rate'] = 1 - (len(unique_questions) / len(state.validated_questions))
    
    return state
```

---

## Example

### Input (Generated Questions)
```
1. "If 3x + 7 = 22, what is x?" 
2. "If 4x + 8 = 24, what is x?"  [DUPLICATE - same structure]
3. "If 3x + 7 = 22, find x."      [DUPLICATE - semantic similarity 98%]
4. "A store sells apples for $3 each. How many can you buy with $15?"
5. "If 2y - 5 = 13, what is y?"
```

### Output (After Filtering)
```
1. "If 3x + 7 = 22, what is x?"
4. "A store sells apples for $3 each. How many can you buy with $15?"
5. "If 2y - 5 = 13, what is y?"

Filtered: 2 duplicates (40% duplication rate)
```

---

## Success Metrics
- **Duplication rate**: <5% structural duplicates
- **Semantic variety**: >0.80 average pairwise distance
- **Speed**: <100ms for 50 questions

---

**Dependencies:** sentence-transformers, sklearn  
**Complexity:** Medium (3-4 days)
