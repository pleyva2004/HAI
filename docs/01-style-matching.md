# Feature: Style Matching System

## Overview

The Style Matching System ensures that AI-generated questions match the exact style, format, and characteristics of uploaded example questions or desired SAT question patterns.

**Priority:** Priority 1 (Core Differentiation)  
**Status:** To Be Implemented  
**Complexity:** Medium  
**Estimated Time:** 3-4 days

---

## Problem Statement

### Current Issue with ChatGPT
When tutors ask ChatGPT to "generate questions like these examples," the results are inconsistent:
- Different word counts (examples are 50 words, generated are 120 words)
- Different vocabulary levels (examples use grade 8 vocab, generated use grade 11)
- Different number types (examples use integers, generated use decimals)
- Different contexts (examples are abstract algebra, generated are real-world)

### Our Solution
A multi-layered style analysis and matching system that:
1. Extracts precise style characteristics from examples
2. Generates 3x the requested questions
3. Scores each for style match
4. Returns only the top matches (>90% similarity)

---

## User Flow

```
Tutor Input
    ↓
[Upload 3-5 example questions]
    ↓
[Style Analyzer extracts patterns]
    ├─ Word count range: 45-55 words
    ├─ Vocabulary level: Grade 9.2
    ├─ Number complexity: Small integers
    ├─ Context type: Real-world scenarios
    ├─ Question structure: "If X, then what is Y?"
    └─ Distractor patterns: Off-by-one errors
    ↓
[Generate 15 candidate questions]
    ↓
[Score each candidate]
    ├─ Candidate 1: 96% match ✓
    ├─ Candidate 2: 88% match ✓
    ├─ Candidate 3: 72% match ✗
    └─ ...
    ↓
[Return top 5 matches]
    └─ All >90% style consistency
```

---

## Technical Architecture

### Components

#### 1. StyleAnalyzer
**Purpose:** Extract style characteristics from example questions

```python
class StyleAnalyzer:
    def analyze(self, examples: List[str]) -> StyleProfile
    
    # Sub-analyzers
    def _analyze_word_counts(self, examples) -> Tuple[int, int]
    def _analyze_vocabulary(self, examples) -> float
    def _analyze_numbers(self, examples) -> str
    def _analyze_context(self, examples) -> str
    def _extract_structure(self, examples) -> str
    def _analyze_distractors(self, examples) -> str
```

#### 2. StyleProfile (Toon Model)
**Purpose:** Store extracted style characteristics

```python
class StyleProfile(Toon):
    word_count_range: tuple[int, int]        # e.g., (45, 55)
    vocabulary_level: float                   # Flesch-Kincaid grade
    number_complexity: str                    # "integers", "decimals", "fractions"
    context_type: str                         # "real_world", "abstract", "geometric"
    question_structure: str                   # Template pattern
    distractor_patterns: str                  # Common wrong answer types
```

#### 3. StyleMatcher
**Purpose:** Score questions against style profile

```python
class StyleMatcher:
    def score_match(self, question: SATQuestion, profile: StyleProfile) -> float
    def filter_by_style(self, questions: List[SATQuestion], profile: StyleProfile, threshold: float) -> List[SATQuestion]
```

---

## Detailed Implementation

### Step 1: Style Analysis

#### Word Count Analysis
```python
def _analyze_word_counts(self, examples: List[str]) -> Tuple[int, int]:
    """Extract word count range from examples"""
    counts = [len(ex.split()) for ex in examples]
    
    # Use min and max with some tolerance
    min_count = min(counts)
    max_count = max(counts)
    
    return (min_count, max_count)

# Example output: (45, 55) means questions should be 45-55 words
```

#### Vocabulary Level Analysis
```python
import textstat

def _analyze_vocabulary(self, examples: List[str]) -> float:
    """Analyze reading level using Flesch-Kincaid"""
    levels = [textstat.flesch_kincaid_grade(ex) for ex in examples]
    avg_level = sum(levels) / len(levels)
    
    return avg_level

# Example output: 9.2 means grade 9.2 reading level
```

#### Number Complexity Analysis
```python
import re

def _analyze_numbers(self, examples: List[str]) -> str:
    """Classify number types used"""
    has_decimals = any('.' in ex and any(c.isdigit() for c in ex) 
                       for ex in examples)
    has_fractions = any('/' in ex and any(c.isdigit() for c in ex) 
                        for ex in examples)
    has_large = any(re.search(r'\d{3,}', ex) for ex in examples)
    
    if has_fractions:
        return "fractions"
    elif has_decimals:
        return "decimals"
    elif has_large:
        return "large_integers"
    else:
        return "small_integers"

# Example output: "small_integers" means use numbers like 3, 7, 12
```

#### Context Classification
```python
def _analyze_context(self, examples: List[str]) -> str:
    """Classify the context type"""
    
    keywords = {
        'real_world': ['store', 'buy', 'sell', 'person', 'car', 
                       'distance', 'time', 'money', 'hours', 'price'],
        'abstract': ['variable', 'function', 'equation', 'expression', 
                     'value', 'solve'],
        'geometric': ['triangle', 'circle', 'angle', 'line', 'point', 
                      'area', 'perimeter']
    }
    
    scores = {ctx: 0 for ctx in keywords}
    
    for example in examples:
        example_lower = example.lower()
        for ctx, words in keywords.items():
            scores[ctx] += sum(1 for word in words if word in example_lower)
    
    return max(scores, key=scores.get)

# Example output: "real_world" means use shopping, travel scenarios
```

#### Question Structure Extraction
```python
def _extract_structure(self, examples: List[str]) -> str:
    """Extract common structural pattern"""
    
    structures = []
    for ex in examples:
        # Replace numbers with N
        structure = re.sub(r'\d+\.?\d*', 'N', ex)
        # Replace single-letter variables with V
        structure = re.sub(r'\b[a-z]\b', 'V', structure)
        # Normalize whitespace
        structure = re.sub(r'\s+', ' ', structure).strip()
        structures.append(structure)
    
    # Find most common structure (simplified)
    from collections import Counter
    structure_counts = Counter(structures)
    most_common = structure_counts.most_common(1)[0][0]
    
    return most_common

# Example output: "If NV + N = N, what is the value of V?"
```

---

### Step 2: Style Matching Scoring

#### Multi-Factor Scoring Algorithm

```python
def score_match(self, question: SATQuestion, profile: StyleProfile) -> float:
    """
    Score question match to profile (0-1 scale)
    
    Scoring breakdown:
    - Word count: 20%
    - Vocabulary: 20%
    - Numbers: 20%
    - Context: 20%
    - Structure: 20%
    """
    
    scores = []
    
    # 1. Word Count Score (20%)
    word_count = len(question.question.split())
    min_wc, max_wc = profile.word_count_range
    
    if min_wc <= word_count <= max_wc:
        scores.append(0.20)
    else:
        # Partial credit based on distance
        distance = min(abs(word_count - min_wc), abs(word_count - max_wc))
        penalty = min(distance / 10, 0.20)
        scores.append(max(0, 0.20 - penalty))
    
    # 2. Vocabulary Score (20%)
    vocab_level = textstat.flesch_kincaid_grade(question.question)
    vocab_diff = abs(vocab_level - profile.vocabulary_level)
    vocab_score = max(0, 0.20 - (vocab_diff * 0.03))
    scores.append(vocab_score)
    
    # 3. Number Complexity Score (20%)
    number_type = self._classify_numbers(question.question)
    if number_type == profile.number_complexity:
        scores.append(0.20)
    else:
        scores.append(0.08)  # Partial credit
    
    # 4. Context Score (20%)
    context = self._classify_context(question.question)
    if context == profile.context_type:
        scores.append(0.20)
    else:
        scores.append(0.05)  # Small partial credit
    
    # 5. Structure Score (20%)
    structure_similarity = self._compare_structures(
        question.question, 
        profile.question_structure
    )
    scores.append(structure_similarity * 0.20)
    
    total = sum(scores)
    return total
```

#### Structure Comparison
```python
def _compare_structures(self, question: str, template: str) -> float:
    """Compare question structure to template (0-1)"""
    
    # Extract structure from question
    q_structure = re.sub(r'\d+\.?\d*', 'N', question)
    q_structure = re.sub(r'\b[a-z]\b', 'V', q_structure)
    q_structure = re.sub(r'\s+', ' ', q_structure).strip()
    
    # Calculate similarity (simple token overlap)
    q_tokens = set(q_structure.split())
    t_tokens = set(template.split())
    
    if not q_tokens or not t_tokens:
        return 0.5
    
    intersection = len(q_tokens & t_tokens)
    union = len(q_tokens | t_tokens)
    
    jaccard = intersection / union if union > 0 else 0
    
    return jaccard
```

---

### Step 3: Filtering Pipeline

```python
def filter_by_style(
    self,
    questions: List[SATQuestion],
    profile: StyleProfile,
    threshold: float = 0.90
) -> List[SATQuestion]:
    """
    Filter questions that meet style threshold
    
    Args:
        questions: Candidate questions
        profile: Target style
        threshold: Minimum match score (default 0.90 = 90%)
    
    Returns:
        Questions meeting threshold
    """
    
    matched = []
    scores = []
    
    for q in questions:
        score = self.score_match(q, profile)
        scores.append((q, score))
        
        if score >= threshold:
            q.style_match_score = score
            matched.append(q)
    
    # Sort by score (highest first)
    matched.sort(key=lambda q: q.style_match_score, reverse=True)
    
    logger.info(f"Filtered {len(matched)}/{len(questions)} questions by style")
    logger.debug(f"Score range: {min(s[1] for s in scores):.2f} - {max(s[1] for s in scores):.2f}")
    
    return matched
```

---

## Integration with LangGraph

```python
def style_matching_node(state: GraphState) -> GraphState:
    """
    LangGraph node for style matching
    
    Input: state.generated_candidates (large list)
    Output: state.style_filtered_questions (filtered list)
    """
    
    # Initialize services
    analyzer = StyleAnalyzer()
    matcher = StyleMatcher()
    
    # Analyze examples to get style profile
    examples = [state.extracted_text] if state.extracted_text else []
    if state.description:
        examples.append(state.description)
    
    style_profile = analyzer.analyze(examples)
    state.style_profile = style_profile
    
    logger.info(f"Style profile: {style_profile.context_type}, "
                f"{style_profile.number_complexity}, "
                f"vocab grade {style_profile.vocabulary_level:.1f}")
    
    # Filter candidates by style
    matched = matcher.filter_by_style(
        questions=state.generated_candidates,
        profile=style_profile,
        threshold=0.85  # 85% threshold
    )
    
    state.style_filtered_questions = matched
    
    # Store metadata
    state.metadata['style_match_rate'] = len(matched) / len(state.generated_candidates)
    state.metadata['avg_style_score'] = sum(q.style_match_score for q in matched) / len(matched)
    
    return state
```

---

## Testing Strategy

### Unit Tests

```python
# test_style_matching.py

def test_word_count_analysis():
    """Test word count extraction"""
    analyzer = StyleAnalyzer()
    
    examples = [
        "If 3x + 7 = 22, what is the value of x?",
        "A store sells notebooks for $4 each. How many can you buy with $20?",
        "Solve the equation 2y - 5 = 13 for y."
    ]
    
    word_range = analyzer._analyze_word_counts(examples)
    
    assert word_range[0] <= 10, "Min word count should be ≤10"
    assert word_range[1] <= 15, "Max word count should be ≤15"


def test_vocabulary_analysis():
    """Test vocabulary level detection"""
    analyzer = StyleAnalyzer()
    
    simple_examples = ["If x + 2 = 5, what is x?"]
    complex_examples = ["Determine the polynomial's derivative utilizing calculus."]
    
    simple_level = analyzer._analyze_vocabulary(simple_examples)
    complex_level = analyzer._analyze_vocabulary(complex_examples)
    
    assert simple_level < complex_level, "Simple should have lower grade level"


def test_number_complexity():
    """Test number type classification"""
    analyzer = StyleAnalyzer()
    
    assert analyzer._analyze_numbers(["2 + 3 = 5"]) == "small_integers"
    assert analyzer._analyze_numbers(["2.5 + 3.7 = 6.2"]) == "decimals"
    assert analyzer._analyze_numbers(["1/2 + 1/4 = 3/4"]) == "fractions"
    assert analyzer._analyze_numbers(["1000 + 2500 = 3500"]) == "large_integers"


def test_style_matching_score():
    """Test complete style matching"""
    matcher = StyleMatcher()
    
    profile = StyleProfile(
        word_count_range=(10, 15),
        vocabulary_level=9.0,
        number_complexity="small_integers",
        context_type="real_world",
        question_structure="If NV + N = N, what is V?",
        distractor_patterns="off_by_one"
    )
    
    good_question = SATQuestion(
        id="q1",
        question="If John has 3 apples and buys 4 more, how many does he have?",
        choices=QuestionChoice(A="5", B="7", C="8", D="12"),
        correct_answer="B",
        explanation="3 + 4 = 7",
        difficulty=50,
        category="arithmetic"
    )
    
    score = matcher.score_match(good_question, profile)
    
    assert score >= 0.80, f"Good question should score ≥80%, got {score:.2f}"
```

### Integration Tests

```python
def test_complete_style_pipeline():
    """Test full style matching pipeline"""
    
    # Setup
    analyzer = StyleAnalyzer()
    matcher = StyleMatcher()
    
    # Example questions
    examples = [
        "If 3x + 7 = 22, what is x?",
        "If 2y - 5 = 13, what is y?",
        "If 4z + 1 = 17, what is z?"
    ]
    
    # Generate profile
    profile = analyzer.analyze(examples)
    
    # Candidate questions (mix of good and bad matches)
    candidates = [
        # Good matches
        SATQuestion(id="q1", question="If 5a + 3 = 18, what is a?", ...),
        SATQuestion(id="q2", question="If 6b - 2 = 10, what is b?", ...),
        # Bad matches (too long, different style)
        SATQuestion(id="q3", question="A store sells pencils for $0.75 each and erasers for $0.50 each. If Maria buys 8 pencils and 4 erasers, what is the total cost before tax?", ...),
    ]
    
    # Filter
    matched = matcher.filter_by_style(candidates, profile, threshold=0.85)
    
    assert len(matched) == 2, "Should match 2 good questions"
    assert all(q.style_match_score >= 0.85 for q in matched)
```

---

## Performance Considerations

### Speed Optimizations

1. **Caching**: Cache style profiles for common patterns
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_profile(examples_hash: str) -> StyleProfile:
    # Cache analyzed profiles
    pass
```

2. **Batch Processing**: Analyze multiple questions in parallel
```python
from concurrent.futures import ThreadPoolExecutor

def batch_score_questions(questions: List[SATQuestion], profile: StyleProfile):
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(
            lambda q: matcher.score_match(q, profile),
            questions
        ))
    return scores
```

3. **Early Filtering**: Quick checks before full analysis
```python
def quick_filter(question: SATQuestion, profile: StyleProfile) -> bool:
    """Fast preliminary filter"""
    word_count = len(question.question.split())
    min_wc, max_wc = profile.word_count_range
    
    # Expand range by 50% for quick filter
    tolerance = (max_wc - min_wc) * 0.5
    if not (min_wc - tolerance <= word_count <= max_wc + tolerance):
        return False
    
    return True
```

---

## Success Metrics

### Quality Metrics
- **Style consistency**: >90% of generated questions match style
- **Score distribution**: Avg match score >0.85
- **Tutor satisfaction**: >4.5/5 rating on style match

### Performance Metrics
- **Analysis time**: <100ms per profile
- **Scoring time**: <10ms per question
- **Filtering time**: <500ms for 50 questions

---

## Example Output

### Input
```
Examples:
1. "If 3x + 7 = 22, what is the value of x?"
2. "If 2y - 5 = 13, what is the value of y?"
3. "If 4z + 1 = 17, what is the value of z?"

Request: Generate 5 similar questions
```

### System Analysis
```json
{
  "style_profile": {
    "word_count_range": [9, 11],
    "vocabulary_level": 6.8,
    "number_complexity": "small_integers",
    "context_type": "abstract",
    "question_structure": "If NV [+/-] N = N, what is the value of V?",
    "distractor_patterns": "arithmetic_errors"
  }
}
```

### Output with Scores
```
Generated Questions:

1. "If 5a + 3 = 18, what is the value of a?" [Style: 96%]
2. "If 6b - 2 = 10, what is the value of b?" [Style: 94%]
3. "If 7c + 4 = 25, what is the value of c?" [Style: 95%]
4. "If 8d - 6 = 34, what is the value of d?" [Style: 93%]
5. "If 9e + 1 = 37, what is the value of e?" [Style: 94%]

Average style match: 94.4%
All questions use: small integers, abstract algebra, 10-word format
```

---

## Future Enhancements

### Phase 2
- **Visual style matching**: Detect use of graphs, tables, diagrams
- **Layout preservation**: Match spacing, formatting, bullet points
- **Multi-lingual style**: Support different language styles

### Phase 3
- **Style learning**: Improve profile from tutor feedback
- **Custom profiles**: Let tutors create named style templates
- **Style blending**: Mix characteristics from multiple examples

---

## Dependencies

```python
# requirements.txt
textstat>=0.7.3          # Vocabulary analysis
numpy>=1.24.0            # Numerical operations
sentence-transformers     # Future: semantic similarity
```

---

## Related Features

- **Feature 2: Difficulty Calibration** - Often used together with style matching
- **Feature 3: Anti-Duplication** - Complements style matching to ensure variety
- **Feature 7: Hybrid Generation** - Uses style profiles for AI variations

---

**Last Updated:** November 24, 2024  
**Status:** Specification Complete, Ready for Implementation  
**Complexity:** Medium (3-4 days)  
**Dependencies:** None (standalone feature)
