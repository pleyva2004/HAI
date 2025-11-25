# Feature: Hybrid Generation (Real + AI)

## Overview
Combines real SAT questions from the question bank with AI-generated variations to provide unlimited supply while maintaining quality.

**Priority:** Phase 2 (Hybrid)  
**Complexity:** Medium  
**Time:** 3-4 days

## Approach
```python
class HybridGenerator:
    def generate_hybrid_set(self, state: GraphState) -> List[SATQuestion]:
        """Generate 50% real, 50% AI variations"""
        
        num_real = state.num_questions // 2
        num_synthetic = state.num_questions - num_real
        
        # 1. Find similar real questions
        real_questions = self.qbank.search_similar(
            query=state.description,
            category=state.analysis.category,
            top_k=num_real
        )
        
        # 2. Use top real questions as templates
        synthetic_questions = []
        for real_q in real_questions[:3]:
            variations = self.generate_variations(real_q, num=num_synthetic // 3)
            synthetic_questions.extend(variations)
        
        # 3. Combine and return
        return self.mix(real_questions, synthetic_questions)
    
    def generate_variations(self, template: OfficialSATQuestion, num: int):
        """Generate AI variations based on real question"""
        
        prompt = f"""Generate {num} NEW questions similar to this OFFICIAL SAT question:

        Question: {template.question_text}
        Category: {template.category}
        Difficulty: {template.difficulty}/100
        National correct rate: {template.national_correct_rate}%
        
        Requirements:
        - Test the SAME concept
        - Match difficulty (based on correct rate)
        - Use DIFFERENT numbers and contexts
        - Follow the SAME structural pattern
        """
        
        return self.llm.with_structured_output(GeneratedQuestions).invoke(prompt).questions
```

## Output Format
```
Question 1 [REAL]
Source: Official SAT Practice Test 7, Q14
Difficulty: 67/100, Correct rate: 54%

Question 2 [AI-GENERATED VARIATION]
Based on: Practice Test 7, Q14
Predicted difficulty: 65/100
Style match: 94%
```

## Metrics
- **Mix ratio**: 50% real, 50% AI
- **Quality consistency**: AI variations within ±5 difficulty points
- **Style match**: >90% similarity to template

**Dependencies:** Question Bank, LLM Service
