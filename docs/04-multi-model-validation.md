# Feature: Multi-Model Validation

## Overview
Cross-validates generated questions using multiple LLMs (GPT-4 + Claude) to ensure correctness, eliminate ambiguity, and improve quality.

**Priority:** Priority 2 (Polish)  
**Complexity:** Medium  
**Time:** 2-3 days

## Problem
- Single LLM can hallucinate wrong answers
- No verification of correctness
- May create ambiguous questions

## Solution
```python
class MultiModelValidator:
    def __init__(self):
        self.models = {
            'gpt4': ChatOpenAI(model="gpt-4o"),
            'claude': ChatAnthropic(model="claude-sonnet-4-20250514")
        }
    
    def validate_question(self, question: SATQuestion) -> ValidationResult:
        """Validate using both models"""
        
        prompt = f"""Validate this SAT question:
        Q: {question.question}
        A) {question.choices.A}  B) {question.choices.B}
        C) {question.choices.C}  D) {question.choices.D}
        Claimed answer: {question.correct_answer}
        
        Check: 1) Is answer correct? 2) Only one correct answer?
        3) Are distractors plausible? 4) Is question unambiguous?
        
        Respond: VALID or INVALID with reason.
        """
        
        results = []
        for name, model in self.models.items():
            result = model.invoke(prompt).content
            results.append(('VALID' in result, result))
        
        # Both must agree on VALID
        is_valid = all(r[0] for r in results)
        
        return ValidationResult(
            is_valid=is_valid,
            validator_agreement=len([r for r in results if r[0]]) / len(results),
            feedback=[r[1] for r in results]
        )
```

## Metrics
- **Validation rate**: >95% questions pass
- **Agreement rate**: >90% between models
- **Time**: <5 seconds per question

**Dependencies:** OpenAI, Anthropic APIs
