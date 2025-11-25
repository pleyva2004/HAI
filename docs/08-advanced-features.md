# Feature: Advanced Features (Style Transfer + Performance Selection)

## Overview
Advanced capabilities for style transfer from real questions and performance-based question selection.

**Priority:** Phase 3 (Advanced)  
**Complexity:** High  
**Time:** 5-6 days

## Style Transfer
```python
class StyleTransfer:
    def generate_with_style_transfer(self, state: GraphState):
        """Use real questions as style templates"""
        
        # Find 5 real questions with target style
        style_templates = self.qbank.search_similar(
            query=state.description,
            category=state.analysis.category,
            top_k=5
        )
        
        # Extract common patterns
        style_patterns = self.extract_patterns(style_templates)
        
        # Generate new questions following patterns
        prompt = f"""Generate questions matching these patterns:
        
        Style analysis of {len(style_templates)} official SAT questions:
        - Structure: {style_patterns.structure}
        - Vocabulary: {style_patterns.vocabulary_level}
        - Numbers: {style_patterns.number_types}
        - Context: {style_patterns.context_type}
        
        Generate questions indistinguishable from official SAT.
        """
        
        return self.llm.invoke(prompt)
```

## Performance-Based Selection
```python
class PerformanceSelector:
    def select_optimal_practice_set(
        self,
        student_level: float,
        category: str,
        num_questions: int
    ):
        """Select questions for optimal learning (Zone of Proximal Development)"""
        
        # Target difficulty: student's level + 10-20 points
        target_range = (student_level + 10, student_level + 20)
        
        # Get questions in ZPD
        questions = self.qbank.get_by_category(
            category=category,
            difficulty_range=target_range
        )
        
        # Mix: 70% focus area, 30% review
        primary = [q for q in questions if q.subcategory == category][:int(num_questions * 0.7)]
        review = [q for q in questions if q.subcategory != category][:int(num_questions * 0.3)]
        
        # Order: easy to hard
        selected = sorted(primary + review, key=lambda q: q.difficulty)
        
        return selected
```

## Features
- **Style Templates**: Learn from 100+ real questions
- **Pattern Extraction**: Identify common structures
- **ZPD Targeting**: Optimal difficulty for learning
- **Adaptive Sequencing**: Order questions for progression

**Dependencies:** Question Bank, ML models
