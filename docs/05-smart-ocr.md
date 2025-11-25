# Feature: Smart OCR with Question Boundaries

## Overview
Enhanced OCR that detects individual question boundaries in PDFs/images, preserves structure, and extracts multiple questions separately.

**Priority:** Priority 2 (Polish)  
**Complexity:** High  
**Time:** 4-5 days

## Problem
- Basic OCR loses structure (tables, graphs)
- Cannot separate multiple questions
- Misses visual elements

## Solution
```python
class SmartOCR:
    def __init__(self):
        self.chandra = Chandra()
        self.layout_detector = LayoutLMv3()  # Document layout AI
    
    def extract_with_structure(self, file_path: str) -> List[ExtractedQuestion]:
        """Extract questions with boundaries detected"""
        
        # 1. Basic OCR
        text = self.chandra.process_document(file_path)
        
        # 2. Detect question regions using layout model
        regions = self.layout_detector.detect_questions(file_path)
        
        # 3. For each region, extract structured data
        questions = []
        for region in regions:
            q = ExtractedQuestion(
                text=self.extract_text(region),
                choices=self.extract_choices(region),
                has_table=self.detect_table(region),
                has_graph=self.detect_graph(region),
                bbox=region.bbox
            )
            questions.append(q)
        
        return questions
    
    def detect_question_boundaries(self, layout):
        """Find question separators (numbers, whitespace)"""
        boundaries = []
        
        # Look for patterns: "1.", "2.", etc.
        for element in layout.elements:
            if element.type == "question_number":
                boundaries.append(element.bbox)
        
        return boundaries
```

## Features
- Detect question numbers (1., 2., Q1, etc.)
- Preserve tables and graphs
- Extract multiple questions from one document
- Handle multi-column layouts

## Metrics
- **Boundary detection accuracy**: >90%
- **Structure preservation**: >85%
- **Speed**: <5 seconds per page

**Dependencies:** Chandra, LayoutLMv3, OpenCV
