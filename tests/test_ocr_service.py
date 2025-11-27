"""Tests for DeepSeek-OCR parsing helpers."""

import pytest

pytest.importorskip("torch")  # Ensure optional dependency is available before importing the service

from src.services.ocr_service import OCRService


def test_parse_markdown_questions_extracts_choices():
    service = OCRService()
    markdown = """
## Question 1. Solve for x: 2x + 3 = 11

- A) 3
- B) 4
- C) 5
- D) 6
"""

    questions = service._parse_markdown_questions(markdown)

    assert len(questions) == 1
    question = questions[0]
    assert "Solve for x" in question.text
    assert question.choices["B"] == "4"


def test_detect_questions_fallback_parses_plain_text():
    service = OCRService()
    text = """
1. What is 5 + 4?
A) 7
B) 8
C) 9
D) 10

2. What is 6 * 3?
A) 12
B) 15
C) 18
D) 21
"""

    questions = service._detect_questions(text)

    assert len(questions) == 2
    assert "5 + 4" in questions[0].text
    assert questions[1].choices["B"] == "15"

