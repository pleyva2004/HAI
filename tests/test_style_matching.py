"""Tests for Style Matching (Feature 1)"""

import pytest
from src.models.toon_models import SATQuestion, QuestionChoice
from src.services.style_analyzer import StyleAnalyzer, StyleMatcher


def test_style_analyzer_word_count():
    """Test word count analysis"""
    analyzer = StyleAnalyzer()

    examples = [
        "If 3x + 7 = 22, what is x?",  # 8 words
        "Solve for y: 2y - 5 = 13",  # 6 words
        "What is the value of z if 4z = 20?",  # 10 words
    ]

    profile = analyzer.analyze(examples)

    assert profile.word_count_range[0] <= 6
    assert profile.word_count_range[1] >= 10


def test_style_analyzer_number_complexity():
    """Test number complexity detection"""
    analyzer = StyleAnalyzer()

    # Fractions
    examples_frac = ["If x/2 + 3/4 = 5/8, find x"]
    profile = analyzer.analyze(examples_frac)
    assert profile.number_complexity == "fractions"

    # Decimals
    examples_dec = ["If x = 3.5 and y = 2.75, what is x + y?"]
    profile = analyzer.analyze(examples_dec)
    assert profile.number_complexity == "decimals"

    # Small integers
    examples_int = ["If x + 5 = 12, what is x?"]
    profile = analyzer.analyze(examples_int)
    assert profile.number_complexity == "small_integers"


def test_style_analyzer_context():
    """Test context classification"""
    analyzer = StyleAnalyzer()

    # Real world
    examples_real = ["A store sells apples for $2 each. If John buys 5 apples, how much does he pay?"]
    profile = analyzer.analyze(examples_real)
    assert profile.context_type == "real_world"

    # Geometric
    examples_geo = ["A triangle has angles of 50° and 60°. What is the third angle?"]
    profile = analyzer.analyze(examples_geo)
    assert profile.context_type == "geometric"

    # Abstract
    examples_abs = ["Solve for x: 3x + 7 = 22"]
    profile = analyzer.analyze(examples_abs)
    assert profile.context_type == "abstract"


def test_style_matcher_scoring():
    """Test style match scoring"""
    analyzer = StyleAnalyzer()
    matcher = StyleMatcher()

    # Create style profile
    examples = [
        "If 3x + 7 = 22, what is the value of x?",
        "Solve for y in the equation 2y - 5 = 13.",
    ]
    profile = analyzer.analyze(examples)

    # Test similar question
    similar_q = SATQuestion(
        id="test1",
        question="If 4x + 9 = 25, what is the value of x?",
        choices=QuestionChoice(A="2", B="4", C="6", D="8"),
        correct_answer="B",
        explanation="Test",
        difficulty=50.0,
        category="algebra",
    )

    score = matcher.score_match(similar_q, profile)
    assert score > 0.5, "Similar question should have good match score"


def test_style_matcher_filtering():
    """Test style-based filtering"""
    analyzer = StyleAnalyzer()
    matcher = StyleMatcher()

    # Create profile
    examples = ["If x + 5 = 12, what is x?"]
    profile = analyzer.analyze(examples)

    # Create questions
    questions = [
        SATQuestion(
            id="match1",
            question="If y + 7 = 15, what is y?",
            choices=QuestionChoice(A="6", B="8", C="10", D="12"),
            correct_answer="B",
            explanation="",
            difficulty=50.0,
            category="algebra",
        ),
        SATQuestion(
            id="nomatch1",
            question="A store sells books for $15.99 each and offers a 25% discount on purchases of 3 or more books. If Sarah buys 4 books, how much does she pay in total before tax?",
            choices=QuestionChoice(A="$47.97", B="$48.00", C="$63.96", D="$64.00"),
            correct_answer="A",
            explanation="",
            difficulty=50.0,
            category="algebra",
        ),
    ]

    filtered = matcher.filter_by_style(questions, profile, threshold=0.5)

    # The simple question should match better than the complex one
    assert any(q.id == "match1" for q in filtered)


def test_style_analyzer_empty_input():
    """Test handling of empty input"""
    analyzer = StyleAnalyzer()

    profile = analyzer.analyze([])

    # Should return default profile
    assert profile.word_count_range == (10, 50)
    assert profile.context_type in ["real_world", "abstract", "geometric"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

