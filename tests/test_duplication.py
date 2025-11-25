"""Tests for Anti-Duplication (Feature 3)"""

import pytest
from src.models.toon_models import SATQuestion, QuestionChoice
from src.services.duplication_detector import DuplicationDetector


def create_test_question(id: str, question: str) -> SATQuestion:
    """Helper to create test question"""
    return SATQuestion(
        id=id,
        question=question,
        choices=QuestionChoice(A="A", B="B", C="C", D="D"),
        correct_answer="A",
        explanation="Test",
        difficulty=50.0,
        category="algebra",
    )


def test_fingerprint_generation():
    """Test structural fingerprint generation"""
    detector = DuplicationDetector()

    question = create_test_question("test1", "If 3x + 7 = 22, what is x?")
    fingerprint = detector.get_fingerprint(question)

    assert fingerprint.structure_hash is not None
    assert fingerprint.concept_pattern is not None
    assert fingerprint.context_type in ["real_world", "abstract", "geometric"]


def test_identical_question_detection():
    """Test detection of identical questions"""
    detector = DuplicationDetector()

    q1 = create_test_question("q1", "If 3x + 7 = 22, what is x?")
    q2 = create_test_question("q2", "If 3x + 7 = 22, what is x?")  # Identical

    detector.add_to_database(q1)

    assert detector.is_duplicate(q2, threshold=0.85)


def test_similar_question_detection():
    """Test detection of very similar questions"""
    detector = DuplicationDetector()

    q1 = create_test_question("q1", "If 3x + 7 = 22, what is the value of x?")
    q2 = create_test_question("q2", "If 3x + 7 = 22, find x?")  # Very similar

    detector.add_to_database(q1)

    assert detector.is_duplicate(q2, threshold=0.85)


def test_different_numbers_same_structure():
    """Test detection of questions with same structure but different numbers"""
    detector = DuplicationDetector()

    q1 = create_test_question("q1", "If 3x + 7 = 22, what is x?")
    q2 = create_test_question("q2", "If 5x + 9 = 34, what is x?")  # Same structure

    detector.add_to_database(q1)

    # Should detect as duplicate due to structural similarity
    is_dup = detector.is_duplicate(q2, threshold=0.80)

    # Structural fingerprints should match
    fp1 = detector.get_fingerprint(q1)
    fp2 = detector.get_fingerprint(q2)
    assert fp1.structure_hash == fp2.structure_hash


def test_different_question_not_duplicate():
    """Test that different questions are not marked as duplicates"""
    detector = DuplicationDetector()

    q1 = create_test_question("q1", "If 3x + 7 = 22, what is x?")
    q2 = create_test_question(
        "q2", "A triangle has angles of 50° and 60°. What is the third angle?"
    )

    detector.add_to_database(q1)

    assert not detector.is_duplicate(q2, threshold=0.85)


def test_filter_duplicates():
    """Test filtering duplicate questions from a list"""
    detector = DuplicationDetector()

    questions = [
        create_test_question("q1", "If 3x + 7 = 22, what is x?"),
        create_test_question("q2", "If 3x + 7 = 22, find x?"),  # Duplicate of q1
        create_test_question("q3", "If 5y - 3 = 17, what is y?"),  # Different
        create_test_question("q4", "If 5y - 3 = 17, find the value of y?"),  # Dup of q3
        create_test_question("q5", "What is the area of a circle with radius 5?"),  # Different
    ]

    unique = detector.filter_duplicates(questions, threshold=0.85)

    # Should have fewer questions after filtering
    assert len(unique) < len(questions)
    # Should have at least the truly unique ones
    assert len(unique) >= 3


def test_similarity_score():
    """Test similarity scoring between questions"""
    detector = DuplicationDetector()

    q1 = create_test_question("q1", "If 3x + 7 = 22, what is x?")
    q2 = create_test_question("q2", "If 3x + 7 = 22, find x?")
    q3 = create_test_question("q3", "What is the area of a square with side 5?")

    # Similar questions should have high similarity
    sim_12 = detector.get_similarity_score(q1, q2)
    assert sim_12 > 0.8

    # Different questions should have lower similarity
    sim_13 = detector.get_similarity_score(q1, q3)
    assert sim_13 < 0.5


def test_find_similar():
    """Test finding similar questions in database"""
    detector = DuplicationDetector()

    questions = [
        create_test_question("q1", "If 3x + 7 = 22, what is x?"),
        create_test_question("q2", "If 5x - 3 = 17, what is x?"),
        create_test_question("q3", "What is the area of a circle with radius 5?"),
    ]

    for q in questions:
        detector.add_to_database(q)

    query = create_test_question("query", "If 4x + 2 = 18, find x?")

    similar = detector.find_similar(query, top_k=2)

    assert len(similar) == 2
    # First result should be algebraic equation questions
    assert similar[0][0].id in ["q1", "q2"]
    # Similarity score should be provided
    assert 0 <= similar[0][1] <= 1


def test_detector_statistics():
    """Test getting detector statistics"""
    detector = DuplicationDetector()

    questions = [
        create_test_question(f"q{i}", f"Test question {i}") for i in range(5)
    ]

    for q in questions:
        detector.add_to_database(q)

    stats = detector.get_statistics()

    assert stats["total_questions"] == 5
    assert stats["cached_embeddings"] == 5
    assert stats["model"] > 0  # Embedding dimension


def test_clear_database():
    """Test clearing the detection database"""
    detector = DuplicationDetector()

    questions = [create_test_question(f"q{i}", f"Question {i}") for i in range(3)]

    for q in questions:
        detector.add_to_database(q)

    assert len(detector.question_database) == 3

    detector.clear_database()

    assert len(detector.question_database) == 0
    assert len(detector.embeddings_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

