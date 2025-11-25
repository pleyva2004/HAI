"""Tests for Difficulty Calibration (Feature 2)"""

import pytest
import numpy as np
from src.models.toon_models import OfficialSATQuestion, SATQuestion, QuestionChoice
from src.services.difficulty_calibrator import DifficultyCalibrator


def create_sample_official_question(difficulty: float, correct_rate: float) -> OfficialSATQuestion:
    """Helper to create sample official question"""
    return OfficialSATQuestion(
        question_id=f"test_{difficulty}",
        source="Test",
        category="algebra",
        subcategory="linear",
        difficulty=difficulty,
        question_text="Test question " * 10,  # Vary length
        choices=QuestionChoice(A="A", B="B", C="C", D="D"),
        correct_answer="A",
        explanation="Test",
        national_correct_rate=correct_rate,
        avg_time_seconds=60,
        common_wrong_answers=[],
        tags=[],
    )


def test_feature_extraction():
    """Test feature extraction from questions"""
    calibrator = DifficultyCalibrator()

    question = SATQuestion(
        id="test1",
        question="If 3x + 7 = 22, what is the value of x?",
        choices=QuestionChoice(A="3", B="5", C="7", D="9"),
        correct_answer="B",
        explanation="Test",
        difficulty=50.0,
        category="algebra_linear",
    )

    features = calibrator.extract_features(question)

    # Should return 7 features
    assert len(features) == 7
    assert all(isinstance(f, (int, float, np.number)) for f in features)

    # Word count should be > 0
    assert features[0] > 0
    # Character count should be > 0
    assert features[1] > 0


def test_difficulty_calibrator_training():
    """Test model training"""
    calibrator = DifficultyCalibrator()

    # Create training data with clear difficulty patterns
    training_questions = [
        create_sample_official_question(20.0, 80.0),  # Easy
        create_sample_official_question(30.0, 70.0),
        create_sample_official_question(50.0, 50.0),  # Medium
        create_sample_official_question(70.0, 30.0),
        create_sample_official_question(80.0, 20.0),  # Hard
    ]

    # Train model
    calibrator.train(training_questions)

    assert calibrator.is_trained
    assert calibrator.model is not None


def test_difficulty_prediction():
    """Test difficulty prediction"""
    calibrator = DifficultyCalibrator()

    # Create and train on sample data
    training_questions = []
    for i in range(10):
        diff = 20 + i * 8  # Range from 20 to 92
        correct_rate = 100 - diff  # Inverse relationship
        training_questions.append(create_sample_official_question(diff, correct_rate))

    calibrator.train(training_questions)

    # Test prediction
    test_question = SATQuestion(
        id="test",
        question="If 3x + 7 = 22, what is x?",
        choices=QuestionChoice(A="3", B="5", C="7", D="9"),
        correct_answer="B",
        explanation="Test",
        difficulty=0.0,  # Will be predicted
        category="algebra",
    )

    difficulty = calibrator.predict(test_question)

    # Should return a valid difficulty score
    assert 0 <= difficulty <= 100


def test_difficulty_calibration_filtering():
    """Test filtering questions by target difficulty"""
    calibrator = DifficultyCalibrator()

    # Train model
    training_questions = [
        create_sample_official_question(i * 10, 100 - i * 10) for i in range(1, 10)
    ]
    calibrator.train(training_questions)

    # Create test questions
    test_questions = [
        SATQuestion(
            id=f"q{i}",
            question=f"Test question {i} " * (5 + i),  # Vary length
            choices=QuestionChoice(A="A", B="B", C="C", D="D"),
            correct_answer="A",
            explanation="",
            difficulty=0.0,
            category="algebra",
        )
        for i in range(10)
    ]

    # Calibrate to target difficulty
    target = 50.0
    tolerance = 15.0

    calibrated = calibrator.calibrate_questions(test_questions, target, tolerance)

    # All returned questions should be within tolerance
    for q in calibrated:
        assert abs(q.difficulty - target) <= tolerance


def test_batch_prediction():
    """Test batch prediction performance"""
    calibrator = DifficultyCalibrator()

    # Train
    training_questions = [
        create_sample_official_question(i * 10, 100 - i * 10) for i in range(1, 10)
    ]
    calibrator.train(training_questions)

    # Create many test questions
    test_questions = [
        SATQuestion(
            id=f"q{i}",
            question=f"Test question number {i}",
            choices=QuestionChoice(A="A", B="B", C="C", D="D"),
            correct_answer="A",
            explanation="",
            difficulty=0.0,
            category="algebra",
        )
        for i in range(50)
    ]

    # Batch predict
    difficulties = calibrator.predict_batch(test_questions)

    assert len(difficulties) == 50
    assert all(0 <= d <= 100 for d in difficulties)


def test_model_persistence(tmp_path):
    """Test saving and loading model"""
    calibrator = DifficultyCalibrator()

    # Train model
    training_questions = [
        create_sample_official_question(i * 10, 100 - i * 10) for i in range(1, 10)
    ]
    calibrator.train(training_questions)

    # Save model
    model_path = str(tmp_path / "test_model.pkl")
    calibrator.save_model(model_path)

    # Load in new calibrator
    calibrator2 = DifficultyCalibrator()
    calibrator2.load_model(model_path)

    assert calibrator2.is_trained

    # Predictions should be same
    test_q = SATQuestion(
        id="test",
        question="Test question",
        choices=QuestionChoice(A="A", B="B", C="C", D="D"),
        correct_answer="A",
        explanation="",
        difficulty=0.0,
        category="algebra",
    )

    diff1 = calibrator.predict(test_q)
    diff2 = calibrator2.predict(test_q)

    assert abs(diff1 - diff2) < 0.01  # Should be essentially the same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

