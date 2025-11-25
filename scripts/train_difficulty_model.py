#!/usr/bin/env python3
"""Train the difficulty calibration model"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.question_bank import QuestionBankService
from src.services.difficulty_calibrator import DifficultyCalibrator
from src.utils.config import get_settings


async def train_model():
    """Train difficulty model on question bank"""
    print("🚀 Difficulty Model Training\n")

    settings = get_settings()

    # Connect to question bank
    print("🔌 Connecting to question bank...")
    qbank = QuestionBankService(settings.database_url)
    await qbank.connect()

    try:
        # Get all questions for training
        print("📚 Loading training data...")
        categories = await qbank.get_categories()

        all_questions = []
        for category in categories:
            questions = await qbank.get_by_category(category, limit=1000)
            all_questions.extend(questions)

        print(f"✅ Loaded {len(all_questions)} questions for training")

        if len(all_questions) < 50:
            print("⚠️  Warning: Small training set. Recommend at least 100+ questions.")

        # Train model
        print("\n🤖 Training Random Forest model...")
        calibrator = DifficultyCalibrator()
        calibrator.train(all_questions)

        # Save model
        model_path = settings.difficulty_model_path
        print(f"\n💾 Saving model to {model_path}...")

        # Create directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        calibrator.save_model(model_path)

        print("\n✅ Training complete!")
        print(f"   Model saved: {model_path}")
        print(f"   Training samples: {len(all_questions)}")

        # Test predictions
        print("\n🧪 Testing predictions on sample questions...")
        test_questions = all_questions[:5]

        for q in test_questions:
            # Convert to SATQuestion for prediction
            from src.models.toon_models import SATQuestion

            sat_q = SATQuestion(
                id=q.question_id,
                question=q.question_text,
                choices=q.choices,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                difficulty=0,  # Will be predicted
                category=q.category,
            )

            predicted = calibrator.predict(sat_q)
            actual = q.difficulty
            error = abs(predicted - actual)

            print(
                f"  {q.question_id}: "
                f"Predicted={predicted:.1f}, Actual={actual:.1f}, "
                f"Error={error:.1f}"
            )

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await qbank.disconnect()


if __name__ == "__main__":
    success = asyncio.run(train_model())
    sys.exit(0 if success else 1)

