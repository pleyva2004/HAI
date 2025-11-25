#!/usr/bin/env python3
"""Load questions into the question bank"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.toon_models import OfficialSATQuestion, QuestionChoice
from src.services.question_bank import QuestionBankService
from src.utils.config import get_settings


async def load_from_json(file_path: str):
    """
    Load questions from JSON file

    Expected format:
    {
      "questions": [
        {
          "question_id": "pt1_q1",
          "source": "Practice Test 1",
          "category": "algebra",
          "subcategory": "linear_equations",
          "difficulty": 45.5,
          "question_text": "If 3x + 7 = 22, what is x?",
          "choices": {"A": "3", "B": "5", "C": "7", "D": "9"},
          "correct_answer": "B",
          "explanation": "Subtract 7, then divide by 3",
          "national_correct_rate": 75.0,
          "avg_time_seconds": 45,
          "common_wrong_answers": ["A", "C"],
          "tags": ["linear", "one_step"]
        },
        ...
      ]
    }
    """
    print(f"📂 Loading questions from {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    questions = []
    for q_data in data.get("questions", []):
        try:
            question = OfficialSATQuestion(
                question_id=q_data["question_id"],
                source=q_data["source"],
                category=q_data["category"],
                subcategory=q_data.get("subcategory", ""),
                difficulty=float(q_data["difficulty"]),
                question_text=q_data["question_text"],
                choices=QuestionChoice(**q_data["choices"]),
                correct_answer=q_data["correct_answer"],
                explanation=q_data.get("explanation", ""),
                national_correct_rate=float(q_data.get("national_correct_rate", 50.0)),
                avg_time_seconds=int(q_data.get("avg_time_seconds", 60)),
                common_wrong_answers=q_data.get("common_wrong_answers", []),
                tags=q_data.get("tags", []),
            )
            questions.append(question)
        except Exception as e:
            print(f"  ⚠️  Failed to parse question {q_data.get('question_id', '?')}: {e}")

    print(f"✅ Parsed {len(questions)} questions")
    return questions


async def load_questions(file_path: str):
    """Main loading function"""
    print("🚀 SAT Question Bank Loader\n")

    # Check file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Load questions from file
    questions = await load_from_json(file_path)

    if not questions:
        print("❌ No questions loaded")
        return False

    # Connect to database
    settings = get_settings()
    qbank = QuestionBankService(settings.database_url)

    try:
        print("\n🔌 Connecting to database...")
        await qbank.connect()

        # Insert questions
        print(f"\n📥 Inserting {len(questions)} questions...")
        for i, question in enumerate(questions, 1):
            try:
                await qbank.insert_question(question)
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(questions)}")
            except Exception as e:
                print(f"  ⚠️  Failed to insert {question.question_id}: {e}")

        # Verify
        total = await qbank.get_count()
        print(f"\n✅ Loading complete! Total questions in database: {total}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    finally:
        await qbank.disconnect()


# Sample data creator
def create_sample_data(output_file: str = "sample_questions.json"):
    """Create sample question data for testing"""
    sample_questions = {
        "questions": [
            {
                "question_id": "sample_alg_1",
                "source": "Sample Data",
                "category": "algebra",
                "subcategory": "linear_equations",
                "difficulty": 30.0,
                "question_text": "If 3x + 7 = 22, what is the value of x?",
                "choices": {"A": "3", "B": "5", "C": "7", "D": "9"},
                "correct_answer": "B",
                "explanation": "Subtract 7 from both sides to get 3x = 15, then divide by 3 to get x = 5.",
                "national_correct_rate": 75.0,
                "avg_time_seconds": 45,
                "common_wrong_answers": ["C", "A"],
                "tags": ["linear", "single_variable", "basic"],
            },
            {
                "question_id": "sample_alg_2",
                "source": "Sample Data",
                "category": "algebra",
                "subcategory": "linear_equations",
                "difficulty": 45.0,
                "question_text": "If 2(x - 4) + 5 = 17, what is the value of x?",
                "choices": {"A": "6", "B": "8", "C": "10", "D": "12"},
                "correct_answer": "C",
                "explanation": "Expand to get 2x - 8 + 5 = 17, simplify to 2x - 3 = 17, add 3 to get 2x = 20, divide by 2 to get x = 10.",
                "national_correct_rate": 60.0,
                "avg_time_seconds": 60,
                "common_wrong_answers": ["B", "D"],
                "tags": ["linear", "distributive", "multi_step"],
            },
            {
                "question_id": "sample_alg_3",
                "source": "Sample Data",
                "category": "algebra",
                "subcategory": "quadratic",
                "difficulty": 65.0,
                "question_text": "If x² - 5x + 6 = 0, what are the possible values of x?",
                "choices": {
                    "A": "2 and 3",
                    "B": "-2 and -3",
                    "C": "1 and 6",
                    "D": "-1 and -6",
                },
                "correct_answer": "A",
                "explanation": "Factor as (x - 2)(x - 3) = 0, so x = 2 or x = 3.",
                "national_correct_rate": 45.0,
                "avg_time_seconds": 90,
                "common_wrong_answers": ["C", "B"],
                "tags": ["quadratic", "factoring", "roots"],
            },
        ]
    }

    with open(output_file, "w") as f:
        json.dump(sample_questions, f, indent=2)

    print(f"✅ Created sample data: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Load questions: python scripts/load_question_bank.py <json_file>")
        print("  Create sample:  python scripts/load_question_bank.py --create-sample")
        sys.exit(1)

    if sys.argv[1] == "--create-sample":
        output = sys.argv[2] if len(sys.argv) > 2 else "sample_questions.json"
        create_sample_data(output)
    else:
        success = asyncio.run(load_questions(sys.argv[1]))
        sys.exit(0 if success else 1)

