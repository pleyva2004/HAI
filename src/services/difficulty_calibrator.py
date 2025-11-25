"""Difficulty Calibration - Feature 2: ML-based difficulty prediction"""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Union

import numpy as np
import textstat
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from ..models.toon_models import OfficialSATQuestion, SATQuestion
from ..utils.helpers import extract_numbers, extract_variables, count_operations

logger = logging.getLogger(__name__)


class DifficultyCalibrator:
    """ML-based difficulty prediction using Random Forest"""

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "word_count",
            "char_count",
            "num_count",
            "var_count",
            "vocab_level",
            "operation_count",
            "concept_depth",
        ]
        logger.info("Difficulty Calibrator initialized")

    def train(self, training_questions: List[OfficialSATQuestion]):
        """
        Train difficulty prediction model on real SAT questions

        Args:
            training_questions: Questions with known difficulty/correct rates
        """
        if not training_questions:
            logger.warning("No training data provided")
            return

        logger.info(f"Training difficulty model on {len(training_questions)} questions")

        # Extract features and labels
        X = []
        y = []

        for q in training_questions:
            try:
                features = self.extract_features(q)
                X.append(features)

                # Convert correct rate to difficulty score (inverse relationship)
                # Higher correct rate = easier = lower difficulty
                difficulty = 100 - q.national_correct_rate
                y.append(difficulty)
            except Exception as e:
                logger.warning(f"Failed to extract features from {q.question_id}: {e}")
                continue

        if not X:
            logger.error("No valid training samples extracted")
            return

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate performance
        y_pred = self.model.predict(X_scaled)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        logger.info(
            f"Training complete - RMSE: {rmse:.2f}, R²: {r2:.3f}, "
            f"Target: RMSE < 10, R² > 0.75"
        )

        # Log feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for name, importance in zip(self.feature_names, importances):
                logger.debug(f"Feature '{name}' importance: {importance:.3f}")

    def extract_features(
        self, question: Union[OfficialSATQuestion, SATQuestion]
    ) -> np.ndarray:
        """
        Extract 7 numerical features from a question

        Features:
        1. Word count
        2. Character count
        3. Number of numerical values
        4. Number of variables
        5. Vocabulary level (Flesch-Kincaid grade)
        6. Operation count (complexity)
        7. Concept depth (category complexity)

        Args:
            question: Question to extract features from

        Returns:
            NumPy array of 7 features
        """
        # Get question text and category
        if isinstance(question, OfficialSATQuestion):
            text = question.question_text
            category = question.category
        else:
            text = question.question
            category = question.category

        # Feature 1: Word count
        word_count = len(text.split())

        # Feature 2: Character count
        char_count = len(text)

        # Feature 3: Number of numerical values
        numbers = extract_numbers(text)
        num_count = len(numbers)

        # Feature 4: Number of variables
        variables = extract_variables(text)
        var_count = len(set(variables))  # Unique variables

        # Feature 5: Vocabulary level
        try:
            vocab_level = textstat.flesch_kincaid_grade(text)
        except Exception:
            vocab_level = 10.0  # Default

        # Feature 6: Operation count (mathematical operations)
        operation_count = count_operations(text)

        # Feature 7: Concept depth (complexity of category)
        # More specific categories (with underscores) are deeper
        concept_depth = len(category.split("_")) if category else 1

        features = np.array(
            [
                word_count,
                char_count,
                num_count,
                var_count,
                vocab_level,
                operation_count,
                concept_depth,
            ]
        )

        return features

    def predict(self, question: SATQuestion) -> float:
        """
        Predict difficulty of a question

        Args:
            question: Question to evaluate

        Returns:
            Predicted difficulty (0-100 scale)
        """
        if not self.is_trained:
            logger.warning(
                "Model not trained, returning default difficulty based on category"
            )
            # Simple heuristic fallback
            category_difficulties = {
                "algebra": 45.0,
                "geometry": 50.0,
                "statistics": 40.0,
                "advanced_math": 65.0,
            }
            return category_difficulties.get(question.category.lower(), 50.0)

        try:
            features = self.extract_features(question)
            features_scaled = self.scaler.transform([features])

            difficulty = self.model.predict(features_scaled)[0]

            # Clip to valid range [0, 100]
            difficulty = np.clip(difficulty, 0, 100)

            logger.debug(f"Predicted difficulty: {difficulty:.1f}/100")
            return float(difficulty)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 50.0  # Default middle difficulty

    def predict_batch(self, questions: List[SATQuestion]) -> List[float]:
        """
        Predict difficulty for multiple questions efficiently

        Args:
            questions: List of questions

        Returns:
            List of predicted difficulties
        """
        if not self.is_trained:
            return [self.predict(q) for q in questions]

        try:
            # Extract all features at once
            X = np.array([self.extract_features(q) for q in questions])
            X_scaled = self.scaler.transform(X)

            # Batch prediction
            difficulties = self.model.predict(X_scaled)
            difficulties = np.clip(difficulties, 0, 100)

            return difficulties.tolist()

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [self.predict(q) for q in questions]

    def calibrate_questions(
        self,
        questions: List[SATQuestion],
        target_difficulty: float,
        tolerance: float = 10.0,
    ) -> List[SATQuestion]:
        """
        Filter questions to match target difficulty

        Args:
            questions: Questions to calibrate
            target_difficulty: Target difficulty (0-100)
            tolerance: Allowed deviation from target

        Returns:
            Filtered questions within target ± tolerance
        """
        calibrated = []

        # Batch predict for efficiency
        predicted_difficulties = self.predict_batch(questions)

        for q, predicted_diff in zip(questions, predicted_difficulties):
            if abs(predicted_diff - target_difficulty) <= tolerance:
                q.difficulty = predicted_diff
                calibrated.append(q)

        logger.info(
            f"Difficulty calibration: {len(calibrated)}/{len(questions)} questions "
            f"matched target {target_difficulty} ± {tolerance}"
        )

        return calibrated

    def rank_by_difficulty(self, questions: List[SATQuestion]) -> List[SATQuestion]:
        """
        Rank questions by predicted difficulty (easiest to hardest)

        Args:
            questions: Questions to rank

        Returns:
            Questions sorted by difficulty
        """
        # Predict difficulties
        difficulties = self.predict_batch(questions)

        # Attach and sort
        for q, diff in zip(questions, difficulties):
            q.difficulty = diff

        ranked = sorted(questions, key=lambda q: q.difficulty)

        logger.info(
            f"Ranked {len(ranked)} questions by difficulty "
            f"(range: {ranked[0].difficulty:.1f} - {ranked[-1].difficulty:.1f})"
            if ranked
            else "No questions to rank"
        )

        return ranked

    def save_model(self, path: str):
        """
        Save trained model to disk

        Args:
            path: File path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                },
                f,
            )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load trained model from disk

        Args:
            path: File path to load model from
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", self.feature_names)
        self.is_trained = True

        logger.info(f"Model loaded from {path}")

