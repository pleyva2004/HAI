"""Style Analysis and Matching - Feature 1"""

import logging
import re
import textstat
from typing import List, Tuple

from ..models.toon_models import StyleProfile, SATQuestion

logger = logging.getLogger(__name__)


class StyleAnalyzer:
    """Analyzes and extracts style characteristics from example questions"""

    def analyze(self, examples: List[str]) -> StyleProfile:
        """
        Analyze style from example questions

        Args:
            examples: List of example question texts

        Returns:
            StyleProfile with extracted characteristics
        """
        if not examples:
            logger.warning("No examples provided, returning default style profile")
            return self._default_profile()

        logger.info(f"Analyzing style from {len(examples)} examples")

        return StyleProfile(
            word_count_range=self._analyze_word_counts(examples),
            vocabulary_level=self._analyze_vocabulary(examples),
            number_complexity=self._analyze_numbers(examples),
            context_type=self._analyze_context(examples),
            question_structure=self._extract_structure(examples),
            distractor_patterns=self._analyze_distractors(examples),
        )

    def _analyze_word_counts(self, examples: List[str]) -> Tuple[int, int]:
        """Analyze word count range"""
        counts = [len(ex.split()) for ex in examples]
        if not counts:
            return (10, 50)

        min_count = min(counts)
        max_count = max(counts)

        # Add some tolerance (±20% or at least 2 words)
        buffer = max(2, int((max_count - min_count) * 0.2))
        return (max(5, min_count - buffer), max_count + buffer)

    def _analyze_vocabulary(self, examples: List[str]) -> float:
        """
        Analyze vocabulary level using Flesch-Kincaid grade

        Returns average grade level across examples
        """
        levels = []
        for ex in examples:
            try:
                level = textstat.flesch_kincaid_grade(ex)
                levels.append(level)
            except Exception:
                # If textstat fails, use a default
                levels.append(10.0)

        return sum(levels) / len(levels) if levels else 10.0

    def _analyze_numbers(self, examples: List[str]) -> str:
        """
        Analyze number complexity in questions

        Returns: "fractions", "decimals", "large_integers", or "small_integers"
        """
        has_fractions = False
        has_decimals = False
        has_large = False

        for ex in examples:
            # Check for fractions (pattern: digit/digit)
            if re.search(r"\d+/\d+", ex):
                has_fractions = True

            # Check for decimals
            if re.search(r"\d+\.\d+", ex):
                has_decimals = True

            # Check for large numbers (3+ digits)
            if re.search(r"\d{3,}", ex):
                has_large = True

        # Priority: fractions > decimals > large > small
        if has_fractions:
            return "fractions"
        elif has_decimals:
            return "decimals"
        elif has_large:
            return "large_integers"
        else:
            return "small_integers"

    def _analyze_context(self, examples: List[str]) -> str:
        """
        Classify the context type of questions

        Returns: "real_world", "geometric", or "abstract"
        """
        keywords = {
            "real_world": [
                "store",
                "buy",
                "sell",
                "person",
                "car",
                "distance",
                "time",
                "money",
                "cost",
                "price",
                "speed",
                "age",
            ],
            "geometric": [
                "triangle",
                "circle",
                "angle",
                "line",
                "point",
                "area",
                "perimeter",
                "radius",
                "diameter",
                "square",
                "rectangle",
            ],
            "abstract": [
                "variable",
                "function",
                "equation",
                "expression",
                "solve",
                "simplify",
                "value",
            ],
        }

        scores = {ctx: 0 for ctx in keywords}

        for example in examples:
            example_lower = example.lower()
            for ctx, words in keywords.items():
                scores[ctx] += sum(1 for word in words if word in example_lower)

        # Return context with highest score
        return max(scores, key=scores.get) if any(scores.values()) else "abstract"

    def _extract_structure(self, examples: List[str]) -> str:
        """
        Extract common question structure template

        Replaces numbers with N and variables with V to find patterns
        """
        if not examples:
            return ""

        # Take the first example as template
        structure = examples[0]

        # Replace numbers with N
        structure = re.sub(r"\d+\.?\d*", "N", structure)

        # Replace single-letter variables with V
        structure = re.sub(r"\b[a-z]\b", "V", structure, flags=re.IGNORECASE)

        # Truncate if too long
        if len(structure) > 200:
            structure = structure[:200] + "..."

        return structure

    def _analyze_distractors(self, examples: List[str]) -> str:
        """
        Analyze patterns in wrong answer construction

        This is a placeholder - full implementation would need actual choices
        """
        # Common patterns in SAT wrong answers:
        # - Off-by-one errors
        # - Sign errors
        # - Arithmetic mistakes
        # - Partial solutions
        return "plausible_near_misses"

    def _default_profile(self) -> StyleProfile:
        """Return default style profile when no examples available"""
        return StyleProfile(
            word_count_range=(10, 50),
            vocabulary_level=10.0,
            number_complexity="small_integers",
            context_type="abstract",
            question_structure="If N * V + N = N, what is V?",
            distractor_patterns="plausible_near_misses",
        )


class StyleMatcher:
    """Matches generated questions against a target style profile"""

    def __init__(self):
        self.analyzer = StyleAnalyzer()

    def score_match(self, question: SATQuestion, profile: StyleProfile) -> float:
        """
        Score how well a question matches the target style

        Uses 5-factor weighted scoring:
        - Word count (20%)
        - Vocabulary level (20%)
        - Number complexity (20%)
        - Context type (20%)
        - Structure similarity (20%)

        Args:
            question: Question to score
            profile: Target style profile

        Returns:
            Match score from 0.0 to 1.0
        """
        scores = []

        # 1. Word count match (20%)
        word_count = len(question.question.split())
        min_wc, max_wc = profile.word_count_range

        if min_wc <= word_count <= max_wc:
            scores.append(0.2)
        else:
            # Partial credit based on distance
            if word_count < min_wc:
                distance = min_wc - word_count
            else:
                distance = word_count - max_wc

            penalty = min(distance / 10, 0.2)  # Max 0.2 penalty
            scores.append(max(0, 0.2 - penalty))

        # 2. Vocabulary match (20%)
        try:
            vocab_level = textstat.flesch_kincaid_grade(question.question)
            vocab_diff = abs(vocab_level - profile.vocabulary_level)
            vocab_score = max(0, 0.2 - (vocab_diff * 0.03))  # 0.03 penalty per grade level
            scores.append(vocab_score)
        except Exception:
            scores.append(0.1)  # Partial credit if textstat fails

        # 3. Number complexity match (20%)
        number_type = self.analyzer._analyze_numbers([question.question])
        if number_type == profile.number_complexity:
            scores.append(0.2)
        else:
            # Partial credit for similar complexity
            complexity_order = ["small_integers", "large_integers", "decimals", "fractions"]
            try:
                target_idx = complexity_order.index(profile.number_complexity)
                actual_idx = complexity_order.index(number_type)
                distance = abs(target_idx - actual_idx)
                scores.append(max(0, 0.2 - (distance * 0.05)))
            except ValueError:
                scores.append(0.1)

        # 4. Context match (20%)
        context = self.analyzer._analyze_context([question.question])
        if context == profile.context_type:
            scores.append(0.2)
        else:
            scores.append(0.05)  # Small partial credit

        # 5. Structure similarity (20%)
        # Simplified - compare presence of similar elements
        q_structure = self.analyzer._extract_structure([question.question])
        profile_structure = profile.question_structure

        # Check for similar patterns (equations, comparisons, etc.)
        similar_patterns = sum(
            1
            for pattern in ["=", "+", "-", "*", "/", "<", ">"]
            if (pattern in q_structure) == (pattern in profile_structure)
        )
        structure_score = min(0.2, similar_patterns * 0.03)
        scores.append(structure_score)

        total_score = sum(scores)
        logger.debug(
            f"Style match score: {total_score:.3f} "
            f"(wc:{scores[0]:.2f}, vocab:{scores[1]:.2f}, "
            f"num:{scores[2]:.2f}, ctx:{scores[3]:.2f}, struct:{scores[4]:.2f})"
        )

        return total_score

    def filter_by_style(
        self,
        questions: List[SATQuestion],
        profile: StyleProfile,
        threshold: float = 0.7,
    ) -> List[SATQuestion]:
        """
        Filter questions that match the style profile

        Args:
            questions: Questions to filter
            profile: Target style profile
            threshold: Minimum match score (0-1)

        Returns:
            Filtered list of questions that meet the threshold
        """
        matched = []

        for q in questions:
            score = self.score_match(q, profile)
            if score >= threshold:
                q.style_match_score = score
                matched.append(q)

        logger.info(
            f"Style filtering: {len(matched)}/{len(questions)} questions "
            f"passed threshold {threshold}"
        )

        return matched

    def rank_by_style(
        self, questions: List[SATQuestion], profile: StyleProfile
    ) -> List[SATQuestion]:
        """
        Rank questions by style match score

        Args:
            questions: Questions to rank
            profile: Target style profile

        Returns:
            Questions sorted by style match (best first)
        """
        for q in questions:
            if q.style_match_score is None:
                q.style_match_score = self.score_match(q, profile)

        ranked = sorted(questions, key=lambda q: q.style_match_score or 0, reverse=True)

        logger.info(
            f"Ranked {len(ranked)} questions by style "
            f"(best: {ranked[0].style_match_score:.3f}, "
            f"worst: {ranked[-1].style_match_score:.3f})"
            if ranked
            else "No questions to rank"
        )

        return ranked

