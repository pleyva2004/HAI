"""Anti-Duplication System - Feature 3: Semantic and structural duplicate detection"""

import hashlib
import logging
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.toon_models import SATQuestion
from ..utils.helpers import normalize_text

logger = logging.getLogger(__name__)


class QuestionFingerprint:
    """Unique structural signature of a question"""

    def __init__(self, structure_hash: str, concept_pattern: str, context_type: str):
        self.structure_hash = structure_hash
        self.concept_pattern = concept_pattern
        self.context_type = context_type

    def __eq__(self, other):
        if not isinstance(other, QuestionFingerprint):
            return False
        return (
            self.structure_hash == other.structure_hash
            and self.context_type == other.context_type
        )

    def __hash__(self):
        return hash((self.structure_hash, self.context_type))


class DuplicationDetector:
    """Detects duplicate/similar questions using semantic and structural analysis"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.question_database: List[SATQuestion] = []
        self.embeddings_cache = {}
        logger.info(f"Duplication Detector initialized with {embedding_model}")

    def get_fingerprint(self, question: SATQuestion) -> QuestionFingerprint:
        """
        Extract structural fingerprint from question

        Creates a normalized structure by:
        1. Replacing numbers with N
        2. Replacing variables with V
        3. Normalizing whitespace
        4. Extracting concepts
        5. Classifying context

        Args:
            question: Question to fingerprint

        Returns:
            QuestionFingerprint object
        """
        # Normalize text
        text = normalize_text(question.question)

        # Create structure hash (numbers and variables replaced)
        structure = re.sub(r"\d+\.?\d*", "N", text)
        structure = re.sub(r"\b[a-zA-Z]\b", "V", structure)
        structure = re.sub(r"\s+", " ", structure).strip()
        structure_hash = hashlib.md5(structure.encode()).hexdigest()

        # Extract concept pattern
        concepts = self._extract_concepts(question)
        concept_pattern = " -> ".join(concepts)

        # Classify context
        context_type = self._classify_context(question.question)

        return QuestionFingerprint(
            structure_hash=structure_hash,
            concept_pattern=concept_pattern,
            context_type=context_type,
        )

    def _extract_concepts(self, question: SATQuestion) -> List[str]:
        """
        Extract mathematical concepts from question

        Args:
            question: Question to analyze

        Returns:
            List of identified concepts
        """
        concepts = []
        text = question.question.lower()

        # Concept keywords
        concept_keywords = {
            "linear_equation": ["equation", "solve for", "=", "x +", "x -"],
            "quadratic": ["quadratic", "x²", "x^2", "squared"],
            "inequality": ["greater", "less", "than", ">", "<", "≥", "≤"],
            "function": ["function", "f(x)", "g(x)", "f(", "g("],
            "geometry": ["triangle", "circle", "angle", "area", "perimeter"],
            "statistics": ["mean", "median", "average", "probability"],
            "exponent": ["exponent", "power", "^", "²", "³"],
            "fraction": ["/", "fraction", "ratio"],
            "percentage": ["percent", "%"],
            "system": ["system", "two equations", "simultaneous"],
        }

        for concept, keywords in concept_keywords.items():
            if any(keyword in text for keyword in keywords):
                concepts.append(concept)

        return concepts if concepts else ["general"]

    def _classify_context(self, text: str) -> str:
        """
        Classify question context type

        Args:
            text: Question text

        Returns:
            Context type: "real_world", "geometric", or "abstract"
        """
        text_lower = text.lower()

        real_world_keywords = [
            "store",
            "buy",
            "sell",
            "person",
            "car",
            "cost",
            "price",
            "speed",
            "distance",
            "time",
            "age",
            "money",
        ]

        geometric_keywords = [
            "triangle",
            "circle",
            "angle",
            "line",
            "point",
            "area",
            "perimeter",
            "square",
            "rectangle",
            "radius",
        ]

        if any(keyword in text_lower for keyword in real_world_keywords):
            return "real_world"
        elif any(keyword in text_lower for keyword in geometric_keywords):
            return "geometric"
        else:
            return "abstract"

    def _get_embedding(self, question: SATQuestion) -> np.ndarray:
        """
        Get or compute embedding for question (with caching)

        Args:
            question: Question to embed

        Returns:
            Embedding vector
        """
        if question.id in self.embeddings_cache:
            return self.embeddings_cache[question.id]

        embedding = self.embedder.encode(question.question)
        self.embeddings_cache[question.id] = embedding
        return embedding

    def is_duplicate(
        self, new_question: SATQuestion, threshold: float = 0.85
    ) -> bool:
        """
        Check if question is a duplicate of any in the database

        Uses both semantic similarity (embeddings) and structural fingerprinting

        Args:
            new_question: Question to check
            threshold: Similarity threshold (0-1)

        Returns:
            True if duplicate detected, False otherwise
        """
        if not self.question_database:
            return False

        # 1. Check semantic similarity
        new_embedding = self._get_embedding(new_question)

        for existing in self.question_database:
            existing_embedding = self._get_embedding(existing)
            similarity = cosine_similarity([new_embedding], [existing_embedding])[0][0]

            if similarity > threshold:
                logger.debug(
                    f"Semantic duplicate detected: {similarity:.3f} > {threshold}"
                )
                return True

        # 2. Check structural similarity
        new_fp = self.get_fingerprint(new_question)

        for existing in self.question_database:
            existing_fp = self.get_fingerprint(existing)

            # Same structure hash + same context = very likely duplicate
            if (
                new_fp.structure_hash == existing_fp.structure_hash
                and new_fp.context_type == existing_fp.context_type
            ):
                logger.debug(
                    f"Structural duplicate detected: same structure and context"
                )
                return True

        return False

    def filter_duplicates(
        self, questions: List[SATQuestion], threshold: float = 0.85
    ) -> List[SATQuestion]:
        """
        Remove duplicate questions from list

        Checks each question against both the internal database and
        other questions in the input list

        Args:
            questions: Questions to filter
            threshold: Similarity threshold (0-1)

        Returns:
            Unique questions only
        """
        unique = []
        initial_count = len(questions)

        for q in questions:
            if not self.is_duplicate(q, threshold):
                unique.append(q)
                self.add_to_database(q)

        duplicates_removed = initial_count - len(unique)
        duplication_rate = (
            duplicates_removed / initial_count * 100 if initial_count > 0 else 0
        )

        logger.info(
            f"Duplicate filtering: removed {duplicates_removed}/{initial_count} "
            f"questions ({duplication_rate:.1f}% duplication rate)"
        )

        return unique

    def add_to_database(self, question: SATQuestion):
        """
        Add question to internal database for future duplicate detection

        Args:
            question: Question to add
        """
        self.question_database.append(question)

        # Pre-compute and cache embedding
        self._get_embedding(question)

    def clear_database(self):
        """Clear the question database and embedding cache"""
        self.question_database = []
        self.embeddings_cache = {}
        logger.info("Question database and cache cleared")

    def get_similarity_score(
        self, question1: SATQuestion, question2: SATQuestion
    ) -> float:
        """
        Calculate semantic similarity between two questions

        Args:
            question1: First question
            question2: Second question

        Returns:
            Similarity score (0-1)
        """
        emb1 = self._get_embedding(question1)
        emb2 = self._get_embedding(question2)

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def find_similar(
        self, query_question: SATQuestion, top_k: int = 5
    ) -> List[tuple[SATQuestion, float]]:
        """
        Find most similar questions in database

        Args:
            query_question: Question to find matches for
            top_k: Number of results to return

        Returns:
            List of (question, similarity_score) tuples
        """
        if not self.question_database:
            return []

        query_embedding = self._get_embedding(query_question)

        # Calculate similarities
        similarities = []
        for q in self.question_database:
            q_embedding = self._get_embedding(q)
            similarity = cosine_similarity([query_embedding], [q_embedding])[0][0]
            similarities.append((q, float(similarity)))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_statistics(self) -> dict:
        """
        Get duplication detector statistics

        Returns:
            Dictionary with stats
        """
        return {
            "total_questions": len(self.question_database),
            "cached_embeddings": len(self.embeddings_cache),
            "model": self.embedder.get_sentence_embedding_dimension(),
        }

