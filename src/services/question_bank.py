"""Question Bank Service - PostgreSQL + pgvector interface"""

import asyncpg
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple

from ..models.toon_models import OfficialSATQuestion, QuestionChoice

logger = logging.getLogger(__name__)


class QuestionBankService:
    """Interface to SAT question bank with vector similarity search"""

    def __init__(self, database_url: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.embedder = SentenceTransformer(embedding_model)
        logger.info(f"Question Bank Service initialized with {embedding_model}")

    async def connect(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def disconnect(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    async def search_similar(
        self,
        query: str,
        category: Optional[str] = None,
        difficulty_range: Optional[Tuple[float, float]] = None,
        top_k: int = 10,
    ) -> List[OfficialSATQuestion]:
        """
        Search for similar questions using vector similarity

        Args:
            query: Search query text
            category: Filter by category (optional)
            difficulty_range: (min, max) difficulty filter (optional)
            top_k: Number of results to return

        Returns:
            List of similar OfficialSATQuestion objects
        """
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        logger.info(f"Searching for similar questions: '{query[:50]}...'")

        # Generate query embedding and convert to pgvector format
        query_embedding_array = self.embedder.encode(query).tolist()
        query_embedding_str = '[' + ','.join(str(x) for x in query_embedding_array) + ']'

        # Build SQL query
        sql = """
            SELECT
                question_id, source, category, subcategory, difficulty,
                question_text, choice_a, choice_b, choice_c, choice_d,
                correct_answer, explanation, national_correct_rate,
                avg_time_seconds, common_wrong_answers, tags,
                1 - (embedding <=> $1::vector) as similarity
            FROM sat_questions
            WHERE ($2::text IS NULL OR category = $2)
              AND ($3::decimal IS NULL OR difficulty >= $3)
              AND ($4::decimal IS NULL OR difficulty <= $4)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                query_embedding_str,
                category,
                difficulty_range[0] if difficulty_range else None,
                difficulty_range[1] if difficulty_range else None,
                top_k,
            )

        questions = [self._parse_row(row) for row in rows]
        logger.info(
            f"Found {len(questions)} similar questions "
            f"(avg similarity: {sum(r['similarity'] for r in rows) / len(rows):.3f})"
            if rows
            else "Found 0 questions"
        )
        return questions

    async def get_by_id(self, question_id: str) -> Optional[OfficialSATQuestion]:
        """Get question by ID"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        sql = "SELECT * FROM sat_questions WHERE question_id = $1"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(sql, question_id)

        if row:
            return self._parse_row(row)
        return None

    async def get_by_category(
        self,
        category: str,
        difficulty: Optional[float] = None,
        limit: int = 50,
    ) -> List[OfficialSATQuestion]:
        """Get questions by category with optional difficulty filter"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        sql = """
            SELECT * FROM sat_questions
            WHERE category = $1
              AND ($2::decimal IS NULL OR ABS(difficulty - $2) < 10)
            ORDER BY difficulty, question_id
            LIMIT $3
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, category, difficulty, limit)

        questions = [self._parse_row(row) for row in rows]
        logger.info(f"Retrieved {len(questions)} questions for category '{category}'")
        return questions

    async def insert_question(self, question: OfficialSATQuestion):
        """Insert a new question into the bank"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Generate embedding and convert to pgvector format
        embedding_array = self.embedder.encode(question.question_text).tolist()
        # Convert to string format that pgvector expects: '[0.1, 0.2, ...]'
        embedding_str = '[' + ','.join(str(x) for x in embedding_array) + ']'

        sql = """
            INSERT INTO sat_questions (
                question_id, source, category, subcategory, difficulty,
                question_text, choice_a, choice_b, choice_c, choice_d,
                correct_answer, explanation, national_correct_rate,
                avg_time_seconds, common_wrong_answers, tags, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::vector)
            ON CONFLICT (question_id) DO NOTHING
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                sql,
                question.question_id,
                question.source,
                question.category,
                question.subcategory,
                question.difficulty,
                question.question_text,
                question.choices.A,
                question.choices.B,
                question.choices.C,
                question.choices.D,
                question.correct_answer,
                question.explanation,
                question.national_correct_rate,
                question.avg_time_seconds,
                question.common_wrong_answers,
                question.tags,
                embedding_str,
            )

        logger.info(f"Inserted question {question.question_id}")

    async def batch_insert(self, questions: List[OfficialSATQuestion]):
        """Batch insert multiple questions"""
        logger.info(f"Batch inserting {len(questions)} questions...")
        for question in questions:
            await self.insert_question(question)
        logger.info("Batch insert complete")

    async def get_count(self) -> int:
        """Get total question count"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM sat_questions")

        return count

    async def get_categories(self) -> List[str]:
        """Get list of all categories"""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")

        sql = "SELECT DISTINCT category FROM sat_questions ORDER BY category"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql)

        return [row["category"] for row in rows]

    def _parse_row(self, row) -> OfficialSATQuestion:
        """Parse database row to OfficialSATQuestion"""
        return OfficialSATQuestion(
            question_id=row["question_id"],
            source=row["source"],
            category=row["category"],
            subcategory=row["subcategory"] or "",
            difficulty=float(row["difficulty"]),
            question_text=row["question_text"],
            choices=QuestionChoice(
                A=row["choice_a"],
                B=row["choice_b"],
                C=row["choice_c"],
                D=row["choice_d"],
            ),
            correct_answer=row["correct_answer"],
            explanation=row["explanation"] or "",
            national_correct_rate=float(row["national_correct_rate"] or 50.0),
            avg_time_seconds=int(row["avg_time_seconds"] or 60),
            common_wrong_answers=row["common_wrong_answers"] or [],
            tags=row["tags"] or [],
        )

