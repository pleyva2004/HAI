"""Database models using SQLAlchemy"""

from datetime import datetime
from typing import List

from sqlalchemy import ARRAY, TIMESTAMP, Column, Numeric, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class SATQuestionDB(Base):
    """Database model for SAT questions"""

    __tablename__ = "sat_questions"

    question_id = Column(String(50), primary_key=True)
    source = Column(String(200), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100))
    difficulty = Column(Numeric(5, 2), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    choice_a = Column(Text, nullable=False)
    choice_b = Column(Text, nullable=False)
    choice_c = Column(Text, nullable=False)
    choice_d = Column(Text, nullable=False)
    correct_answer = Column(String(1), nullable=False)
    explanation = Column(Text)
    national_correct_rate = Column(Numeric(5, 2))
    avg_time_seconds = Column(Numeric)
    common_wrong_answers = Column(ARRAY(Text))
    tags = Column(ARRAY(Text), index=True)
    embedding = Column(Vector(384))  # Dimension for all-MiniLM-L6-v2
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<SATQuestion(id={self.question_id}, category={self.category})>"

