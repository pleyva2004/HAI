#!/usr/bin/env python3
"""Database setup script - creates tables and indexes"""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def setup_database():
    """Create database schema with pgvector extension"""

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return False

    print("🔌 Connecting to database...")
    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("   Make sure PostgreSQL is running and DATABASE_URL is correct")
        return False

    try:
        print("📦 Creating pgvector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✅ pgvector extension ready")

        print("📋 Creating sat_questions table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sat_questions (
                question_id VARCHAR(50) PRIMARY KEY,
                source VARCHAR(200) NOT NULL,
                category VARCHAR(100) NOT NULL,
                subcategory VARCHAR(100),
                difficulty DECIMAL(5,2) NOT NULL,
                question_text TEXT NOT NULL,
                choice_a TEXT NOT NULL,
                choice_b TEXT NOT NULL,
                choice_c TEXT NOT NULL,
                choice_d TEXT NOT NULL,
                correct_answer CHAR(1) NOT NULL CHECK (correct_answer IN ('A', 'B', 'C', 'D')),
                explanation TEXT,
                national_correct_rate DECIMAL(5,2),
                avg_time_seconds INTEGER,
                common_wrong_answers TEXT[],
                tags TEXT[],
                embedding vector(384),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        print("✅ sat_questions table created")

        print("🔍 Creating indexes...")

        # Category index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_category
            ON sat_questions(category)
        """)
        print("  ✓ Category index")

        # Difficulty index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_difficulty
            ON sat_questions(difficulty)
        """)
        print("  ✓ Difficulty index")

        # Tags GIN index
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags
            ON sat_questions USING GIN(tags)
        """)
        print("  ✓ Tags GIN index")

        # Vector index (IVFFlat for approximate nearest neighbor search)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding
            ON sat_questions USING ivfflat(embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("  ✓ Vector embedding index")

        # Check if table has data
        count = await conn.fetchval("SELECT COUNT(*) FROM sat_questions")
        print(f"\n📊 Current question count: {count}")

        print("\n✅ Database setup complete!")
        return True

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False
    finally:
        await conn.close()


if __name__ == "__main__":
    print("🚀 SAT Question Generator - Database Setup\n")
    success = asyncio.run(setup_database())
    sys.exit(0 if success else 1)

