# Configuration settings for the agent
import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Claude
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

LANGSMITH_PROJECT = "sat-question-generator"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Workflow
MAX_GENERATION_ATTEMPTS = 3
RETRIEVAL_LIMIT = 5
SIMILARITY_THRESHOLD = 0.75
