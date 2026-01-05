# Configuration settings for the agent
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

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

# Claude API Token Limits
CLAUDE_MAX_TOKENS_DESCRIPTION_EXTRACTION = 1000
CLAUDE_MAX_TOKENS_IMAGE_EXTRACTION = 2000
CLAUDE_MAX_TOKENS_CLASSIFICATION = 500
CLAUDE_MAX_TOKENS_GENERATION = 2500
