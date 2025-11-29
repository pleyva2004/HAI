"""
Embedding generation service using OpenAI.

Handles creating embeddings for text content and building query strings
for semantic search.
"""

from typing import List, Optional
from openai import OpenAI

from backend.config import OPENAI_API_KEY, EMBEDDING_MODEL


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def build_query_text(
    extracted_text: Optional[str] = None,
    equation_content: Optional[str] = None,
    visual_description: Optional[str] = None,
    skills: Optional[List[str]] = None
) -> str:
    """
    Build a query string from extracted question features.

    Combines all available text into a single string for embedding.
    """

    query_parts = []

    if extracted_text:
        query_parts.append(extracted_text)

    if equation_content:
        query_parts.append(equation_content)

    if visual_description:
        query_parts.append(visual_description)

    if skills and len(skills) > 0:
        skills_text = ", ".join(skills)
        query_parts.append(f"Skills: {skills_text}")

    query_text = " ".join(query_parts)

    return query_text


def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text.

    Returns a list of floats representing the embedding.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    embedding = response.data[0].embedding

    return embedding
