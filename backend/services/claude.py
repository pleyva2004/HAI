"""
Claude API service for SAT question generation.

Handles all interactions with the Anthropic Claude API including:
- Vision-based extraction from images
- Text processing and classification
- Question generation
"""

from typing import Dict, Any, List
import json

from anthropic import Anthropic

from backend.config import CLAUDE_API_KEY, CLAUDE_MODEL
from backend.workflows.state import Question


# Initialize Claude client
client = Anthropic(api_key=CLAUDE_API_KEY)


def extract_from_image(image_base64: str) -> Dict[str, Any]:
    """
    Extract structured information from an image using Claude Vision.

    Returns dict with keys: text, equation, table, visual
    """

    prompt = """
    Analyze this SAT question image and extract the following information:

    1. The main question text
    2. Any equations or formulas (convert to LaTeX format)
    3. Any table data (as structured JSON)
    4. Description of any graphs, diagrams, or visual elements

    Return your response as JSON with these keys:
    {
        "text": "the main question text",
        "equation": "LaTeX formatted equations if present, null otherwise",
        "table": {"headers": [...], "rows": [...]} if table present, null otherwise,
        "visual": "description of visual elements if present, null otherwise"
    }
    """

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    response_text = message.content[0].text
    result = json.loads(response_text)

    return result


def extract_from_description(description: str) -> Dict[str, Any]:
    """
    Extract structured information from a text description.

    Returns dict with keys: text, equation, table, visual
    """

    prompt = f"""
    Analyze this SAT question description and extract the following information:

    Description: {description}

    Return your response as JSON with these keys:
    {{
        "text": "the main question text or concept",
        "equation": "any equations mentioned (LaTeX format), null if none",
        "table": null (tables not possible from text description),
        "visual": "description if visual elements are mentioned, null otherwise"
    }}
    """

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = message.content[0].text
    result = json.loads(response_text)

    return result


def classify_question(extracted_text: str, equation_content: str, visual_description: str) -> Dict[str, Any]:
    """
    Classify a question into SAT taxonomy.

    Returns dict with keys: section, domain, skill, difficulty
    """

    # Build the content to classify
    content_parts = []

    if extracted_text:
        content_parts.append(f"Question: {extracted_text}")

    if equation_content:
        content_parts.append(f"Equations: {equation_content}")

    if visual_description:
        content_parts.append(f"Visual: {visual_description}")

    content = "\n".join(content_parts)

    prompt = f"""
    Classify this SAT question into the appropriate categories.

    Content:
    {content}

    Classify into:
    - Section: "Math" or "Reading and Writing"
    - Domain (if Math): "Algebra", "Advanced Math", "Problem-Solving and Data Analysis", or "Geometry and Trigonometry"
    - Skills: List of specific skills tested (e.g., ["linear equations", "word problems"])
    - Difficulty: "Easy", "Medium", or "Hard"

    Return JSON format:
    {{
        "section": "...",
        "domain": "...",
        "skill": ["skill1", "skill2"],
        "difficulty": "..."
    }}
    """

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = message.content[0].text
    result = json.loads(response_text)

    return result


def generate_question(user_description: str, section: str, domain: str, difficulty: str, skill: List[str], similar_questions: List[Question]) -> Dict[str, Any]:
    """
    Generate a new SAT question using Claude.

    Returns dict with question data including:
    question_text, equation_content, table_data, visual_description,
    answer_choices, correct_answer, explanation
    """

    # Build the generation prompt
    prompt_parts = []

    prompt_parts.append("You are an expert SAT question writer.")
    prompt_parts.append("")

    # Add examples if we have them
    if similar_questions and len(similar_questions) > 0:
        prompt_parts.append("Here are some example SAT questions for reference:")
        prompt_parts.append("")

        for i, example in enumerate(similar_questions, 1):
            prompt_parts.append(f"EXAMPLE {i}:")
            prompt_parts.append(f"Question: {example.question_text}")

            if example.equation_content:
                prompt_parts.append(f"Equation: {example.equation_content}")

            if example.answer_choices:
                prompt_parts.append("Answer Choices:")
                for choice_key, choice_value in example.answer_choices.items():
                    prompt_parts.append(f"  {choice_key}: {choice_value}")

            prompt_parts.append(f"Correct Answer: {example.correct_answer}")
            prompt_parts.append("")

    # Add user request
    prompt_parts.append("USER REQUEST:")

    if user_description:
        prompt_parts.append(user_description)
    else:
        prompt_parts.append("Generate a similar question based on the extracted content.")

    prompt_parts.append("")

    # Add constraints
    prompt_parts.append("CONSTRAINTS:")

    if section:
        prompt_parts.append(f"- Section: {section}")

    if domain:
        prompt_parts.append(f"- Domain: {domain}")

    if difficulty:
        prompt_parts.append(f"- Difficulty: {difficulty}")

    if skills and len(skills) > 0:
        skills_text = ", ".join(skills)
        prompt_parts.append(f"- Skills: {skills_text}")

    prompt_parts.append("")

    # Add format instructions
    format_instructions = """
    OUTPUT FORMAT (return as valid JSON):
    {
        "question_text": "the main question text",
        "equation_content": "LaTeX formatted equations if applicable, null otherwise",
        "table_data": {"headers": [...], "rows": [...]} if applicable, null otherwise,
        "visual_description": "description of any visual elements if applicable, null otherwise",
        "answer_choices": {
            "A": "first choice",
            "B": "second choice",
            "C": "third choice",
            "D": "fourth choice"
        },
        "correct_answer": "A" or "B" or "C" or "D",
        "explanation": "step-by-step explanation of how to solve the question"
    }
    """

    prompt_parts.append(format_instructions)

    # Combine all parts
    full_prompt = "\n".join(prompt_parts)

    # Call Claude
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    )

    response_text = message.content[0].text
    result = json.loads(response_text)

    return result
