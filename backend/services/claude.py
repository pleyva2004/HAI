"""
Claude API service for SAT question generation.

Handles all interactions with the Anthropic Claude API including:
- Vision-based extraction from images
- Text processing and classification
- Question generation
"""

from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from anthropic import Anthropic

from backend.config import CLAUDE_MODEL
from backend.workflows.state import BaseQuestion, MathQuestionExtraction, QuestionClassification, GeneratedQuestion


# Initialize Claude client
client = Anthropic()

def extract_from_image(image_base64: str) -> MathQuestionExtraction:
    prompt = """
    Analyze this SAT question image and extract the following information:

    1. The main question text
    2. Any equations or formulas (convert to LaTeX format)
    3. Any table data (as structured JSON)
    4. Description of any graphs, diagrams, or visual elements
    """

    print("HAI is analyzing the image")
    response = client.beta.messages.parse(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=2000,
        betas=['structured-outputs-2025-11-13'],
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
        ],
        output_format=MathQuestionExtraction
    )

    print("HAI is providing analysis")

    if response.parsed_output is None:
        print("HAI failed to analyze the image")
        raise ValueError("Failed to analyze the image")
    return response.parsed_output # type: ignore


def extract_from_description(description: str) -> MathQuestionExtraction:

    prompt = f"""
    Analyze this SAT question description and extract the following information:

    Description: {description}

    Extract:
    1. The main question text or concept
    2. Any equations mentioned (convert to LaTeX format)
    3. Any visual elements described
    """

    response = client.beta.messages.parse(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=1000,
        betas=['structured-outputs-2025-11-13'],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        output_format=MathQuestionExtraction
    )

    # Convert Pydantic model to dict (nested models are automatically converted)
    if response.parsed_output is None:
        print("HAI failed to analyze the description")
        raise ValueError("Failed to analyze the description")
    return response.parsed_output # type: ignore


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
    """

    response = client.beta.messages.parse(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=500,
        betas=['structured-outputs-2025-11-13'],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        output_format=QuestionClassification
    )

    # Convert Pydantic model to dict
    if response.parsed_output is None:
        raise ValueError("Failed to parse structured output from Claude response")
    return response.parsed_output.model_dump()


def generate_question(user_description: str, section: str, domain: str, difficulty: str, skill: List[str], similar_questions: List[BaseQuestion]) -> Dict[str, Any]:
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

    if skill and len(skill) > 0:
        skills_text = ", ".join(skill)
        prompt_parts.append(f"- Skills: {skills_text}")

    prompt_parts.append("")

    # Combine all parts
    full_prompt = "\n".join(prompt_parts)

    # Call Claude with structured output
    response = client.beta.messages.parse(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=2500,
        betas=['structured-outputs-2025-11-13'],
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        output_format=GeneratedQuestion
    )

    # Convert Pydantic model to dict (nested models are automatically converted)
    if response.parsed_output is None:
        raise ValueError("Failed to parse structured output from Claude response")
    return response.parsed_output.model_dump()
