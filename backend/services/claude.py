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

from backend.config import CLAUDE_MODEL, CLAUDE_API_KEY
from backend.workflows.state import BaseQuestion, MathQuestionExtraction, QuestionClassification, GeneratedQuestion


# Initialize Claude client
client = Anthropic(api_key=CLAUDE_API_KEY)


# TODO: Implement this
# This is called when the user provides a description of the type of question they want to generate
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

# These are only called when the user provides an image
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

def classify_question(extracted_features: str) -> QuestionClassification:
    """
    Classify a question into SAT taxonomy.

    Returns dict with keys: section, domain, skill, difficulty
    """
    # Build the content to classify

    prompt = f"""
    Classify this SAT question into the appropriate categories.

    Content:
    {extracted_features}

    Classify into:
    - Section: "Math" or "Reading and Writing"
    - Domain (if Math): "Algebra", "Advanced Math", "Problem-Solving and Data Analysis", or "Geometry and Trigonometry"
    - Skills: List of specific skills tested (e.g., ["linear equations", "word problems"])
    - Difficulty: "Easy", "Medium", or "Hard"
    """

    print("HAI is checking itself")
    response = client.beta.messages.parse(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=500,
        betas=['structured-outputs-2025-11-13'],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        output_format=QuestionClassification
    )

    print("HAI is providing final analysis")

    # Convert Pydantic model to dict
    if response.parsed_output is None:
        raise ValueError("Failed to parse structured output from Claude response")
    return response.parsed_output # type: ignore

def generate_question(extracted_features: str, classified_features: str, similar_questions: List[BaseQuestion] = []) -> GeneratedQuestion:


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
            prompt_parts.append(f"Question: {example.text}")

            if example.equation:
                prompt_parts.append(f"Equation: {example.equation}")

            if example.answer_choices:
                prompt_parts.append("Answer Choices:")
                for choice_key, choice_value in example.answer_choices.items():
                    prompt_parts.append(f"  {choice_key}: {choice_value}")

            prompt_parts.append(f"Correct Answer: {example.correct_answer}")
            prompt_parts.append("")

    # Add user request
    prompt_parts.append("USER REQUEST:")
    prompt_parts.append(extracted_features)
    prompt_parts.append("")

    # Add constraints
    prompt_parts.append("CONSTRAINTS:")
    prompt_parts.append(classified_features)
    prompt_parts.append("")

    # Combine all parts
    full_prompt = "\n".join(prompt_parts)

    print("HAI is the prompt:")
    print("--------------------------------")
    print(full_prompt)
    print("--------------------------------")

    print("HAI is generating the question")

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

    print("HAI finished generating the question")

    # Convert Pydantic model to dict (nested models are automatically converted)
    if response.parsed_output is None:
        print("HAI failed to geenearte the question")
        raise ValueError("Failed to parse structured output from Claude response")
    return response.parsed_output # type: ignore
