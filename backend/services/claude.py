"""
Claude API service for SAT question generation.

Handles all interactions with the Anthropic Claude API including:
- Vision-based extraction from images
- Text processing and classification
- Question generation
"""

from time import process_time_ns
from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from anthropic import Anthropic

from backend.config import CLAUDE_MODEL, CLAUDE_API_KEY
from backend.workflows.state import BaseQuestion, MathQuestionExtraction, QuestionClassification, GeneratedQuestion
from backend.services.parse import parse_generated_question


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
    5. Question Multiple Choice anser selection (as structured JSON)
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

    prompt_parts.append("You are an expert SAT question writer. Your task is to generate a new SAT question that follows the same style and structure as an original question, but uses different words, numbers, and context.")
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
    prompt_parts.append("Here is the original question you will use as a template:")

    prompt_parts.append("<original_question>")
    prompt_parts.append(extracted_features)
    prompt_parts.append("</original_question>")



    # Add constraints
    prompt_parts.append("Here are the constraints that your new question must satisfy:")

    prompt_parts.append("<constraints>")
    prompt_parts.append(classified_features)
    prompt_parts.append("</constraints>")


    # Add Guidelines
    prompt_parts.append("""Your goal is to create a new question that:

MUST PRESERVE:
- The same section, domain, skill set, and difficulty level as specified in the constraints
- The same question structure and format (including table structure if present)
- The same type of statistical/mathematical concept being tested
- The same style of answer choices (e.g., if the original has 4 choices A-D, yours should too)
- Similar complexity and reasoning requirements

MUST CHANGE:
- All specific numbers and percentages (use different realistic values)
- The context and scenario (e.g., if the original is about cell phone use, choose a completely different topic like social media habits, study hours, exercise frequency, etc.)
- All specific terminology and words related to the context
- The specific categories in any tables (while maintaining the same table dimensions)
- The wording of answer choices (while testing the same conceptual understanding)

IMPORTANT GUIDELINES:
- Ensure all numbers are internally consistent (totals must add up correctly in tables)
- Make the scenario realistic and appropriate for high school students
- Keep the margin of error concept and interpretation central to the question
- Ensure answer choices test the same misconceptions and correct understanding as the original""")

    # Add Chain of Thought
    prompt_parts.append("""Before writing your final question, use a scratchpad to plan out your new scenario and verify your numbers.

<scratchpad>
Plan your new question here:
- Choose a new context/scenario
- Determine new numbers that are realistic and internally consistent
- Sketch out the table structure with new categories
- Draft the main question text
- Create answer choices that parallel the original
- Double-check all arithmetic
</scratchpad>""")

    prompt_parts.append("""After planning, provide your complete new question in the following format:

<new_question>
<text>
[Write the main question text here, including the scenario and what is being asked]
</text>

<equation>
[Write "null" if no equation is needed, otherwise provide the equation]
</equation>

<table>
<headers>
[List all column headers separated by commas]
</headers>
<rows>
[Provide each row of data, with values in quotes separated by commas]
</rows>
</table>

<visual>
[Write "null" if no visual is needed, otherwise describe it]
</visual>

<answer_choices>
[List each answer choice as A., B., C., D., etc.]
</answer_choices>
</new_question>

Make sure your question is completely original in content while maintaining the exact same educational objectives and difficulty level as the original.""")

    # Combine all parts
    full_prompt = "\n".join(prompt_parts)

    print("HAI is the prompt:")
    print("--------------------------------")
    print(full_prompt)
    print("--------------------------------")

    print("HAI is generating the question")

    # General Claude API call without structured output
    response = client.beta.messages.create(  # type: ignore
        model=CLAUDE_MODEL,
        max_tokens=2500,
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    )

    print("HAI finished generating the question")

    # Extract text content from response and parse it
    if response.content and len(response.content) > 0:
        text_content = getattr(response.content[0], "text", None)
        if text_content:
            print("HAI is parsing the generated question")
            print("--------------------------------")
            print(text_content)
            print("--------------------------------")
            return parse_generated_question(text_content)
        else:
            raise ValueError("Response content is not text")
    else:
        raise ValueError("No content in Claude response")
