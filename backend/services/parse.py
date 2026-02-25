import re
from typing import Dict, List
from backend.workflows.state import GeneratedQuestion, TableData


def parse_generated_question(raw_response: str) -> GeneratedQuestion:
    """
    Parse Claude's XML-like response into GeneratedQuestion model.

    Expected format:
    <new_question>
    <text>...</text>
    <equation>... or null</equation>
    <table><headers>...</headers><rows>...</rows></table>
    <visual>... or null</visual>
    <answer_choices>A. ... B. ... etc.</answer_choices>
    </new_question>
    """

    # Extract text
    text_match = re.search(r'<text>(.*?)</text>', raw_response, re.DOTALL)
    text = ""
    if text_match:
        text = text_match.group(1).strip()

    # Extract equation (None if "null")
    equation = None
    equation_match = re.search(r'<equation>(.*?)</equation>', raw_response, re.DOTALL)
    if equation_match:
        equation_content = equation_match.group(1).strip()
        if equation_content.lower() != "null":
            equation = equation_content

    # Extract visual (None if "null")
    visual = None
    visual_match = re.search(r'<visual>(.*?)</visual>', raw_response, re.DOTALL)
    if visual_match:
        visual_content = visual_match.group(1).strip()
        if visual_content.lower() != "null":
            visual = visual_content

    # Extract table
    table = None
    table_match = re.search(r'<table>(.*?)</table>', raw_response, re.DOTALL)
    if table_match:
        table_content = table_match.group(1)

        # Extract headers
        headers = []
        headers_match = re.search(r'<headers>(.*?)</headers>', table_content, re.DOTALL)
        if headers_match:
            headers_text = headers_match.group(1).strip()
            headers = [h.strip() for h in headers_text.split(',')]

        # Extract rows
        rows = []
        rows_match = re.search(r'<rows>(.*?)</rows>', table_content, re.DOTALL)
        if rows_match:
            rows_text = rows_match.group(1).strip()
            for line in rows_text.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Remove leading "- " and parse values
                    line = line[1:].strip()
                    # Split by comma and clean up quotes
                    values = []
                    for part in line.split(','):
                        part = part.strip().strip('"').strip("'")
                        values.append(part)
                    rows.append(values)

        if headers or rows:
            table = TableData(headers=headers, rows=rows)

    # Extract answer choices
    answer_choices = {}
    choices_match = re.search(r'<answer_choices>(.*?)</answer_choices>', raw_response, re.DOTALL)
    if choices_match:
        choices_text = choices_match.group(1).strip()
        # Match patterns like "A. text" where text continues until next choice or end
        choice_pattern = re.compile(r'([A-D])[.\:]\s*(.+?)(?=\n[A-D][.\:]|\Z)', re.DOTALL)
        for match in choice_pattern.finditer(choices_text):
            letter = match.group(1)
            choice_text = match.group(2).strip()
            answer_choices[letter] = choice_text

    # Extract correct answer
    correct_answer = None
    answer_match = re.search(r'<correct_answer>(.*?)</correct_answer>', raw_response, re.DOTALL)
    if answer_match:
        raw_ans = answer_match.group(1).strip()
        # Look for A, B, C, or D as a standalone word
        letter_match = re.search(r'\b([A-D])\b', raw_ans)
        if letter_match:
            correct_answer = letter_match.group(1)

    # Extract explanation
    explanation = None
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', raw_response, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return GeneratedQuestion(
        text=text,
        equation=equation,
        table=table,
        visual=visual,
        answer_choices=answer_choices,
        correct_answer=correct_answer,
        explanation=explanation
    )
