# Parse Claude Generation Response

## Status: ✅ IMPLEMENTED

## Overview
Parse Claude's XML-like text response into the `GeneratedQuestion` Pydantic model.

---

## Files Modified

| File | Change |
|------|--------|
| `backend/workflows/state.py` | Made `correct_answer` and `explanation` optional |
| `backend/services/parse.py` | **NEW** - Parser function for XML-like response |
| `backend/services/claude.py` | Import parser and update `generate_question()` return |

---

## Implementation Details

### 1. Updated GeneratedQuestion Model

**File**: `backend/workflows/state.py`

```python
class GeneratedQuestion(BaseModel):
    text: str
    equation: Optional[str] = Field(None, description="The LaTeX formatted equations of the question")
    table: Optional[TableData] = None
    visual: Optional[str] = None

    answer_choices: Dict[str, str]  # {"A": "...", "B": "...", etc.}
    correct_answer: Optional[Literal["A", "B", "C", "D"]] = None  # Made optional
    explanation: Optional[str] = None  # Made optional
```

---

### 2. Parser Function

**File**: `backend/services/parse.py`

```python
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

    return GeneratedQuestion(
        text=text,
        equation=equation,
        table=table,
        visual=visual,
        answer_choices=answer_choices,
        correct_answer=None,
        explanation=None
    )
```

---

### 3. Updated Claude Service

**File**: `backend/services/claude.py`

Added import:
```python
from backend.services.parse import parse_generated_question
```

Updated function signature:
```python
def generate_question(extracted_features: str, classified_features: str, similar_questions: List[BaseQuestion] = []) -> GeneratedQuestion:
```

Updated return logic:
```python
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
```

---

## Verification

Test by running:
```bash
python backend/test_claude.py
```

The output should now be a `GeneratedQuestion` object with parsed fields.

---

## Expected Input/Output

**Input (Claude's raw response):**
```xml
<new_question>
<text>
A random sample of 850 US college students...
</text>
<equation>
null
</equation>
<table>
<headers>
Shopping frequency,Uses mobile app for shopping,Does not use mobile app,Total
</headers>
<rows>
- Rare,"90","130","220"
- Occasional,"140","150","290"
- Frequent,"220","120","340"
- Total,"450","400","850"
</rows>
</table>
<visual>
null
</visual>
<answer_choices>
A. Approximately 4% of the students in the study...
B. It is not possible that the percent...
C. The percent of all US college students...
D. It is doubtful that the percent...
</answer_choices>
</new_question>
```

**Output (GeneratedQuestion object):**
```python
GeneratedQuestion(
    text="A random sample of 850 US college students...",
    equation=None,
    table=TableData(
        headers=["Shopping frequency", "Uses mobile app for shopping", "Does not use mobile app", "Total"],
        rows=[
            ["Rare", "90", "130", "220"],
            ["Occasional", "140", "150", "290"],
            ["Frequent", "220", "120", "340"],
            ["Total", "450", "400", "850"]
        ]
    ),
    visual=None,
    answer_choices={
        "A": "Approximately 4% of the students in the study...",
        "B": "It is not possible that the percent...",
        "C": "The percent of all US college students...",
        "D": "It is doubtful that the percent..."
    },
    correct_answer=None,
    explanation=None
)
```
