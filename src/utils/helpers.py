"""Helper utility functions"""

import hashlib
import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def hash_text(text: str) -> str:
    """Generate MD5 hash of text"""
    return hashlib.md5(text.encode()).hexdigest()


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text"""
    return re.findall(r'\d+\.?\d*', text)


def extract_variables(text: str) -> List[str]:
    """Extract algebraic variables (single letters)"""
    # Find single letters that are likely variables
    variables = re.findall(r'\b[a-zA-Z]\b', text)
    # Filter out common words
    common_words = {'a', 'A', 'i', 'I'}
    return [v for v in variables if v not in common_words]


def count_operations(text: str) -> int:
    """Count mathematical operations in text"""
    operations = re.findall(r'[+\-*/=<>≤≥]', text)
    return len(operations)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

