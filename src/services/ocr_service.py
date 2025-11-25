"""OCR Service - Document text extraction using Chandra"""

import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class ExtractedQuestion:
    """Structured question extracted from OCR"""

    def __init__(self, text: str, choices: Dict[str, str], region_bbox: tuple):
        self.text = text
        self.choices = choices
        self.region_bbox = region_bbox


class OCRService:
    """Handles PDF and image OCR with structure detection"""

    def __init__(self):
        # Note: Chandra integration would go here
        # For now, we'll provide a working implementation that can be extended
        logger.info("OCR Service initialized")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF or image

        Args:
            file_path: Path to file (PDF, PNG, JPG, etc.)

        Returns:
            Extracted text

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting text from {file_path_obj.name}")

        try:
            # TODO: Replace with actual Chandra integration
            # from chandra import Chandra
            # chandra = Chandra()

            suffix = file_path_obj.suffix.lower()

            if suffix == ".pdf":
                text = self._extract_pdf(file_path)
            elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                text = self._extract_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            logger.info(f"Extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

    def extract_with_structure(self, file_path: str) -> List[ExtractedQuestion]:
        """
        Extract questions with structure detection

        Args:
            file_path: Path to file

        Returns:
            List of ExtractedQuestion objects with boundaries and choices
        """
        # Basic text extraction
        text = self.extract_text(file_path)

        # Detect question boundaries
        questions = self._detect_questions(text)

        logger.info(f"Detected {len(questions)} questions with structure")
        return questions

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF (placeholder for Chandra)"""
        # TODO: Implement actual Chandra PDF extraction
        # For now, return empty string as placeholder
        logger.warning("PDF extraction not yet implemented - using placeholder")
        return ""

    def _extract_image(self, file_path: str) -> str:
        """Extract text from image (placeholder for Chandra)"""
        # TODO: Implement actual Chandra image OCR
        # For now, return empty string as placeholder
        logger.warning("Image OCR not yet implemented - using placeholder")
        return ""

    def _detect_questions(self, text: str) -> List[ExtractedQuestion]:
        """
        Detect individual questions in extracted text

        Looks for numbered questions (1., 2., etc.) and extracts choices

        Args:
            text: Extracted text

        Returns:
            List of ExtractedQuestion objects
        """
        questions = []

        # Pattern to match question numbers (1., 2., etc.)
        pattern = r"(?:^|\n)(\d+)\.\s+"
        parts = re.split(pattern, text)

        # Process pairs of (question_number, question_text)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                q_num = parts[i]
                q_text = parts[i + 1]

                # Extract choices from the question text
                choices = self._extract_choices(q_text)

                # Create ExtractedQuestion
                questions.append(
                    ExtractedQuestion(
                        text=q_text.strip(),
                        choices=choices,
                        region_bbox=(0, 0, 0, 0),  # Placeholder for actual bbox
                    )
                )

        return questions

    def _extract_choices(self, text: str) -> Dict[str, str]:
        """
        Extract A, B, C, D choices from question text

        Handles patterns like:
        - A) choice text
        - A. choice text
        - (A) choice text

        Args:
            text: Question text containing choices

        Returns:
            Dict mapping choice letter to choice text
        """
        choices = {}

        # Match patterns: A) text, A. text, (A) text
        patterns = [
            r"([A-D])[\.\)]\s+([^\n]+)",  # A) or A.
            r"\(([A-D])\)\s+([^\n]+)",  # (A)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for letter, choice_text in matches:
                if letter not in choices:  # Don't override if already found
                    choices[letter] = choice_text.strip()

        return choices

    def extract_from_text(self, text: str) -> List[ExtractedQuestion]:
        """
        Extract questions directly from text (no OCR needed)

        Useful when user provides text description instead of file

        Args:
            text: Text containing questions

        Returns:
            List of ExtractedQuestion objects
        """
        return self._detect_questions(text)

