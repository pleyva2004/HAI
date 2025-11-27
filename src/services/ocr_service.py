"""OCR Service - Document text extraction using DeepSeek-OCR."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

DEEPSEEK_PROMPT = "<image>\\n<|grounding|>Convert the document to markdown."
MAX_PAGES = 5  # Avoid excessive inference time on very large uploads
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class ExtractedQuestion:
    """Structured question extracted from OCR."""

    def __init__(self, text: str, choices: Dict[str, str], region_bbox: tuple):
        self.text = text
        self.choices = choices
        self.region_bbox = region_bbox


class DeepSeekOCRClient:
    """Lazy-loading wrapper around the DeepSeek-OCR transformers model."""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info("Loading DeepSeek-OCR model (%s) on %s", self.model_name, self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }

        # Flash attention is only applicable on CUDA builds
        if torch.cuda.is_available():
            model_kwargs["_attn_implementation"] = "flash_attention_2"

        model = AutoModel.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        model = model.to(self.device).eval()

        self._tokenizer = tokenizer
        self._model = model
        logger.info("DeepSeek-OCR model loaded successfully")

    def extract_markdown(self, image: Image.Image, prompt: str) -> str:
        """
        Run DeepSeek-OCR on a single image and return markdown output.
        """
        self._ensure_loaded()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "page.png"
            image.save(tmp_path, format="PNG")

            output_dir = Path(tmp_dir) / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = self._model.infer(  # type: ignore[attr-defined]
                self._tokenizer,
                prompt=prompt,
                image_file=str(tmp_path),
                output_path=str(output_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
            )

        return self._normalize_output(result)

    @staticmethod
    def _normalize_output(result) -> str:
        """
        DeepSeek-OCR can return strings, dicts, or custom objects.
        Normalize everything to a markdown string.
        """
        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            for key in ("markdown", "text", "output", "result"):
                value = result.get(key)
                if isinstance(value, str):
                    return value.strip()

        if isinstance(result, Sequence) and not isinstance(result, (bytes, bytearray)):
            return "\n\n".join(str(item) for item in result if item).strip()

        return str(result or "").strip()


class OCRService:
    """Handles PDF and image OCR with structure detection."""

    def __init__(self):
        self._ocr_client = DeepSeekOCRClient()
        logger.info("OCR Service initialized with DeepSeek-OCR")

    def extract_text(self, file_path: str) -> str:
        """
        Extract plain text rendering of all detected questions.
        """
        questions = self.extract_with_structure(file_path)
        if not questions:
            return ""

        lines = []
        for idx, question in enumerate(questions, start=1):
            lines.append(f"{idx}. {question.text}")
            for label, choice in question.choices.items():
                lines.append(f"{label}) {choice}")
            lines.append("")

        rendered = "\n".join(lines).strip()
        logger.info("Rendered %d questions into plain text", len(questions))
        return rendered

    def extract_with_structure(self, file_path: str) -> List[ExtractedQuestion]:
        """
        Extract questions with structure detection using DeepSeek-OCR markdown.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        images = self._load_document_images(path)
        if not images:
            logger.warning("No images could be derived from %s", path.name)
            return []

        markdown_chunks = []
        for page_index, image in enumerate(images[:MAX_PAGES], start=1):
            try:
                markdown = self._ocr_client.extract_markdown(image, DEEPSEEK_PROMPT)
                logger.info(
                    "DeepSeek-OCR processed page %d/%d (%d chars)",
                    page_index,
                    min(len(images), MAX_PAGES),
                    len(markdown),
                )
                markdown_chunks.append(markdown)
            except Exception as ocr_error:  # pragma: no cover - dependent on model at runtime
                logger.error("DeepSeek-OCR failed on page %d: %s", page_index, ocr_error)

        markdown_output = "\n\n".join(chunk for chunk in markdown_chunks if chunk).strip()

        if not markdown_output:
            logger.warning("DeepSeek-OCR produced no output for %s", path.name)
            return []

        questions = self._parse_markdown_questions(markdown_output)

        if not questions:
            logger.warning(
                "Markdown parsing produced no structured questions; "
                "falling back to regex-based detection"
            )
            plain_text = self._strip_markdown(markdown_output)
            return self._detect_questions(plain_text)

        logger.info("Detected %d structured questions from markdown", len(questions))
        return questions

    def extract_from_text(self, text: str) -> List[ExtractedQuestion]:
        """
        Extract questions directly from text (no OCR needed).
        """
        return self._detect_questions(text)

    def _load_document_images(self, file_path: Path) -> List[Image.Image]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            if convert_from_path is None:
                raise RuntimeError(
                    "pdf2image is required for PDF OCR but is not installed. "
                    "Install pdf2image and ensure poppler is available."
                )
            logger.info("Converting PDF %s to images for OCR", file_path.name)
            images = convert_from_path(str(file_path))
            return images

        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return [Image.open(file_path).convert("RGB")]

        raise ValueError(f"Unsupported file type for OCR: {suffix}")

    def _parse_markdown_questions(self, markdown: str) -> List[ExtractedQuestion]:
        question_blocks = self._split_markdown_into_blocks(markdown)
        questions: List[ExtractedQuestion] = []

        for block in question_blocks:
            text, choices = self._parse_markdown_block(block)
            if text:
                questions.append(
                    ExtractedQuestion(
                        text=text,
                        choices=choices,
                        region_bbox=(0, 0, 0, 0),
                    )
                )

        return questions

    def _split_markdown_into_blocks(self, markdown: str) -> List[str]:
        """
        Split markdown into blocks based on numbered headings.
        """
        pattern = re.compile(
            r"(?m)(?:^|\n)(?:##\s*)?(?:Question\s+)?\d+[\.)]\s+.+"
        )
        matches = list(pattern.finditer(markdown))

        if not matches:
            return [markdown.strip()] if markdown.strip() else []

        blocks = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            blocks.append(markdown[start:end].strip())

        return blocks

    def _parse_markdown_block(self, block: str) -> tuple[str, Dict[str, str]]:
        """
        Parse a markdown block into question text and choices.
        """
        lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            return "", {}

        # First line contains the question number/title - remove the marker
        first_line = re.sub(
            r"^(?:##\s*)?(?:Question\s+)?\d+[\.)]\s*", "", lines[0], flags=re.IGNORECASE
        )
        question_lines = [first_line]
        choice_lines = []
        in_choices = False

        for line in lines[1:]:
            if re.match(r"^\s*(?:[-*]|\|)", line):
                in_choices = True
            if in_choices:
                choice_lines.append(line)
            else:
                question_lines.append(line)

        question_text = "\n".join(question_lines).strip()
        choices = self._parse_choice_lines(choice_lines)

        return question_text, choices

    def _parse_choice_lines(self, choice_lines: List[str]) -> Dict[str, str]:
        """
        Parse markdown bullet or table rows into answer choices.
        """
        choices: Dict[str, str] = {}
        bullet_pattern = re.compile(r"^\s*(?:[-*]|[0-9]+\.)\s*([A-D])[\.\)]?\s+(.*)$")

        for line in choice_lines:
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                label, value = bullet_match.groups()
                choices.setdefault(label.upper(), value.strip())
                continue

            # Handle simple markdown tables (| A | text |)
            if line.startswith("|") and "|" in line[1:]:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) >= 2 and cells[0].upper() in {"A", "B", "C", "D"}:
                    choices.setdefault(cells[0].upper(), cells[1])

        return choices

    def _detect_questions(self, text: str) -> List[ExtractedQuestion]:
        """
        Regex-based fallback question detection on plain text.
        """
        questions: List[ExtractedQuestion] = []
        pattern = r"(?:(?:^|\n)(\d+)[\.\)])\s+"
        parts = re.split(pattern, text)

        for i in range(1, len(parts), 2):
            question_body = parts[i + 1] if i + 1 < len(parts) else ""
            if not question_body.strip():
                continue
            choices = self._extract_choices(question_body)
            questions.append(
                ExtractedQuestion(
                    text=question_body.strip(),
                    choices=choices,
                    region_bbox=(0, 0, 0, 0),
                )
            )

        return questions

    def _extract_choices(self, text: str) -> Dict[str, str]:
        """
        Extract A, B, C, D choices from question text.
        """
        choices = {}
        patterns = [
            r"([A-D])[\.\)]\s+([^\n]+)",
            r"\(([A-D])\)\s+([^\n]+)",
            r"^\s*[•-]\s*([A-D])\s*[:.-]\s+([^\n]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for letter, choice_text in matches:
                choices.setdefault(letter.upper(), choice_text.strip())

        return choices

    @staticmethod
    def _strip_markdown(markdown: str) -> str:
        """
        Convert markdown to approximate plain text for regex fallback.
        """
        text = re.sub(r"`{1,3}.*?`{1,3}", "", markdown, flags=re.DOTALL)
        text = re.sub(r"\!\[.*?\]\(.*?\)", "", text)  # Remove images
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Replace links with label
        text = re.sub(r"[*_#>-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
