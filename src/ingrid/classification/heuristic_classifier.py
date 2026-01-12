"""Rule-based heuristic document classifier.

This module implements a fast, rule-based classifier that uses text patterns,
extraction result hints, and language detection to classify documents.
Used as a fallback when vision LLM classification is unavailable or fails.
"""

import logging
import re
import time
from pathlib import Path

from PIL import Image

from ..extraction.models import ContentType, ExtractionResult, ExtractorType
from .language_detector import detect_languages
from .models import ClassificationResult, ClassifierType, DocType

logger = logging.getLogger(__name__)


class HeuristicClassifier:
    """Rule-based classifier using text patterns and extraction hints.

    This classifier uses heuristic rules to classify documents based on:
    - Text structure and patterns (greetings, signatures, columns)
    - Extraction result confidence hints (TrOCR vs Docling)
    - Language detection on extracted text

    Attributes:
        None required - this is a stateless classifier.
    """

    def __init__(self) -> None:
        """Initialize heuristic classifier."""
        logger.info("Initialized HeuristicClassifier")

    def classify(
        self,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
        text: str | None = None,
        extraction_results: list[ExtractionResult] | None = None,
    ) -> ClassificationResult:
        """Classify a document using heuristic rules.

        Args:
            image: PIL Image object (not used by heuristics, reserved for future).
            image_path: Path to image file (not used by heuristics, reserved for future).
            text: Extracted text to analyze.
            extraction_results: List of extraction results from Phase 2 (for content type hints).

        Returns:
            ClassificationResult with doc_type, content_type, languages, and confidence scores.
        """
        start_time = time.time()

        # Use provided text or empty string
        text = text or ""

        # Detect doc_type from text patterns
        doc_type, doc_type_confidence = self._detect_doc_type(text)

        # Detect content_type from extraction results
        content_type, content_type_confidence = self._detect_content_type(
            text, extraction_results
        )

        # Detect languages
        languages = self._detect_languages(text)

        # Calculate overall confidence (average)
        overall_confidence = (doc_type_confidence + content_type_confidence) / 2.0

        processing_time = time.time() - start_time

        logger.info(
            f"Heuristic classification complete: {doc_type.value}, {content_type.value}, "
            f"languages={languages}, confidence={overall_confidence:.2f}"
        )

        return ClassificationResult(
            classifier=ClassifierType.HEURISTIC,
            doc_type=doc_type,
            content_type=content_type,
            languages=languages,
            confidence=overall_confidence,
            confidence_scores={
                "doc_type": doc_type_confidence,
                "content_type": content_type_confidence,
            },
            processing_time=processing_time,
            success=True,
            reasoning="Rule-based classification using text patterns and extraction hints",
        )

    def _detect_doc_type(self, text: str) -> tuple[DocType, float]:
        """Detect document type using text pattern heuristics.

        Args:
            text: Extracted text to analyze.

        Returns:
            Tuple of (DocType, confidence_score).
        """
        if not text or len(text.strip()) < 20:
            logger.debug("Text too short for doc_type detection")
            return (DocType.UNKNOWN, 0.3)

        text_lower = text.lower()
        score_letter = 0.0
        score_newspaper = 0.0

        # Letter patterns
        # Greetings
        greeting_patterns = [
            r"\b(dear|beste|lieve|geachte)\b",
            r"\b(hi|hello|hoi|hallo)\b",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, text_lower):
                score_letter += 0.2
                break

        # Closings/signatures
        closing_patterns = [
            r"\b(sincerely|yours|regards|liefs|groeten|groet|mvg)\b",
            r"\b(love|kisses|kus|kussen)\b",
        ]
        for pattern in closing_patterns:
            if re.search(pattern, text_lower):
                score_letter += 0.2
                break

        # Newspaper patterns
        # Headlines (all caps words, short lines)
        lines = text.split("\n")
        short_uppercase_lines = sum(
            1 for line in lines[:5] if len(line.strip()) < 50 and line.strip().isupper()
        )
        if short_uppercase_lines >= 1:
            score_newspaper += 0.2

        # Date at top (common in newspapers)
        if len(lines) > 0:
            first_line = lines[0].lower()
            if re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", first_line):
                score_newspaper += 0.1
            # Month names
            if re.search(
                r"\b(januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\b",
                first_line,
            ):
                score_newspaper += 0.1
            if re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                first_line,
            ):
                score_newspaper += 0.1

        # Multiple columns indicator (many short lines)
        avg_line_length = sum(len(line.strip()) for line in lines) / max(len(lines), 1)
        if avg_line_length < 40 and len(lines) > 20:
            score_newspaper += 0.1

        # Determine doc_type
        if score_letter > score_newspaper and score_letter >= 0.2:
            # Confidence capped at 0.75 for heuristics
            confidence = min(0.5 + score_letter, 0.75)
            return (DocType.LETTER, confidence)
        elif score_newspaper > score_letter and score_newspaper >= 0.2:
            confidence = min(0.5 + score_newspaper, 0.75)
            return (DocType.NEWSPAPER_ARTICLE, confidence)
        else:
            # Not enough signal to determine type
            return (DocType.OTHER, 0.4)

    def _detect_content_type(
        self, text: str, extraction_results: list[ExtractionResult] | None
    ) -> tuple[ContentType, float]:
        """Detect content type using extraction result hints.

        Args:
            text: Extracted text (not currently used, reserved for future).
            extraction_results: List of extraction results from Phase 2.

        Returns:
            Tuple of (ContentType, confidence_score).
        """
        if not extraction_results or len(extraction_results) == 0:
            logger.debug("No extraction results provided for content type detection")
            return (ContentType.UNKNOWN, 0.3)

        # Find TrOCR and Docling results
        trocr_result = None
        docling_result = None

        for result in extraction_results:
            if result.extractor == ExtractorType.TROCR_HTR and result.success:
                trocr_result = result
            elif result.extractor == ExtractorType.DOCLING_OCR and result.success:
                docling_result = result

        # Logic: If TrOCR has higher confidence, likely handwritten
        # If Docling has higher confidence, likely typed
        # If both similar, likely mixed
        if trocr_result and docling_result:
            trocr_conf = trocr_result.confidence
            docling_conf = docling_result.confidence
            diff = abs(trocr_conf - docling_conf)

            if diff < 0.1:
                # Similar confidence - mixed
                return (ContentType.MIXED, 0.65)
            elif trocr_conf > docling_conf:
                # TrOCR better - handwritten
                confidence = min(0.5 + (trocr_conf - docling_conf), 0.75)
                return (ContentType.HANDWRITTEN, confidence)
            else:
                # Docling better - typed
                confidence = min(0.5 + (docling_conf - trocr_conf), 0.75)
                return (ContentType.TYPED, confidence)

        elif trocr_result:
            # Only TrOCR succeeded - likely handwritten
            return (ContentType.HANDWRITTEN, 0.65)

        elif docling_result:
            # Only Docling succeeded - likely typed
            return (ContentType.TYPED, 0.65)

        else:
            # No successful extractions
            return (ContentType.UNKNOWN, 0.3)

    def _detect_languages(self, text: str) -> list[str]:
        """Detect languages in text using language detector.

        Args:
            text: Extracted text to analyze.

        Returns:
            List of language codes (e.g., ['nl', 'en']).
        """
        if not text or len(text.strip()) < 20:
            logger.debug("Text too short for language detection, defaulting to ['nl']")
            # Default to Dutch (most common in dataset)
            return ["nl"]

        detected = detect_languages(text, top_n=2)

        if not detected:
            logger.debug("Language detection failed, defaulting to ['nl']")
            return ["nl"]

        # Return language codes (ignore confidence scores)
        return [lang_code for lang_code, _ in detected]
