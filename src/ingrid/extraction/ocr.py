"""OCR extraction using Docling.

Docling is optimized for document layout analysis and multi-column text
extraction (ideal for newspaper articles and structured documents).
"""

import logging
import tempfile
import time
from pathlib import Path

from docling.document_converter import DocumentConverter
from PIL import Image

from .models import ContentType, ExtractionResult, ExtractorType

logger = logging.getLogger(__name__)


class DoclingOCRExtractor:
    """OCR extractor using Docling.

    Docling provides layout-aware OCR that handles multi-column layouts,
    tables, and structured documents effectively.

    Attributes:
        languages: List of language codes (e.g., ["en", "nl"]).
        confidence_threshold: Minimum confidence to accept results.
        converter: Docling DocumentConverter instance.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize Docling OCR extractor.

        Args:
            languages: List of language codes (e.g., ["en", "nl"]).
            confidence_threshold: Minimum confidence to accept results.
        """
        self.languages = languages or ["en", "nl"]
        self.confidence_threshold = confidence_threshold
        self.converter = DocumentConverter()
        logger.info(f"Initialized Docling OCR with languages: {self.languages}")

    def extract(
        self,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
    ) -> ExtractionResult:
        """Extract text from image using Docling OCR.

        Args:
            image: PIL Image object (preferred).
            image_path: Path to image file (fallback).

        Returns:
            ExtractionResult with extracted text and metadata.

        Raises:
            ValueError: If neither image nor image_path provided.
        """
        start_time = time.time()

        if image is None and image_path is None:
            raise ValueError("Must provide either image or image_path")

        try:
            # Docling requires file path, so save temp if needed
            cleanup_temp = False
            if image_path is None:
                # Save PIL image to temp file
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as tmp:
                    image.save(tmp.name)  # type: ignore
                    image_path = Path(tmp.name)
                    cleanup_temp = True
            else:
                image_path = Path(image_path)

            # Run Docling conversion
            result = self.converter.convert(str(image_path))

            # Extract text in different formats
            text = result.document.export_to_markdown()
            raw_text = result.document.export_to_text()

            # Calculate confidence (Docling doesn't provide per-char confidence,
            # so we estimate based on text quality heuristics)
            confidence = self._estimate_confidence(text)

            # Detect content type hint (typed documents typically have uniform spacing)
            content_type_hint = ContentType.TYPED

            processing_time = time.time() - start_time

            extraction_result = ExtractionResult(
                extractor=ExtractorType.DOCLING_OCR,
                text=text,
                raw_text=raw_text,
                confidence=confidence,
                character_count=len(text),
                word_count=len(text.split()),
                detected_languages=self.languages,  # Docling doesn't detect, use config
                content_type_hint=content_type_hint,
                processing_time=processing_time,
                success=True,
                metadata={
                    "docling_version": "2.x",
                    "layout_detected": True,
                },
            )

            logger.info(
                f"Docling extraction: {len(text)} chars, "
                f"confidence={confidence:.2f}, time={processing_time:.2f}s"
            )

            # Cleanup temp file if created
            if cleanup_temp:
                image_path.unlink(missing_ok=True)

            return extraction_result

        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            return ExtractionResult(
                extractor=ExtractorType.DOCLING_OCR,
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _estimate_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text quality heuristics.

        Args:
            text: Extracted text.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not text or len(text) < 10:
            return 0.0

        # Heuristics:
        # - Presence of common words boosts confidence
        # - Too many single-character "words" lowers confidence
        # - Reasonable word length distribution

        words = text.split()
        if not words:
            return 0.0

        single_char_words = sum(1 for w in words if len(w) == 1)
        single_char_ratio = single_char_words / len(words)

        avg_word_length = sum(len(w) for w in words) / len(words)

        # Base confidence
        confidence = 0.7

        # Penalize high single-char ratio
        confidence -= single_char_ratio * 0.3

        # Reward reasonable avg word length (3-7 chars is normal)
        if 3 <= avg_word_length <= 7:
            confidence += 0.2

        return max(0.0, min(1.0, confidence))
