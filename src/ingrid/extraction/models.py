"""Data models for extraction pipeline.

This module defines all data structures used throughout the extraction process,
including extraction results, preprocessed images, and extraction jobs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image


class ExtractorType(Enum):
    """Types of extraction methods."""

    DOCLING_OCR = "docling_ocr"
    TROCR_HTR = "trocr_htr"
    VISION_LLM = "vision_llm"
    TESSERACT = "tesseract"  # Future
    EASYOCR = "easyocr"  # Future


class ContentType(Enum):
    """Document content types (preliminary detection)."""

    HANDWRITTEN = "handwritten"
    TYPED = "typed"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Result from a single extraction attempt.

    Attributes:
        extractor: Type of extractor that produced this result.
        timestamp: When this extraction was performed.
        text: Extracted text (cleaned/processed).
        raw_text: Unprocessed output from the extractor (for debugging).
        confidence: Overall confidence score (0.0-1.0).
        character_count: Number of characters in extracted text.
        word_count: Number of words in extracted text.
        detected_languages: List of detected language codes (e.g., ['nl', 'en']).
        content_type_hint: Preliminary content type detection.
        processing_time: Time taken for extraction in seconds.
        success: Whether extraction was successful.
        error: Error message if extraction failed.
        metadata: Additional extractor-specific metadata.
    """

    # Identification
    extractor: ExtractorType
    timestamp: datetime = field(default_factory=datetime.now)

    # Extracted content
    text: str = ""
    raw_text: str | None = None

    # Quality metrics
    confidence: float = 0.0  # Overall confidence (0.0-1.0)
    character_count: int = 0
    word_count: int = 0

    # Detection hints (preliminary, Phase 3 will refine)
    detected_languages: list[str] = field(default_factory=list)
    content_type_hint: ContentType = ContentType.UNKNOWN

    # Performance
    processing_time: float = 0.0  # Seconds

    # Errors (if any)
    error: str | None = None
    success: bool = True

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedImage:
    """Result of image preprocessing.

    Attributes:
        original_path: Path to the original image file.
        image: PIL Image object (preprocessed).
        deskewed: Whether image was deskewed.
        deskew_angle: Angle of rotation applied (degrees).
        contrast_enhanced: Whether contrast was enhanced.
        original_size: Original image dimensions (width, height).
        processed_size: Processed image dimensions (width, height).
        file_size_bytes: Original file size in bytes.
        processing_time: Time taken for preprocessing in seconds.
    """

    original_path: Path
    image: Image.Image  # PIL Image object

    # Transformations applied
    deskewed: bool = False
    deskew_angle: float = 0.0
    contrast_enhanced: bool = False

    # Metadata
    original_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)
    file_size_bytes: int = 0

    processing_time: float = 0.0


@dataclass
class ExtractionJob:
    """Complete extraction job for a single document.

    Attributes:
        image_path: Path to the input image file.
        config: Configuration dictionary used for extraction.
        preprocessed: Preprocessed image result (if preprocessing succeeded).
        results: List of extraction results from all extractors.
        primary_result: Best extraction result (selected by orchestrator).
        success: Whether at least one extraction succeeded.
        total_processing_time: Total time for entire job in seconds.
        errors: List of error messages encountered during job.
    """

    # Input
    image_path: Path
    config: dict[str, Any] = field(default_factory=dict)

    # Preprocessing
    preprocessed: PreprocessedImage | None = None

    # Extraction results (multiple strategies)
    results: list[ExtractionResult] = field(default_factory=list)

    # Best result (selected by orchestrator)
    primary_result: ExtractionResult | None = None

    # Overall status
    success: bool = False
    total_processing_time: float = 0.0

    # Error tracking
    errors: list[str] = field(default_factory=list)
