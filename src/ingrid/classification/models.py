"""Data models for document classification.

This module defines all data structures used throughout the classification process,
including classification results, classification jobs, and document type enums.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..extraction.models import ContentType


class ClassifierType(Enum):
    """Types of classification methods."""

    VISION_LLM = "vision_llm"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"  # Future: combination


class DocType(Enum):
    """Document type categories."""

    LETTER = "letter"
    NEWSPAPER_ARTICLE = "newspaper_article"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result from a single classification attempt.

    Attributes:
        classifier: Type of classifier that produced this result.
        timestamp: When this classification was performed.
        doc_type: Detected document type.
        content_type: Detected content type (handwritten/typed/mixed).
        languages: List of detected language codes (e.g., ['nl', 'en']).
        confidence: Overall confidence score (0.0-1.0).
        confidence_scores: Per-field confidence scores.
        processing_time: Time taken for classification in seconds.
        success: Whether classification was successful.
        error: Error message if classification failed.
        metadata: Additional classifier-specific metadata.
        reasoning: Classifier's reasoning (if available, e.g., from LLM).
    """

    # Identification
    classifier: ClassifierType
    timestamp: datetime = field(default_factory=datetime.now)

    # Classification results
    doc_type: DocType = DocType.UNKNOWN
    content_type: ContentType = ContentType.UNKNOWN
    languages: list[str] = field(default_factory=list)

    # Quality metrics
    confidence: float = 0.0  # Overall confidence (0.0-1.0)
    confidence_scores: dict[str, float] = field(default_factory=dict)  # Per-field

    # Performance
    processing_time: float = 0.0  # Seconds

    # Status
    success: bool = True
    error: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None  # LLM's reasoning (if available)


@dataclass
class ClassificationJob:
    """Complete classification job for a single document.

    Attributes:
        image_path: Path to the input image file.
        extracted_text: Optional extracted text from Phase 2.
        config: Configuration dictionary used for classification.
        results: List of classification results from all classifiers.
        primary_result: Best classification result (selected by orchestrator).
        success: Whether at least one classification succeeded.
        total_processing_time: Total time for entire job in seconds.
        flagged_for_review: Whether result should be reviewed (low confidence).
        manual_override: User-provided corrections to classification.
        errors: List of error messages encountered during job.
    """

    # Input
    image_path: Path
    extracted_text: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # Classification results (multiple classifiers)
    results: list[ClassificationResult] = field(default_factory=list)

    # Best result (selected by orchestrator)
    primary_result: ClassificationResult | None = None

    # Overall status
    success: bool = False
    total_processing_time: float = 0.0

    # Flags
    flagged_for_review: bool = False  # If confidence < threshold
    manual_override: dict[str, str] | None = None  # User corrections

    # Error tracking
    errors: list[str] = field(default_factory=list)
