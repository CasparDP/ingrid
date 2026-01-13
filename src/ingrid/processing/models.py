"""Data models for document processing.

This module defines the data structures used throughout the processing pipeline,
including metadata extraction, text cleanup results, and processing jobs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..extraction.models import ExtractionJob
from ..classification.models import ClassificationJob


class ProcessorType(Enum):
    """Types of processing operations."""

    TEXT_CLEANUP = "text_cleanup"
    METADATA_EXTRACTION = "metadata_extraction"
    SUMMARIZATION = "summarization"


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from document.

    Attributes:
        date: Document date (ISO format string or original format).
        sender: Sender name (for letters).
        recipient: Recipient name (for letters).
        location: Location mentioned or where written.
        topics: Key topics/themes (3-5 keywords).
        people_mentioned: List of people mentioned in document.
        organizations_mentioned: List of organizations mentioned.
        date_confidence: Confidence in date extraction (0.0-1.0).
        sender_confidence: Confidence in sender extraction (0.0-1.0).
        recipient_confidence: Confidence in recipient extraction (0.0-1.0).
        location_confidence: Confidence in location extraction (0.0-1.0).
        processing_time: Time taken for metadata extraction.
        success: Whether extraction was successful.
        error: Error message if extraction failed.
        raw_response: Raw LLM response (for debugging).
    """

    # Core metadata fields
    date: str | None = None
    sender: str | None = None
    recipient: str | None = None
    location: str | None = None
    topics: list[str] = field(default_factory=list)
    people_mentioned: list[str] = field(default_factory=list)
    organizations_mentioned: list[str] = field(default_factory=list)

    # Confidence scores per field
    date_confidence: float = 0.0
    sender_confidence: float = 0.0
    recipient_confidence: float = 0.0
    location_confidence: float = 0.0

    # Processing metadata
    processing_time: float = 0.0
    success: bool = True
    error: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class ProcessingResult:
    """Complete processing result for a document.

    Attributes:
        image_path: Path to the original image file.
        extraction_job: Original extraction job from Phase 2.
        classification_job: Original classification job from Phase 3.
        config: Configuration snapshot used for processing.

        cleaned_text: LLM-cleaned text (OCR errors fixed).
        summary: Document summary in original language (2-3 sentences).
        summary_english: English translation of summary.
        summary_language: Language code of original summary (e.g., 'nl').
        metadata: Extracted structured metadata.

        cleanup_success: Whether text cleanup succeeded.
        cleanup_error: Error message if cleanup failed.
        cleanup_time: Time taken for cleanup.

        summary_success: Whether summarization succeeded.
        summary_error: Error message if summarization failed.
        summary_time: Time taken for summarization.

        translation_success: Whether translation succeeded.
        translation_error: Error message if translation failed.
        translation_time: Time taken for translation.

        metadata_success: Whether metadata extraction succeeded.
        metadata_error: Error message if metadata extraction failed.
        metadata_time: Time taken for metadata extraction.

        markdown_path: Path to generated markdown file.
        markdown_success: Whether markdown generation succeeded.
        markdown_error: Error message if markdown failed.

        success: Whether all enabled processing steps succeeded.
        total_processing_time: Total time for all processing.
    """

    # Input
    image_path: Path
    extraction_job: ExtractionJob
    classification_job: ClassificationJob
    config: dict[str, Any] = field(default_factory=dict)

    # Processing outputs
    cleaned_text: str = ""
    summary: str = ""
    summary_english: str = ""  # English translation of summary
    summary_language: str = ""  # Language code of original summary (e.g., 'nl')
    metadata: DocumentMetadata | None = None

    # Per-stage status
    cleanup_success: bool = False
    cleanup_error: str | None = None
    cleanup_time: float = 0.0

    summary_success: bool = False
    summary_error: str | None = None
    summary_time: float = 0.0

    translation_success: bool = False
    translation_error: str | None = None
    translation_time: float = 0.0

    metadata_success: bool = False
    metadata_error: str | None = None
    metadata_time: float = 0.0

    # Markdown output
    markdown_path: Path | None = None
    markdown_success: bool = False
    markdown_error: str | None = None

    # Overall status
    success: bool = False
    total_processing_time: float = 0.0
