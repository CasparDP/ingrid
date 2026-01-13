"""Metadata extractor for structured information extraction.

This module provides LLM-based metadata extraction to parse dates, people,
locations, topics, and other structured information from cleaned document text.
"""

import json
import logging
import time

from ..llm.base import BaseLLMProvider, LLMError
from .models import DocumentMetadata

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a metadata extraction specialist for historical documents. Extract structured information accurately and provide confidence scores for each field."""


def build_metadata_prompt(cleaned_text: str, doc_type: str, languages: list[str]) -> str:
    """Build metadata extraction prompt.

    Args:
        cleaned_text: Cleaned document text.
        doc_type: Document type.
        languages: Detected languages.

    Returns:
        Formatted metadata extraction prompt.
    """
    language_str = ", ".join(languages) if languages else "unknown"

    # Customize fields based on doc_type
    if doc_type == "letter":
        fields_description = """
- date: The date of the letter (preserve original format, provide ISO if parseable)
- sender: Who wrote the letter
- recipient: Who the letter was addressed to
- location: Where the letter was written or references
- topics: Main topics/themes (3-5 keywords)
- people_mentioned: Names of people mentioned in the letter
- organizations_mentioned: Organizations or institutions mentioned"""
    else:
        fields_description = """
- date: The date of the article or document (preserve original format, provide ISO if parseable)
- sender: Author or writer (if applicable)
- recipient: N/A for newspaper articles
- location: Location mentioned or where event occurred
- topics: Main topics/themes (3-5 keywords)
- people_mentioned: Names of people mentioned
- organizations_mentioned: Organizations or institutions mentioned"""

    return f"""Analyze this {doc_type} (in {language_str}) and extract metadata.

Text:
{cleaned_text}

Extract the following fields:
{fields_description}

Respond in JSON format:
{{
  "date": "string or null",
  "sender": "string or null",
  "recipient": "string or null",
  "location": "string or null",
  "topics": ["topic1", "topic2"],
  "people_mentioned": ["name1", "name2"],
  "organizations_mentioned": ["org1", "org2"],
  "confidence": {{
    "date": 0.0-1.0,
    "sender": 0.0-1.0,
    "recipient": 0.0-1.0,
    "location": 0.0-1.0
  }}
}}

If a field cannot be determined, use null. Provide confidence scores for each extracted field."""


class MetadataExtractor:
    """LLM-based metadata extractor.

    This processor uses an LLM to extract structured metadata from cleaned
    document text, including dates, people, locations, and topics.

    Attributes:
        llm: LLM provider instance.
        temperature: Low temperature for deterministic output (default: 0.2).
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        """Initialize metadata extractor.

        Args:
            llm: LLM provider instance.
            temperature: Temperature for LLM calls (default: 0.2).
            max_retries: Maximum number of retry attempts (default: 3).
        """
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries
        logger.info(f"Initialized MetadataExtractor with temperature={temperature}")

    def extract(
        self,
        cleaned_text: str,
        doc_type: str,
        languages: list[str],
    ) -> DocumentMetadata:
        """Extract metadata from cleaned text.

        Args:
            cleaned_text: Cleaned document text.
            doc_type: Document type.
            languages: Detected languages.

        Returns:
            DocumentMetadata with extracted fields and confidence scores.
        """
        start_time = time.time()

        # Validate input
        if not cleaned_text or len(cleaned_text.strip()) < 20:
            logger.warning("Text too short for metadata extraction")
            return DocumentMetadata(
                success=False,
                error="Text too short for metadata extraction",
                processing_time=0.0,
            )

        try:
            # Build prompt
            prompt = build_metadata_prompt(cleaned_text, doc_type, languages)

            # Call LLM with retry mechanism
            response = self.llm.retry_with_backoff(
                lambda: self.llm.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=1000,
                ),
                max_retries=self.max_retries,
            )

            # Parse JSON response
            try:
                result_dict = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response.content}")
                return DocumentMetadata(
                    success=False,
                    error=f"Failed to parse JSON: {e}",
                    processing_time=time.time() - start_time,
                    raw_response={"raw": response.content},
                )

            # Build DocumentMetadata from response
            confidence_dict = result_dict.get("confidence", {})

            metadata = DocumentMetadata(
                date=result_dict.get("date"),
                sender=result_dict.get("sender"),
                recipient=result_dict.get("recipient"),
                location=result_dict.get("location"),
                topics=result_dict.get("topics", []),
                people_mentioned=result_dict.get("people_mentioned", []),
                organizations_mentioned=result_dict.get("organizations_mentioned", []),
                date_confidence=confidence_dict.get("date", 0.5),
                sender_confidence=confidence_dict.get("sender", 0.5),
                recipient_confidence=confidence_dict.get("recipient", 0.5),
                location_confidence=confidence_dict.get("location", 0.5),
                processing_time=time.time() - start_time,
                success=True,
                raw_response=result_dict,
            )

            logger.info(f"Metadata extraction complete in {metadata.processing_time:.2f}s")

            return metadata

        except LLMError as e:
            logger.error(f"LLM error during metadata extraction: {e}")
            return DocumentMetadata(
                success=False,
                error=f"LLM error: {e}",
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error during metadata extraction: {e}", exc_info=True)
            return DocumentMetadata(
                success=False,
                error=f"Unexpected error: {e}",
                processing_time=time.time() - start_time,
            )
