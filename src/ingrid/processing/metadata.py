"""Metadata extractor for structured information extraction.

This module provides LLM-based metadata extraction to parse dates, people,
locations, topics, and other structured information from cleaned document text.

IMPORTANT: All extraction is strictly grounded in the source text.
The LLM is instructed to NEVER hallucinate or infer information not explicitly present.
"""

import json
import logging
import time

from ..llm.base import BaseLLMProvider, LLMError
from ..llm.structured import extract_with_retry, parse_metadata_response
from .models import DocumentMetadata

logger = logging.getLogger(__name__)


# Note: clean_json_response moved to llm.structured module
# Keeping this for backward compatibility if needed
from ..llm.structured import clean_json_response

SYSTEM_PROMPT = """You are a metadata extraction specialist for historical documents.

CRITICAL RULES - FOLLOW EXACTLY:
1. Extract ONLY information that is EXPLICITLY written in the document text.
2. NEVER guess, infer, or hallucinate information that is not clearly stated.
3. If a field cannot be determined from the text, use null - do NOT make up values.
4. For sender/recipient: Only extract if there is a clear signature or salutation with a name.
5. For people_mentioned: Only include actual proper names of people, NOT pronouns (I, you, ich, du, hij, zij, etc.) or generic terms (God, author, writer).
6. For dates: Only extract if a specific date is written in the document.
7. For locations: Only extract if a place name is explicitly mentioned.
8. Set confidence scores LOW (0.1-0.3) when information is ambiguous or partially visible.
9. Set confidence scores HIGH (0.7-1.0) only when information is clearly and unambiguously stated.

Your accuracy depends on being conservative - it is better to return null than to guess wrong."""


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

DOCUMENT TEXT:
---
{cleaned_text}
---

Extract the following fields ONLY from the text above:
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

STRICT RULES:
- Extract ONLY what is EXPLICITLY written in the document - do NOT infer or guess
- If a field cannot be determined from the text, use null (not "null" string, not "N/A", not "Unknown")
- For people_mentioned: Include ONLY actual proper names, NOT pronouns (ich, du, I, you, hij, zij) or generic terms
- For sender/recipient: Use null unless there is a clear signature or explicit "Dear [Name]" / "From [Name]"
- Confidence scores: Use LOW scores (0.1-0.3) for uncertain/partial information, HIGH (0.7-1.0) only for clearly stated facts
- Respond with ONLY the JSON object, no extra text before or after
- Do NOT use trailing commas in arrays or objects
- Ensure all strings are properly quoted
- Use empty arrays [] for lists with no items, not null"""


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

            # Call LLM with structured parsing and retry
            def llm_call(feedback: str = "") -> str:
                """LLM call wrapper for retry logic."""
                full_prompt = prompt + feedback
                response = self.llm.retry_with_backoff(
                    lambda: self.llm.complete(
                        prompt=full_prompt,
                        system_prompt=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_tokens=1000,
                    ),
                    max_retries=1,  # Single retry per parse attempt
                )
                return response.content

            # Use structured parsing with retry
            parsed_result = extract_with_retry(
                llm_call=llm_call,
                parser=parse_metadata_response,
                max_retries=self.max_retries,
            )

            # Check if parsing succeeded
            if parsed_result is None:
                # Fallback: try old JSON parsing method
                logger.warning("Structured parsing failed, falling back to JSON parsing")
                response = self.llm.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=1000,
                )

                try:
                    cleaned_response = clean_json_response(response.content)
                    result_dict = json.loads(cleaned_response)
                except json.JSONDecodeError as e:
                    logger.error(f"All parsing methods failed: {e}")
                    logger.error(f"Raw response (first 500 chars): {response.content[:500]}")
                    return DocumentMetadata(
                        success=False,
                        error=f"Failed to parse response after all attempts: {e}",
                        processing_time=time.time() - start_time,
                        raw_response={"raw": response.content[:1000]},
                    )
            else:
                # Convert Pydantic model to dict
                result_dict = parsed_result.model_dump()

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
