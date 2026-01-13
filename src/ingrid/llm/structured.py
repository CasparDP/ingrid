"""Structured LLM response parsing with Pydantic validation.

This module provides robust parsing of LLM responses into structured data
using Pydantic models. It includes fallback strategies for handling malformed
JSON responses from smaller local models.
"""

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Pydantic Response Models
# =============================================================================


# Common pronouns and non-name words to filter out from people_mentioned
# These are often incorrectly extracted by LLMs as "names"
PRONOUNS_AND_NON_NAMES = {
    # German pronouns
    "ich",
    "du",
    "er",
    "sie",
    "es",
    "wir",
    "ihr",
    "sie",
    "Sie",
    "ICH",
    "DU",
    "ER",
    "SIE",
    "ES",
    "WIR",
    "IHR",
    "SIE",
    # Dutch pronouns
    "ik",
    "jij",
    "je",
    "hij",
    "zij",
    "ze",
    "het",
    "wij",
    "we",
    "jullie",
    "IK",
    "JIJ",
    "JE",
    "HIJ",
    "ZIJ",
    "ZE",
    "HET",
    "WIJ",
    "WE",
    "JULLIE",
    # English pronouns
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "I",
    "You",
    "He",
    "She",
    "It",
    "We",
    "They",
    "Me",
    "Him",
    "Her",
    "Us",
    "Them",
    # Common non-name words often mistakenly extracted
    "GOD",
    "God",
    "god",
    "GOTT",
    "Gott",
    "gott",
    "unknown",
    "Unknown",
    "UNKNOWN",
    "n/a",
    "N/A",
    "null",
    "none",
    "None",
    "NONE",
    "author",
    "Author",
    "AUTHOR",
    "sender",
    "Sender",
    "SENDER",
    "recipient",
    "Recipient",
    "RECIPIENT",
    "writer",
    "Writer",
    "WRITER",
}


class MetadataResponse(BaseModel):
    """Pydantic model for metadata extraction response.

    All fields must be extracted ONLY from the provided document text.
    Do not infer, guess, or hallucinate information not explicitly present.
    """

    date: str | None = Field(
        default=None,
        description="Document date ONLY if explicitly written in the text. Use null if not found.",
    )
    sender: str | None = Field(
        default=None,
        description="Name of sender/author ONLY if explicitly signed or stated. Use null if unknown.",
    )
    recipient: str | None = Field(
        default=None,
        description="Name of recipient ONLY if explicitly addressed. Use null if unknown.",
    )
    location: str | None = Field(
        default=None,
        description="Location ONLY if explicitly mentioned in the text. Use null if not found.",
    )
    topics: list[str] = Field(
        default_factory=list, description="3-5 main topics/themes discussed in the document."
    )
    people_mentioned: list[str] = Field(
        default_factory=list,
        description="Actual names of people mentioned. Do NOT include pronouns (I, you, ich, du, etc.) or generic terms.",
    )
    organizations_mentioned: list[str] = Field(
        default_factory=list,
        description="Organizations or institutions explicitly named in the text.",
    )
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores (0.0-1.0) for date, sender, recipient, location fields.",
    )

    @field_validator("people_mentioned", mode="before")
    @classmethod
    def filter_pronouns_and_non_names(cls, v: list[str]) -> list[str]:
        """Filter out pronouns and non-name words from people_mentioned."""
        if not v:
            return []
        return [
            name
            for name in v
            if name
            and name.strip() not in PRONOUNS_AND_NON_NAMES
            and len(name.strip()) > 1  # Filter single characters
        ]

    @field_validator("sender", "recipient", mode="before")
    @classmethod
    def normalize_unknown_values(cls, v: str | None) -> str | None:
        """Normalize 'unknown', 'n/a', etc. to None."""
        if v is None:
            return None
        v_lower = v.strip().lower()
        if v_lower in {"unknown", "n/a", "null", "none", "", "unbekannt", "onbekend"}:
            return None
        return v.strip()

    class Config:
        """Pydantic config."""

        # Allow extra fields for forward compatibility
        extra = "allow"


class ClassificationResponse(BaseModel):
    """Pydantic model for document classification response."""

    doc_type: str
    content_type: str
    languages: list[str]
    confidence: dict[str, float]
    reasoning: str | None = None

    class Config:
        """Pydantic config."""

        extra = "allow"


class SummaryResponse(BaseModel):
    """Pydantic model for summary response."""

    summary: str
    language: str | None = None

    class Config:
        """Pydantic config."""

        extra = "allow"


# =============================================================================
# JSON Cleaning Utilities
# =============================================================================


def clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON.

    Handles common issues:
    - Markdown code blocks (```json ... ```)
    - Leading/trailing whitespace
    - Text before/after the JSON object
    - Trailing commas in arrays/objects

    Args:
        response: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*$", "", response)
    response = re.sub(r"```", "", response)

    # Strip whitespace
    response = response.strip()

    # Try to find JSON object boundaries
    # Look for first { and last }
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1 and end > start:
        response = response[start : end + 1]

    # Fix trailing commas before closing brackets/braces (common LLM error)
    response = re.sub(r",\s*}", "}", response)
    response = re.sub(r",\s*]", "]", response)

    # Fix unescaped newlines in strings (rare but happens)
    # This is a simple fix that may not handle all cases
    response = response.replace("\n", " ")

    return response


def extract_json_manually(response: str, model: type[T]) -> dict[str, Any]:
    """Manually extract fields from response text as last resort.

    This function uses regex to extract field values when JSON parsing fails.
    It's a fallback for very malformed responses.

    Args:
        response: Raw LLM response
        model: Pydantic model class

    Returns:
        Dictionary with extracted fields (best effort)
    """
    logger.warning("Falling back to manual field extraction")

    result: dict[str, Any] = {}

    # Get field names from Pydantic model
    if hasattr(model, "model_fields"):
        field_names = model.model_fields.keys()
    else:
        field_names = []

    # Try to extract each field using common patterns
    for field_name in field_names:
        # Pattern: "field_name": "value" or 'field_name': 'value'
        pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, response)
        if match:
            result[field_name] = match.group(1)
            continue

        # Pattern: "field_name": ["item1", "item2"]
        pattern = rf'"{field_name}"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, response)
        if match:
            items_str = match.group(1)
            items = re.findall(r'"([^"]*)"', items_str)
            result[field_name] = items
            continue

        # Pattern: "field_name": value (no quotes)
        pattern = rf'"{field_name}"\s*:\s*([^,\}}\]]*)'
        match = re.search(pattern, response)
        if match:
            value = match.group(1).strip()
            if value.lower() == "null":
                result[field_name] = None
            elif value.lower() == "true":
                result[field_name] = True
            elif value.lower() == "false":
                result[field_name] = False
            else:
                try:
                    result[field_name] = float(value)
                except ValueError:
                    result[field_name] = value

    logger.debug(f"Manually extracted fields: {list(result.keys())}")
    return result


# =============================================================================
# Main Parsing Functions
# =============================================================================


def parse_json_with_validation(
    response: str,
    model: type[T],
    fallback_to_regex: bool = True,
    fallback_to_manual: bool = True,
) -> T | None:
    """Parse and validate JSON response with Pydantic model.

    Strategy:
    1. Try direct JSON parse + Pydantic validation
    2. If fails, try clean_json_response() + parse
    3. If fails, try manual field extraction (if enabled)
    4. If all fail, return None

    Args:
        response: Raw LLM response
        model: Pydantic model class
        fallback_to_regex: Try regex-based cleaning on failure
        fallback_to_manual: Try manual field extraction as last resort

    Returns:
        Validated Pydantic model instance or None
    """
    # Step 1: Direct parse
    try:
        data = json.loads(response)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.debug(f"Direct parse failed: {e}")

    # Step 2: Cleaned parse
    if fallback_to_regex:
        try:
            cleaned = clean_json_response(response)
            data = json.loads(cleaned)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Cleaned parse failed: {e}")

    # Step 3: Manual extraction
    if fallback_to_manual:
        try:
            data = extract_json_manually(response, model)
            if data:
                return model.model_validate(data)
        except ValidationError as e:
            logger.debug(f"Manual extraction validation failed: {e}")

    # All attempts failed
    logger.error(f"All parsing attempts failed for response: {response[:200]}...")
    return None


def parse_metadata_response(response: str) -> MetadataResponse | None:
    """Parse metadata extraction response.

    Args:
        response: Raw LLM response

    Returns:
        MetadataResponse or None if parsing fails
    """
    return parse_json_with_validation(response, MetadataResponse)


def parse_classification_response(response: str) -> ClassificationResponse | None:
    """Parse classification response.

    Args:
        response: Raw LLM response

    Returns:
        ClassificationResponse or None if parsing fails
    """
    return parse_json_with_validation(response, ClassificationResponse)


def parse_summary_response(response: str) -> SummaryResponse | None:
    """Parse summary response.

    Args:
        response: Raw LLM response

    Returns:
        SummaryResponse or None if parsing fails
    """
    return parse_json_with_validation(response, SummaryResponse)


# =============================================================================
# Retry with Feedback
# =============================================================================


def extract_with_retry(
    llm_call: callable,
    parser: callable,
    max_retries: int = 3,
) -> Any | None:
    """Extract structured data with retry on parse failure.

    Args:
        llm_call: Callable that returns LLM response
        parser: Callable that parses response into structured data
        max_retries: Maximum retry attempts

    Returns:
        Parsed structured data or None
    """
    feedback = ""

    for attempt in range(max_retries):
        # Call LLM
        response = llm_call(feedback)

        # Try to parse
        result = parser(response.content if hasattr(response, "content") else response)

        if result is not None:
            if attempt > 0:
                logger.info(f"Parse succeeded on attempt {attempt + 1}")
            return result

        # On failure, add feedback for next attempt
        if attempt < max_retries - 1:
            feedback = (
                "\n\nPREVIOUS ATTEMPT FAILED - JSON PARSE ERROR.\n"
                "Please respond with ONLY valid JSON, no extra text.\n"
                "Ensure: no trailing commas, proper quotes, valid structure."
            )
            logger.debug(f"Parse failed on attempt {attempt + 1}, retrying with feedback")

    logger.error(f"Failed to parse response after {max_retries} attempts")
    return None
