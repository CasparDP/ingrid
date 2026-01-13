"""Text cleanup processor for OCR error correction.

This module provides LLM-based text cleanup to fix OCR errors while preserving
the original structure, meaning, and language of the document.
"""

import logging
import time

from ..llm.base import BaseLLMProvider, LLMError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a document transcription specialist with expertise in historical documents. Your task is to correct OCR errors while maintaining complete fidelity to the original document's content, structure, and language."""


def build_cleanup_prompt(
    raw_text: str, doc_type: str, content_type: str, languages: list[str]
) -> str:
    """Build cleanup prompt with context.

    Args:
        raw_text: Raw OCR text to clean.
        doc_type: Document type (letter, newspaper_article, etc.).
        content_type: Content type (handwritten, typed, mixed).
        languages: List of detected languages.

    Returns:
        Formatted cleanup prompt.
    """
    language_str = ", ".join(languages) if languages else "unknown"

    return f"""Clean up the following OCR output from a {doc_type} ({content_type}).

The document is in {language_str}.

OCR Output:
{raw_text}

Instructions:
1. Fix obvious OCR errors (e.g., "tlie" -> "the", "rn" -> "m")
2. Maintain original paragraph structure and line breaks
3. Keep any dates, names, and places exactly as written (do not modernize spelling)
4. Mark unclear sections with [unclear]
5. Do not add any information not present in the original
6. Preserve the original language (do not translate)
7. For handwritten documents, be conservative - if uncertain, preserve original
8. Fix punctuation only if clearly wrong

Output the cleaned text only, no explanations."""


class TextCleanupProcessor:
    """LLM-based text cleanup processor for OCR error correction.

    This processor uses an LLM to intelligently correct OCR errors while preserving
    the original document's structure, meaning, and language.

    Attributes:
        llm: LLM provider instance.
        temperature: Low temperature for consistency (default: 0.3).
        max_retries: Maximum retry attempts (from config).
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        """Initialize text cleanup processor.

        Args:
            llm: LLM provider instance.
            temperature: Temperature for LLM calls (default: 0.3).
            max_retries: Maximum number of retry attempts (default: 3).
        """
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries
        logger.info(f"Initialized TextCleanupProcessor with temperature={temperature}")

    def cleanup(
        self,
        raw_text: str,
        doc_type: str,
        content_type: str,
        languages: list[str],
    ) -> tuple[str, float, str | None]:
        """Clean up OCR text using LLM.

        Args:
            raw_text: Raw OCR text to clean.
            doc_type: Document type (letter, newspaper_article, etc.).
            content_type: Content type (handwritten, typed, mixed).
            languages: List of detected languages.

        Returns:
            Tuple of (cleaned_text, processing_time, error_message).
            If cleanup fails, returns original text with error message.
        """
        start_time = time.time()

        # Validate input
        if not raw_text or len(raw_text.strip()) < 10:
            logger.warning("Text too short for cleanup")
            return (raw_text, 0.0, "Text too short for cleanup")

        try:
            # Build prompt
            prompt = build_cleanup_prompt(raw_text, doc_type, content_type, languages)

            # Call LLM with retry mechanism
            response = self.llm.retry_with_backoff(
                lambda: self.llm.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=None,  # Allow full response
                ),
                max_retries=self.max_retries,
            )

            cleaned_text = response.content.strip()
            processing_time = time.time() - start_time

            logger.info(
                f"Text cleanup complete: {len(raw_text)} -> {len(cleaned_text)} chars "
                f"in {processing_time:.2f}s"
            )

            return (cleaned_text, processing_time, None)

        except LLMError as e:
            logger.error(f"LLM error during cleanup: {e}")
            return (raw_text, time.time() - start_time, f"LLM error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}", exc_info=True)
            return (raw_text, time.time() - start_time, f"Unexpected error: {e}")
