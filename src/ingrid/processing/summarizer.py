"""Document summarizer for generating concise summaries.

This module provides LLM-based summarization to create 2-3 sentence summaries
of document content in the original language.
"""

import logging
import time

from ..llm.base import BaseLLMProvider, LLMError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a document summarization specialist. Create concise, accurate summaries that capture the essential content and context of historical documents."""


def build_summary_prompt(cleaned_text: str, doc_type: str, languages: list[str]) -> str:
    """Build summarization prompt.

    Args:
        cleaned_text: Cleaned document text.
        doc_type: Document type.
        languages: Detected languages.

    Returns:
        Formatted summarization prompt.
    """
    primary_language = languages[0] if languages else "English"
    language_names = {"nl": "Dutch", "en": "English", "de": "German", "fr": "French"}
    language_name = language_names.get(primary_language, primary_language)

    if doc_type == "letter":
        summary_instructions = """Summarize this letter in 2-3 sentences. Include:
- Who wrote to whom
- Main purpose or content of the letter
- Any notable context or circumstances"""
    else:
        summary_instructions = """Summarize this document in 2-3 sentences. Include:
- Main topic or event
- Key details or findings
- Any notable context"""

    return f"""Summarize the following {doc_type}.

Text:
{cleaned_text}

{summary_instructions}

Write the summary in {language_name} (the original language of the document).
Keep it concise (2-3 sentences maximum).
Output only the summary, no additional formatting or explanation."""


class DocumentSummarizer:
    """LLM-based document summarizer.

    This processor uses an LLM to generate concise 2-3 sentence summaries
    in the same language as the original document.

    Attributes:
        llm: LLM provider instance.
        temperature: Moderate temperature for creative summarization (default: 0.5).
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        temperature: float = 0.5,
        max_retries: int = 3,
    ) -> None:
        """Initialize document summarizer.

        Args:
            llm: LLM provider instance.
            temperature: Temperature for LLM calls (default: 0.5).
            max_retries: Maximum number of retry attempts (default: 3).
        """
        self.llm = llm
        self.temperature = temperature
        self.max_retries = max_retries
        logger.info(f"Initialized DocumentSummarizer with temperature={temperature}")

    def summarize(
        self,
        cleaned_text: str,
        doc_type: str,
        languages: list[str],
    ) -> tuple[str, float, str | None]:
        """Generate summary from cleaned text.

        Args:
            cleaned_text: Cleaned document text.
            doc_type: Document type.
            languages: Detected languages.

        Returns:
            Tuple of (summary, processing_time, error_message).
            If summarization fails, returns empty summary with error message.
        """
        start_time = time.time()

        # Validate input
        if not cleaned_text or len(cleaned_text.strip()) < 20:
            logger.warning("Text too short for summarization")
            return ("", 0.0, "Text too short for summarization")

        try:
            # Build prompt
            prompt = build_summary_prompt(cleaned_text, doc_type, languages)

            # Call LLM with retry mechanism
            response = self.llm.retry_with_backoff(
                lambda: self.llm.complete(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_tokens=300,  # ~2-3 sentences
                ),
                max_retries=self.max_retries,
            )

            summary = response.content.strip()
            processing_time = time.time() - start_time

            logger.info(
                f"Summarization complete in {processing_time:.2f}s: {len(summary)} chars"
            )

            return (summary, processing_time, None)

        except LLMError as e:
            logger.error(f"LLM error during summarization: {e}")
            return ("", time.time() - start_time, f"LLM error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {e}", exc_info=True)
            return ("", time.time() - start_time, f"Unexpected error: {e}")
