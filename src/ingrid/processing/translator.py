"""Summary translation for generating English translations."""

import logging
import time

from ingrid.llm.base import BaseLLMProvider, LLMError

logger = logging.getLogger(__name__)


# Language code to full name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "nl": "Dutch",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
}


class SummaryTranslator:
    """Translates document summaries to English."""

    SYSTEM_PROMPT = """You are a translation specialist with expertise in historical documents and multilingual content.

Your task is to translate document summaries while preserving their meaning, tone, and historical context."""

    def __init__(self, llm: BaseLLMProvider):
        """Initialize translator.

        Args:
            llm: LLM provider for translation
        """
        self.llm = llm
        logger.debug("SummaryTranslator initialized")

    def translate_to_english(
        self, summary: str, source_language: str, max_retries: int = 3
    ) -> tuple[str, float, str | None]:
        """Translate summary to English if not already in English.

        Args:
            summary: Summary text in original language
            source_language: ISO language code of the summary (e.g., 'nl', 'de')
            max_retries: Maximum retry attempts for LLM calls

        Returns:
            Tuple of (translated_summary, processing_time, error_message)
            - If source_language is 'en', returns (summary, 0.0, None)
            - On success: (translated_text, time, None)
            - On failure: ("", time, error_message)
        """
        start_time = time.time()

        # Skip translation if already in English
        if source_language == "en":
            logger.debug("Summary already in English, skipping translation")
            return summary, 0.0, None

        # Validate input
        if not summary or len(summary.strip()) < 10:
            error = "Summary too short for translation"
            logger.warning(error)
            return "", time.time() - start_time, error

        # Get full language name
        source_lang_name = LANGUAGE_NAMES.get(source_language, source_language.upper())

        # Build translation prompt
        prompt = self._build_translation_prompt(summary, source_lang_name)

        try:
            logger.debug(
                f"Translating {len(summary)} character summary from {source_lang_name} to English"
            )

            # Call LLM with retry logic
            response = self.llm.retry_with_backoff(
                lambda: self.llm.complete(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=0.3,  # Low temperature for deterministic translation
                    max_tokens=500,  # Summaries should be concise
                ),
                max_retries=max_retries,
            )

            translation = response.content.strip()
            processing_time = time.time() - start_time

            # Validate translation
            if not translation or len(translation) < 10:
                error = "Translation too short or empty"
                logger.error(error)
                return "", processing_time, error

            logger.info(
                f"Translation completed: {len(translation)} characters "
                f"from {source_lang_name} in {processing_time:.1f}s"
            )
            return translation, processing_time, None

        except LLMError as e:
            processing_time = time.time() - start_time
            error = f"LLM error during translation: {str(e)}"
            logger.error(error)
            return "", processing_time, error

        except Exception as e:
            processing_time = time.time() - start_time
            error = f"Unexpected error during translation: {str(e)}"
            logger.error(error)
            return "", processing_time, error

    def _build_translation_prompt(self, summary: str, source_language: str) -> str:
        """Build translation prompt for LLM.

        Args:
            summary: Summary text to translate
            source_language: Full name of source language

        Returns:
            Formatted prompt for LLM
        """
        return f"""Translate the following {source_language} document summary to English.

Guidelines:
1. Preserve the original meaning and tone exactly
2. Maintain the same level of detail and structure
3. Keep it concise (2-3 sentences)
4. Do not add information that is not in the original
5. Do not explain or comment on the translation
6. Output ONLY the English translation

{source_language} Summary:
{summary}

English Translation:"""
