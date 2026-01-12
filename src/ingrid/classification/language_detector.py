"""Language detection utilities for document classification.

This module provides language detection functionality using the langdetect library,
with support for Dutch, English, German, and French.
"""

import logging
from typing import List, Tuple

from langdetect import LangDetectException, detect_langs

logger = logging.getLogger(__name__)


def detect_languages(text: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """Detect languages in text using langdetect.

    Args:
        text: Text to analyze for language detection.
        top_n: Maximum number of languages to return.

    Returns:
        List of (language_code, confidence) tuples, sorted by confidence descending.
        Returns empty list if detection fails.

    Examples:
        >>> detect_languages("Dit is een Nederlandse tekst")
        [('nl', 0.99), ('af', 0.01)]

        >>> detect_languages("This is English text")
        [('en', 0.99)]
    """
    if not text or len(text.strip()) < 10:
        logger.debug("Text too short for language detection (< 10 chars)")
        return []

    try:
        # detect_langs returns list of Language objects with .lang and .prob attributes
        detected = detect_langs(text)

        # Convert to list of tuples and limit to top_n
        results = [(lang.lang, lang.prob) for lang in detected][:top_n]

        logger.debug(f"Detected languages: {results}")
        return results

    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}")
        return []


def is_dutch(text: str, threshold: float = 0.5) -> bool:
    """Quick check if text is likely Dutch.

    Args:
        text: Text to analyze.
        threshold: Minimum confidence threshold (0.0-1.0).

    Returns:
        True if Dutch is detected with confidence >= threshold.

    Examples:
        >>> is_dutch("Dit is een Nederlandse tekst")
        True

        >>> is_dutch("This is English text")
        False
    """
    languages = detect_languages(text, top_n=1)

    if not languages:
        return False

    lang_code, confidence = languages[0]
    return lang_code == "nl" and confidence >= threshold


def is_english(text: str, threshold: float = 0.5) -> bool:
    """Quick check if text is likely English.

    Args:
        text: Text to analyze.
        threshold: Minimum confidence threshold (0.0-1.0).

    Returns:
        True if English is detected with confidence >= threshold.

    Examples:
        >>> is_english("This is English text")
        True

        >>> is_english("Dit is een Nederlandse tekst")
        False
    """
    languages = detect_languages(text, top_n=1)

    if not languages:
        return False

    lang_code, confidence = languages[0]
    return lang_code == "en" and confidence >= threshold


def get_primary_language(text: str) -> str | None:
    """Get the most likely language code for the given text.

    Args:
        text: Text to analyze.

    Returns:
        ISO 639-1 language code (e.g., 'nl', 'en') or None if detection fails.

    Examples:
        >>> get_primary_language("Dit is Nederlands")
        'nl'

        >>> get_primary_language("This is English")
        'en'
    """
    languages = detect_languages(text, top_n=1)

    if not languages:
        return None

    return languages[0][0]
