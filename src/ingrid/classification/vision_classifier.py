"""Vision LLM-based document classifier.

This module implements document classification using vision-capable language models
to analyze document images and determine document type, content type, and languages.
"""

import json
import logging
import time
from pathlib import Path

from PIL import Image

from ..extraction.models import ContentType
from ..llm import BaseLLMProvider, LLMError
from .models import ClassificationResult, ClassifierType, DocType

logger = logging.getLogger(__name__)


# Classification prompts
SYSTEM_PROMPT = """You are a precise document classification system specializing in historical documents. Analyze documents carefully and provide accurate classifications with confidence scores."""

CLASSIFICATION_PROMPT = """Analyze this image and classify the document.

Determine:
1. doc_type: Is this a "letter", "newspaper_article", or "other"?
2. content_type: Is the text "handwritten", "typed", or "mixed"?
3. languages: What language(s) is the text in? (use ISO codes: en, nl, de, etc.)

Consider:
- Letters typically have a greeting, body, and signature
- Newspaper articles have headlines, columns, and formal structure
- Handwritten text has variable letterforms and connected writing
- Typed text has uniform characters

Respond in JSON format:
{
  "doc_type": "letter|newspaper_article|other",
  "content_type": "handwritten|typed|mixed",
  "languages": ["nl", "en"],
  "confidence": {
    "doc_type": 0.0-1.0,
    "content_type": 0.0-1.0
  },
  "reasoning": "Brief explanation of classification decision"
}"""


class VisionLLMClassifier:
    """Vision LLM-based classifier for document type, content type, and language detection.

    Uses a vision-capable language model to analyze document images and classify them.
    This is the primary, most accurate classification method.

    Attributes:
        llm: LLM provider instance with vision capabilities.
    """

    def __init__(self, llm: BaseLLMProvider) -> None:
        """Initialize vision LLM classifier.

        Args:
            llm: LLM provider instance with vision capabilities.
        """
        self.llm = llm
        logger.info(f"Initialized VisionLLMClassifier with {type(llm).__name__}")

    def classify(
        self,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
        text: str | None = None,
    ) -> ClassificationResult:
        """Classify a document using vision LLM.

        Args:
            image: PIL Image object to classify.
            image_path: Path to image file (used if image not provided).
            text: Optional extracted text hint (not currently used, reserved for future).

        Returns:
            ClassificationResult with doc_type, content_type, languages, and confidence scores.

        Raises:
            ValueError: If neither image nor image_path is provided.
        """
        start_time = time.time()

        # Validate inputs
        if image is None and image_path is None:
            raise ValueError("Either image or image_path must be provided")

        # Convert image to path if needed (vision() expects path, not PIL Image)
        if image_path is None and image is not None:
            # Save temporary image and use that path
            # For now, raise error - caller should provide image_path
            raise ValueError("image_path must be provided for vision LLM classification")

        # Ensure image_path is a Path object
        if isinstance(image_path, str):
            image_path = Path(image_path)

        # Call vision LLM
        try:
            logger.debug(f"Calling vision LLM for classification on {image_path}")
            response = self.llm.vision(
                image_path=image_path,
                prompt=CLASSIFICATION_PROMPT,
                system_prompt=SYSTEM_PROMPT,
            )

            logger.debug(f"Vision LLM response: {response.content[:200]}...")

            # Parse JSON response
            try:
                result_dict = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response.content}")
                return ClassificationResult(
                    classifier=ClassifierType.VISION_LLM,
                    success=False,
                    error=f"Failed to parse JSON response: {e}",
                    processing_time=time.time() - start_time,
                    metadata={"raw_response": response.content},
                )

            # Extract classification fields
            doc_type = self._parse_doc_type(result_dict.get("doc_type"))
            content_type = self._parse_content_type(result_dict.get("content_type"))
            languages = result_dict.get("languages", [])
            confidence_dict = result_dict.get("confidence", {})
            reasoning = result_dict.get("reasoning")

            # Calculate overall confidence (average of doc_type and content_type confidence)
            doc_type_conf = confidence_dict.get("doc_type", 0.5)
            content_type_conf = confidence_dict.get("content_type", 0.5)
            overall_confidence = (doc_type_conf + content_type_conf) / 2.0

            processing_time = time.time() - start_time

            logger.info(
                f"Vision LLM classification complete: {doc_type.value}, {content_type.value}, "
                f"languages={languages}, confidence={overall_confidence:.2f}"
            )

            return ClassificationResult(
                classifier=ClassifierType.VISION_LLM,
                doc_type=doc_type,
                content_type=content_type,
                languages=languages,
                confidence=overall_confidence,
                confidence_scores={
                    "doc_type": doc_type_conf,
                    "content_type": content_type_conf,
                },
                processing_time=processing_time,
                success=True,
                reasoning=reasoning,
                metadata={
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "raw_response": result_dict,
                },
            )

        except LLMError as e:
            logger.error(f"LLM error during classification: {e}")
            return ClassificationResult(
                classifier=ClassifierType.VISION_LLM,
                success=False,
                error=f"LLM error: {e}",
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error during classification: {e}", exc_info=True)
            return ClassificationResult(
                classifier=ClassifierType.VISION_LLM,
                success=False,
                error=f"Unexpected error: {e}",
                processing_time=time.time() - start_time,
            )

    def _parse_doc_type(self, value: str | None) -> DocType:
        """Parse doc_type string to DocType enum.

        Args:
            value: Doc type string from LLM response.

        Returns:
            DocType enum value, defaults to UNKNOWN if invalid.
        """
        if value is None:
            return DocType.UNKNOWN

        value = value.lower().strip()

        # Map common variations
        if value in ("letter", "letters"):
            return DocType.LETTER
        elif value in ("newspaper_article", "newspaper", "article"):
            return DocType.NEWSPAPER_ARTICLE
        elif value in ("other",):
            return DocType.OTHER
        else:
            logger.warning(f"Unknown doc_type value: {value}, defaulting to UNKNOWN")
            return DocType.UNKNOWN

    def _parse_content_type(self, value: str | None) -> ContentType:
        """Parse content_type string to ContentType enum.

        Args:
            value: Content type string from LLM response.

        Returns:
            ContentType enum value, defaults to UNKNOWN if invalid.
        """
        if value is None:
            return ContentType.UNKNOWN

        value = value.lower().strip()

        # Map common variations
        if value in ("handwritten", "handwriting", "hand-written"):
            return ContentType.HANDWRITTEN
        elif value in ("typed", "typewritten", "printed"):
            return ContentType.TYPED
        elif value in ("mixed", "both"):
            return ContentType.MIXED
        else:
            logger.warning(f"Unknown content_type value: {value}, defaulting to UNKNOWN")
            return ContentType.UNKNOWN
