"""Vision LLM extraction using configured vision model.

Leverages the existing LLM abstraction to use vision models (qwen3-vl, llama-vision, etc.)
for document transcription. Works with any vision-capable provider.
"""

import logging
import tempfile
import time
from pathlib import Path

from PIL import Image

from ..llm import BaseLLMProvider, LLMError, VisionResponse
from .models import ContentType, ExtractionResult, ExtractorType

logger = logging.getLogger(__name__)


# Extraction prompts
TRANSCRIPTION_PROMPT = """You are a document transcription expert. Your task is to accurately transcribe all text visible in this document image.

Instructions:
1. Transcribe ALL text exactly as written, preserving spelling, capitalization, and punctuation
2. Maintain the original layout and paragraph structure
3. If text is unclear or illegible, mark it as [unclear]
4. Do not translate or interpret - transcribe literally
5. Include any dates, addresses, signatures, or marginalia
6. If the document is in multiple languages, transcribe all of them

Output only the transcribed text, no explanations or commentary."""

SYSTEM_PROMPT = """You are a precise historical document transcription system. Accuracy and faithfulness to the original text are paramount."""


class VisionLLMExtractor:
    """Extractor using vision-capable LLM.

    This extractor leverages the existing LLM abstraction layer to use
    vision models for document transcription. It can work with any provider
    that supports vision capabilities (Ollama Cloud, GPT-4V, Gemini, etc.).

    Attributes:
        llm: Initialized LLM provider with vision support.
    """

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """Initialize vision LLM extractor.

        Args:
            llm_provider: Initialized LLM provider with vision support.
        """
        self.llm = llm_provider
        logger.info(
            f"Initialized VisionLLM extractor with provider: {type(llm_provider).__name__}"
        )

    def extract(
        self,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
    ) -> ExtractionResult:
        """Extract text from image using vision LLM.

        Args:
            image: PIL Image object.
            image_path: Path to image file (required - LLM providers need file path).
            prompt: Custom prompt (defaults to TRANSCRIPTION_PROMPT).
            system_prompt: Custom system prompt (defaults to SYSTEM_PROMPT).

        Returns:
            ExtractionResult with extracted text and metadata.

        Raises:
            ValueError: If image_path not provided.
        """
        start_time = time.time()

        # Vision LLM providers typically need file paths
        cleanup_temp = False
        if image_path is None:
            # Save PIL image to temp if needed
            if image is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as tmp:
                    image.save(tmp.name)
                    image_path = Path(tmp.name)
                    cleanup_temp = True
            else:
                raise ValueError("Must provide image_path for vision LLM")
        else:
            image_path = Path(image_path)

        prompt = prompt or TRANSCRIPTION_PROMPT
        system_prompt = system_prompt or SYSTEM_PROMPT

        try:
            # Call vision LLM
            response: VisionResponse = self.llm.vision(
                image_path=image_path,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            text = response.content.strip()

            # Extract confidence if provided
            confidence = 0.8  # Default for LLM extraction
            if response.confidence and "overall" in response.confidence:
                confidence = response.confidence["overall"]

            # Detect content type from text characteristics
            content_type_hint = self._infer_content_type(text)

            processing_time = time.time() - start_time

            result = ExtractionResult(
                extractor=ExtractorType.VISION_LLM,
                text=text,
                raw_text=text,
                confidence=confidence,
                character_count=len(text),
                word_count=len(text.split()),
                detected_languages=[],  # Could add language detection here
                content_type_hint=content_type_hint,
                processing_time=processing_time,
                success=True,
                metadata={
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "llm_metadata": response.metadata,
                },
            )

            logger.info(
                f"VisionLLM extraction: {len(text)} chars, "
                f"confidence={confidence:.2f}, time={processing_time:.2f}s"
            )

            # Cleanup temp file if created
            if cleanup_temp:
                image_path.unlink(missing_ok=True)

            return result

        except LLMError as e:
            logger.error(f"VisionLLM extraction failed: {e}")
            return ExtractionResult(
                extractor=ExtractorType.VISION_LLM,
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _infer_content_type(self, text: str) -> ContentType:
        """Infer whether text is handwritten or typed based on characteristics.

        This is a rough heuristic - Phase 3 classification will be more robust.

        Args:
            text: Extracted text.

        Returns:
            Content type hint.
        """
        # Simple heuristic: presence of [unclear] suggests handwritten
        if "[unclear]" in text.lower():
            return ContentType.HANDWRITTEN

        # Very short text is ambiguous
        if len(text) < 50:
            return ContentType.UNKNOWN

        # Check for informal language patterns (handwritten letters)
        informal_markers = ["dear ", "sincerely", "love,", "yours truly"]
        if any(marker in text.lower() for marker in informal_markers):
            return ContentType.HANDWRITTEN

        # Default to unknown (Phase 3 will decide)
        return ContentType.UNKNOWN
