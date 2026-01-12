"""Handwritten text recognition using Microsoft TrOCR.

TrOCR is a transformer-based HTR model that works well on handwritten
historical documents. It uses a vision encoder-decoder architecture.
"""

import logging
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .models import ContentType, ExtractionResult, ExtractorType

logger = logging.getLogger(__name__)


class TrOCRExtractor:
    """HTR extractor using Microsoft TrOCR.

    TrOCR is a transformer-based model specifically designed for handwritten
    text recognition. It supports multiple models for different use cases.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Device to run on ("cpu", "cuda", "mps").
        processor: TrOCR processor for image preprocessing.
        model: TrOCR vision-encoder-decoder model.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-large-handwritten",
        device: str | None = None,
    ) -> None:
        """Initialize TrOCR extractor.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run on ("cpu", "cuda", "mps", or None for auto).
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading TrOCR model: {model_name} on {device}")

        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        logger.info("TrOCR model loaded successfully")

    def extract(
        self,
        image: Image.Image | None = None,
        image_path: Path | str | None = None,
    ) -> ExtractionResult:
        """Extract handwritten text from image using TrOCR.

        Note: TrOCR processes the entire image as a single line/block. For multi-line
        documents, results may be suboptimal. Line segmentation will improve accuracy
        (future enhancement).

        Args:
            image: PIL Image object (preferred).
            image_path: Path to image file (fallback).

        Returns:
            ExtractionResult with extracted text and metadata.

        Raises:
            ValueError: If neither image nor image_path provided.
        """
        start_time = time.time()

        if image is None and image_path is None:
            raise ValueError("Must provide either image or image_path")

        if image is None:
            image_path = Path(image_path)  # type: ignore
            image = Image.open(image_path).convert("RGB")

        try:
            # Preprocess image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            # Decode to text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # TrOCR doesn't provide confidence scores directly
            # We estimate based on output quality
            confidence = self._estimate_confidence(text, image)

            processing_time = time.time() - start_time

            result = ExtractionResult(
                extractor=ExtractorType.TROCR_HTR,
                text=text,
                raw_text=text,  # TrOCR output is already "clean"
                confidence=confidence,
                character_count=len(text),
                word_count=len(text.split()),
                detected_languages=[],  # TrOCR doesn't detect language
                content_type_hint=ContentType.HANDWRITTEN,
                processing_time=processing_time,
                success=True,
                metadata={
                    "model": self.model_name,
                    "device": self.device,
                },
            )

            logger.info(
                f"TrOCR extraction: {len(text)} chars, "
                f"confidence={confidence:.2f}, time={processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return ExtractionResult(
                extractor=ExtractorType.TROCR_HTR,
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _estimate_confidence(self, text: str, image: Image.Image) -> float:
        """Estimate HTR confidence.

        TrOCR doesn't provide token-level confidence, so we use heuristics
        based on text quality and image characteristics.

        Args:
            text: Extracted text.
            image: Input image.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not text or len(text) < 5:
            return 0.3  # Very short results are suspicious

        # Heuristics:
        # - Reasonable text length relative to image size
        # - Presence of common words
        # - Low ratio of special characters

        img_area = image.size[0] * image.size[1]
        chars_per_pixel = len(text) / img_area if img_area > 0 else 0

        # Base confidence
        confidence = 0.6

        # Adjust based on text/image ratio
        # Typical handwritten text: 0.0001 - 0.001 chars/pixel
        if 0.00005 < chars_per_pixel < 0.002:
            confidence += 0.2

        # Check for reasonable word structure
        words = text.split()
        if words and 3 <= sum(len(w) for w in words) / len(words) <= 10:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))
