"""Extraction module for Ingrid document processing pipeline.

Provides unified interface for document text extraction using multiple strategies:
- Docling OCR (typed documents)
- TrOCR HTR (handwritten documents)
- Vision LLM (fallback/validation)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ..config import Config
from ..llm import BaseLLMProvider
from .htr import TrOCRExtractor
from .models import ExtractionJob, ExtractionResult, PreprocessedImage
from .ocr import DoclingOCRExtractor
from .preprocessing import ImagePreprocessor
from .vision_extract import VisionLLMExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractionOrchestrator",
    "ExtractionJob",
    "ExtractionResult",
    "PreprocessedImage",
]


class ExtractionOrchestrator:
    """Orchestrates multiple extraction strategies for a document.

    The orchestrator initializes extractors based on configuration, preprocesses
    images, runs extractions (in parallel or sequentially), and selects the best
    result based on confidence and text quality metrics.

    Attributes:
        config: Application configuration.
        llm_provider: Optional LLM provider for vision extraction.
        preprocessor: Image preprocessor instance.
        extractors: Dictionary of enabled extractors.
    """

    def __init__(
        self,
        config: Config,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        """Initialize extraction orchestrator.

        Args:
            config: Application configuration.
            llm_provider: Optional LLM provider for vision extraction.
        """
        self.config = config
        self.llm_provider = llm_provider

        # Initialize extractors based on config
        self.preprocessor = ImagePreprocessor(
            deskew=config.ocr.preprocessing.deskew,
            enhance_contrast=config.ocr.preprocessing.enhance_contrast,
        )

        self.extractors: dict[str, DoclingOCRExtractor | TrOCRExtractor | VisionLLMExtractor] = {}

        # OCR extractor
        if config.ocr.enable_docling:
            self.extractors["docling"] = DoclingOCRExtractor(
                languages=config.ocr.languages,
            )
            logger.info("Enabled Docling OCR extractor")

        # HTR extractor
        if config.ocr.enable_trocr:
            self.extractors["trocr"] = TrOCRExtractor(
                model_name=config.ocr.htr_model,
            )
            logger.info("Enabled TrOCR HTR extractor")

        # Vision LLM extractor
        if config.ocr.enable_vision_llm and llm_provider:
            self.extractors["vision_llm"] = VisionLLMExtractor(llm_provider)
            logger.info("Enabled VisionLLM extractor")

        if not self.extractors:
            raise ValueError("No extractors enabled in configuration")

        logger.info(f"Initialized ExtractionOrchestrator with {len(self.extractors)} extractors")

    def extract(
        self,
        image_path: Path | str,
        run_parallel: bool | None = None,
    ) -> ExtractionJob:
        """Extract text from document image using all enabled extractors.

        Args:
            image_path: Path to document image.
            run_parallel: Run extractors in parallel (faster, more resource-intensive).
                         If None, uses config default.

        Returns:
            ExtractionJob with all results and selected primary result.
        """
        start_time = time.time()
        image_path = Path(image_path)

        # Use config default if not specified
        if run_parallel is None:
            run_parallel = self.config.ocr.run_extractors_parallel

        logger.info(f"Starting extraction for: {image_path.name}")

        job = ExtractionJob(
            image_path=image_path,
            config=self.config.model_dump(),
        )

        try:
            # Step 1: Preprocess image
            preprocessed = self.preprocessor.process(image_path)
            job.preprocessed = preprocessed
            logger.info(
                f"Preprocessing complete: deskewed={preprocessed.deskewed}, "
                f"contrast_enhanced={preprocessed.contrast_enhanced}"
            )

            # Step 2: Run extractors
            if run_parallel:
                results = self._extract_parallel(preprocessed)
            else:
                results = self._extract_sequential(preprocessed)

            job.results = results

            # Step 3: Select primary result
            primary = self._select_primary_result(results)
            job.primary_result = primary

            if primary:
                job.success = True
                logger.info(
                    f"Extraction complete: primary extractor={primary.extractor.value}, "
                    f"confidence={primary.confidence:.2f}, chars={primary.character_count}"
                )
            else:
                job.success = False
                job.errors.append("All extractors failed")
                logger.warning("Extraction failed: no successful results")

        except Exception as e:
            job.success = False
            job.errors.append(str(e))
            logger.error(f"Extraction failed for {image_path.name}: {e}")

        job.total_processing_time = time.time() - start_time
        logger.info(f"Total extraction time: {job.total_processing_time:.2f}s")

        return job

    def _extract_parallel(self, preprocessed: PreprocessedImage) -> list[ExtractionResult]:
        """Run extractors in parallel using ThreadPoolExecutor.

        Args:
            preprocessed: Preprocessed image.

        Returns:
            List of extraction results.
        """
        results = []

        with ThreadPoolExecutor(max_workers=len(self.extractors)) as executor:
            futures = {
                executor.submit(
                    extractor.extract,
                    image=preprocessed.image,
                    image_path=preprocessed.original_path,
                ): name
                for name, extractor in self.extractors.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Extractor '{name}' completed")
                except Exception as e:
                    logger.error(f"Extractor '{name}' failed: {e}")

        return results

    def _extract_sequential(self, preprocessed: PreprocessedImage) -> list[ExtractionResult]:
        """Run extractors sequentially.

        Args:
            preprocessed: Preprocessed image.

        Returns:
            List of extraction results.
        """
        results = []

        for name, extractor in self.extractors.items():
            try:
                result = extractor.extract(
                    image=preprocessed.image,
                    image_path=preprocessed.original_path,
                )
                results.append(result)
                logger.info(f"Extractor '{name}' completed")
            except Exception as e:
                logger.error(f"Extractor '{name}' failed: {e}")

        return results

    def _select_primary_result(
        self, results: list[ExtractionResult]
    ) -> ExtractionResult | None:
        """Select the best extraction result based on confidence and text length.

        Scoring algorithm: 70% confidence + 30% normalized text length

        Args:
            results: List of extraction results.

        Returns:
            Best result, or None if all failed.
        """
        # Filter successful results
        successful = [r for r in results if r.success and r.text]

        if not successful:
            return None

        # Score results: 70% confidence, 30% text length
        def score(result: ExtractionResult) -> float:
            normalized_length = min(result.character_count / 1000, 1.0)
            return 0.7 * result.confidence + 0.3 * normalized_length

        # Return highest scoring result
        best = max(successful, key=score)
        logger.info(
            f"Selected primary result: {best.extractor.value} (score={score(best):.2f})"
        )
        return best
