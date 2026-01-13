"""Classification module for document type, content type, and language detection.

This module provides a unified interface for classifying documents using multiple
classification strategies (vision LLM, heuristic rules) and selecting the best result.

Public API:
    - ClassificationOrchestrator: Main entry point for classification
    - ClassificationResult: Individual classifier result
    - ClassificationJob: Complete classification job with all results
    - ClassifierType: Enum of classifier types
    - DocType: Enum of document types
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import Config
from ..extraction.models import ExtractionResult
from ..llm import BaseLLMProvider, get_provider_for_task
from .heuristic_classifier import HeuristicClassifier
from .models import ClassificationJob, ClassificationResult, ClassifierType
from .vision_classifier import VisionLLMClassifier

# Re-export public API
from .models import ClassificationResult, ClassifierType, ClassificationJob, DocType
from ..extraction.models import ContentType

__all__ = [
    "ClassificationOrchestrator",
    "ClassificationResult",
    "ClassificationJob",
    "ClassifierType",
    "DocType",
    "ContentType",
]

logger = logging.getLogger(__name__)


class ClassificationOrchestrator:
    """Orchestrates document classification using multiple classifiers.

    The orchestrator:
    1. Initializes classifiers based on configuration
    2. Runs classifiers (parallel or sequential)
    3. Selects the primary result based on confidence and classifier type
    4. Flags low-confidence results for manual review
    5. Returns a complete ClassificationJob

    Attributes:
        config: Configuration object.
        llm: LLM provider instance.
        vision_classifier: Vision LLM classifier (if enabled).
        heuristic_classifier: Heuristic classifier (if enabled).
    """

    def __init__(self, config: Config, llm: BaseLLMProvider) -> None:
        """Initialize classification orchestrator.

        Args:
            config: Configuration object with classification settings.
            llm: LLM provider instance (default provider) for vision classification.
        """
        self.config = config
        self.llm = llm

        # Get task-specific LLM provider for classification if configured
        classification_llm = get_provider_for_task("classification", config.llm, llm)

        # Log if task-specific model is being used
        if classification_llm is not llm:
            logger.info("Using task-specific model for classification")

        # Initialize classifiers based on config
        self.vision_classifier = None
        self.heuristic_classifier = None

        if config.classification.enable_vision_classifier:
            self.vision_classifier = VisionLLMClassifier(classification_llm)
            logger.info("Vision LLM classifier enabled")

        if config.classification.enable_heuristic_classifier:
            self.heuristic_classifier = HeuristicClassifier()
            logger.info("Heuristic classifier enabled")

        if not self.vision_classifier and not self.heuristic_classifier:
            logger.warning("No classifiers enabled! Classification will always fail.")

    def classify(
        self,
        image_path: Path | str,
        extracted_text: str | None = None,
        extraction_results: list[ExtractionResult] | None = None,
    ) -> ClassificationJob:
        """Classify a document using available classifiers.

        Args:
            image_path: Path to the document image.
            extracted_text: Optional extracted text from Phase 2.
            extraction_results: Optional extraction results from Phase 2 (hints).

        Returns:
            ClassificationJob with all results and selected primary result.
        """
        start_time = time.time()
        image_path = Path(image_path)

        logger.info(f"Starting classification for {image_path.name}")

        # Create job
        job = ClassificationJob(
            image_path=image_path,
            extracted_text=extracted_text,
            config=self.config.classification.model_dump(),
        )

        # Run classifiers (passing image_path, not PIL Image)
        if self.config.classification.run_parallel:
            results = self._classify_parallel(image_path, extracted_text, extraction_results)
        else:
            results = self._classify_sequential(image_path, extracted_text, extraction_results)

        job.results = results

        # Select primary result
        if results:
            job.primary_result = self._select_primary_result(results)
            if job.primary_result and job.primary_result.success:
                job.success = True

                # Check confidence threshold
                if (
                    self.config.classification.flag_for_review_below_threshold
                    and job.primary_result.confidence
                    < self.config.classification.confidence_threshold
                ):
                    job.flagged_for_review = True
                    logger.info(
                        f"Classification flagged for review: confidence "
                        f"{job.primary_result.confidence:.2f} < "
                        f"{self.config.classification.confidence_threshold:.2f}"
                    )
        else:
            job.errors.append("No classifiers produced results")

        job.total_processing_time = time.time() - start_time

        logger.info(
            f"Classification complete: success={job.success}, "
            f"flagged={job.flagged_for_review}, time={job.total_processing_time:.2f}s"
        )

        return job

    def _classify_sequential(
        self,
        image_path: Path,
        extracted_text: str | None,
        extraction_results: list[ExtractionResult] | None,
    ) -> list[ClassificationResult]:
        """Run classifiers sequentially (vision first, heuristic if vision fails).

        Args:
            image_path: Path to image file.
            extracted_text: Optional extracted text.
            extraction_results: Optional extraction results.

        Returns:
            List of classification results.
        """
        results = []

        # Try vision classifier first (most accurate)
        if self.vision_classifier:
            try:
                logger.debug("Running vision classifier")
                result = self.vision_classifier.classify(
                    image_path=image_path, text=extracted_text
                )
                results.append(result)

                # If vision classifier succeeded with good confidence, we're done
                if result.success and result.confidence >= 0.7:
                    logger.debug(
                        f"Vision classifier succeeded with confidence {result.confidence:.2f}"
                    )
                    return results

            except Exception as e:
                logger.error(f"Vision classifier failed: {e}", exc_info=True)

        # Fall back to heuristic classifier
        if self.heuristic_classifier:
            try:
                logger.debug("Running heuristic classifier")
                result = self.heuristic_classifier.classify(
                    image_path=image_path,
                    text=extracted_text,
                    extraction_results=extraction_results,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Heuristic classifier failed: {e}", exc_info=True)

        return results

    def _classify_parallel(
        self,
        image_path: Path,
        extracted_text: str | None,
        extraction_results: list[ExtractionResult] | None,
    ) -> list[ClassificationResult]:
        """Run classifiers in parallel using ThreadPoolExecutor.

        Args:
            image_path: Path to image file.
            extracted_text: Optional extracted text.
            extraction_results: Optional extraction results.

        Returns:
            List of classification results.
        """
        results = []

        # Build list of classifier functions to run
        tasks = []

        if self.vision_classifier:
            tasks.append(
                (
                    "vision",
                    lambda: self.vision_classifier.classify(
                        image_path=image_path, text=extracted_text
                    ),
                )
            )

        if self.heuristic_classifier:
            tasks.append(
                (
                    "heuristic",
                    lambda: self.heuristic_classifier.classify(
                        image_path=image_path,
                        text=extracted_text,
                        extraction_results=extraction_results,
                    ),
                )
            )

        # Run in parallel
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {executor.submit(task_fn): name for name, task_fn in tasks}

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"{name} classifier completed")
                except Exception as e:
                    logger.error(f"{name} classifier failed: {e}", exc_info=True)

        return results

    def _select_primary_result(
        self, results: list[ClassificationResult]
    ) -> ClassificationResult | None:
        """Select the primary result from multiple classifier results.

        Selection algorithm:
        - Only consider successful results
        - Score = base confidence + bonus for vision LLM (+0.1)
        - Select result with highest score

        Args:
            results: List of classification results.

        Returns:
            Best classification result, or None if no successful results.
        """
        successful = [r for r in results if r.success]

        if not successful:
            logger.warning("No successful classification results")
            return None

        def score(result: ClassificationResult) -> float:
            """Calculate weighted score for result."""
            base = result.confidence

            # Bonus for vision LLM (more accurate)
            if result.classifier == ClassifierType.VISION_LLM:
                base += 0.1

            return base

        best = max(successful, key=score)
        logger.info(
            f"Selected primary result: {best.classifier.value} "
            f"(confidence={best.confidence:.2f}, score={score(best):.2f})"
        )

        return best
