"""Document processing module for text cleanup, metadata extraction, and summarization.

This module orchestrates LLM-based processing steps to clean OCR text, extract
structured metadata, generate summaries, and create markdown output files.
"""

import logging
import time

from ..config import Config
from ..extraction.models import ExtractionJob
from ..classification.models import ClassificationJob
from ..llm.base import BaseLLMProvider

from .cleanup import TextCleanupProcessor
from .markdown import MarkdownGenerator
from .metadata import MetadataExtractor
from .models import DocumentMetadata, ProcessingResult, ProcessorType
from .summarizer import DocumentSummarizer
from .translator import SummaryTranslator

# Public exports
__all__ = [
    "ProcessingOrchestrator",
    "ProcessingResult",
    "DocumentMetadata",
    "ProcessorType",
]

logger = logging.getLogger(__name__)


class ProcessingOrchestrator:
    """Orchestrates document processing steps: cleanup, metadata, summary, translation, markdown.

    The orchestrator:
    1. Validates that extraction and classification succeeded
    2. Runs text cleanup (required)
    3. Runs metadata extraction (if enabled)
    4. Runs summarization (if enabled)
    5. Runs English translation (if summary succeeded and not English)
    6. Generates markdown output (if enabled)

    All processing is sequential (cleanup must complete before metadata/summary).

    Attributes:
        config: Configuration object.
        llm: LLM provider instance.
        text_cleanup: Text cleanup processor.
        metadata_extractor: Metadata extractor.
        summarizer: Document summarizer.
        translator: Summary translator.
        markdown_generator: Markdown generator.
    """

    def __init__(self, config: Config, llm: BaseLLMProvider) -> None:
        """Initialize processing orchestrator.

        Args:
            config: Configuration object with processing settings.
            llm: LLM provider instance.
        """
        self.config = config
        self.llm = llm

        # Initialize processors
        self.text_cleanup = TextCleanupProcessor(
            llm=llm, temperature=0.3, max_retries=config.processing.max_retries
        )

        self.metadata_extractor = (
            MetadataExtractor(
                llm=llm, temperature=0.2, max_retries=config.processing.max_retries
            )
            if config.processing.extract_metadata
            else None
        )

        self.summarizer = (
            DocumentSummarizer(
                llm=llm, temperature=0.5, max_retries=config.processing.max_retries
            )
            if config.processing.generate_summaries
            else None
        )

        self.translator = SummaryTranslator(llm=llm)

        self.markdown_generator = MarkdownGenerator(output_path=config.storage.output_path)

        logger.info(
            f"Initialized ProcessingOrchestrator: "
            f"cleanup=enabled, metadata={config.processing.extract_metadata}, "
            f"summaries={config.processing.generate_summaries}, translation=enabled"
        )

    def process(
        self,
        extraction_job: ExtractionJob,
        classification_job: ClassificationJob,
        write_markdown: bool = True,
    ) -> ProcessingResult:
        """Process document through all enabled processing steps.

        Args:
            extraction_job: Extraction job from Phase 2.
            classification_job: Classification job from Phase 3.
            write_markdown: Whether to write markdown file (default: True).

        Returns:
            ProcessingResult with all processing outputs and status.
        """
        start_time = time.time()

        # Validate inputs
        if not extraction_job.success or not extraction_job.primary_result:
            logger.error("Cannot process: extraction failed")
            return ProcessingResult(
                image_path=extraction_job.image_path,
                extraction_job=extraction_job,
                classification_job=classification_job,
                success=False,
                cleanup_error="Extraction failed",
                total_processing_time=0.0,
                config=self.config.processing.model_dump(),
            )

        if not classification_job.success or not classification_job.primary_result:
            logger.warning("Classification failed, processing will continue with defaults")

        # Get data from jobs
        raw_text = extraction_job.primary_result.text
        doc_type = (
            classification_job.primary_result.doc_type.value
            if classification_job.primary_result
            else "other"
        )
        content_type = (
            classification_job.primary_result.content_type.value
            if classification_job.primary_result
            else "unknown"
        )
        languages = (
            classification_job.primary_result.languages
            if classification_job.primary_result
            else ["en"]
        )

        # Initialize result
        result = ProcessingResult(
            image_path=extraction_job.image_path,
            extraction_job=extraction_job,
            classification_job=classification_job,
            config=self.config.processing.model_dump(),
        )

        logger.info(f"Starting processing for {extraction_job.image_path.name}")

        # Step 1: Text Cleanup (REQUIRED)
        logger.info("Step 1: Text cleanup")
        cleaned_text, cleanup_time, cleanup_error = self.text_cleanup.cleanup(
            raw_text=raw_text,
            doc_type=doc_type,
            content_type=content_type,
            languages=languages,
        )

        result.cleaned_text = cleaned_text
        result.cleanup_time = cleanup_time
        result.cleanup_error = cleanup_error
        result.cleanup_success = cleanup_error is None

        if not result.cleanup_success:
            logger.error(f"Text cleanup failed: {cleanup_error}")
            result.success = False
            result.total_processing_time = time.time() - start_time
            return result

        # Step 2: Metadata Extraction (if enabled)
        if self.metadata_extractor:
            logger.info("Step 2: Metadata extraction")
            metadata = self.metadata_extractor.extract(
                cleaned_text=cleaned_text,
                doc_type=doc_type,
                languages=languages,
            )

            result.metadata = metadata
            result.metadata_time = metadata.processing_time
            result.metadata_error = metadata.error
            result.metadata_success = metadata.success

            if not result.metadata_success:
                logger.warning(f"Metadata extraction failed: {metadata.error}")
        else:
            logger.info("Step 2: Metadata extraction (skipped - disabled)")

        # Step 3: Summarization (if enabled)
        if self.summarizer:
            logger.info("Step 3: Summarization")
            summary, summary_time, summary_error = self.summarizer.summarize(
                cleaned_text=cleaned_text,
                doc_type=doc_type,
                languages=languages,
            )

            result.summary = summary
            result.summary_time = summary_time
            result.summary_error = summary_error
            result.summary_success = summary_error is None

            # Determine summary language (use primary detected language)
            result.summary_language = languages[0] if languages else "en"

            if not result.summary_success:
                logger.warning(f"Summarization failed: {summary_error}")
        else:
            logger.info("Step 3: Summarization (skipped - disabled)")

        # Step 4: English Translation (if summary succeeded and not already English)
        if result.summary_success and result.summary_language != "en":
            logger.info(
                f"Step 4: Translation to English (from {result.summary_language})"
            )
            translation, translation_time, translation_error = (
                self.translator.translate_to_english(
                    result.summary,
                    result.summary_language,
                    max_retries=self.config.processing.max_retries,
                )
            )

            result.summary_english = translation
            result.translation_time = translation_time
            result.translation_error = translation_error
            result.translation_success = translation_error is None

            if not result.translation_success:
                logger.warning(f"Translation failed: {translation_error}")
        elif result.summary_success and result.summary_language == "en":
            # Summary already in English, no translation needed
            logger.info("Step 4: Translation (skipped - summary already in English)")
            result.summary_english = result.summary
            result.translation_success = True
            result.translation_time = 0.0
        else:
            # No summary or summary failed
            logger.info("Step 4: Translation (skipped - no summary)")

        # Step 5: Markdown Generation (if requested)
        if write_markdown:
            logger.info("Step 5: Markdown generation")
            markdown_path, markdown_error = self.markdown_generator.generate(result)

            result.markdown_path = markdown_path
            result.markdown_error = markdown_error
            result.markdown_success = markdown_error is None

            if not result.markdown_success:
                logger.warning(f"Markdown generation failed: {markdown_error}")
        else:
            logger.info("Step 5: Markdown generation (skipped)")

        # Final status
        result.success = (
            result.cleanup_success
            and (not self.metadata_extractor or result.metadata_success)
            and (not self.summarizer or result.summary_success)
            and (
                not result.summary_success
                or result.summary_language == "en"
                or result.translation_success
            )  # Translation required if summary exists and not English
            and (not write_markdown or result.markdown_success)
        )

        result.total_processing_time = time.time() - start_time

        logger.info(
            f"Processing complete: success={result.success}, "
            f"time={result.total_processing_time:.2f}s"
        )

        return result
