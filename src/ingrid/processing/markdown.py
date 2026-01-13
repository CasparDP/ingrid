"""Markdown file generator with YAML frontmatter.

This module generates markdown output files from processing results, following
the specification in CLAUDE.md with structured YAML frontmatter.
"""

import logging
from pathlib import Path

from ..classification.models import DocType
from .models import ProcessingResult

logger = logging.getLogger(__name__)


def generate_markdown(result: ProcessingResult) -> str:
    """Generate markdown content with YAML frontmatter.

    Args:
        result: Processing result with all document data.

    Returns:
        Complete markdown content as string.
    """
    # Extract data
    extraction_job = result.extraction_job
    classification_job = result.classification_job
    primary_extraction = extraction_job.primary_result
    primary_classification = classification_job.primary_result
    metadata = result.metadata

    # Build YAML frontmatter
    frontmatter_lines = ["---"]

    # Generate unique ID (use filename without extension)
    doc_id = result.image_path.stem
    frontmatter_lines.append(f"id: {doc_id}")
    frontmatter_lines.append(f"filename: {result.image_path.name}")

    if primary_classification:
        frontmatter_lines.append(f"doc_type: {primary_classification.doc_type.value}")
        frontmatter_lines.append(
            f"content_type: {primary_classification.content_type.value}"
        )
        if primary_classification.languages:
            langs_str = "[" + ", ".join(primary_classification.languages) + "]"
            frontmatter_lines.append(f"languages: {langs_str}")

    if metadata:
        if metadata.date:
            frontmatter_lines.append(f"date: {metadata.date}")
        if metadata.sender:
            frontmatter_lines.append(f"sender: {metadata.sender}")
        if metadata.recipient:
            frontmatter_lines.append(f"recipient: {metadata.recipient}")
        if metadata.location:
            frontmatter_lines.append(f"location: {metadata.location}")
        if metadata.topics:
            topics_str = "[" + ", ".join(metadata.topics) + "]"
            frontmatter_lines.append(f"topics: {topics_str}")
        if metadata.people_mentioned:
            people_str = "[" + ", ".join(metadata.people_mentioned) + "]"
            frontmatter_lines.append(f"people_mentioned: {people_str}")

    # Add summaries if available
    if result.summary:
        # Escape quotes in summary for YAML
        summary_escaped = result.summary.replace('"', '\\"')
        frontmatter_lines.append(f'summary: "{summary_escaped}"')
    if result.summary_language:
        frontmatter_lines.append(f"summary_language: {result.summary_language}")
    if result.summary_english:
        # Escape quotes in summary for YAML
        summary_en_escaped = result.summary_english.replace('"', '\\"')
        frontmatter_lines.append(f'summary_english: "{summary_en_escaped}"')

    # Add confidence scores
    if primary_classification and metadata:
        frontmatter_lines.append("confidence:")
        frontmatter_lines.append(
            f"  doc_type: {primary_classification.confidence_scores.get('doc_type', 0.0):.2f}"
        )
        frontmatter_lines.append(
            f"  content_type: {primary_classification.confidence_scores.get('content_type', 0.0):.2f}"
        )
        if metadata.date:
            frontmatter_lines.append(f"  date: {metadata.date_confidence:.2f}")
        if metadata.sender:
            frontmatter_lines.append(f"  sender: {metadata.sender_confidence:.2f}")
        if metadata.recipient:
            frontmatter_lines.append(f"  recipient: {metadata.recipient_confidence:.2f}")
        if metadata.location:
            frontmatter_lines.append(f"  location: {metadata.location_confidence:.2f}")

    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    # Build document header
    if metadata and metadata.sender and metadata.recipient:
        title = f"# Letter from {metadata.sender} to {metadata.recipient}"
    elif primary_classification and primary_classification.doc_type == DocType.LETTER:
        title = "# Letter"
    elif (
        primary_classification
        and primary_classification.doc_type == DocType.NEWSPAPER_ARTICLE
    ):
        title = "# Newspaper Article"
    else:
        title = "# Document"

    # Build metadata section
    meta_lines = []
    if metadata:
        if metadata.date:
            meta_lines.append(f"**Date:** {metadata.date}")
        if metadata.location:
            meta_lines.append(f"**Location:** {metadata.location}")

    meta_section = "\n".join(meta_lines) if meta_lines else ""

    # Build summary section
    summary_section = ""
    if result.summary:
        # Determine language label for summary
        lang_label = result.summary_language.upper() if result.summary_language else ""

        # If summary language is English, only show one section
        if result.summary_language == "en":
            summary_section = f"""## Summary

{result.summary}"""
        # If we have both original and English, show both
        elif result.summary_english:
            summary_section = f"""## Summary ({lang_label})

{result.summary}

## Summary (English)

{result.summary_english}"""
        # Only original summary available
        else:
            summary_section = f"""## Summary ({lang_label})

{result.summary}"""

    # Build transcription section
    transcription_section = f"""## Transcription

{result.cleaned_text}"""

    # Build raw OCR section (collapsible)
    raw_ocr_section = ""
    if primary_extraction and primary_extraction.raw_text:
        raw_ocr_section = f"""## Original OCR Output

<details>
<summary>Raw OCR text (click to expand)</summary>

{primary_extraction.raw_text}

</details>"""

    # Combine all sections
    sections = [
        frontmatter,
        "",
        title,
        "",
        meta_section,
        "",
        summary_section,
        "",
        transcription_section,
        "",
        raw_ocr_section,
    ]

    # Filter out empty sections and join
    markdown_content = "\n".join(section for section in sections if section.strip())

    return markdown_content


class MarkdownGenerator:
    """Markdown file generator with YAML frontmatter.

    This class generates markdown output files from processing results, following
    the specification in CLAUDE.md.

    Attributes:
        output_path: Base output directory for markdown files.
    """

    def __init__(self, output_path: Path) -> None:
        """Initialize markdown generator.

        Args:
            output_path: Output directory for markdown files.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized MarkdownGenerator: output_path={output_path}")

    def generate(self, result: ProcessingResult) -> tuple[Path | None, str | None]:
        """Generate and write markdown file.

        Args:
            result: Processing result with all data.

        Returns:
            Tuple of (markdown_path, error_message).
        """
        try:
            # Generate markdown content
            markdown_content = generate_markdown(result)

            # Determine output filename (same as input, but .md extension)
            output_filename = result.image_path.stem + ".md"
            output_path = self.output_path / output_filename

            # Write to file
            output_path.write_text(markdown_content, encoding="utf-8")

            logger.info(f"Markdown file written: {output_path}")

            return (output_path, None)

        except Exception as e:
            logger.error(f"Error generating markdown: {e}", exc_info=True)
            return (None, f"Markdown generation error: {e}")
