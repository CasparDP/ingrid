"""CLI interface for Ingrid document processing pipeline.

Provides command-line interface using Typer with Rich formatting for
processing documents, testing LLM connectivity, and validating configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .classification import ClassificationOrchestrator
from .config import Config, load_config
from .extraction import ExtractionOrchestrator
from .llm import BaseLLMProvider, LLMError, get_provider
from .processing import ProcessingOrchestrator
from .storage import StorageOrchestrator, DatabaseManager

# Initialize Typer app
app = typer.Typer(
    name="ingrid",
    help="Document extraction pipeline for historical letters and newspapers",
    add_completion=False,
)

# Rich console for pretty output
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with Rich handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def load_and_validate_config(config_path: Path) -> Config:
    """Load configuration and display validation results.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Validated configuration object.

    Raises:
        typer.Exit: If configuration loading fails.
    """
    try:
        with console.status("[bold blue]Loading configuration..."):
            config = load_config(config_path)

        console.print("[green]✓[/green] Configuration loaded successfully")
        return config

    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] Configuration file not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        raise typer.Exit(1)


@app.command()
def process(
    file_path: str = typer.Option(
        "",
        "--file",
        "-f",
        help="Path to a single file to process",
    ),
    batch: bool = typer.Option(
        False, "--batch", "-b", help="Process all files in the scans directory"
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Override LLM provider from config (ollama, anthropic, google)",
    ),
    config_path: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed extraction results"
    ),
    doc_type: Optional[str] = typer.Option(
        None, "--doc-type", help="Override document type (letter, newspaper_article, other)"
    ),
    content_type: Optional[str] = typer.Option(
        None, "--content-type", help="Override content type (handwritten, typed, mixed)"
    ),
    skip_cleanup: bool = typer.Option(
        False, "--skip-cleanup", help="Skip text cleanup step"
    ),
    skip_metadata: bool = typer.Option(
        False, "--skip-metadata", help="Skip metadata extraction"
    ),
    skip_summary: bool = typer.Option(
        False, "--skip-summary", help="Skip summarization"
    ),
    write_markdown: bool = typer.Option(
        True, "--write-markdown/--no-markdown", help="Generate markdown files"
    ),
    skip_storage: bool = typer.Option(
        False, "--skip-storage", help="Skip database storage and embedding generation"
    ),
) -> None:
    """Process document(s) through the extraction pipeline.

    Examples:
        ingrid process --file scans/letter.jpg
        ingrid process --batch
        ingrid process --batch --provider anthropic
    """
    # Validate arguments
    if not batch and not file_path:
        console.print("[red]✗[/red] Either provide --file or use --batch flag")
        raise typer.Exit(1)

    if batch and file_path:
        console.print("[yellow]![/yellow] Ignoring --file when --batch is used")

    # Load configuration
    config = load_and_validate_config(Path(config_path))
    setup_logging(config.logging.level)

    # Override provider if specified
    if provider:
        valid_providers = ["ollama", "ollama_cloud", "anthropic", "google", "huggingface"]
        if provider not in valid_providers:
            console.print(f"[red]✗[/red] Invalid provider: {provider}")
            raise typer.Exit(1)
        config.llm.provider = provider  # type: ignore
        console.print(f"[yellow]![/yellow] Using provider override: {provider}")

    # Display configuration summary
    console.print(
        Panel.fit(
            f"[bold]Configuration Summary[/bold]\n"
            f"Provider: {config.llm.provider}\n"
            f"Input path: {config.storage.input_path}\n"
            f"Output path: {config.storage.output_path}",
            border_style="blue",
        )
    )

    # Initialize LLM provider
    try:
        with console.status("[bold blue]Initializing LLM provider..."):
            llm_config = config.get_active_llm_config()
            llm = get_provider(config.llm.provider, llm_config)

        console.print(f"[green]✓[/green] LLM provider initialized: {config.llm.provider}")

        # Health check
        with console.status("[bold blue]Running health check..."):
            healthy = llm.health_check()

        if healthy:
            console.print("[green]✓[/green] LLM provider is healthy")
        else:
            console.print("[red]✗[/red] LLM provider health check failed")
            console.print(
                "[yellow]![/yellow] Check that Ollama is running and models are available"
            )
            raise typer.Exit(1)

    except LLMError as e:
        console.print(f"[red]✗[/red] LLM provider error: {e}")
        raise typer.Exit(1)

    # Process files
    if batch:
        process_batch(
            config, llm, verbose, doc_type, content_type,
            skip_cleanup, skip_metadata, skip_summary, write_markdown, skip_storage
        )
    else:
        if file_path:
            process_single(
                Path(file_path), config, llm, verbose, doc_type, content_type,
                skip_cleanup, skip_metadata, skip_summary, write_markdown, skip_storage
            )


def process_single(
    file_path: Path,
    config: Config,
    llm: BaseLLMProvider,
    verbose: bool,
    doc_type_override: str | None = None,
    content_type_override: str | None = None,
    skip_cleanup: bool = False,
    skip_metadata: bool = False,
    skip_summary: bool = False,
    write_markdown: bool = True,
    skip_storage: bool = False,
) -> None:
    """Process a single file through extraction, classification, and processing pipeline.

    Args:
        file_path: Path to the file to process.
        config: Configuration object.
        llm: LLM provider instance.
        verbose: Show detailed extraction results.
        doc_type_override: Manual override for document type.
        content_type_override: Manual override for content type.
        skip_cleanup: Skip text cleanup step.
        skip_metadata: Skip metadata extraction.
        skip_summary: Skip summarization.
        write_markdown: Generate markdown files.
    """
    console.print(f"\n[bold]Processing:[/bold] {file_path.name}")

    try:
        # Initialize orchestrator
        orchestrator = ExtractionOrchestrator(config, llm)

        # Run extraction
        with console.status("[bold blue]Extracting text..."):
            job = orchestrator.extract(file_path)

        # Display extraction results
        if job.success and job.primary_result:
            result = job.primary_result

            console.print(f"[green]✓[/green] Extraction successful")
            console.print(f"  Extractor: {result.extractor.value}")
            console.print(f"  Confidence: {result.confidence:.2%}")
            console.print(f"  Text length: {result.character_count} chars")
            console.print(f"  Processing time: {job.total_processing_time:.2f}s")

            if verbose:
                console.print("\n[bold]Extracted Text Preview:[/bold]")
                preview = result.text[:500] + "..." if len(result.text) > 500 else result.text
                console.print(Panel(preview, border_style="blue"))

                # Show all extractor results
                if len(job.results) > 1:
                    console.print("\n[bold]All Extractor Results:[/bold]")
                    table = Table()
                    table.add_column("Extractor", style="cyan")
                    table.add_column("Success", style="green")
                    table.add_column("Confidence", style="magenta")
                    table.add_column("Length", style="yellow")
                    table.add_column("Time", style="blue")

                    for r in job.results:
                        table.add_row(
                            r.extractor.value,
                            "✓" if r.success else "✗",
                            f"{r.confidence:.2%}" if r.success else "-",
                            str(r.character_count) if r.success else "-",
                            f"{r.processing_time:.2f}s",
                        )
                    console.print(table)

            # Phase 3: Classification
            if config.classification.auto_detect:
                console.print("\n[bold]Classification:[/bold]")
                classification_orch = ClassificationOrchestrator(config, llm)

                with console.status("[bold blue]Classifying document..."):
                    classification_job = classification_orch.classify(
                        image_path=file_path,
                        extracted_text=result.text,
                        extraction_results=job.results,
                    )

                # Apply manual overrides if provided
                if doc_type_override or content_type_override:
                    classification_job.manual_override = {}
                    if doc_type_override:
                        classification_job.manual_override["doc_type"] = doc_type_override
                    if content_type_override:
                        classification_job.manual_override["content_type"] = content_type_override
                    console.print("[yellow]![/yellow] Using manual overrides")

                # Display classification results
                if classification_job.success and classification_job.primary_result:
                    cls = classification_job.primary_result

                    console.print(f"[green]✓[/green] Classification successful")
                    console.print(f"  Document type: {cls.doc_type.value}")
                    console.print(f"  Content type: {cls.content_type.value}")
                    console.print(
                        f"  Languages: {', '.join(cls.languages) if cls.languages else 'unknown'}"
                    )
                    console.print(f"  Confidence: {cls.confidence:.2%}")
                    console.print(f"  Classifier: {cls.classifier.value}")
                    console.print(f"  Processing time: {classification_job.total_processing_time:.2f}s")

                    # Flag for review
                    if classification_job.flagged_for_review:
                        console.print(
                            f"[yellow]![/yellow] Low confidence ({cls.confidence:.2%} < "
                            f"{config.classification.confidence_threshold:.2%}) - flagged for review"
                        )

                    # Show reasoning if available
                    if verbose and cls.reasoning:
                        console.print(f"\n[bold]Reasoning:[/bold]")
                        console.print(Panel(cls.reasoning, border_style="blue"))

                    # Show all classifier results if verbose
                    if verbose and len(classification_job.results) > 1:
                        console.print("\n[bold]All Classifier Results:[/bold]")
                        table = Table()
                        table.add_column("Classifier", style="cyan")
                        table.add_column("Doc Type", style="green")
                        table.add_column("Content Type", style="magenta")
                        table.add_column("Confidence", style="yellow")
                        table.add_column("Time", style="blue")

                        for r in classification_job.results:
                            table.add_row(
                                r.classifier.value,
                                r.doc_type.value if r.success else "-",
                                r.content_type.value if r.success else "-",
                                f"{r.confidence:.2%}" if r.success else "-",
                                f"{r.processing_time:.2f}s",
                            )
                        console.print(table)
                else:
                    console.print("[yellow]![/yellow] Classification failed")
                    if classification_job.errors:
                        for error in classification_job.errors:
                            console.print(f"  Error: {error}")

                # Phase 4: Processing (if classification succeeded or was disabled)
                if not skip_cleanup and result.text:
                    console.print("\n[bold]Processing:[/bold]")

                    # Temporarily override config flags based on CLI options
                    original_metadata_flag = config.processing.extract_metadata
                    original_summary_flag = config.processing.generate_summaries

                    if skip_metadata:
                        config.processing.extract_metadata = False
                    if skip_summary:
                        config.processing.generate_summaries = False

                    # Initialize orchestrator
                    processing_orch = ProcessingOrchestrator(config, llm)

                    with console.status("[bold blue]Processing document..."):
                        processing_result = processing_orch.process(
                            extraction_job=job,
                            classification_job=classification_job,
                            write_markdown=write_markdown,
                        )

                    # Restore original config flags
                    config.processing.extract_metadata = original_metadata_flag
                    config.processing.generate_summaries = original_summary_flag

                    # Display processing results
                    if processing_result.success:
                        console.print(f"[green]✓[/green] Processing successful")
                        console.print(f"  Cleaned text: {len(processing_result.cleaned_text)} chars")

                        if processing_result.summary:
                            console.print(f"  Summary: {len(processing_result.summary)} chars")

                        if processing_result.metadata and processing_result.metadata.success:
                            console.print(f"  Metadata extracted:")
                            if processing_result.metadata.date:
                                console.print(f"    - Date: {processing_result.metadata.date}")
                            if processing_result.metadata.sender:
                                console.print(f"    - Sender: {processing_result.metadata.sender}")
                            if processing_result.metadata.recipient:
                                console.print(f"    - Recipient: {processing_result.metadata.recipient}")
                            if processing_result.metadata.topics:
                                console.print(
                                    f"    - Topics: {', '.join(processing_result.metadata.topics)}"
                                )

                        if processing_result.markdown_path:
                            console.print(f"  Markdown: {processing_result.markdown_path.name}")

                        console.print(
                            f"  Processing time: {processing_result.total_processing_time:.2f}s"
                        )

                        # Show summary in verbose mode
                        if verbose and processing_result.summary:
                            console.print("\n[bold]Summary:[/bold]")
                            console.print(Panel(processing_result.summary, border_style="blue"))

                        # Phase 5: Storage (if processing succeeded and not skipped)
                        if not skip_storage and config.processing.generate_embeddings:
                            console.print("\n[bold]Storage & Indexing:[/bold]")

                            try:
                                # Initialize embedding provider
                                embedding_config = config.get_embedding_config()
                                embedding_provider = get_provider(
                                    config.embeddings.provider, embedding_config
                                )

                                # Initialize storage orchestrator
                                storage_orch = StorageOrchestrator(config, embedding_provider)

                                with console.status("[bold blue]Storing and indexing..."):
                                    doc_id, storage_time, storage_success, storage_error = (
                                        storage_orch.store_document(processing_result, file_path)
                                    )

                                if storage_success:
                                    console.print(f"[green]✓[/green] Stored with ID: {doc_id[:8]}...")
                                    console.print(f"  Storage time: {storage_time:.1f}s")
                                    console.print(f"  Database: {config.storage.database_path}")
                                    console.print(f"  Vector store: {config.storage.chroma_path}")
                                else:
                                    console.print(f"[red]✗[/red] Storage failed: {storage_error}")

                            except Exception as e:
                                console.print(f"[red]✗[/red] Storage error: {e}")
                        elif skip_storage:
                            console.print("\n[blue]ℹ[/blue] Storage skipped (--skip-storage)")
                        elif not config.processing.generate_embeddings:
                            console.print("\n[blue]ℹ[/blue] Embeddings disabled in config")

                    else:
                        console.print(f"[yellow]![/yellow] Processing incomplete")
                        if processing_result.cleanup_error:
                            console.print(f"  Cleanup error: {processing_result.cleanup_error}")
                        if processing_result.metadata_error:
                            console.print(f"  Metadata error: {processing_result.metadata_error}")
                        if processing_result.summary_error:
                            console.print(f"  Summary error: {processing_result.summary_error}")
                        if processing_result.translation_error:
                            console.print(f"  Translation error: {processing_result.translation_error}")
            else:
                console.print("\n[blue]ℹ[/blue] Auto-classification disabled")

        else:
            console.print(f"[red]✗[/red] Extraction failed")
            if job.errors:
                for error in job.errors:
                    console.print(f"  Error: {error}")

    except Exception as e:
        console.print(f"[red]✗[/red] Processing failed: {e}")
        raise typer.Exit(1)


def process_batch(
    config: Config,
    llm: BaseLLMProvider,
    verbose: bool,
    doc_type_override: str | None = None,
    content_type_override: str | None = None,
    skip_cleanup: bool = False,
    skip_metadata: bool = False,
    skip_summary: bool = False,
    write_markdown: bool = True,
    skip_storage: bool = False,
) -> None:
    """Process all files in scans directory.

    Args:
        config: Configuration object.
        llm: LLM provider instance.
        verbose: Show detailed results for each file.
        doc_type_override: Manual override for document type.
        content_type_override: Manual override for content type.
        skip_cleanup: Skip text cleanup step.
        skip_metadata: Skip metadata extraction.
        skip_summary: Skip summarization.
        write_markdown: Generate markdown files.
    """
    scan_dir = Path(config.storage.input_path)

    if not scan_dir.exists():
        console.print(f"[red]✗[/red] Scan directory not found: {scan_dir}")
        raise typer.Exit(1)

    # Find all image files
    image_files = (
        list(scan_dir.glob("*.jpg"))
        + list(scan_dir.glob("*.JPG"))
        + list(scan_dir.glob("*.jpeg"))
        + list(scan_dir.glob("*.JPEG"))
        + list(scan_dir.glob("*.png"))
        + list(scan_dir.glob("*.PNG"))
    )

    if not image_files:
        console.print(f"[yellow]![/yellow] No image files found in {scan_dir}")
        return

    console.print(f"\n[bold]Found {len(image_files)} files to process[/bold]\n")

    # Initialize orchestrators
    orchestrator = ExtractionOrchestrator(config, llm)
    classification_orch = ClassificationOrchestrator(config, llm) if config.classification.auto_detect else None

    # Temporarily override config flags based on CLI options
    original_metadata_flag = config.processing.extract_metadata
    original_summary_flag = config.processing.generate_summaries

    if skip_metadata:
        config.processing.extract_metadata = False
    if skip_summary:
        config.processing.generate_summaries = False

    processing_orch = ProcessingOrchestrator(config, llm) if not skip_cleanup else None

    # Restore original config flags
    config.processing.extract_metadata = original_metadata_flag
    config.processing.generate_summaries = original_summary_flag

    # Initialize storage orchestrator if needed
    storage_orch = None
    if not skip_storage and config.processing.generate_embeddings:
        try:
            embedding_config = config.get_embedding_config()
            embedding_provider = get_provider(config.embeddings.provider, embedding_config)
            storage_orch = StorageOrchestrator(config, embedding_provider)
        except Exception as e:
            console.print(f"[red]✗[/red] Storage initialization failed: {e}")
            console.print("[yellow]![/yellow] Continuing without storage")

    # Process with progress bar
    results_summary = {
        "success": 0,
        "failed": 0,
        "classified": 0,
        "classification_failed": 0,
        "processed": 0,
        "processing_failed": 0,
        "stored": 0,
        "storage_failed": 0,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(image_files))

        for file in image_files:
            progress.update(task, description=f"[cyan]Processing {file.name}")

            try:
                job = orchestrator.extract(file)

                if job.success:
                    results_summary["success"] += 1

                    # Run classification if enabled
                    if classification_orch and job.primary_result:
                        try:
                            classification_job = classification_orch.classify(
                                image_path=file,
                                extracted_text=job.primary_result.text,
                                extraction_results=job.results,
                            )

                            # Apply manual overrides if provided
                            if doc_type_override or content_type_override:
                                classification_job.manual_override = {}
                                if doc_type_override:
                                    classification_job.manual_override["doc_type"] = doc_type_override
                                if content_type_override:
                                    classification_job.manual_override["content_type"] = content_type_override

                            if classification_job.success:
                                results_summary["classified"] += 1
                                if verbose and classification_job.primary_result:
                                    cls = classification_job.primary_result
                                    console.print(
                                        f"[green]✓[/green] {file.name}: "
                                        f"{job.primary_result.character_count} chars "
                                        f"({job.primary_result.extractor.value}, {job.primary_result.confidence:.2%}) | "
                                        f"{cls.doc_type.value}, {cls.content_type.value} "
                                        f"({cls.classifier.value}, {cls.confidence:.2%})"
                                    )
                            else:
                                results_summary["classification_failed"] += 1
                                if verbose:
                                    console.print(
                                        f"[yellow]![/yellow] {file.name}: extracted but classification failed"
                                    )

                        except Exception as e:
                            results_summary["classification_failed"] += 1
                            if verbose:
                                console.print(f"[yellow]![/yellow] {file.name}: classification error: {e}")

                    # Phase 4: Processing (if enabled)
                    if processing_orch and job.primary_result and classification_job.success:
                        try:
                            processing_result = processing_orch.process(
                                extraction_job=job,
                                classification_job=classification_job,
                                write_markdown=write_markdown,
                            )

                            if processing_result.success:
                                results_summary["processed"] += 1
                                if verbose:
                                    summary_preview = (
                                        processing_result.summary[:50] + "..."
                                        if len(processing_result.summary) > 50
                                        else processing_result.summary
                                    )
                                    console.print(
                                        f"[green]✓[/green] {file.name}: processed | {summary_preview}"
                                    )

                                # Phase 5: Storage (if enabled and processing succeeded)
                                if storage_orch:
                                    try:
                                        doc_id, storage_time, storage_success, storage_error = (
                                            storage_orch.store_document(processing_result, file)
                                        )
                                        if storage_success:
                                            results_summary["stored"] += 1
                                            if verbose:
                                                console.print(
                                                    f"[green]✓[/green] {file.name}: stored | ID: {doc_id[:8]}..."
                                                )
                                        else:
                                            results_summary["storage_failed"] += 1
                                            if verbose:
                                                console.print(f"[red]✗[/red] {file.name}: storage failed: {storage_error}")
                                    except Exception as e:
                                        results_summary["storage_failed"] += 1
                                        if verbose:
                                            console.print(f"[red]✗[/red] {file.name}: storage error: {e}")

                            else:
                                results_summary["processing_failed"] += 1
                                if verbose:
                                    console.print(
                                        f"[yellow]![/yellow] {file.name}: processing failed"
                                    )

                        except Exception as e:
                            results_summary["processing_failed"] += 1
                            if verbose:
                                console.print(f"[yellow]![/yellow] {file.name}: processing error: {e}")

                    elif verbose and job.primary_result:
                        console.print(
                            f"[green]✓[/green] {file.name}: {job.primary_result.character_count} chars "
                            f"({job.primary_result.extractor.value}, {job.primary_result.confidence:.2%})"
                        )
                else:
                    results_summary["failed"] += 1
                    if verbose:
                        console.print(f"[red]✗[/red] {file.name}: {job.errors}")

            except Exception as e:
                results_summary["failed"] += 1
                if verbose:
                    console.print(f"[red]✗[/red] {file.name}: {e}")

            progress.advance(task)

    # Display summary
    console.print("\n[bold]Processing Complete[/bold]")
    console.print(f"  [green]Extraction Success:[/green] {results_summary['success']}")
    console.print(f"  [red]Extraction Failed:[/red] {results_summary['failed']}")
    if classification_orch:
        console.print(f"  [green]Classified:[/green] {results_summary['classified']}")
        console.print(
            f"  [yellow]Classification Failed:[/yellow] {results_summary['classification_failed']}"
        )
    if processing_orch:
        console.print(f"  [green]Processed:[/green] {results_summary['processed']}")
        console.print(f"  [yellow]Processing Failed:[/yellow] {results_summary['processing_failed']}")
    if storage_orch:
        console.print(f"  [green]Stored:[/green] {results_summary['stored']}")
        console.print(f"  [red]Storage Failed:[/red] {results_summary['storage_failed']}")


@app.command()
def test_llm(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Provider to test (defaults to config setting)"
    ),
    config_path: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Test LLM provider connectivity and configuration.

    This command validates:
    - Configuration is valid
    - Provider is accessible
    - Required models are available
    - Basic operations work (completion, embedding)
    """
    config = load_and_validate_config(Path(config_path))
    setup_logging(config.logging.level)

    # Override provider if specified
    if provider:
        config.llm.provider = provider  # type: ignore

    console.print(f"\n[bold]Testing LLM Provider:[/bold] {config.llm.provider}")

    try:
        # Initialize provider
        with console.status("[bold blue]Initializing provider..."):
            llm_config = config.get_active_llm_config()
            llm = get_provider(config.llm.provider, llm_config)
        console.print("[green]✓[/green] Provider initialized")

        # Health check
        with console.status("[bold blue]Running health check..."):
            healthy = llm.health_check()

        if not healthy:
            console.print("[red]✗[/red] Health check failed")
            raise typer.Exit(1)
        console.print("[green]✓[/green] Health check passed")

        # Test completion
        with console.status("[bold blue]Testing text completion..."):
            response = llm.complete(
                prompt="Say 'Hello, Ingrid!' and nothing else.", temperature=0.0
            )
        console.print("[green]✓[/green] Text completion works")
        console.print(f"  Response: {response.content[:100]}")

        # Test embeddings
        with console.status("[bold blue]Testing embeddings..."):
            embedding = llm.embed("Test text for embeddings")
        console.print("[green]✓[/green] Embeddings work")
        console.print(f"  Dimensions: {embedding.dimensions}")

        # Summary
        console.print(
            Panel.fit(
                "[bold green]All tests passed![/bold green]\n"
                f"Provider '{config.llm.provider}' is ready to use.",
                border_style="green",
            )
        )

    except LLMError as e:
        console.print(f"[red]✗[/red] Test failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        raise typer.Exit(1)


@app.command()
def config_check(
    config_path: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to configuration file"
    )
) -> None:
    """Validate configuration file without connecting to services."""
    console.print("[bold]Validating configuration...[/bold]\n")

    try:
        config = load_config(Path(config_path))

        # Display configuration details
        table = Table(title="Configuration Summary", show_header=True)
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Setting", style="magenta")
        table.add_column("Value", style="green")

        table.add_row("LLM", "Provider", config.llm.provider)
        table.add_row("Storage", "Input", str(config.storage.input_path))
        table.add_row("Storage", "Output", str(config.storage.output_path))
        table.add_row("Storage", "Database", str(config.storage.database_path))
        table.add_row("Processing", "Batch Size", str(config.processing.batch_size))
        table.add_row("Logging", "Level", config.logging.level)

        console.print(table)
        console.print("\n[green]✓[/green] Configuration is valid")

    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        raise typer.Exit(1)


@app.command()
def stats(
    config_path: str = typer.Option(
        "config.yaml", "--config", "-c", help="Path to configuration file"
    ),
) -> None:
    """Show database and vector store statistics.

    Displays:
    - Total document count
    - Documents by type and content type
    - Documents by language
    - Flagged documents count
    - Average processing times
    - Vector collection sizes
    """
    config = load_and_validate_config(Path(config_path))
    setup_logging(config.logging.level)

    try:
        # Initialize database manager
        from .storage import DatabaseManager, VectorStoreManager

        db = DatabaseManager(config.storage.database_path)
        db_stats = db.get_statistics()

        # Display database statistics
        console.print("\n[bold]Database Statistics[/bold]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")

        table.add_row("Total Documents", str(db_stats["total"]))
        table.add_row("", "")  # Spacer

        # By type
        for doc_type, count in db_stats["by_type"].items():
            table.add_row(f"  {doc_type}", str(count))

        table.add_row("", "")  # Spacer

        # By content type
        for content_type, count in db_stats["by_content_type"].items():
            table.add_row(f"  {content_type}", str(count))

        table.add_row("", "")  # Spacer

        # By language
        for lang, count in db_stats["by_language"].items():
            table.add_row(f"  {lang}", str(count))

        table.add_row("", "")  # Spacer
        table.add_row("Flagged for Review", str(db_stats["flagged_count"]))

        console.print(table)

        # Average processing times
        if db_stats["total"] > 0:
            console.print("\n[bold]Average Processing Times[/bold]")
            times_table = Table()
            times_table.add_column("Phase", style="cyan")
            times_table.add_column("Time", style="yellow", justify="right")

            times_table.add_row("Extraction", f"{db_stats['avg_extraction_time']:.1f}s")
            times_table.add_row("Classification", f"{db_stats['avg_classification_time']:.1f}s")
            times_table.add_row("Processing", f"{db_stats['avg_processing_time']:.1f}s")
            times_table.add_row("Storage", f"{db_stats['avg_storage_time']:.1f}s")

            console.print(times_table)

        # Try to get vector store stats if initialized
        try:
            embedding_config = config.get_embedding_config()
            embedding_provider = get_provider(config.embeddings.provider, embedding_config)
            vectorstore = VectorStoreManager(
                config.storage.chroma_path, embedding_provider
            )
            vector_stats = vectorstore.get_collection_stats()

            console.print("\n[bold]Vector Collections[/bold]")
            vector_table = Table()
            vector_table.add_column("Collection", style="cyan")
            vector_table.add_column("Documents", style="yellow", justify="right")

            for coll_name, count in vector_stats.items():
                vector_table.add_row(coll_name, str(count))

            console.print(vector_table)

        except Exception as e:
            console.print(f"\n[yellow]![/yellow] Vector store stats unavailable: {e}")

    except FileNotFoundError:
        console.print(f"[red]✗[/red] Database not found at {config.storage.database_path}")
        console.print("[yellow]![/yellow] Run 'ingrid process --batch' to create database")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option(
        "cleaned_text",
        "--collection",
        "-c",
        help="Collection to search: cleaned_text, summaries, summaries_english",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    config_path: str = typer.Option("config.yaml", "--config", help="Path to configuration file"),
) -> None:
    """Semantic search across processed documents.

    Search collections:
    - cleaned_text: Full document text
    - summaries: Original language summaries
    - summaries_english: English translated summaries

    Examples:
        ingrid search "wartime food shortages"
        ingrid search "family letters" --collection summaries_english
        ingrid search "Amsterdam 1943" --top-k 5
    """
    config = load_and_validate_config(Path(config_path))
    setup_logging(config.logging.level)

    try:
        # Initialize embedding provider and vector store
        embedding_config = config.get_embedding_config()
        embedding_provider = get_provider(config.embeddings.provider, embedding_config)

        from .storage import VectorStoreManager
        vectorstore = VectorStoreManager(
            config.storage.chroma_path,
            embedding_provider,
        )

        # Perform search
        console.print(f"\n[bold]Searching for:[/bold] '{query}'")
        console.print(f"[bold]Collection:[/bold] ingrid_{collection}")

        with console.status("[cyan]Searching..."):
            results = vectorstore.search_text(query, collection, top_k)

        if not results:
            console.print("\n[yellow]![/yellow] No results found")
            return

        # Display results
        console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Score", style="green", width=6)
        table.add_column("Document", style="yellow", width=30)
        table.add_column("Preview", style="white", width=60)

        for i, result in enumerate(results, 1):
            filename = result["metadata"].get("filename", "Unknown")
            doc_type = result["metadata"].get("doc_type", "unknown")
            preview = result["document"][:100] + "..." if len(result["document"]) > 100 else result["document"]

            table.add_row(
                str(i),
                f"{result['score']:.3f}",
                f"{filename}\n({doc_type})",
                preview
            )

        console.print(table)

        # Show how to get more details
        console.print(f"\n[blue]ℹ[/blue] Use 'ingrid show <doc_id>' to view full document details")

    except FileNotFoundError:
        console.print(f"[red]✗[/red] Vector store not found at {config.storage.chroma_path}")
        console.print("[yellow]![/yellow] Run 'ingrid process --batch' to create vector store")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Search error: {e}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]![/yellow] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]✗[/red] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
