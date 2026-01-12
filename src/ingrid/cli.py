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
        process_batch(config, llm, verbose, doc_type, content_type)
    else:
        if file_path:
            process_single(Path(file_path), config, llm, verbose, doc_type, content_type)


def process_single(
    file_path: Path,
    config: Config,
    llm: BaseLLMProvider,
    verbose: bool,
    doc_type_override: str | None = None,
    content_type_override: str | None = None,
) -> None:
    """Process a single file through extraction and classification pipeline.

    Args:
        file_path: Path to the file to process.
        config: Configuration object.
        llm: LLM provider instance.
        verbose: Show detailed extraction results.
        doc_type_override: Manual override for document type.
        content_type_override: Manual override for content type.
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
) -> None:
    """Process all files in scans directory.

    Args:
        config: Configuration object.
        llm: LLM provider instance.
        verbose: Show detailed results for each file.
        doc_type_override: Manual override for document type.
        content_type_override: Manual override for content type.
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

    # Process with progress bar
    results_summary = {"success": 0, "failed": 0, "classified": 0, "classification_failed": 0}

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
        console.print(f"  [yellow]Classification Failed:[/yellow] {results_summary['classification_failed']}")


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
