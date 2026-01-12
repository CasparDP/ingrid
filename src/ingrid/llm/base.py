"""Abstract base class and interfaces for LLM providers.

Defines the common interface that all LLM providers must implement, along with
response models and exception hierarchy.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar


# =============================================================================
# Exception hierarchy
# =============================================================================


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class LLMConfigError(LLMError):
    """Configuration error (missing keys, invalid settings)."""

    pass


class LLMConnectionError(LLMError):
    """Cannot connect to LLM provider."""

    pass


class LLMModelError(LLMError):
    """Model not found or invalid."""

    pass


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM."""

    pass


class LLMTimeoutError(LLMError):
    """Request timeout."""

    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""

    pass


# =============================================================================
# Response models
# =============================================================================


@dataclass
class LLMResponse:
    """Standard response wrapper for all LLM operations."""

    content: str
    model: str
    tokens_used: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, object] | None = None


@dataclass
class VisionResponse(LLMResponse):
    """Response for vision-based operations."""

    confidence: dict[str, float] | None = None


@dataclass
class EmbeddingResponse:
    """Response for embedding operations."""

    embedding: list[float]
    model: str
    dimensions: int


# =============================================================================
# Abstract base class
# =============================================================================

T = TypeVar("T")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers.

    All concrete LLM provider implementations must inherit from this class
    and implement all abstract methods.
    """

    def __init__(self, config: dict[str, str | int | bool]) -> None:
        """Initialize the LLM provider.

        Args:
            config: Provider-specific configuration dictionary.
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration.

        Raises:
            LLMConfigError: If configuration is invalid or incomplete.
        """
        pass

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text completion.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMConnectionError: If connection to provider fails.
            LLMResponseError: If response is invalid.
            LLMTimeoutError: If request times out.
        """
        pass

    @abstractmethod
    def vision(
        self,
        image_path: Path | str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> VisionResponse:
        """Analyze image with vision model.

        Args:
            image_path: Path to the image file.
            prompt: The prompt describing what to analyze.
            system_prompt: Optional system prompt to set context.

        Returns:
            VisionResponse containing the analysis and metadata.

        Raises:
            LLMConnectionError: If connection to provider fails.
            LLMResponseError: If response is invalid.
            FileNotFoundError: If image file doesn't exist.
        """
        pass

    @abstractmethod
    def embed(
        self, text: str | list[str]
    ) -> EmbeddingResponse | list[EmbeddingResponse]:
        """Generate embeddings for text.

        Args:
            text: Single text string or list of text strings to embed.

        Returns:
            Single EmbeddingResponse for string input, or list of
            EmbeddingResponse for list input.

        Raises:
            LLMConnectionError: If connection to provider fails.
            LLMResponseError: If response is invalid.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is accessible and configured correctly.

        Returns:
            True if provider is healthy and ready to use, False otherwise.
        """
        pass

    def retry_with_backoff(
        self,
        func: Callable[[], T],
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> T:
        """Retry a function with exponential backoff.

        Args:
            func: Function to retry.
            max_retries: Maximum number of retry attempts.
            initial_delay: Initial delay in seconds.
            backoff_factor: Multiplier for delay on each retry.

        Returns:
            The result of the function call.

        Raises:
            The last exception encountered if all retries fail.
        """
        last_exception: Exception | None = None
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                return func()
            except (LLMConnectionError, LLMTimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    raise

        # This should never be reached due to the raise in the except block
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected retry logic error")
