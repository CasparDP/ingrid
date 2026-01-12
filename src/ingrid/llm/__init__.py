"""LLM integration layer for Ingrid document processing pipeline.

Provides a unified interface for working with various LLM providers including
Ollama, Anthropic Claude, Google Gemini, and HuggingFace.
"""

from .base import (
    BaseLLMProvider,
    EmbeddingResponse,
    LLMConnectionError,
    LLMConfigError,
    LLMError,
    LLMModelError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    VisionResponse,
)
from .ollama import OllamaProvider

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMResponse",
    "VisionResponse",
    "EmbeddingResponse",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMModelError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    # Providers
    "OllamaProvider",
    # Factory function
    "get_provider",
]

# Provider registry for dynamic instantiation
PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    # Future providers will be added here:
    # "ollama_cloud": OllamaCloudProvider,
    # "anthropic": AnthropicProvider,
    # "google": GoogleProvider,
    # "huggingface": HuggingFaceProvider,
}


def get_provider(provider_name: str, config: dict[str, str | int | bool]) -> BaseLLMProvider:
    """Factory function to instantiate LLM provider by name.

    Args:
        provider_name: Name of the provider (ollama, anthropic, etc.).
        config: Provider-specific configuration dictionary.

    Returns:
        Instantiated provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available providers: {available}"
        )

    provider_class = PROVIDERS[provider_name]
    return provider_class(config)
