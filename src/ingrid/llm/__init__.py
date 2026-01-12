"""LLM integration layer for Ingrid document processing pipeline.

Provides a unified interface for working with various LLM providers including
Ollama (local and cloud), Anthropic Claude, Google Gemini, and HuggingFace.
"""

from .base import (
    BaseLLMProvider,
    EmbeddingResponse,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    VisionResponse,
)
from .ollama import OllamaProvider
from .ollama_cloud import OllamaCloudProvider

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
    "OllamaCloudProvider",
    # Factory function
    "get_provider",
]

# Provider registry for dynamic instantiation
PROVIDERS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "ollama_cloud": OllamaCloudProvider,
    # Future providers will be added here:
    # "anthropic": AnthropicProvider,
    # "google": GoogleProvider,
    # "huggingface": HuggingFaceProvider,
}


def get_provider(provider_name: str, config: dict[str, str | int | bool]) -> BaseLLMProvider:
    """Factory function to instantiate LLM provider by name.

    Args:
        provider_name: Name of the provider (ollama, ollama_cloud, anthropic, etc.).
        config: Provider-specific configuration dictionary.

    Returns:
        Instantiated provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: {available}")

    provider_class = PROVIDERS[provider_name]
    return provider_class(config)
