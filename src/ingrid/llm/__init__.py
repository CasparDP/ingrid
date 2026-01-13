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
    # Factory functions
    "get_provider",
    "get_provider_for_task",
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


def get_provider_for_task(
    task: str,
    llm_config: "LLMConfig",  # type: ignore  # Forward reference to avoid circular import
    default_provider: BaseLLMProvider | None = None,
) -> BaseLLMProvider:
    """Get LLM provider with task-specific model override if configured.

    This function checks if a task-specific model is configured in llm.task_models.
    If found, it creates a new provider instance with that model.
    Otherwise, returns the default provider or creates one with the default model.

    Args:
        task: Task name (e.g., "metadata_extraction", "classification", "summarization")
        llm_config: LLMConfig object containing provider and task_models settings
        default_provider: Optional default provider to return if no override exists

    Returns:
        LLM provider instance (either new with overridden model or the default)

    Example:
        ```python
        from ingrid.config import load_config
        from ingrid.llm import get_provider, get_provider_for_task

        config = load_config()
        default_llm = get_provider(config.llm.provider, config.get_active_llm_config())

        # Use task-specific model if configured
        metadata_llm = get_provider_for_task("metadata_extraction", config.llm, default_llm)
        ```
    """
    # Check if task has a model override
    task_model = llm_config.get_model_for_task(task)

    if task_model is None:
        # No override - return default provider or create standard one
        if default_provider is not None:
            return default_provider

        # Create provider with default config
        # Import here to avoid circular dependency
        from ..config import Config
        if isinstance(llm_config, dict):
            config_obj = Config(**{"llm": llm_config})  # type: ignore
        else:
            # Get active provider config
            provider_config = {}
            provider_name = llm_config.provider

            if provider_name == "ollama" and llm_config.ollama:
                provider_config = llm_config.ollama.model_dump()
            elif provider_name == "ollama_cloud" and llm_config.ollama_cloud:
                provider_config = llm_config.ollama_cloud.model_dump()
            elif provider_name == "anthropic" and llm_config.anthropic:
                provider_config = llm_config.anthropic.model_dump()
            elif provider_name == "google" and llm_config.google:
                provider_config = llm_config.google.model_dump()
            elif provider_name == "huggingface" and llm_config.huggingface:
                provider_config = llm_config.huggingface.model_dump()

            return get_provider(provider_name, provider_config)

    # Task-specific model override exists - create new provider with overridden model
    provider_name = llm_config.provider
    provider_config = {}

    if provider_name == "ollama" and llm_config.ollama:
        provider_config = llm_config.ollama.model_dump()
        provider_config["model"] = task_model  # Override model
    elif provider_name == "ollama_cloud" and llm_config.ollama_cloud:
        provider_config = llm_config.ollama_cloud.model_dump()
        provider_config["model"] = task_model
    elif provider_name == "anthropic" and llm_config.anthropic:
        provider_config = llm_config.anthropic.model_dump()
        provider_config["model"] = task_model
    elif provider_name == "google" and llm_config.google:
        provider_config = llm_config.google.model_dump()
        provider_config["model"] = task_model
    elif provider_name == "huggingface" and llm_config.huggingface:
        provider_config = llm_config.huggingface.model_dump()
        provider_config["model"] = task_model

    return get_provider(provider_name, provider_config)
