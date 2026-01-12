"""Configuration system for Ingrid document processing pipeline.

Provides type-safe configuration loading using Pydantic models with environment variable
substitution support.
"""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Provider-specific configuration models
# =============================================================================


class OllamaConfig(BaseModel):
    """Configuration for local Ollama instance."""

    base_url: str = "http://localhost:11434"
    model: str
    embedding_model: str


class OllamaCloudConfig(BaseModel):
    """Configuration for Ollama Cloud."""

    api_key: str
    model: str


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic Claude API."""

    api_key: str
    model: str


class GoogleConfig(BaseModel):
    """Configuration for Google AI (Gemini)."""

    api_key: str
    model: str


class HuggingFaceConfig(BaseModel):
    """Configuration for HuggingFace Inference API."""

    api_key: str
    model: str


# =============================================================================
# Main configuration section models
# =============================================================================


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["ollama", "ollama_cloud", "anthropic", "google", "huggingface"]
    ollama: OllamaConfig | None = None
    ollama_cloud: OllamaCloudConfig | None = None
    anthropic: AnthropicConfig | None = None
    google: GoogleConfig | None = None
    huggingface: HuggingFaceConfig | None = None

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that provider is a recognized value."""
        valid_providers = ["ollama", "ollama_cloud", "anthropic", "google", "huggingface"]
        if v not in valid_providers:
            raise ValueError(
                f"Invalid provider '{v}'. Must be one of: {', '.join(valid_providers)}"
            )
        return v


class OCRConfig(BaseModel):
    """OCR engine configuration."""

    engine: Literal["docling", "tesseract", "easyocr"]
    htr_model: str
    languages: list[str]


class ClassificationConfig(BaseModel):
    """Document classification configuration."""

    auto_detect: bool = True
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.7)


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""

    provider: str
    model: str
    dimensions: int = Field(gt=0)


class StorageConfig(BaseModel):
    """Storage paths configuration."""

    database_path: Path
    chroma_path: Path
    output_path: Path
    input_path: Path


class ProcessingConfig(BaseModel):
    """Processing options configuration."""

    batch_size: int = Field(gt=0, default=10)
    generate_summaries: bool = True
    extract_metadata: bool = True
    generate_embeddings: bool = True
    max_retries: int = Field(ge=0, le=10, default=3)
    cache_responses: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path


# =============================================================================
# Root configuration model
# =============================================================================


class Config(BaseModel):
    """Root configuration model for Ingrid pipeline."""

    llm: LLMConfig
    ocr: OCRConfig
    classification: ClassificationConfig
    embeddings: EmbeddingsConfig
    storage: StorageConfig
    processing: ProcessingConfig
    logging: LoggingConfig

    def get_active_llm_config(self) -> dict[str, str | int]:
        """Get configuration dictionary for the active LLM provider.

        Returns:
            Configuration dictionary for the active provider.

        Raises:
            ValueError: If no configuration exists for the active provider.
        """
        provider = self.llm.provider
        # Replace hyphens with underscores for attribute access
        provider_attr = provider.replace("-", "_")
        config = getattr(self.llm, provider_attr, None)

        if config is None:
            raise ValueError(f"No configuration found for provider: {provider}")

        return config.model_dump()


# =============================================================================
# Configuration loading functions
# =============================================================================


def substitute_env_vars(config_dict: dict[str, object] | list[object] | str | object) -> object:
    """Recursively substitute ${VAR_NAME} with environment variables.

    Args:
        config_dict: Configuration dictionary, list, string, or other value to process.

    Returns:
        Processed configuration with environment variables substituted.
    """
    if isinstance(config_dict, dict):
        return {k: substitute_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [substitute_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Match ${VAR_NAME} pattern
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, config_dict)
        result = config_dict
        for var_name in matches:
            env_value = os.getenv(var_name, "")
            result = result.replace(f"${{{var_name}}}", env_value)
        return result
    return config_dict


def load_config(config_path: Path | str = "config.yaml") -> Config:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Validated configuration object.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid.
    """
    # Load environment variables from .env
    load_dotenv()

    # Load YAML
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    processed_config = substitute_env_vars(raw_config)

    # Validate with Pydantic
    config = Config.model_validate(processed_config)

    return config


# =============================================================================
# Global configuration singleton
# =============================================================================

_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The global configuration object.

    Raises:
        RuntimeError: If configuration hasn't been loaded yet.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance (primarily for testing).

    Args:
        config: Configuration object to set globally.
    """
    global _config
    _config = config
