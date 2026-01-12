"""Ingrid - Document Extraction Pipeline for Historical Letters and Newspapers.

A comprehensive pipeline for extracting text from scanned historical documents,
with support for handwritten and typed text, multiple languages, and semantic search.
"""

__version__ = "0.1.0"
__author__ = "Caspar Moolenaar"

from .config import Config, get_config, load_config
from .llm import BaseLLMProvider, LLMError, get_provider

__all__ = [
    # Configuration
    "load_config",
    "get_config",
    "Config",
    # LLM
    "get_provider",
    "BaseLLMProvider",
    "LLMError",
]
