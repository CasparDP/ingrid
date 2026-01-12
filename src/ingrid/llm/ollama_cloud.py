"""Ollama Cloud LLM provider implementation.

Provides integration with Ollama Cloud (https://ollama.com) for LLM operations
including text completion, vision analysis, and embeddings.

Usage requires OLLAMA_API_KEY environment variable to be set.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Any

from ollama import Client

from .base import (
    BaseLLMProvider,
    EmbeddingResponse,
    LLMConfigError,
    LLMConnectionError,
    LLMModelError,
    LLMResponse,
    LLMResponseError,
    VisionResponse,
)

logger = logging.getLogger(__name__)

OLLAMA_CLOUD_HOST = "https://ollama.com"


class OllamaCloudProvider(BaseLLMProvider):
    """Ollama Cloud LLM provider implementation.

    Connects to Ollama's hosted cloud service for running LLM inference
    without requiring a local Ollama installation.
    """

    def __init__(self, config: dict[str, str | int | bool]) -> None:
        """Initialize Ollama Cloud provider.

        Args:
            config: Provider configuration containing api_key (or uses env var),
                   model, and optionally embedding_model.
        """
        super().__init__(config)

        # Get API key from config or environment variable
        api_key = config.get("api_key")
        if api_key and str(api_key).startswith("${") and str(api_key).endswith("}"):
            # Config references an environment variable
            env_var = str(api_key)[2:-1]
            api_key = os.environ.get(env_var)

        if not api_key:
            api_key = os.environ.get("OLLAMA_API_KEY")

        if not api_key:
            raise LLMConfigError(
                "Ollama Cloud API key not found. Set OLLAMA_API_KEY environment "
                "variable or provide api_key in config."
            )

        self.client = Client(
            host=OLLAMA_CLOUD_HOST,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.model = str(config["model"])
        self.embedding_model = str(config.get("embedding_model", "nomic-embed-text"))

        logger.info(f"Initialized Ollama Cloud provider with model: {self.model}")

    def _validate_config(self) -> None:
        """Validate Ollama Cloud-specific configuration.

        Raises:
            ValueError: If required configuration keys are missing.
        """
        if "model" not in self.config:
            raise ValueError("Missing required config key: model")

    def _encode_image(self, image_path: Path | str) -> str:
        """Encode image to base64 for Ollama vision API.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64-encoded image string.

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text completion using Ollama Cloud.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMConnectionError: If connection to Ollama Cloud fails.
            LLMResponseError: If response is invalid.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            response = self.client.chat(model=self.model, messages=messages, options=options)

            return LLMResponse(
                content=response["message"]["content"],
                model=self.model,
                tokens_used=response.get("eval_count"),
                finish_reason=response.get("done_reason"),
                metadata={"response": response},
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg:
                raise LLMConnectionError(f"Failed to connect to Ollama Cloud: {e}") from e
            if "model" in error_msg and "not found" in error_msg:
                raise LLMModelError(f"Model not found on Ollama Cloud: {e}") from e
            raise LLMResponseError(f"Ollama Cloud error: {e}") from e

    def complete_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """Generate streaming text completion using Ollama Cloud.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Chunks of generated text as they arrive.

        Raises:
            LLMConnectionError: If connection to Ollama Cloud fails.
            LLMResponseError: If response is invalid.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        try:
            for part in self.client.chat(
                model=self.model, messages=messages, options=options, stream=True
            ):
                yield part["message"]["content"]

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg:
                raise LLMConnectionError(f"Failed to connect to Ollama Cloud: {e}") from e
            raise LLMResponseError(f"Ollama Cloud streaming error: {e}") from e

    def vision(
        self,
        image_path: Path | str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> VisionResponse:
        """Analyze image using Ollama Cloud vision model.

        Args:
            image_path: Path to the image file.
            prompt: The prompt describing what to analyze.
            system_prompt: Optional system prompt to set context.

        Returns:
            VisionResponse containing the analysis and metadata.

        Raises:
            LLMConnectionError: If connection to Ollama Cloud fails.
            LLMResponseError: If response is invalid.
            FileNotFoundError: If image file doesn't exist.
        """
        image_b64 = self._encode_image(image_path)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt, "images": [image_b64]})

        try:
            response = self.client.chat(model=self.model, messages=messages)

            return VisionResponse(
                content=response["message"]["content"],
                model=self.model,
                tokens_used=response.get("eval_count"),
                finish_reason=response.get("done_reason"),
                metadata={"response": response},
            )

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg:
                raise LLMConnectionError(f"Failed to connect to Ollama Cloud: {e}") from e
            if "model" in error_msg and "not found" in error_msg:
                raise LLMModelError(f"Vision model not found on Ollama Cloud: {e}") from e
            raise LLMResponseError(f"Ollama Cloud vision error: {e}") from e

    def embed(self, text: str | list[str]) -> EmbeddingResponse | list[EmbeddingResponse]:
        """Generate embeddings using Ollama Cloud.

        Args:
            text: Single text string or list of text strings to embed.

        Returns:
            Single EmbeddingResponse for string input, or list of
            EmbeddingResponse for list input.

        Raises:
            LLMConnectionError: If connection to Ollama Cloud fails.
            LLMResponseError: If response is invalid.
        """
        single_text = isinstance(text, str)
        texts = [text] if single_text else text

        try:
            responses: list[EmbeddingResponse] = []
            for txt in texts:
                response = self.client.embeddings(model=self.embedding_model, prompt=txt)
                responses.append(
                    EmbeddingResponse(
                        embedding=response["embedding"],
                        model=self.embedding_model,
                        dimensions=len(response["embedding"]),
                    )
                )

            return responses[0] if single_text else responses

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "network" in error_msg:
                raise LLMConnectionError(f"Failed to connect to Ollama Cloud: {e}") from e
            raise LLMResponseError(f"Ollama Cloud embedding error: {e}") from e

    def health_check(self) -> bool:
        """Check Ollama Cloud connectivity.

        Note: Unlike local Ollama, we can't list available models on the cloud.
        We just verify we can make a basic request.

        Returns:
            True if Ollama Cloud is accessible with valid credentials.
        """
        try:
            # Try a minimal completion to verify connectivity and auth
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 1},
            )
            logger.info(f"Ollama Cloud health check passed. Model: {self.model}")
            return True

        except Exception as e:
            logger.error(f"Ollama Cloud health check failed: {e}")
            return False
