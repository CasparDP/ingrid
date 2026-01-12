"""Ollama LLM provider implementation.

Provides integration with local Ollama instances for LLM operations including
text completion, vision analysis, and embeddings.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import ollama

from .base import (
    BaseLLMProvider,
    EmbeddingResponse,
    LLMConnectionError,
    LLMModelError,
    LLMResponse,
    LLMResponseError,
    VisionResponse,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, config: dict[str, str | int | bool]) -> None:
        """Initialize Ollama provider.

        Args:
            config: Provider configuration containing base_url, model, and embedding_model.
        """
        super().__init__(config)
        base_url = config.get("base_url", "http://localhost:11434")
        self.client = ollama.Client(host=str(base_url))
        self.model = str(config["model"])
        self.embedding_model = str(config["embedding_model"])

        logger.info(f"Initialized Ollama provider with model: {self.model}")

    def _validate_config(self) -> None:
        """Validate Ollama-specific configuration.

        Raises:
            ValueError: If required configuration keys are missing.
        """
        required = ["model", "embedding_model"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

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
        """Generate text completion using Ollama.

        Args:
            prompt: The user prompt to complete.
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMConnectionError: If connection to Ollama fails.
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

        except ollama.RequestError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except ollama.ResponseError as e:
            raise LLMResponseError(f"Invalid response from Ollama: {e}") from e
        except KeyError as e:
            raise LLMResponseError(f"Missing expected key in response: {e}") from e
        except Exception as e:
            raise LLMResponseError(f"Unexpected error: {e}") from e

    def vision(
        self,
        image_path: Path | str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> VisionResponse:
        """Analyze image using Ollama vision model.

        Args:
            image_path: Path to the image file.
            prompt: The prompt describing what to analyze.
            system_prompt: Optional system prompt to set context.

        Returns:
            VisionResponse containing the analysis and metadata.

        Raises:
            LLMConnectionError: If connection to Ollama fails.
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

        except ollama.RequestError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except ollama.ResponseError as e:
            raise LLMResponseError(f"Invalid response from Ollama: {e}") from e
        except KeyError as e:
            raise LLMResponseError(f"Missing expected key in response: {e}") from e
        except Exception as e:
            raise LLMResponseError(f"Unexpected error: {e}") from e

    def embed(
        self, text: str | list[str]
    ) -> EmbeddingResponse | list[EmbeddingResponse]:
        """Generate embeddings using Ollama.

        Args:
            text: Single text string or list of text strings to embed.

        Returns:
            Single EmbeddingResponse for string input, or list of
            EmbeddingResponse for list input.

        Raises:
            LLMConnectionError: If connection to Ollama fails.
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

        except ollama.RequestError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except ollama.ResponseError as e:
            raise LLMResponseError(f"Invalid response from Ollama: {e}") from e
        except KeyError as e:
            raise LLMResponseError(f"Missing expected key in response: {e}") from e
        except Exception as e:
            raise LLMResponseError(f"Unexpected error: {e}") from e

    def health_check(self) -> bool:
        """Check Ollama connectivity and model availability.

        Returns:
            True if Ollama is accessible and required models are available.
        """
        try:
            # Check if we can connect to Ollama
            models_response = self.client.list()
            available_models = [m["name"] for m in models_response.get("models", [])]

            # Check if required models are available
            required_models = [self.model, self.embedding_model]
            for model in required_models:
                # Model names might have tags, so check prefix
                model_base = model.split(":")[0]
                if not any(m.startswith(model_base) for m in available_models):
                    logger.warning(f"Model not found: {model}")
                    logger.info(f"Available models: {available_models}")
                    return False

            logger.info(f"Ollama health check passed. Models available: {required_models}")
            return True

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
