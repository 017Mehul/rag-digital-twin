"""
LLM provider abstractions and concrete implementations.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.exceptions import ErrorCode, ResponseGenerationError


class LLMProvider(ABC):
    """
    Base interface for language-model providers.
    """

    provider_name: str = "base"

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name
        self._loaded_model: Optional[Any] = None
        self._loaded_backend = "uninitialized"

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load any underlying client or model implementation.
        """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Generate text from a prompt.
        """

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for provider-agnostic tests and validation.
        """
        return len(self._validate_text(text).split())

    def get_config(self) -> Dict[str, Any]:
        """
        Return provider configuration metadata.
        """
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "backend": self._loaded_backend,
        }

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ResponseGenerationError(
                "LLM input must be a string",
                ErrorCode.LLM_GENERATION_FAILED,
                self.model_name,
            )

        normalized = text.strip()
        if not normalized:
            raise ResponseGenerationError(
                "LLM input cannot be empty",
                ErrorCode.LLM_GENERATION_FAILED,
                self.model_name,
            )

        return normalized

    def _mock_response(self, prompt: str) -> str:
        normalized_prompt = self._validate_text(prompt)
        digest = hashlib.sha256(f"{self.provider_name}:{self.model_name}:{normalized_prompt}".encode("utf-8")).hexdigest()
        return f"Context-grounded answer generated for request {digest[:12]}."


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI chat-completion provider with optional deterministic mock mode.
    """

    provider_name = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        mock_responses: bool = False,
    ) -> None:
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client
        self.mock_responses = mock_responses

    def load_model(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model

        if self.mock_responses:
            self._loaded_model = "mock-openai-llm"
            self._loaded_backend = "mock"
            return self._loaded_model

        if self.client is not None:
            self._loaded_model = self.client
            self._loaded_backend = "client"
            return self._loaded_model

        if not self.api_key:
            raise ResponseGenerationError(
                "OpenAI API key is required to load the LLM provider",
                ErrorCode.LLM_MODEL_NOT_FOUND,
                self.model_name,
            )

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ResponseGenerationError(
                "OpenAI client library is not available",
                ErrorCode.LLM_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

        self._loaded_model = OpenAI(api_key=self.api_key)
        self._loaded_backend = "openai"
        return self._loaded_model

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        normalized_prompt = self._validate_text(prompt)
        client = self.load_model()

        if self.mock_responses:
            return self._mock_response(normalized_prompt)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": normalized_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except ResponseGenerationError:
            raise
        except Exception as exc:
            raise ResponseGenerationError(
                f"OpenAI response generation failed: {exc}",
                ErrorCode.LLM_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc


class HuggingFaceLLMProvider(LLMProvider):
    """
    Hugging Face text-generation provider with optional deterministic mock mode.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        generator: Optional[Any] = None,
        mock_responses: bool = False,
    ) -> None:
        super().__init__(model_name=model_name)
        self.generator = generator
        self.mock_responses = mock_responses

    def load_model(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model

        if self.mock_responses:
            self._loaded_model = "mock-huggingface-llm"
            self._loaded_backend = "mock"
            return self._loaded_model

        if self.generator is not None:
            self._loaded_model = self.generator
            self._loaded_backend = "generator"
            return self._loaded_model

        try:
            from transformers import pipeline
        except Exception as exc:
            raise ResponseGenerationError(
                "Transformers pipeline is not available",
                ErrorCode.LLM_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

        try:
            self._loaded_model = pipeline("text-generation", model=self.model_name)
            self._loaded_backend = "transformers"
            return self._loaded_model
        except Exception as exc:
            raise ResponseGenerationError(
                f"Unable to load Hugging Face language model: {exc}",
                ErrorCode.LLM_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        normalized_prompt = self._validate_text(prompt)
        generator = self.load_model()

        if self.mock_responses:
            return self._mock_response(normalized_prompt)

        try:
            response = generator(
                normalized_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            if isinstance(response, list) and response:
                return str(response[0].get("generated_text", "")).strip()
            return str(response).strip()
        except ResponseGenerationError:
            raise
        except Exception as exc:
            raise ResponseGenerationError(
                f"Hugging Face response generation failed: {exc}",
                ErrorCode.LLM_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc
