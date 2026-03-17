"""
Embedding provider abstractions and concrete implementations.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.exceptions import EmbeddingGenerationError, ErrorCode


class EmbeddingModel(ABC):
    """
    Base interface for embedding providers.
    """

    provider_name: str = "base"

    def __init__(self, model_name: str, dimension: int) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.model_name = model_name
        self.dimension = dimension
        self._loaded_model: Optional[Any] = None
        self._loaded_backend = "uninitialized"

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load any underlying client or model implementation.
        """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text input.
        """

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """
        return [self.embed_text(text) for text in texts]

    def get_config(self) -> Dict[str, Any]:
        """
        Return provider configuration metadata.
        """
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "backend": self._loaded_backend,
        }

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise EmbeddingGenerationError(
                "Embedding input must be a string",
                ErrorCode.EMBEDDING_GENERATION_FAILED,
                self.model_name,
            )

        normalized = text.strip()
        if not normalized:
            raise EmbeddingGenerationError(
                "Embedding input cannot be empty",
                ErrorCode.EMBEDDING_GENERATION_FAILED,
                self.model_name,
            )

        return normalized

    def _mock_embedding(self, text: str) -> List[float]:
        normalized = self._validate_text(text)
        seed = f"{self.provider_name}:{self.model_name}:{normalized}".encode("utf-8")
        values: List[float] = []
        counter = 0

        while len(values) < self.dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for index in range(0, len(digest), 4):
                if len(values) >= self.dimension:
                    break
                chunk = digest[index:index + 4]
                raw_value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                values.append((raw_value * 2.0) - 1.0)
            counter += 1

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class OpenAIEmbeddingProvider(EmbeddingModel):
    """
    OpenAI embeddings provider with optional deterministic mock mode.
    """

    provider_name = "openai"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        dimension: int = 1536,
        mock_embeddings: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client
        self.mock_embeddings = mock_embeddings

    def load_model(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model

        if self.mock_embeddings:
            self._loaded_model = "mock-openai-model"
            self._loaded_backend = "mock"
            return self._loaded_model

        if self.client is not None:
            self._loaded_model = self.client
            self._loaded_backend = "client"
            return self._loaded_model

        if not self.api_key:
            raise EmbeddingGenerationError(
                "OpenAI API key is required to load embeddings",
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                self.model_name,
            )

        try:
            from openai import OpenAI
        except Exception as exc:
            raise EmbeddingGenerationError(
                "OpenAI client library is not available",
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

        self._loaded_model = OpenAI(api_key=self.api_key)
        self._loaded_backend = "openai"
        return self._loaded_model

    def embed_text(self, text: str) -> List[float]:
        normalized = self._validate_text(text)
        client = self.load_model()

        if self.mock_embeddings:
            return self._mock_embedding(normalized)

        try:
            response = client.embeddings.create(model=self.model_name, input=normalized)
            vector = list(response.data[0].embedding)
            return self._coerce_dimension(vector)
        except EmbeddingGenerationError:
            raise
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"OpenAI embedding generation failed: {exc}",
                ErrorCode.EMBEDDING_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        normalized = [self._validate_text(text) for text in texts]
        client = self.load_model()

        if self.mock_embeddings:
            return [self._mock_embedding(text) for text in normalized]

        try:
            response = client.embeddings.create(model=self.model_name, input=normalized)
            return [self._coerce_dimension(list(item.embedding)) for item in response.data]
        except EmbeddingGenerationError:
            raise
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"OpenAI batch embedding generation failed: {exc}",
                ErrorCode.EMBEDDING_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc

    def _coerce_dimension(self, vector: List[float]) -> List[float]:
        if len(vector) == self.dimension:
            return vector
        if len(vector) > self.dimension:
            return vector[:self.dimension]
        return vector + [0.0] * (self.dimension - len(vector))


class HuggingFaceEmbeddingProvider(EmbeddingModel):
    """
    Hugging Face embeddings provider with optional deterministic mock mode.
    """

    provider_name = "huggingface"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Optional[Any] = None,
        dimension: int = 384,
        mock_embeddings: bool = False,
    ) -> None:
        super().__init__(model_name=model_name, dimension=dimension)
        self.model = model
        self.mock_embeddings = mock_embeddings

    def load_model(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model

        if self.mock_embeddings:
            self._loaded_model = "mock-huggingface-model"
            self._loaded_backend = "mock"
            return self._loaded_model

        if self.model is not None:
            self._loaded_model = self.model
            self._loaded_backend = "model"
            return self._loaded_model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingGenerationError(
                "SentenceTransformer is not available",
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

        try:
            self._loaded_model = SentenceTransformer(self.model_name)
            self._loaded_backend = "sentence-transformers"
            return self._loaded_model
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Unable to load Hugging Face embedding model: {exc}",
                ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
                self.model_name,
                cause=exc,
            ) from exc

    def embed_text(self, text: str) -> List[float]:
        normalized = self._validate_text(text)
        model = self.load_model()

        if self.mock_embeddings:
            return self._mock_embedding(normalized)

        try:
            vector = model.encode(normalized)
            return self._coerce_dimension([float(value) for value in vector])
        except EmbeddingGenerationError:
            raise
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Hugging Face embedding generation failed: {exc}",
                ErrorCode.EMBEDDING_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        normalized = [self._validate_text(text) for text in texts]
        model = self.load_model()

        if self.mock_embeddings:
            return [self._mock_embedding(text) for text in normalized]

        try:
            vectors = model.encode(normalized)
            return [self._coerce_dimension([float(value) for value in vector]) for vector in vectors]
        except EmbeddingGenerationError:
            raise
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Hugging Face batch embedding generation failed: {exc}",
                ErrorCode.EMBEDDING_API_ERROR,
                self.model_name,
                cause=exc,
            ) from exc

    def _coerce_dimension(self, vector: List[float]) -> List[float]:
        if len(vector) == self.dimension:
            return vector
        if len(vector) > self.dimension:
            return vector[:self.dimension]
        return vector + [0.0] * (self.dimension - len(vector))
