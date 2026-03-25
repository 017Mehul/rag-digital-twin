"""
Tests for centralized provider factory configuration and fallback behavior.
"""

from __future__ import annotations

from typing import Any, List

import pytest
from hypothesis import given, strategies as st

from src.embedding_generator import EmbeddingGenerator
from src.exceptions import ConfigurationError, EmbeddingGenerationError, ErrorCode, ResponseGenerationError
from src.models.rag_config import RAGConfig
from src.models.search_results import RetrievedContext
from src.providers import EmbeddingModel, LLMProvider, ProviderFactory
from src.response_generator import ResponseGenerator


class BrokenEmbeddingProvider(EmbeddingModel):
    provider_name = "broken-embedding"

    def __init__(self, model_name: str = "broken-embedding-model", dimension: int = 16) -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def load_model(self) -> Any:
        raise EmbeddingGenerationError(
            "Primary embedding backend is unavailable",
            ErrorCode.EMBEDDING_API_ERROR,
            self.model_name,
        )

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        raise EmbeddingGenerationError(
            "Primary embedding backend is unavailable",
            ErrorCode.EMBEDDING_API_ERROR,
            self.model_name,
        )


class BrokenLLMProvider(LLMProvider):
    provider_name = "broken-llm"

    def __init__(self, model_name: str = "broken-llm-model") -> None:
        super().__init__(model_name=model_name)

    def load_model(self) -> Any:
        raise ResponseGenerationError(
            "Primary LLM backend is unavailable",
            ErrorCode.LLM_API_ERROR,
            self.model_name,
        )

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        self._validate_text(prompt)
        raise ResponseGenerationError(
            "Primary LLM backend is unavailable",
            ErrorCode.LLM_API_ERROR,
            self.model_name,
        )


class EchoEmbeddingProvider(EmbeddingModel):
    provider_name = "echo-embedding"

    def __init__(self, model_name: str = "echo-embedding-model", dimension: int = 12) -> None:
        super().__init__(model_name=model_name, dimension=dimension)

    def load_model(self) -> Any:
        self._loaded_backend = "test"
        return self

    def embed_text(self, text: str) -> List[float]:
        normalized = self._validate_text(text)
        return [float((len(normalized) + index) % 7) for index in range(self.dimension)]


def _retrieved_context(body: str) -> RetrievedContext:
    return RetrievedContext(
        formatted_context=f"[Source 1] doc.txt\n{body}",
        source_chunks=[],
        total_tokens=max(1, len(body.split())),
        sources=["doc.txt"],
        insufficient_context=False,
    )


class TestProviderFactory:
    def test_register_custom_embedding_provider_supports_dynamic_loading(self):
        ProviderFactory.register_embedding_provider(
            provider_name="echo-embedding",
            provider_class=EchoEmbeddingProvider,
            default_model_name="echo-embedding-model",
            supported_kwargs={"dimension"},
        )

        provider = ProviderFactory.create_embedding_provider(
            provider_name="echo-embedding",
            model_name="echo-embedding-model",
            provider_config={"dimension": 10},
        )

        vector = provider.embed_text("dynamic loading works")

        assert len(vector) == 10
        assert provider.get_config()["provider"] == "echo-embedding"

    def test_invalid_provider_specific_config_raises_configuration_error(self):
        with pytest.raises(ConfigurationError) as exc_info:
            ProviderFactory.create_embedding_provider(
                provider_name="openai",
                model_name="text-embedding-3-small",
                provider_config={"unsupported_flag": True},
            )

        assert exc_info.value.error_code == ErrorCode.CONFIG_VALIDATION_FAILED
        assert exc_info.value.details["config_key"] == "embedding_provider_config"

    def test_embedding_factory_falls_back_when_primary_provider_fails(self):
        ProviderFactory.register_embedding_provider(
            provider_name="broken-embedding",
            provider_class=BrokenEmbeddingProvider,
            default_model_name="broken-embedding-model",
            supported_kwargs={"dimension"},
        )

        config = RAGConfig(
            embedding_provider="broken-embedding",
            embedding_model="broken-embedding-model",
            embedding_provider_config={"dimension": 16},
            embedding_fallbacks=[
                {
                    "provider": "huggingface",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "config": {"mock_embeddings": True, "dimension": 16},
                }
            ],
        )

        generator = EmbeddingGenerator.from_config(config)
        vector = generator.generate_embedding("fallback embeddings")
        provider_info = generator.get_provider_info()

        assert len(vector) == 16
        assert provider_info["provider"] == "huggingface"
        assert provider_info["fallback_enabled"] is True

    def test_llm_factory_falls_back_when_primary_provider_fails(self):
        ProviderFactory.register_llm_provider(
            provider_name="broken-llm",
            provider_class=BrokenLLMProvider,
            default_model_name="broken-llm-model",
            supported_kwargs=set(),
        )

        config = RAGConfig(
            llm_provider="broken-llm",
            llm_model="broken-llm-model",
            llm_fallbacks=[
                {
                    "provider": "huggingface",
                    "model_name": "distilgpt2",
                    "config": {"mock_responses": True},
                }
            ],
        )

        generator = ResponseGenerator.from_config(config)
        response = generator.generate_response(
            "What happened?",
            _retrieved_context("Fallback LLMs should preserve the same interface contract."),
        )
        provider_info = generator.get_provider_info()

        assert response.context_used is True
        assert response.sources == ["doc.txt"]
        assert provider_info["provider"] == "huggingface"
        assert provider_info["fallback_enabled"] is True


class TestProviderFactoryProperties:
    @given(
        embedding_provider=st.sampled_from(["openai", "huggingface"]),
        llm_provider=st.sampled_from(["openai", "huggingface"]),
        text=st.text(min_size=1, max_size=120).filter(lambda value: value.strip()),
        question=st.text(min_size=1, max_size=80).filter(lambda value: value.strip()),
    )
    def test_provider_abstraction(self, embedding_provider, llm_provider, text, question):
        """
        Property 24: Consumers should work uniformly across supported provider implementations.
        """
        embedding_model = (
            "text-embedding-3-small"
            if embedding_provider == "openai"
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        llm_model = "gpt-4o-mini" if llm_provider == "openai" else "distilgpt2"

        config = RAGConfig(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_provider_config={"mock_embeddings": True, "dimension": 14},
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_provider_config={"mock_responses": True},
        )

        embedding_generator = EmbeddingGenerator.from_config(config)
        response_generator = ResponseGenerator.from_config(config)

        vector = embedding_generator.generate_embedding(text)
        response = response_generator.generate_response(question, _retrieved_context(text))

        assert len(vector) == 14
        assert embedding_generator.get_provider_info()["provider"] == embedding_provider
        assert response_generator.get_provider_info()["provider"] == llm_provider
        assert response.context_used is True
        assert response.sources == ["doc.txt"]
