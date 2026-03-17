"""
Tests for embedding generation and provider abstraction.
"""

from typing import Any, List

import pytest
from hypothesis import given, strategies as st

from src.embedding_generator import EmbeddingGenerator
from src.exceptions import EmbeddingGenerationError, ErrorCode
from src.models.document_chunk import DocumentChunk
from src.models.rag_config import RAGConfig
from src.providers import EmbeddingModel, HuggingFaceEmbeddingProvider, OpenAIEmbeddingProvider


class FlakyEmbeddingProvider(EmbeddingModel):
    provider_name = "flaky"

    def __init__(self, fail_times: int = 1, dimension: int = 8) -> None:
        super().__init__(model_name="flaky-model", dimension=dimension)
        self.fail_times = fail_times
        self.calls = 0

    def load_model(self) -> Any:
        self._loaded_backend = "test"
        return self

    def embed_text(self, text: str) -> List[float]:
        self.calls += 1
        self._validate_text(text)
        if self.calls <= self.fail_times:
            raise RuntimeError("temporary embedding service failure")
        return self._mock_embedding(text)


class AlwaysFailEmbeddingProvider(EmbeddingModel):
    provider_name = "failing"

    def __init__(self) -> None:
        super().__init__(model_name="failing-model", dimension=6)

    def load_model(self) -> Any:
        self._loaded_backend = "test"
        return self

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        raise RuntimeError("service unavailable")


class TestEmbeddingGenerator:
    def test_create_generator_from_config_openai(self):
        config = RAGConfig(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            batch_size=4,
            max_retries=2,
        )

        generator = EmbeddingGenerator.from_config(
            config,
            provider_kwargs={"mock_embeddings": True, "dimension": 24},
        )

        info = generator.get_provider_info()

        assert info["provider"] == "openai"
        assert info["model_name"] == "text-embedding-3-small"
        assert generator.batch_size == 4
        assert generator.max_retries == 2

    def test_chunk_embedding_preserves_metadata(self, sample_document_chunk):
        provider = OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            dimension=16,
            mock_embeddings=True,
        )
        generator = EmbeddingGenerator(provider=provider)

        result = generator.generate_chunk_embedding(sample_document_chunk)

        assert len(result["embedding"]) == 16
        assert result["chunk"].chunk_id == sample_document_chunk.chunk_id
        assert result["metadata"].chunk_id == sample_document_chunk.chunk_id
        assert result["metadata"].source_file == sample_document_chunk.source_file
        assert result["metadata"].embedding_model == "text-embedding-3-small"

    def test_batch_generation_reports_progress(self):
        provider = HuggingFaceEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=12,
            mock_embeddings=True,
        )
        generator = EmbeddingGenerator(provider=provider, batch_size=2)
        progress_updates = []

        vectors = generator.generate_embeddings(
            ["alpha", "beta", "gamma"],
            progress_callback=lambda processed, total: progress_updates.append((processed, total)),
        )

        assert len(vectors) == 3
        assert progress_updates == [(2, 3), (3, 3)]

    def test_retry_logic_recovers_from_transient_failure(self):
        provider = FlakyEmbeddingProvider(fail_times=1, dimension=10)
        generator = EmbeddingGenerator(provider=provider, max_retries=2)

        vector = generator.generate_embedding("retry me")

        assert len(vector) == 10
        assert provider.calls == 2


class TestEmbeddingGeneratorProperties:
    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda value: value.strip()),
            min_size=1,
            max_size=10,
        )
    )
    def test_embedding_dimension_consistency(self, texts):
        """
        Property 6: Embeddings should always match the configured dimension.
        """
        generator = EmbeddingGenerator(
            provider=OpenAIEmbeddingProvider(
                model_name="text-embedding-3-small",
                dimension=20,
                mock_embeddings=True,
            ),
            batch_size=3,
        )

        vectors = generator.generate_embeddings(texts)

        assert len(vectors) == len(texts)
        assert all(len(vector) == 20 for vector in vectors)

    @given(text=st.text(min_size=1, max_size=200).filter(lambda value: value.strip()))
    def test_embedding_determinism(self, text):
        """
        Property 7: Same text with the same model should produce the same embedding.
        """
        generator = EmbeddingGenerator(
            provider=HuggingFaceEmbeddingProvider(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=18,
                mock_embeddings=True,
            )
        )

        first = generator.generate_embedding(text)
        second = generator.generate_embedding(text)

        assert first == second

    @given(text=st.text(min_size=1, max_size=200).filter(lambda value: value.strip()))
    def test_descriptive_error_handling(self, text):
        """
        Property 8: Failures should surface descriptive embedding errors.
        """
        generator = EmbeddingGenerator(
            provider=AlwaysFailEmbeddingProvider(),
            max_retries=1,
        )

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            generator.generate_embedding(text)

        error = exc_info.value
        assert error.error_code == ErrorCode.EMBEDDING_GENERATION_FAILED
        assert error.details["model_name"] == "failing-model"
        assert "single embedding generation" in error.message
        assert "service unavailable" in error.message
