"""
Embedding generation orchestration with provider abstraction, caching, and retries.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from .exceptions import EmbeddingGenerationError, ErrorCode
from .models.document_chunk import DocumentChunk
from .models.embedding_metadata import EmbeddingMetadata
from .models.rag_config import RAGConfig
from .providers import (
    EmbeddingModel,
    ProviderFactory,
)


ProgressCallback = Callable[[int, int], None]


class EmbeddingGenerator:
    """
    Generate embeddings for text and document chunks using configurable providers.
    """

    def __init__(
        self,
        provider: EmbeddingModel,
        batch_size: int = 10,
        max_retries: int = 3,
        cache_size: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")

        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache_size = cache_size
        self._embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()

    @classmethod
    def from_config(
        cls,
        config: RAGConfig,
        provider: Optional[EmbeddingModel] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        cache_size: int = 1000,
    ) -> "EmbeddingGenerator":
        if provider is None:
            merged_provider_config = dict(config.embedding_provider_config)
            merged_provider_config.update(provider_kwargs or {})
            provider = ProviderFactory.create_embedding_provider(
                provider_name=config.embedding_provider,
                model_name=config.embedding_model,
                provider_config=merged_provider_config,
                fallback_providers=config.embedding_fallbacks,
            )

        return cls(
            provider=provider,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            cache_size=cache_size,
        )

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Return active provider metadata.
        """
        return self.provider.get_config()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text with caching and retries.
        """
        if text in self._embedding_cache:
            self._embedding_cache.move_to_end(text)
            return list(self._embedding_cache[text])

        vector = self._with_retries(
            operation=lambda: self.provider.embed_text(text),
            failure_context="single embedding generation",
        )
        self._store_in_cache(text, vector)
        return list(vector)

    def generate_embeddings(
        self,
        texts: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        """
        results: List[List[float]] = []
        total = len(texts)
        processed = 0

        for batch_start in range(0, total, self.batch_size):
            batch = texts[batch_start:batch_start + self.batch_size]
            batch_results = self._generate_batch(batch)
            results.extend(batch_results)
            processed += len(batch)

            if progress_callback:
                progress_callback(processed, total)

        return results

    def generate_chunk_embedding(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Generate embedding output for a single document chunk.
        """
        embedding = self.generate_embedding(chunk.content)
        metadata = EmbeddingMetadata.from_document_chunk(chunk, self.provider.model_name)
        return {
            "chunk": chunk,
            "embedding": embedding,
            "metadata": metadata,
        }

    def generate_chunk_embeddings(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings and metadata for a list of chunks.
        """
        embeddings = self.generate_embeddings(
            [chunk.content for chunk in chunks],
            progress_callback=progress_callback,
        )
        return [
            {
                "chunk": chunk,
                "embedding": embedding,
                "metadata": EmbeddingMetadata.from_document_chunk(chunk, self.provider.model_name),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results: List[Optional[List[float]]] = [None] * len(texts)

        for index, text in enumerate(texts):
            if text in self._embedding_cache:
                self._embedding_cache.move_to_end(text)
                results[index] = list(self._embedding_cache[text])
            else:
                uncached_indices.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            generated_vectors = self._with_retries(
                operation=lambda: self.provider.embed_batch(uncached_texts),
                failure_context=f"batch embedding generation for {len(uncached_texts)} texts",
            )
            for index, text, vector in zip(uncached_indices, uncached_texts, generated_vectors):
                self._store_in_cache(text, vector)
                results[index] = list(vector)

        return [vector for vector in results if vector is not None]

    def _with_retries(
        self,
        operation: Callable[[], Any],
        failure_context: str,
    ) -> Any:
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except EmbeddingGenerationError as exc:
                last_error = exc
                if attempt == attempts:
                    raise
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    raise EmbeddingGenerationError(
                        f"Failed during {failure_context} after {attempts} attempts: {exc}",
                        ErrorCode.EMBEDDING_GENERATION_FAILED,
                        self.provider.model_name,
                        cause=exc,
                    ) from exc

        raise EmbeddingGenerationError(
            f"Failed during {failure_context}: {last_error}",
            ErrorCode.EMBEDDING_GENERATION_FAILED,
            self.provider.model_name,
            cause=last_error,
        )

    def _store_in_cache(self, text: str, vector: List[float]) -> None:
        if self.cache_size == 0:
            return

        self._embedding_cache[text] = list(vector)
        self._embedding_cache.move_to_end(text)
        while len(self._embedding_cache) > self.cache_size:
            self._embedding_cache.popitem(last=False)
