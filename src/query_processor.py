"""
Query processing and similarity search orchestration for the RAG system.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .embedding_generator import EmbeddingGenerator
from .exceptions import ErrorCode, QueryProcessingError
from .models.document_chunk import DocumentChunk
from .models.rag_config import RAGConfig
from .models.search_results import QueryResults, SearchResults
from .vector_store import VectorStore


class QueryProcessor:
    """
    Process user queries into embeddings and ranked retrieval results.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        top_k_results: int = 5,
        similarity_threshold: float = 0.7,
        max_query_length: int = 4000,
        insufficient_context_message: str = "Insufficient context to answer this query reliably.",
    ) -> None:
        if top_k_results <= 0:
            raise ValueError("top_k_results must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_query_length <= 0:
            raise ValueError("max_query_length must be positive")

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k_results = top_k_results
        self.similarity_threshold = similarity_threshold
        self.max_query_length = max_query_length
        self.insufficient_context_message = insufficient_context_message

    @classmethod
    def from_config(
        cls,
        config: RAGConfig,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        max_query_length: int = 4000,
        insufficient_context_message: str = "Insufficient context to answer this query reliably.",
    ) -> "QueryProcessor":
        return cls(
            embedding_generator=embedding_generator,
            vector_store=vector_store,
            top_k_results=config.top_k_results,
            similarity_threshold=config.similarity_threshold,
            max_query_length=max_query_length,
            insufficient_context_message=insufficient_context_message,
        )

    def process_query(
        self,
        query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> QueryResults:
        """
        Embed a query, search the vector store, and return ranked results.
        """
        normalized_query = self._validate_query(query)
        limit = self._resolve_top_k(k)
        active_threshold = self._resolve_threshold(threshold)

        try:
            query_embedding = self.embed_query(normalized_query)
            search_results = self.vector_store.search(query_embedding, top_k=limit)
            filtered_results = self.filter_results(search_results, active_threshold)
        except QueryProcessingError:
            raise
        except Exception as exc:
            raise QueryProcessingError(
                f"Failed to process query: {exc}",
                ErrorCode.QUERY_PROCESSING_FAILED,
                normalized_query,
                cause=exc,
            ) from exc

        retrieved_chunks = [self._metadata_to_chunk(item) for item in filtered_results.metadata]
        relevance_scores = [self._calculate_relevance_score(distance) for distance in filtered_results.distances]
        message = "" if retrieved_chunks else self.insufficient_context_message

        return QueryResults(
            query=normalized_query,
            retrieved_chunks=retrieved_chunks,
            relevance_scores=relevance_scores,
            total_results=len(retrieved_chunks),
            message=message,
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a user query using the active embedding model.
        """
        normalized_query = self._validate_query(query)

        try:
            return self.embedding_generator.generate_embedding(normalized_query)
        except Exception as exc:
            raise QueryProcessingError(
                f"Failed to embed query: {exc}",
                ErrorCode.QUERY_PROCESSING_FAILED,
                normalized_query,
                cause=exc,
            ) from exc

    def filter_results(
        self,
        results: SearchResults,
        threshold: Optional[float] = None,
    ) -> SearchResults:
        """
        Filter ranked similarity results by the configured relevance threshold.
        """
        active_threshold = self._resolve_threshold(threshold)
        filtered_indices: List[int] = []
        filtered_distances: List[float] = []
        filtered_metadata: List[Dict[str, Any]] = []

        for index, distance, metadata in zip(results.indices, results.distances, results.metadata):
            relevance_score = self._calculate_relevance_score(distance)
            if relevance_score >= active_threshold:
                filtered_indices.append(index)
                filtered_distances.append(relevance_score)
                filtered_metadata.append(metadata)

        return SearchResults(
            indices=filtered_indices,
            distances=filtered_distances,
            metadata=filtered_metadata,
        )

    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise QueryProcessingError(
                "Query must be a string",
                ErrorCode.QUERY_PROCESSING_FAILED,
                str(query),
            )

        normalized_query = query.strip()
        if not normalized_query:
            raise QueryProcessingError(
                "Query cannot be empty",
                ErrorCode.QUERY_EMPTY,
                query,
            )
        if len(normalized_query) > self.max_query_length:
            raise QueryProcessingError(
                f"Query exceeds maximum supported length of {self.max_query_length} characters",
                ErrorCode.QUERY_TOO_LONG,
                normalized_query,
            )

        return normalized_query

    def _resolve_top_k(self, value: Optional[int]) -> int:
        resolved = self.top_k_results if value is None else value
        if resolved <= 0:
            raise QueryProcessingError(
                "top-k result limit must be positive",
                ErrorCode.QUERY_PROCESSING_FAILED,
            )
        return resolved

    def _resolve_threshold(self, value: Optional[float]) -> float:
        resolved = self.similarity_threshold if value is None else value
        if not 0.0 <= resolved <= 1.0:
            raise QueryProcessingError(
                "Similarity threshold must be between 0.0 and 1.0",
                ErrorCode.QUERY_PROCESSING_FAILED,
            )
        return float(resolved)

    @staticmethod
    def _calculate_relevance_score(distance: float) -> float:
        return max(-1.0, min(1.0, float(distance)))

    @staticmethod
    def _metadata_to_chunk(metadata: Dict[str, Any]) -> DocumentChunk:
        chunk_payload = metadata.get("chunk")
        if isinstance(chunk_payload, dict):
            return DocumentChunk.from_dict(chunk_payload)

        embedding_metadata = metadata.get("embedding_metadata", {})
        if not isinstance(embedding_metadata, dict):
            embedding_metadata = {}

        source_file = (
            metadata.get("source_file")
            or metadata.get("source")
            or embedding_metadata.get("source_file")
            or "<vector-store>"
        )
        content = (
            metadata.get("content")
            or metadata.get("text")
            or embedding_metadata.get("content_preview")
            or json.dumps(metadata, sort_keys=True)
        )

        return DocumentChunk(
            content=str(content).strip(),
            metadata=dict(metadata),
            source_file=str(source_file),
            chunk_id=str(
                metadata.get("chunk_id")
                or embedding_metadata.get("chunk_id")
                or f"query-result-{datetime.now().timestamp()}"
            ),
        )
