"""
Durable Pinecone-backed vector storage for serverless deployments.

The class intentionally mirrors the small interface used by RAGPipeline and
QueryProcessor so local development can continue using FAISS unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List\nfrom pathlib import Path

from .exceptions import ErrorCode, VectorStoreError
from .models.document_chunk import DocumentChunk
from .models.embedding_metadata import EmbeddingMetadata
from .models.search_results import SearchResults


class PineconeVectorStore:
    """Shared durable vector store backed by a Pinecone serverless index."""

    backend = "pinecone"

    def __init__(
        self,
        dimension: int,
        index_host: str | None = None,
        api_key: str | None = None,
        namespace: str | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self.index_type = "serverless"
        self.nlist = 0
        self.hnsw_m = 0
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "default")
        self.index_host = (index_host or os.getenv("PINECONE_INDEX_HOST", "")).strip()
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "")

        if not self.api_key or not self.index_host:
            raise VectorStoreError(
                "Pinecone requires PINECONE_API_KEY and PINECONE_INDEX_HOST",
                ErrorCode.VECTOR_STORE_LOAD_FAILED,
                "pinecone",
            )

        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise VectorStoreError(
                "Pinecone support requires the 'pinecone' package",
                ErrorCode.VECTOR_STORE_LOAD_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

        try:
            self._index = Pinecone(api_key=self.api_key).Index(host=self.index_host)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize Pinecone index",
                ErrorCode.VECTOR_STORE_LOAD_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        if len(embeddings) != len(metadata):
            raise VectorStoreError(
                "Embeddings and metadata must have the same length",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                "pinecone",
            )
        if not embeddings:
            return

        records = []
        for embedding, item in zip(embeddings, metadata):
            if len(embedding) != self.dimension:
                raise VectorStoreError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}",
                    ErrorCode.VECTOR_STORE_SAVE_FAILED,
                    "pinecone",
                )
            payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            record_id = self._record_id(item)
            records.append(
                {
                    "id": record_id,
                    "values": [float(value) for value in embedding],
                    "metadata": {
                        "payload": payload,
                        "document_id": str(item.get("document_id", "")),
                        "source_file": str(item.get("source_file", "")),
                        "chunk_index": int(item.get("chunk", {}).get("metadata", {}).get("chunk_index", 0)),
                    },
                }
            )

        try:
            # Pinecone's current data-plane SDK handles request batching internally
            # for this small ingestion path; chunking keeps requests comfortably sized.
            for start in range(0, len(records), 100):
                self._index.upsert(
                    vectors=records[start : start + 100],
                    namespace=self.namespace,
                )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to upsert embeddings to Pinecone",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    def add_documents(self, entries: List[Dict[str, Any]]) -> None:
        embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for entry in entries:
            chunk = entry["chunk"]
            embedding = entry["embedding"]
            embedding_metadata = entry["metadata"]

            chunk_payload = chunk.to_dict() if isinstance(chunk, DocumentChunk) else dict(chunk)
            if isinstance(embedding_metadata, EmbeddingMetadata):
                metadata_payload = embedding_metadata.to_dict()
            else:
                metadata_payload = dict(embedding_metadata)

            metadata.append(
                {
                    "chunk": chunk_payload,
                    "embedding_metadata": metadata_payload,
                    "source_file": metadata_payload.get(
                        "source_file", chunk_payload.get("source_file", "")
                    ),
                    "chunk_id": metadata_payload.get(
                        "chunk_id", chunk_payload.get("chunk_id", "")
                    ),
                    "document_id": metadata_payload.get(
                        "document_id", chunk_payload.get("metadata", {}).get("document_id", "")
                    ),
                }
            )
            embeddings.append(embedding)

        self.add_embeddings(embeddings, metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> SearchResults:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(query_embedding) != self.dimension:
            raise VectorStoreError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {len(query_embedding)}",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                "pinecone",
            )

        try:
            response = self._index.query(
                vector=[float(value) for value in query_embedding],
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
            )
            matches = getattr(response, "matches", None)
            if matches is None and isinstance(response, dict):
                matches = response.get("matches", [])
            matches = matches or []
        except Exception as exc:
            raise VectorStoreError(
                "Vector similarity search failed against Pinecone",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

        result_indices: List[int] = []
        result_distances: List[float] = []
        result_metadata: List[Dict[str, Any]] = []

        for rank, match in enumerate(matches):
            raw_metadata = getattr(match, "metadata", None)
            score = getattr(match, "score", None)
            if isinstance(match, dict):
                raw_metadata = match.get("metadata", raw_metadata)
                score = match.get("score", score)

            if not isinstance(raw_metadata, dict):
                continue
            payload = raw_metadata.get("payload")
            try:
                item = json.loads(payload) if isinstance(payload, str) else dict(raw_metadata)
            except (TypeError, ValueError):
                item = dict(raw_metadata)

            result_indices.append(rank)
            result_distances.append(float(score or 0.0))
            result_metadata.append(item)

        return SearchResults(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata,
        )

    def delete_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a logical document."""
        if not document_id:
            return
        try:
            self._index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=self.namespace,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to delete document from Pinecone",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return document summaries using Pinecone IDs and stored metadata."""
        try:
            ids = []
            for page in self._index.list(namespace=self.namespace):
                page_ids = getattr(page, "ids", None)
                if page_ids is None and isinstance(page, dict):
                    page_ids = page.get("ids", [])
                ids.extend(page_ids or [])
            documents: Dict[str, Dict[str, Any]] = {}
            for start in range(0, len(ids), 100):
                response = self._index.fetch(ids=ids[start:start + 100], namespace=self.namespace)
                vectors = getattr(response, "vectors", None)
                if vectors is None and isinstance(response, dict):
                    vectors = response.get("vectors", {})
                for item in (vectors or {}).values():
                    metadata = getattr(item, "metadata", None)
                    if metadata is None and isinstance(item, dict):
                        metadata = item.get("metadata", {})
                    metadata = metadata or {}
                    document_id = str(metadata.get("document_id", ""))
                    source_file = str(metadata.get("source_file", "Unknown document"))
                    key = document_id or source_file
                    row = documents.setdefault(key, {
                        "name": Path(source_file).name,
                        "path": source_file,
                        "type": Path(source_file).suffix.lstrip(".").upper() or "OTHER",
                        "chunks": 0,
                        "status": "Indexed",
                    })
                    row["chunks"] += 1
            return list(documents.values())
        except Exception as exc:
            raise VectorStoreError(
                "Unable to list Pinecone documents",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    def get_stats(self) -> Dict[str, Any]:
        stats = self._index.describe_index_stats(namespace=self.namespace)
        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces", {})
        current = (namespaces or {}).get(self.namespace, {})
        count = getattr(current, "vector_count", None)
        if count is None and isinstance(current, dict):
            count = current.get("vector_count", 0)
        return {"backend": self.backend, "namespace": self.namespace, "vectors": int(count or 0)}

    def save(self, directory: str) -> Dict[str, str]:
        """No-op for compatibility: Pinecone persists vectors remotely."""
        return {
            "backend": "pinecone",
            "namespace": self.namespace,
            "index_host": self.index_host,
        }

    def validate(self) -> bool:
        try:
            self._index.describe_index_stats(namespace=self.namespace)
        except Exception as exc:
            raise VectorStoreError(
                "Unable to validate Pinecone index",
                ErrorCode.VECTOR_STORE_INDEX_CORRUPTED,
                "pinecone",
                cause=exc,
            ) from exc
        return True

    def is_empty(self) -> bool:
        try:
            stats = self._index.describe_index_stats(namespace=self.namespace)
            namespace_stats = getattr(stats, "namespaces", None)
            if namespace_stats is None and isinstance(stats, dict):
                namespace_stats = stats.get("namespaces", {})
            namespace_stats = namespace_stats or {}
            current = namespace_stats.get(self.namespace, {})
            count = getattr(current, "vector_count", None)
            if count is None and isinstance(current, dict):
                count = current.get("vector_count", 0)
            return int(count or 0) == 0
        except Exception as exc:
            raise VectorStoreError(
                "Unable to inspect Pinecone vector count",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    def __len__(self) -> int:
        try:
            stats = self._index.describe_index_stats(namespace=self.namespace)
            namespace_stats = getattr(stats, "namespaces", None)
            if namespace_stats is None and isinstance(stats, dict):
                namespace_stats = stats.get("namespaces", {})
            namespace_stats = namespace_stats or {}
            current = namespace_stats.get(self.namespace, {})
            count = getattr(current, "vector_count", None)
            if count is None and isinstance(current, dict):
                count = current.get("vector_count", 0)
            return int(count or 0)
        except Exception as exc:
            raise VectorStoreError(
                "Unable to inspect Pinecone vector count",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                "pinecone",
                cause=exc,
            ) from exc

    @staticmethod
    def _record_id(metadata: Dict[str, Any]) -> str:
        chunk_id = str(metadata.get("chunk_id", ""))
        source_file = str(metadata.get("source_file", ""))
        seed = f"{source_file}\0{chunk_id}\0{json.dumps(metadata, sort_keys=True, default=str)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()
