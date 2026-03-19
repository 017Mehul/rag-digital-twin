"""
Vector storage and similarity search built on FAISS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from .exceptions import ErrorCode, VectorStoreError
from .models.document_chunk import DocumentChunk
from .models.embedding_metadata import EmbeddingMetadata
from .models.search_results import SearchResults


class VectorStore:
    """
    Store embeddings and metadata in a FAISS index with persistence support.
    """

    SUPPORTED_INDEX_TYPES = {"flat", "ivf", "hnsw"}

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        nlist: int = 10,
        hnsw_m: int = 32,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(f"Unsupported index_type: {index_type}")

        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.hnsw_m = hnsw_m
        self.metadata_store: List[Dict[str, Any]] = []
        self._vector_data = np.empty((0, self.dimension), dtype=np.float32)
        self.backend = "faiss" if faiss is not None else "numpy"
        self.index = self._create_index()

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add embeddings and aligned metadata to the index.
        """
        if len(embeddings) != len(metadata):
            raise VectorStoreError(
                "Embeddings and metadata must have the same length",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
            )
        if not embeddings:
            return

        vectors = self._prepare_vectors(embeddings)

        try:
            self._train_if_needed(vectors)
            if self.backend == "faiss":
                self.index.add(vectors)
            else:
                self._vector_data = np.vstack([self._vector_data, vectors])
            self.metadata_store.extend(dict(item) for item in metadata)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add embeddings to vector store: {exc}",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
                cause=exc,
            ) from exc

    def add_documents(self, entries: List[Dict[str, Any]]) -> None:
        """
        Add embedding generator outputs that include chunk and metadata objects.
        """
        embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for entry in entries:
            chunk = entry["chunk"]
            embedding = entry["embedding"]
            embedding_metadata = entry["metadata"]

            if isinstance(chunk, DocumentChunk):
                chunk_payload = chunk.to_dict()
            else:
                chunk_payload = chunk

            if isinstance(embedding_metadata, EmbeddingMetadata):
                metadata_payload = embedding_metadata.to_dict()
            else:
                metadata_payload = dict(embedding_metadata)

            metadata.append(
                {
                    "chunk": chunk_payload,
                    "embedding_metadata": metadata_payload,
                    "source_file": metadata_payload.get("source_file", chunk_payload.get("source_file", "")),
                    "chunk_id": metadata_payload.get("chunk_id", chunk_payload.get("chunk_id", "")),
                }
            )
            embeddings.append(embedding)

        self.add_embeddings(embeddings, metadata)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> SearchResults:
        """
        Search the index using cosine similarity over normalized vectors.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.is_empty():
            return SearchResults(indices=[], distances=[], metadata=[])

        query = self._prepare_vectors([query_embedding])
        limit = min(top_k, len(self.metadata_store))

        try:
            if self.backend == "faiss":
                distances, indices = self.index.search(query, limit)
            else:
                scores = np.dot(self._vector_data, query[0])
                ranked_indices = np.argsort(scores)[::-1][:limit]
                distances = np.asarray([[float(scores[index]) for index in ranked_indices]], dtype=np.float32)
                indices = np.asarray([[int(index) for index in ranked_indices]], dtype=np.int64)
        except Exception as exc:
            raise VectorStoreError(
                f"Vector similarity search failed: {exc}",
                ErrorCode.VECTOR_STORE_SEARCH_FAILED,
                self.index_type,
                cause=exc,
            ) from exc

        result_indices: List[int] = []
        result_distances: List[float] = []
        result_metadata: List[Dict[str, Any]] = []

        for index, distance in zip(indices[0].tolist(), distances[0].tolist()):
            if index < 0:
                continue
            result_indices.append(index)
            result_distances.append(float(distance))
            result_metadata.append(dict(self.metadata_store[index]))

        return SearchResults(
            indices=result_indices,
            distances=result_distances,
            metadata=result_metadata,
        )

    def save(self, directory: str) -> Dict[str, str]:
        """
        Persist the FAISS index and metadata to disk.
        """
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / "vector_store.faiss"
        vectors_path = output_dir / "vector_store.npy"
        metadata_path = output_dir / "vector_store_metadata.json"

        try:
            if self.backend == "faiss":
                faiss.write_index(self.index, str(index_path))
            else:
                np.save(vectors_path, self._vector_data)
            metadata_payload = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "hnsw_m": self.hnsw_m,
                "metadata_store": self.metadata_store,
                "vector_count": len(self.metadata_store),
                "backend": self.backend,
            }
            metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to persist vector store: {exc}",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
                cause=exc,
            ) from exc

        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
        }

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a persisted vector store from disk and validate it.
        """
        input_dir = Path(directory)
        index_path = input_dir / "vector_store.faiss"
        vectors_path = input_dir / "vector_store.npy"
        metadata_path = input_dir / "vector_store_metadata.json"

        if not metadata_path.exists():
            raise VectorStoreError(
                "Persisted vector store files are missing",
                ErrorCode.VECTOR_STORE_LOAD_FAILED,
                "unknown",
            )

        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            store = cls(
                dimension=metadata_payload["dimension"],
                index_type=metadata_payload["index_type"],
                nlist=metadata_payload.get("nlist", 10),
                hnsw_m=metadata_payload.get("hnsw_m", 32),
            )
            expected_backend = metadata_payload.get("backend", "faiss")
            if expected_backend == "faiss":
                if faiss is None or not index_path.exists():
                    raise VectorStoreError(
                        "Persisted FAISS index cannot be loaded in the current environment",
                        ErrorCode.VECTOR_STORE_LOAD_FAILED,
                        metadata_payload["index_type"],
                    )
                store.index = faiss.read_index(str(index_path))
            else:
                if not vectors_path.exists():
                    raise VectorStoreError(
                        "Persisted vector matrix is missing",
                        ErrorCode.VECTOR_STORE_LOAD_FAILED,
                        metadata_payload["index_type"],
                    )
                store._vector_data = np.load(vectors_path).astype(np.float32)
            store.metadata_store = metadata_payload["metadata_store"]
            store.validate()
            return store
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load vector store: {exc}",
                ErrorCode.VECTOR_STORE_LOAD_FAILED,
                "unknown",
                cause=exc,
            ) from exc

    def validate(self) -> bool:
        """
        Validate the loaded store for corruption and metadata consistency.
        """
        try:
            index_count = int(self.index.ntotal) if self.backend == "faiss" else int(len(self._vector_data))
        except Exception as exc:
            raise VectorStoreError(
                f"Unable to inspect vector store index: {exc}",
                ErrorCode.VECTOR_STORE_INDEX_CORRUPTED,
                self.index_type,
                cause=exc,
            ) from exc

        if index_count != len(self.metadata_store):
            raise VectorStoreError(
                "Vector store metadata count does not match index size",
                ErrorCode.VECTOR_STORE_INDEX_CORRUPTED,
                self.index_type,
            )

        for item in self.metadata_store:
            if not isinstance(item, dict):
                raise VectorStoreError(
                    "Vector store metadata contains invalid entries",
                    ErrorCode.VECTOR_STORE_INDEX_CORRUPTED,
                    self.index_type,
                )

        return True

    def is_empty(self) -> bool:
        return len(self.metadata_store) == 0

    def __len__(self) -> int:
        return len(self.metadata_store)

    def _create_index(self):
        if faiss is None:
            return None

        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)

        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                max(1, self.nlist),
                faiss.METRIC_INNER_PRODUCT,
            )

        if self.index_type == "hnsw":
            try:
                return faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            except TypeError:
                index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
                return index

        raise VectorStoreError(
            f"Unsupported vector store index type: {self.index_type}",
            ErrorCode.VECTOR_STORE_SAVE_FAILED,
            self.index_type,
        )

    def _train_if_needed(self, vectors: np.ndarray) -> None:
        if self.backend != "faiss":
            return

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            try:
                self.index.train(vectors)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to train FAISS index: {exc}",
                    ErrorCode.VECTOR_STORE_SAVE_FAILED,
                    self.index_type,
                    cause=exc,
                ) from exc

    def _prepare_vectors(self, embeddings: List[List[float]]) -> np.ndarray:
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            raise VectorStoreError(
                f"Invalid embedding matrix: {exc}",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
                cause=exc,
            ) from exc

        if vectors.ndim != 2:
            raise VectorStoreError(
                "Embeddings must be a 2D matrix",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
            )
        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}",
                ErrorCode.VECTOR_STORE_SAVE_FAILED,
                self.index_type,
            )

        if faiss is not None:
            faiss.normalize_L2(vectors)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms
        return vectors
