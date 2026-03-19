"""
Tests for FAISS-backed vector storage.
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from src.embedding_generator import EmbeddingGenerator
from src.exceptions import ErrorCode, VectorStoreError
from src.models.document_chunk import DocumentChunk
from src.providers import OpenAIEmbeddingProvider
from src.vector_store import VectorStore


def _embedding_generator(dimension: int = 16) -> EmbeddingGenerator:
    return EmbeddingGenerator(
        provider=OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            dimension=dimension,
            mock_embeddings=True,
        )
    )


class TestVectorStore:
    def test_supports_multiple_faiss_index_types(self):
        for index_type in ("flat", "ivf", "hnsw"):
            store = VectorStore(dimension=8, index_type=index_type)
            assert store.index_type == index_type

    def test_similarity_search_returns_metadata(self):
        generator = _embedding_generator(dimension=12)
        store = VectorStore(dimension=12, index_type="flat")
        texts = ["alpha signal", "beta signal", "gamma context"]
        metadata = [{"label": text} for text in texts]
        embeddings = generator.generate_embeddings(texts)

        store.add_embeddings(embeddings, metadata)
        results = store.search(generator.generate_embedding("alpha signal"), top_k=2)

        assert len(results) == 2
        assert results.metadata[0]["label"] == "alpha signal"
        assert results.distances[0] >= results.distances[1]

    def test_load_detects_corrupted_metadata(self, temp_directory):
        generator = _embedding_generator(dimension=10)
        store = VectorStore(dimension=10, index_type="flat")
        embeddings = generator.generate_embeddings(["alpha", "beta"])
        store.add_embeddings(embeddings, [{"id": 1}, {"id": 2}])
        store.save(temp_directory)

        metadata_path = Path(temp_directory) / "vector_store_metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["metadata_store"] = payload["metadata_store"][:-1]
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(VectorStoreError) as exc_info:
            VectorStore.load(temp_directory)

        assert exc_info.value.error_code == ErrorCode.VECTOR_STORE_INDEX_CORRUPTED


class TestVectorStoreProperties:
    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=100).filter(lambda value: value.strip()),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    def test_storage_and_retrieval_round_trip(self, texts):
        """
        Property 9: Stored vectors should be retrievable with aligned metadata.
        """
        generator = _embedding_generator(dimension=14)
        store = VectorStore(dimension=14, index_type="flat")
        embeddings = generator.generate_embeddings(texts)
        metadata = [{"text": text, "position": index} for index, text in enumerate(texts)]

        store.add_embeddings(embeddings, metadata)
        results = store.search(generator.generate_embedding(texts[0]), top_k=len(texts))

        assert len(store) == len(texts)
        assert results.metadata[0]["text"] == texts[0]
        assert set(item["text"] for item in results.metadata) == set(texts)

    @given(
        base=st.text(min_size=1, max_size=60).filter(lambda value: value.strip()),
        other=st.text(min_size=1, max_size=60).filter(lambda value: value.strip()),
    )
    def test_similarity_search_ordering(self, base, other):
        """
        Property 10: The most similar vector should rank ahead of less similar ones.
        """
        generator = _embedding_generator(dimension=18)
        store = VectorStore(dimension=18, index_type="flat")
        query_text = f"{base} query"
        exact_match = query_text
        different_text = f"{other} unrelated"

        embeddings = generator.generate_embeddings([exact_match, different_text])
        store.add_embeddings(
            embeddings,
            [{"text": exact_match}, {"text": different_text}],
        )

        results = store.search(generator.generate_embedding(query_text), top_k=2)

        assert results.metadata[0]["text"] == exact_match
        assert results.distances[0] >= results.distances[1]

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=60).filter(lambda value: value.strip()),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(deadline=None)
    def test_index_persistence_round_trip(self, texts):
        """
        Property 11: Persisted vector stores should load back with identical results.
        """
        generator = _embedding_generator(dimension=20)
        store = VectorStore(dimension=20, index_type="flat")
        embeddings = generator.generate_embeddings(texts)
        metadata = [{"text": text} for text in texts]
        query = texts[0]

        store.add_embeddings(embeddings, metadata)
        before = store.search(generator.generate_embedding(query), top_k=len(texts))

        with tempfile.TemporaryDirectory() as temp_directory:
            store.save(temp_directory)
            restored = VectorStore.load(temp_directory)
            after = restored.search(generator.generate_embedding(query), top_k=len(texts))

            assert before.indices == after.indices
            assert before.metadata == after.metadata
            assert before.distances == pytest.approx(after.distances)

    @given(
        initial=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda value: value.strip()),
            min_size=1,
            max_size=4,
            unique=True,
        ),
        additional=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda value: value.strip()),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )
    def test_incremental_knowledge_base_updates(self, initial, additional):
        """
        Property 26: Incremental additions should preserve earlier data and include new data.
        """
        extra = [text for text in additional if text not in initial]
        if not extra:
            extra = ["synthetic extra entry"]

        generator = _embedding_generator(dimension=22)
        store = VectorStore(dimension=22, index_type="flat")

        first_embeddings = generator.generate_embeddings(initial)
        store.add_embeddings(first_embeddings, [{"text": text} for text in initial])
        initial_count = len(store)

        second_embeddings = generator.generate_embeddings(extra)
        store.add_embeddings(second_embeddings, [{"text": text} for text in extra])

        assert len(store) == initial_count + len(extra)
        all_results = store.search(
            generator.generate_embedding(initial[0]),
            top_k=len(store),
        )
        all_texts = {item["text"] for item in all_results.metadata}
        assert set(initial).issubset(all_texts)
        assert set(extra).issubset(all_texts)

    def test_document_entries_round_trip_through_vector_store(self):
        generator = _embedding_generator(dimension=24)
        store = VectorStore(dimension=24, index_type="flat")
        chunks = [
            DocumentChunk(content="alpha content", source_file="doc1.txt"),
            DocumentChunk(content="beta content", source_file="doc2.txt"),
        ]
        entries = generator.generate_chunk_embeddings(chunks)

        store.add_documents(entries)
        results = store.search(generator.generate_embedding("alpha content"), top_k=2)

        assert results.metadata[0]["chunk"]["source_file"] == "doc1.txt"
        assert results.metadata[0]["embedding_metadata"]["chunk_id"] == chunks[0].chunk_id
