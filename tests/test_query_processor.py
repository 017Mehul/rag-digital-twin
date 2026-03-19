"""
Tests for query processing and ranked retrieval.
"""

from typing import List

import pytest
from hypothesis import given, strategies as st

from src.embedding_generator import EmbeddingGenerator
from src.exceptions import ErrorCode, QueryProcessingError
from src.models.document_chunk import DocumentChunk
from src.providers import OpenAIEmbeddingProvider
from src.query_processor import QueryProcessor
from src.vector_store import VectorStore


def _build_query_processor(texts: List[str], dimension: int = 24) -> QueryProcessor:
    generator = EmbeddingGenerator(
        provider=OpenAIEmbeddingProvider(
            model_name="text-embedding-3-small",
            dimension=dimension,
            mock_embeddings=True,
        )
    )
    store = VectorStore(dimension=dimension, index_type="flat")
    chunks = [
        DocumentChunk(
            content=text,
            source_file=f"doc_{index}.txt",
            metadata={"chunk_index": index},
        )
        for index, text in enumerate(texts)
    ]
    store.add_documents(generator.generate_chunk_embeddings(chunks))
    return QueryProcessor(
        embedding_generator=generator,
        vector_store=store,
        top_k_results=max(1, min(5, len(texts))),
        similarity_threshold=0.0,
    )


class TestQueryProcessor:
    def test_process_query_returns_ranked_chunks_and_scores(self):
        processor = _build_query_processor(
            [
                "semantic retrieval with vectors",
                "cooking recipe for soup",
                "database indexing with faiss",
            ],
            dimension=32,
        )

        results = processor.process_query("semantic retrieval with vectors", k=2, threshold=0.0)

        assert results.total_results == len(results.retrieved_chunks) == len(results.relevance_scores)
        assert results.retrieved_chunks[0].content == "semantic retrieval with vectors"
        assert results.relevance_scores[0] >= results.relevance_scores[-1]
        assert results.message == ""

    def test_process_query_reports_insufficient_context_when_no_results(self):
        processor = _build_query_processor(
            ["vector databases", "retrieval augmented generation"],
            dimension=20,
        )

        results = processor.process_query("completely unrelated astronomy question", threshold=1.0)

        assert results.is_empty()
        assert results.message == "Insufficient context to answer this query reliably."

    def test_empty_query_raises_descriptive_error(self):
        processor = _build_query_processor(["alpha", "beta"], dimension=16)

        with pytest.raises(QueryProcessingError) as exc_info:
            processor.process_query("   ")

        assert exc_info.value.error_code == ErrorCode.QUERY_EMPTY


class TestQueryProcessorProperties:
    @given(text=st.text(min_size=1, max_size=200).filter(lambda value: value.strip()))
    def test_query_document_embedding_consistency(self, text):
        """
        Property 12: Query embeddings should match document embeddings for the same text.
        """
        processor = _build_query_processor([text, f"{text} extra context"], dimension=18)

        query_embedding = processor.embed_query(text)
        document_embedding = processor.embedding_generator.generate_embedding(text)

        assert query_embedding == document_embedding

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=80).filter(lambda value: value.strip()),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        k=st.integers(min_value=1, max_value=6),
    )
    def test_top_k_result_limiting(self, texts, k):
        """
        Property 13: Query results should contain at most k items.
        """
        processor = _build_query_processor(texts, dimension=22)

        results = processor.process_query(texts[0], k=k, threshold=0.0)

        assert len(results) <= k

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=80).filter(lambda value: value.strip()),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_configurable_threshold_filtering(self, texts, threshold):
        """
        Property 14: Returned results should satisfy the active relevance threshold.
        """
        processor = _build_query_processor(texts, dimension=26)

        results = processor.process_query(texts[0], k=len(texts), threshold=threshold)

        assert all(score >= threshold for score in results.relevance_scores)

    @given(
        texts=st.lists(
            st.text(min_size=1, max_size=80).filter(lambda value: value.strip()),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    def test_relevance_score_provision(self, texts):
        """
        Property 15: Each returned result should include a numerical relevance score.
        """
        processor = _build_query_processor(texts, dimension=28)

        results = processor.process_query(texts[0], k=len(texts), threshold=0.0)

        assert results.total_results == len(results.retrieved_chunks) == len(results.relevance_scores)
        assert all(isinstance(score, float) for score in results.relevance_scores)
        assert all(-1.0 <= score <= 1.0 for score in results.relevance_scores)
