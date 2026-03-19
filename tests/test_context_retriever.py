"""
Tests for context retrieval and insufficient-context handling.
"""

import pytest
from hypothesis import given, strategies as st

from src.context_retriever import ContextRetriever
from src.embedding_generator import EmbeddingGenerator
from src.models.document_chunk import DocumentChunk
from src.providers import OpenAIEmbeddingProvider
from src.query_processor import QueryProcessor
from src.vector_store import VectorStore


def _build_context_retriever(texts, dimension: int = 24, max_context_length: int = 4000) -> ContextRetriever:
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
    if chunks:
        store.add_documents(generator.generate_chunk_embeddings(chunks))

    processor = QueryProcessor(
        embedding_generator=generator,
        vector_store=store,
        top_k_results=max(1, min(5, len(texts) or 1)),
        similarity_threshold=0.0,
    )
    return ContextRetriever(
        query_processor=processor,
        max_context_length=max_context_length,
        min_overlap_chars=10,
    )


class TestContextRetriever:
    def test_retrieve_context_formats_sources_and_respects_length(self):
        retriever = _build_context_retriever(
            [
                "FAISS enables efficient similarity search for embeddings.",
                "RAG systems combine retrieval with generation.",
            ],
            dimension=30,
            max_context_length=120,
        )

        context = retriever.retrieve_context("FAISS enables efficient similarity search for embeddings.")

        assert not context.insufficient_context
        assert len(context.formatted_context) <= 120
        assert context.sources
        assert context.sources == [chunk.source_file for chunk in context.source_chunks]
        assert "[Source 1]" in context.formatted_context

    def test_deduplicate_chunks_trims_overlapping_content(self):
        retriever = _build_context_retriever([], dimension=16)
        chunks = [
            DocumentChunk(
                content="FAISS powers retrieval with dense vectors and semantic search.",
                source_file="doc.txt",
                metadata={"chunk_index": 0},
            ),
            DocumentChunk(
                content="dense vectors and semantic search. Additional ranking details follow.",
                source_file="doc.txt",
                metadata={"chunk_index": 1},
            ),
        ]

        deduplicated = retriever.deduplicate_chunks(chunks)

        assert len(deduplicated) == 2
        assert "dense vectors and semantic search." not in deduplicated[1].content
        assert "Additional ranking details follow." in deduplicated[1].content


class TestContextRetrieverProperties:
    @given(query=st.text(min_size=1, max_size=200).filter(lambda value: value.strip()))
    def test_insufficient_context_handling(self, query):
        """
        Property 27: Queries with no relevant context should report uncertainty.
        """
        retriever = _build_context_retriever([], dimension=18)

        context = retriever.retrieve_context(query)

        assert context.insufficient_context is True
        assert "Insufficient context" in context.formatted_context
        assert context.source_chunks == []
        assert context.sources == []
        assert context.total_tokens > 0
