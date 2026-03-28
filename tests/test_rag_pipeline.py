"""
Tests for RAG pipeline orchestration, monitoring, and audit logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from src.exceptions import ConfigurationError
from src.models.document_chunk import DocumentChunk
from src.models.embedding_metadata import EmbeddingMetadata
from src.models.rag_config import RAGConfig
from src.models.search_results import GeneratedResponse, QueryResults, RetrievedContext
from src.models.system_status import IngestionResults, SystemHealth
from src.rag_pipeline import RAGPipeline


PROPERTY_TEST_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _build_pipeline_config(temp_directory: str, **overrides: Any) -> RAGConfig:
    run_root = Path(temp_directory) / f"run_{uuid4().hex}"
    base = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 120,
        "chunk_overlap": 20,
        "max_context_length": 400,
        "top_k_results": 3,
        "similarity_threshold": 0.0,
        "max_response_tokens": 200,
        "temperature": 0.2,
        "batch_size": 4,
        "max_retries": 1,
        "timeout_seconds": 15,
        "data_directory": str(run_root / "data"),
        "embeddings_directory": str(run_root / "embeddings"),
        "logs_directory": str(run_root / "logs"),
    }
    base.update(overrides)
    return RAGConfig(**base)


class _StubProvider:
    def __init__(self, provider: str, model_name: str, dimension: int = 8) -> None:
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())


class _SpyDocumentProcessor:
    def __init__(self, events: List[str]) -> None:
        self.events = events

    def process_batch(
        self,
        file_paths: List[str],
        metadata_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.events.append("document_processing")
        processed_documents = {}
        for index, file_path in enumerate(file_paths):
            metadata = dict((metadata_by_file or {}).get(file_path, {}))
            metadata["chunk_index"] = index
            processed_documents[file_path] = [
                DocumentChunk(
                    content=f"content for {file_path}",
                    source_file=file_path,
                    metadata=metadata,
                )
            ]
        return {
            "processed_documents": processed_documents,
            "failed_documents": {},
            "total_documents": len(file_paths),
            "successful_documents": len(processed_documents),
            "failed_count": 0,
        }


class _SpyEmbeddingGenerator:
    def __init__(self, events: List[str], dimension: int = 8, should_fail: bool = False) -> None:
        self.events = events
        self.provider = _StubProvider("embedding-stub", "mock-embedding", dimension)
        self.should_fail = should_fail

    def generate_chunk_embeddings(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        self.events.append("embedding_generation")
        if self.should_fail:
            raise RuntimeError("embedding generation failed")

        zero_tail = [0.0] * (self.provider.dimension - 1)
        return [
            {
                "chunk": chunk,
                "embedding": [1.0] + zero_tail,
                "metadata": EmbeddingMetadata.from_document_chunk(chunk, self.provider.model_name),
            }
            for chunk in chunks
        ]


class _SpyVectorStore:
    def __init__(self, events: List[str], dimension: int = 8) -> None:
        self.events = events
        self.dimension = dimension
        self.backend = "test"
        self.index_type = "flat"
        self.metadata_store: List[Dict[str, Any]] = []

    def add_documents(self, entries: List[Dict[str, Any]]) -> None:
        self.events.append("vector_storage")
        for entry in entries:
            chunk = entry["chunk"]
            if isinstance(chunk, DocumentChunk):
                chunk_payload = chunk.to_dict()
            else:
                chunk_payload = dict(chunk)
            self.metadata_store.append({"chunk": chunk_payload, "source_file": chunk_payload["source_file"]})

    def save(self, directory: str) -> Dict[str, str]:
        return {"directory": directory}

    def is_empty(self) -> bool:
        return len(self.metadata_store) == 0

    def __len__(self) -> int:
        return len(self.metadata_store)


class _SpyQueryProcessor:
    def __init__(self, events: List[str], vector_store: _SpyVectorStore, should_fail: bool = False) -> None:
        self.events = events
        self.vector_store = vector_store
        self.embedding_generator: Optional[_SpyEmbeddingGenerator] = None
        self.top_k_results = 3
        self.similarity_threshold = 0.0
        self.should_fail = should_fail

    def process_query(self, query: str, k: Optional[int] = None, threshold: Optional[float] = None) -> QueryResults:
        self.events.append("query_processing")
        if self.should_fail:
            raise RuntimeError("query processing failed")

        return QueryResults(
            query=query,
            retrieved_chunks=[
                DocumentChunk(
                    content="retrieved knowledge",
                    source_file="doc.txt",
                    metadata={"chunk_index": 0},
                )
            ],
            relevance_scores=[0.95],
            total_results=1,
        )


class _SpyContextRetriever:
    def __init__(self, events: List[str]) -> None:
        self.events = events
        self.query_processor: Optional[_SpyQueryProcessor] = None
        self.max_context_length = 200

    def retrieve_context_from_results(
        self,
        query_results: QueryResults,
        max_context_length: Optional[int] = None,
    ) -> RetrievedContext:
        self.events.append("context_retrieval")
        return RetrievedContext(
            formatted_context="[Source 1] doc.txt\nretrieved knowledge",
            source_chunks=query_results.retrieved_chunks,
            total_tokens=4,
            sources=["doc.txt"],
            insufficient_context=False,
        )


class _SpyResponseGenerator:
    def __init__(self, events: List[str], should_fail: bool = False) -> None:
        self.events = events
        self.provider = _StubProvider("llm-stub", "mock-llm")
        self.max_response_tokens = 50
        self.temperature = 0.1
        self.should_fail = should_fail

    def generate_response(self, query: str, context: RetrievedContext) -> GeneratedResponse:
        self.events.append("response_generation")
        if self.should_fail:
            raise RuntimeError("response generation failed")

        return GeneratedResponse(
            response_text="Grounded answer.\n\nSources:\n- doc.txt",
            sources=["doc.txt"],
            confidence_score=0.8,
            context_used=True,
            token_count=5,
            model_used=self.provider.model_name,
            generation_time=0.01,
        )


def _build_spy_pipeline(
    temp_directory: str,
    embedding_should_fail: bool = False,
    query_should_fail: bool = False,
    response_should_fail: bool = False,
) -> tuple[RAGPipeline, List[str]]:
    events: List[str] = []
    vector_store = _SpyVectorStore(events)
    pipeline = RAGPipeline(
        config=_build_pipeline_config(temp_directory),
        document_processor=_SpyDocumentProcessor(events),
        embedding_generator=_SpyEmbeddingGenerator(events, should_fail=embedding_should_fail),
        vector_store=vector_store,
        query_processor=_SpyQueryProcessor(events, vector_store=vector_store, should_fail=query_should_fail),
        context_retriever=_SpyContextRetriever(events),
        response_generator=_SpyResponseGenerator(events, should_fail=response_should_fail),
    )
    return pipeline, events


class TestRAGPipeline:
    def test_pipeline_end_to_end_with_mock_providers(self, temp_directory):
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 32},
            llm_provider_kwargs={"mock_responses": True},
        )
        file_path = Path(temp_directory) / "knowledge.txt"
        file_path.write_text(
            "RAG pipelines combine retrieval with response generation to ground answers in source material.",
            encoding="utf-8",
        )

        ingestion = pipeline.ingest_documents([str(file_path)])
        response = pipeline.query("How do RAG pipelines improve answers?")
        status = pipeline.get_system_status()

        assert ingestion.is_successful()
        assert response.context_used is True
        assert response.sources == [str(file_path)]
        assert "Sources:" in response.response_text
        assert status.health == SystemHealth.HEALTHY
        assert status.performance_metrics["vector_store_size"] >= 1.0

    def test_empty_vector_store_returns_graceful_response(self, temp_directory):
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 24},
            llm_provider_kwargs={"mock_responses": True},
        )

        response = pipeline.query("What knowledge is available?")

        assert response.context_used is False
        assert response.sources == []
        assert "indexed knowledge" in response.response_text.lower()


class TestRAGPipelineProperties:
    @PROPERTY_TEST_SETTINGS
    @given(file_count=st.integers(min_value=1, max_value=3))
    def test_pipeline_component_ordering(self, temp_directory, file_count):
        """
        Property 18: Pipeline components should execute in the expected order.
        """
        pipeline, events = _build_spy_pipeline(temp_directory)
        file_paths = [f"doc_{index}.txt" for index in range(file_count)]

        ingestion = pipeline.ingest_documents(file_paths)
        response = pipeline.query("What does the indexed knowledge contain?")

        assert isinstance(ingestion, IngestionResults)
        assert isinstance(response, GeneratedResponse)
        expected_events = ["document_processing"]
        for _ in range(file_count):
            expected_events.extend(["embedding_generation", "vector_storage"])
        expected_events.extend(["query_processing", "context_retrieval", "response_generation"])

        assert events == expected_events

    @PROPERTY_TEST_SETTINGS
    @given(stage=st.sampled_from(["ingestion", "query"]))
    def test_graceful_error_propagation(self, temp_directory, stage):
        """
        Property 19: Pipeline failures should surface gracefully without crashing the caller.
        """
        pipeline, _ = _build_spy_pipeline(
            temp_directory,
            embedding_should_fail=stage == "ingestion",
            response_should_fail=stage == "query",
        )

        if stage == "ingestion":
            results = pipeline.ingest_documents(["doc.txt"])

            assert isinstance(results, IngestionResults)
            assert results.has_errors()
            assert pipeline.get_system_status().health in {SystemHealth.DEGRADED, SystemHealth.UNHEALTHY}
        else:
            pipeline.ingest_documents(["doc.txt"])
            response = pipeline.query("What failed?")

            assert isinstance(response, GeneratedResponse)
            assert response.context_used is False
            assert "could not complete the full retrieval pipeline" in response.response_text.lower()
            assert pipeline.get_system_status().health in {SystemHealth.DEGRADED, SystemHealth.UNHEALTHY}

    @PROPERTY_TEST_SETTINGS
    @given(
        queries=st.lists(
            st.text(min_size=1, max_size=80).filter(lambda value: value.strip()),
            min_size=1,
            max_size=3,
        )
    )
    def test_processing_mode_support(self, temp_directory, queries):
        """
        Property 21: The pipeline should support both real-time and batch processing modes.
        """
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 28},
            llm_provider_kwargs={"mock_responses": True},
        )
        file_path = Path(temp_directory) / "modes.txt"
        file_path.write_text(
            "Batch and real-time processing are both supported in this RAG pipeline.",
            encoding="utf-8",
        )
        pipeline.ingest_documents([str(file_path)])

        single_response = pipeline.query(queries[0])
        batch_responses = pipeline.query_batch(queries)

        assert isinstance(single_response, GeneratedResponse)
        assert len(batch_responses) == len(queries)
        assert all(isinstance(response, GeneratedResponse) for response in batch_responses)

    @PROPERTY_TEST_SETTINGS
    @given(
        top_k=st.integers(min_value=1, max_value=8),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        max_context_length=st.integers(min_value=50, max_value=600),
        max_response_tokens=st.integers(min_value=50, max_value=400),
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_configuration_application(
        self,
        temp_directory,
        top_k,
        threshold,
        max_context_length,
        max_response_tokens,
        temperature,
    ):
        """
        Property 23: Updated configuration values should be applied across dependent components.
        """
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 30},
            llm_provider_kwargs={"mock_responses": True},
        )

        updated_config = pipeline.update_configuration(
            {
                "top_k_results": top_k,
                "similarity_threshold": threshold,
                "max_context_length": max_context_length,
                "max_response_tokens": max_response_tokens,
                "temperature": temperature,
            }
        )

        assert updated_config.top_k_results == top_k
        assert updated_config.similarity_threshold == threshold
        assert pipeline.query_processor.top_k_results == top_k
        assert pipeline.query_processor.similarity_threshold == threshold
        assert pipeline.context_retriever.max_context_length == max_context_length
        assert pipeline.response_generator.max_response_tokens == max_response_tokens
        assert pipeline.response_generator.temperature == temperature

    @PROPERTY_TEST_SETTINGS
    @given(
        invalid_threshold=st.one_of(
            st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
        )
    )
    def test_configuration_validation(self, temp_directory, invalid_threshold):
        """
        Property 25: Invalid configuration updates should be rejected with a configuration error.
        """
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 24},
            llm_provider_kwargs={"mock_responses": True},
        )

        with pytest.raises(ConfigurationError) as exc_info:
            pipeline.update_configuration({"similarity_threshold": invalid_threshold})

        assert exc_info.value.error_code == exc_info.value.error_code.CONFIG_VALIDATION_FAILED

    @PROPERTY_TEST_SETTINGS
    @given(operation=st.sampled_from(["ingest", "query", "update_config"]))
    def test_operation_logging_completeness(self, temp_directory, operation):
        """
        Property 20: Each major operation should produce audit log entries.
        """
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 26},
            llm_provider_kwargs={"mock_responses": True},
        )
        file_path = Path(temp_directory) / "audit.txt"
        file_path.write_text("Audit logging should record key decision points.", encoding="utf-8")

        if operation == "ingest":
            pipeline.ingest_documents([str(file_path)])
            events = {entry["event"] for entry in pipeline.get_audit_trail()}
            assert {"ingestion_started", "embedding_generation_started", "vector_store_updated", "ingestion_completed"} <= events
        elif operation == "query":
            pipeline.ingest_documents([str(file_path)])
            pipeline.query("What should be logged?")
            events = {entry["event"] for entry in pipeline.get_audit_trail()}
            assert {"query_started", "query_processed", "context_retrieved", "response_generated"} <= events
        else:
            pipeline.update_configuration({"top_k_results": 2})
            events = {entry["event"] for entry in pipeline.get_audit_trail()}
            assert {"configuration_update_started", "configuration_updated"} <= events

    @PROPERTY_TEST_SETTINGS
    @given(
        texts=st.lists(
            st.text(min_size=5, max_size=80).filter(lambda value: value.strip()),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    def test_state_consistency_maintenance(self, temp_directory, texts):
        """
        Property 22: Internal component references and reported state should remain consistent.
        """
        pipeline = RAGPipeline(
            config=_build_pipeline_config(temp_directory),
            embedding_provider_kwargs={"mock_embeddings": True, "dimension": 32},
            llm_provider_kwargs={"mock_responses": True},
        )
        file_paths: List[str] = []
        for index, text in enumerate(texts):
            file_path = Path(temp_directory) / f"state_{index}.txt"
            file_path.write_text(text, encoding="utf-8")
            file_paths.append(str(file_path))

        pipeline.ingest_documents(file_paths)
        pipeline.update_configuration({"top_k_results": 2, "max_context_length": 250})
        status = pipeline.get_system_status()

        assert pipeline.query_processor.vector_store is pipeline.vector_store
        assert pipeline.query_processor.embedding_generator is pipeline.embedding_generator
        assert pipeline.context_retriever.query_processor is pipeline.query_processor
        assert status.performance_metrics["vector_store_size"] == float(len(pipeline.vector_store))
        assert status.performance_metrics["audit_event_count"] == float(len(pipeline.get_audit_trail()))
        assert status.health == SystemHealth.HEALTHY
