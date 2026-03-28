"""
End-to-end integration and system validation tests for the RAG pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import string
from typing import List
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from src.exceptions import ErrorCode, ResponseGenerationError
from src.models.rag_config import RAGConfig
from src.models.system_status import SystemHealth
from src.rag_pipeline import RAGPipeline


INTEGRATION_PROPERTY_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

SAFE_DOCUMENT_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + " .,-",
    min_size=16,
    max_size=90,
).filter(lambda value: value.strip())


def _build_pipeline_config(temp_directory: str, **overrides: object) -> RAGConfig:
    run_root = Path(temp_directory) / f"integration_{uuid4().hex}"
    base = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "max_context_length": 800,
        "top_k_results": 1,
        "similarity_threshold": 0.0,
        "max_response_tokens": 240,
        "temperature": 0.1,
        "batch_size": 8,
        "max_retries": 1,
        "timeout_seconds": 20,
        "data_directory": str(run_root / "data"),
        "embeddings_directory": str(run_root / "embeddings"),
        "logs_directory": str(run_root / "logs"),
    }
    base.update(overrides)
    return RAGConfig(**base)


def _build_pipeline(
    temp_directory: str,
    embedding_dimension: int = 32,
    **config_overrides: object,
) -> RAGPipeline:
    return RAGPipeline(
        config=_build_pipeline_config(temp_directory, **config_overrides),
        embedding_provider_kwargs={"mock_embeddings": True, "dimension": embedding_dimension},
        llm_provider_kwargs={"mock_responses": True},
    )


def _write_documents(root: Path, contents: List[str]) -> List[str]:
    root.mkdir(parents=True, exist_ok=True)
    file_paths: List[str] = []

    for index, content in enumerate(contents):
        file_path = root / f"doc_{index}.txt"
        file_path.write_text(content, encoding="utf-8")
        file_paths.append(str(file_path))

    return file_paths


@pytest.mark.integration
class TestEndToEndSystemValidation:
    def test_complete_pipeline_with_sample_documents_and_queries(self, temp_directory):
        pipeline = _build_pipeline(temp_directory)
        documents = [
            "Architecture guidance states that grounded answers should cite retrieved sources for every response.",
            "Operations guidance requires audit trails, health reporting, and resilient retry handling in production.",
            "Security guidance recommends access reviews, incident response drills, and least-privilege defaults.",
        ]
        file_paths = _write_documents(Path(temp_directory) / "corpus", documents)
        metadata_by_file = {
            file_path: {"topic": f"topic-{index}", "owner": "integration-suite"}
            for index, file_path in enumerate(file_paths)
        }

        ingestion = pipeline.ingest_documents(file_paths, metadata_by_file=metadata_by_file)
        first_response = pipeline.query(documents[0], k=1, threshold=0.0)
        second_response = pipeline.query(documents[1], k=1, threshold=0.0)
        status = pipeline.get_system_status()
        audit_events = {entry["event"] for entry in pipeline.get_audit_trail()}

        assert ingestion.is_successful()
        assert ingestion.successful_documents == 3
        assert ingestion.failed_documents == 0
        assert first_response.context_used is True
        assert second_response.context_used is True
        assert first_response.sources == [file_paths[0]]
        assert second_response.sources == [file_paths[1]]
        assert "Sources:" in first_response.response_text
        assert "Sources:" in second_response.response_text
        assert status.health == SystemHealth.HEALTHY
        assert status.performance_metrics["vector_store_size"] == float(len(pipeline.vector_store))
        assert status.performance_metrics["queries_processed_total"] == 2.0
        assert {
            "pipeline_initialized",
            "ingestion_started",
            "embedding_generation_started",
            "vector_store_updated",
            "query_started",
            "query_processed",
            "context_retrieved",
            "response_generated",
        } <= audit_events

    def test_cross_component_data_flow_and_consistency(self, temp_directory):
        pipeline = _build_pipeline(temp_directory, chunk_size=320, chunk_overlap=16)
        documents = [
            "Controls engineering notes describe closed-loop monitoring and calibrated digital twin feedback paths.",
            "Observability notes describe dashboards, anomaly alarms, and source-attributed incident summaries.",
        ]
        file_paths = _write_documents(Path(temp_directory) / "flow", documents)
        metadata_by_file = {
            file_paths[0]: {"topic": "controls", "revision": 1},
            file_paths[1]: {"topic": "observability", "revision": 2},
        }

        ingestion = pipeline.ingest_documents(file_paths, metadata_by_file=metadata_by_file)
        stored_entries = list(pipeline.vector_store.metadata_store)
        stored_chunk_ids = {entry["chunk_id"] for entry in stored_entries}
        query_results = pipeline.query_processor.process_query(documents[0], k=1, threshold=0.0)
        context = pipeline.context_retriever.retrieve_context_from_results(
            query_results,
            max_context_length=pipeline.config.max_context_length,
        )
        response = pipeline.response_generator.generate_response(documents[0], context)

        assert ingestion.is_successful()
        assert len(stored_entries) == len(file_paths)
        assert {entry["source_file"] for entry in stored_entries} == set(file_paths)
        assert {entry["chunk"]["metadata"]["topic"] for entry in stored_entries} == {
            "controls",
            "observability",
        }
        assert query_results.total_results == 1
        assert query_results.retrieved_chunks[0].source_file == file_paths[0]
        assert query_results.retrieved_chunks[0].metadata["topic"] == "controls"
        assert query_results.retrieved_chunks[0].chunk_id in stored_chunk_ids
        assert context.sources == [file_paths[0]]
        assert response.sources == [file_paths[0]]
        assert file_paths[0] in response.response_text

    def test_error_propagation_and_recovery_scenarios(self, temp_directory, monkeypatch):
        pipeline = _build_pipeline(temp_directory)
        documents = [
            "Recovery guidance explains how fallback responses should preserve system availability after failures.",
            "Validation guidance explains how good documents should still index when a bad file is supplied.",
        ]
        file_paths = _write_documents(Path(temp_directory) / "recovery", documents)
        invalid_path = Path(temp_directory) / "recovery" / "unsupported.bin"
        invalid_path.write_bytes(b"\x00\x01\x02\x03")

        ingestion = pipeline.ingest_documents([file_paths[0], str(invalid_path), file_paths[1]])

        assert ingestion.successful_documents == 2
        assert ingestion.failed_documents == 1
        assert str(invalid_path) in ingestion.failed_files
        assert any(ErrorCode.DOCUMENT_INVALID_FORMAT.value in error for error in ingestion.errors)

        original_generate = pipeline.response_generator.provider.generate
        failure_state = {"triggered": False}

        def flaky_generate(prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
            if not failure_state["triggered"]:
                failure_state["triggered"] = True
                raise ResponseGenerationError(
                    "Injected response failure",
                    ErrorCode.LLM_API_ERROR,
                    pipeline.response_generator.provider.model_name,
                )
            return original_generate(prompt, max_tokens=max_tokens, temperature=temperature)

        monkeypatch.setattr(pipeline.response_generator.provider, "generate", flaky_generate)
        failed_response = pipeline.query(documents[0], k=1, threshold=0.0)
        monkeypatch.setattr(pipeline.response_generator.provider, "generate", original_generate)
        recovered_response = pipeline.query(documents[0], k=1, threshold=0.0)
        status = pipeline.get_system_status()

        assert failed_response.context_used is False
        assert "could not complete the full retrieval pipeline" in failed_response.response_text.lower()
        assert recovered_response.context_used is True
        assert recovered_response.sources == [file_paths[0]]
        assert status.error_count >= 1
        assert status.health == SystemHealth.DEGRADED


@pytest.mark.integration
@pytest.mark.property
class TestIntegrationProperties:
    @INTEGRATION_PROPERTY_SETTINGS
    @given(
        documents=st.lists(
            SAFE_DOCUMENT_TEXT,
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    def test_data_consistency_across_pipeline_boundaries(self, temp_directory, documents):
        """
        Property 28: Ingested documents should remain internally consistent across storage, retrieval, and response stages.
        """
        pipeline = _build_pipeline(
            temp_directory,
            chunk_size=512,
            chunk_overlap=32,
            top_k_results=1,
        )
        file_paths = _write_documents(Path(temp_directory) / "property_consistency", documents)
        metadata_by_file = {
            file_path: {"ordinal": index, "suite": "integration-property"}
            for index, file_path in enumerate(file_paths)
        }

        ingestion = pipeline.ingest_documents(file_paths, metadata_by_file=metadata_by_file)
        responses = pipeline.query_batch(documents, k=1, threshold=0.0)
        stored_entries = list(pipeline.vector_store.metadata_store)
        status = pipeline.get_system_status()

        assert ingestion.is_successful()
        assert len(pipeline.vector_store) == len(documents)
        assert {entry["source_file"] for entry in stored_entries} == set(file_paths)
        assert status.performance_metrics["vector_store_size"] == float(len(documents))
        assert status.performance_metrics["batch_queries_processed_total"] == float(len(documents))

        for index, response in enumerate(responses):
            assert response.context_used is True
            assert response.sources == [file_paths[index]]
            assert file_paths[index] in response.response_text

        query_results = pipeline.query_processor.process_query(documents[0], k=1, threshold=0.0)
        assert query_results.retrieved_chunks[0].source_file == file_paths[0]
        assert query_results.retrieved_chunks[0].metadata["ordinal"] == 0

    @INTEGRATION_PROPERTY_SETTINGS
    @given(
        documents=st.lists(
            SAFE_DOCUMENT_TEXT,
            min_size=2,
            max_size=5,
            unique=True,
        ),
        data=st.data(),
    )
    def test_concurrent_queries_preserve_results_and_state(self, temp_directory, documents, data):
        """
        Property 29: Concurrent queries should complete without corrupting shared pipeline state or losing metrics.
        """
        pipeline = _build_pipeline(
            temp_directory,
            chunk_size=512,
            chunk_overlap=32,
            top_k_results=1,
        )
        file_paths = _write_documents(Path(temp_directory) / "property_concurrency", documents)
        pipeline.ingest_documents(file_paths)
        pipeline.embedding_generator.provider.load_model()
        pipeline.response_generator.provider.load_model()

        query_indices = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=len(documents) - 1),
                min_size=2,
                max_size=6,
            )
        )
        queries = [documents[index] for index in query_indices]
        expected_sources = [file_paths[index] for index in query_indices]

        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
            responses = list(executor.map(lambda query: pipeline.query(query, k=1, threshold=0.0), queries))

        audit_trail = pipeline.get_audit_trail()
        query_started_events = [entry for entry in audit_trail if entry["event"] == "query_started"]
        status = pipeline.get_system_status()

        assert len(responses) == len(queries)
        assert len(pipeline.vector_store) == len(documents)
        assert status.error_count == 0
        assert status.performance_metrics["queries_processed_total"] == float(len(queries))
        assert len(query_started_events) == len(queries)

        for response, expected_source in zip(responses, expected_sources):
            assert response.context_used is True
            assert response.sources == [expected_source]
