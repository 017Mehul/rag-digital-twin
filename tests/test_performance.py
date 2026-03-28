"""
Performance and scalability benchmarks for the RAG pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import tracemalloc
from typing import Callable, List, Tuple
from uuid import uuid4

import pytest

from src.models.system_status import SystemHealth
from src.models.rag_config import RAGConfig
from src.rag_pipeline import RAGPipeline


MAX_INGEST_SECONDS = 10.0
MAX_CONCURRENT_QUERY_SECONDS = 5.0
MAX_SEARCH_SECONDS = 4.0
MAX_PEAK_MEMORY_MB = 64.0


def _build_pipeline_config(temp_directory: str, **overrides: object) -> RAGConfig:
    run_root = Path(temp_directory) / f"performance_{uuid4().hex}"
    base = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "chunk_size": 512,
        "chunk_overlap": 32,
        "max_context_length": 1000,
        "top_k_results": 1,
        "similarity_threshold": 0.0,
        "max_response_tokens": 240,
        "temperature": 0.1,
        "batch_size": 16,
        "max_retries": 1,
        "timeout_seconds": 20,
        "data_directory": str(run_root / "data"),
        "embeddings_directory": str(run_root / "embeddings"),
        "logs_directory": str(run_root / "logs"),
    }
    base.update(overrides)
    return RAGConfig(**base)


def _build_pipeline(temp_directory: str, embedding_dimension: int = 48) -> RAGPipeline:
    return RAGPipeline(
        config=_build_pipeline_config(temp_directory),
        embedding_provider_kwargs={"mock_embeddings": True, "dimension": embedding_dimension},
        llm_provider_kwargs={"mock_responses": True},
    )


def _write_large_corpus(root: Path, count: int) -> Tuple[List[str], List[str]]:
    root.mkdir(parents=True, exist_ok=True)
    file_paths: List[str] = []
    contents: List[str] = []

    for index in range(count):
        content = (
            f"Benchmark document {index} records digital twin telemetry, retrieval grounding, "
            f"audit evidence, and source attribution for scenario {index}."
        )
        file_path = root / f"benchmark_{index}.txt"
        file_path.write_text(content, encoding="utf-8")
        file_paths.append(str(file_path))
        contents.append(content)

    return file_paths, contents


def _measure(operation: Callable[[], object]) -> Tuple[float, object]:
    start_time = time.perf_counter()
    result = operation()
    return time.perf_counter() - start_time, result


def _run_concurrent_queries(pipeline: RAGPipeline, queries: List[str]):
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(lambda query: pipeline.query(query, k=1, threshold=0.0), queries))


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
class TestPipelinePerformance:
    def test_ingestion_and_concurrent_query_benchmarks(self, temp_directory):
        pipeline = _build_pipeline(temp_directory)
        file_paths, contents = _write_large_corpus(Path(temp_directory) / "bench_ingest", count=60)

        ingest_elapsed, ingestion = _measure(lambda: pipeline.ingest_documents(file_paths))
        pipeline.embedding_generator.provider.load_model()
        pipeline.response_generator.provider.load_model()

        queries = contents[:12]
        expected_sources = file_paths[:12]
        concurrent_elapsed, responses = _measure(lambda: _run_concurrent_queries(pipeline, queries))
        status = pipeline.get_system_status()

        assert ingestion.is_successful()
        assert all(response.context_used is True for response in responses)
        assert [response.sources[0] for response in responses] == expected_sources
        assert ingest_elapsed < MAX_INGEST_SECONDS
        assert concurrent_elapsed < MAX_CONCURRENT_QUERY_SECONDS
        assert status.performance_metrics["queries_processed_total"] == float(len(queries))
        assert status.performance_metrics["vector_store_size"] == float(len(file_paths))

    def test_large_document_collection_search_remains_responsive(self, temp_directory):
        pipeline = _build_pipeline(temp_directory)
        file_paths, contents = _write_large_corpus(Path(temp_directory) / "bench_search", count=140)

        ingestion = pipeline.ingest_documents(file_paths)
        sampled_queries = contents[::10]
        expected_sources = file_paths[::10]
        search_elapsed, query_results = _measure(
            lambda: [
                pipeline.query_processor.process_query(query, k=1, threshold=0.0)
                for query in sampled_queries
            ]
        )
        status = pipeline.get_system_status()

        assert ingestion.successful_documents == 140
        assert all(result.total_results == 1 for result in query_results)
        assert [result.retrieved_chunks[0].source_file for result in query_results] == expected_sources
        assert search_elapsed < MAX_SEARCH_SECONDS
        assert status.health == SystemHealth.HEALTHY
        assert status.performance_metrics["vector_store_size"] == float(len(file_paths))

    def test_large_ingestion_keeps_peak_python_memory_bounded(self, temp_directory):
        pipeline = _build_pipeline(temp_directory)
        file_paths, _ = _write_large_corpus(Path(temp_directory) / "bench_memory", count=120)

        tracemalloc.start()
        try:
            ingest_elapsed, ingestion = _measure(lambda: pipeline.ingest_documents(file_paths))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_memory_mb = peak / (1024 * 1024)
        status = pipeline.get_system_status()

        assert ingestion.is_successful()
        assert ingest_elapsed < MAX_INGEST_SECONDS
        assert peak_memory_mb < MAX_PEAK_MEMORY_MB
        assert status.performance_metrics["vector_store_size"] == float(len(file_paths))
