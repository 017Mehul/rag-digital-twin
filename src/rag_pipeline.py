"""
Main RAG pipeline orchestration with monitoring, audit logging, and recovery.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .context_retriever import ContextRetriever
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .exceptions import ConfigurationError, ErrorCode, ErrorHandler
from .models.rag_config import RAGConfig
from .models.search_results import GeneratedResponse
from .models.system_status import IngestionResults, SystemHealth, SystemStatus
from .query_processor import QueryProcessor
from .response_generator import ResponseGenerator
from .utils.logging_utils import get_logger
from .vector_store import VectorStore


class RAGPipeline:
    """
    Coordinate ingestion, retrieval, response generation, and system monitoring.
    """

    def __init__(
        self,
        config: RAGConfig,
        document_processor: Optional[DocumentProcessor] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        query_processor: Optional[QueryProcessor] = None,
        context_retriever: Optional[ContextRetriever] = None,
        response_generator: Optional[ResponseGenerator] = None,
        logger: Optional[logging.Logger] = None,
        embedding_provider_kwargs: Optional[Dict[str, Any]] = None,
        llm_provider_kwargs: Optional[Dict[str, Any]] = None,
        vector_store_index_type: str = "flat",
    ) -> None:
        try:
            config.validate()
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid pipeline configuration: {exc}",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                cause=exc,
            ) from exc

        self.config = config
        self.logger = logger or get_logger("RAGPipeline")
        self.error_handler = ErrorHandler(self.logger)
        self.system_status = SystemStatus(health=SystemHealth.UNKNOWN)
        self.audit_trail: List[Dict[str, Any]] = []
        self.started_at = time.perf_counter()
        self.embedding_provider_kwargs = dict(embedding_provider_kwargs or {})
        self.llm_provider_kwargs = dict(llm_provider_kwargs or {})
        self.vector_store_index_type = vector_store_index_type

        self.document_processor = (
            document_processor if document_processor is not None else self._create_document_processor()
        )
        self.embedding_generator = (
            embedding_generator if embedding_generator is not None else self._create_embedding_generator()
        )
        self.vector_store = vector_store if vector_store is not None else self._load_or_create_vector_store()
        self.query_processor = query_processor if query_processor is not None else self._create_query_processor()
        self.context_retriever = (
            context_retriever if context_retriever is not None else self._create_context_retriever()
        )
        self.response_generator = (
            response_generator if response_generator is not None else self._create_response_generator()
        )

        self._synchronize_dependencies()
        self._refresh_system_status()
        self.system_status.health = SystemHealth.HEALTHY
        self._record_audit(
            "pipeline_initialized",
            {
                "embedding_provider": self.config.embedding_provider,
                "embedding_model": self.config.embedding_model,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
            },
        )

    def ingest_documents(
        self,
        file_paths: List[str],
        metadata_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
        persist: bool = True,
    ) -> IngestionResults:
        """
        Ingest a batch of documents into the vector store.
        """
        start_time = time.perf_counter()
        results = IngestionResults(total_documents=len(file_paths))
        self._record_audit(
            "ingestion_started",
            {"document_count": len(file_paths), "persist": persist},
        )

        if not file_paths:
            results.processing_time = 0.0
            self._update_metrics(last_ingestion_seconds=0.0, last_ingestion_documents=0.0)
            self._record_audit("ingestion_completed", results.to_dict())
            return results

        try:
            batch_result = self.document_processor.process_batch(file_paths, metadata_by_file=metadata_by_file)

            processed_documents = batch_result["processed_documents"]
            failed_documents = batch_result["failed_documents"]
            all_chunks = []

            for file_path, chunks in processed_documents.items():
                results.add_successful_document(file_path, len(chunks))
                all_chunks.extend(chunks)
            for file_path, error_message in failed_documents.items():
                results.add_failed_document(file_path, error_message)

            if all_chunks:
                self._record_audit(
                    "embedding_generation_started",
                    {"chunk_count": len(all_chunks)},
                )
                entries = self.embedding_generator.generate_chunk_embeddings(all_chunks)
                results.total_embeddings = len(entries)

                self.vector_store.add_documents(entries)
                self._record_audit(
                    "vector_store_updated",
                    {
                        "added_embeddings": len(entries),
                        "vector_store_size": len(self.vector_store),
                    },
                )

                if persist:
                    self.vector_store.save(self.config.embeddings_directory)
                    self._record_audit(
                        "vector_store_persisted",
                        {"directory": self.config.embeddings_directory},
                    )
            else:
                self._record_audit("ingestion_no_chunks", {"reason": "no valid chunks were produced"})
        except Exception as exc:
            handled = self.error_handler.handle_error(exc, {"operation": "ingest_documents"})
            results.errors.append(handled["message"])
            self.system_status.record_error(handled["message"])
            self._record_audit("ingestion_failed", handled, level="error")
        finally:
            results.processing_time = time.perf_counter() - start_time
            self._update_metrics(
                last_ingestion_seconds=results.processing_time,
                last_ingestion_documents=float(results.total_documents),
                documents_ingested_total=float(
                    self.system_status.performance_metrics.get("documents_ingested_total", 0.0)
                    + results.successful_documents
                ),
                chunks_ingested_total=float(
                    self.system_status.performance_metrics.get("chunks_ingested_total", 0.0)
                    + results.total_chunks
                ),
                vector_store_size=float(len(self.vector_store)),
            )
            self._refresh_system_status()
            self._record_audit("ingestion_completed", results.to_dict())

        return results

    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> GeneratedResponse:
        """
        Process a single real-time query through retrieval and generation.
        """
        start_time = time.perf_counter()
        self._record_audit(
            "query_started",
            {
                "query": user_query[:200] if isinstance(user_query, str) else str(user_query),
                "top_k": k if k is not None else self.query_processor.top_k_results,
                "threshold": threshold if threshold is not None else self.query_processor.similarity_threshold,
            },
        )

        if self.vector_store.is_empty():
            response = self._build_insufficient_context_response(
                "I do not have any indexed knowledge available yet.",
                generation_time=time.perf_counter() - start_time,
            )
            self._record_audit(
                "query_insufficient_context",
                {"reason": "vector store is empty"},
                level="warning",
            )
            self._finalize_query_metrics(start_time)
            return response

        try:
            query_results = self.query_processor.process_query(user_query, k=k, threshold=threshold)
            self._record_audit(
                "query_processed",
                {
                    "retrieved_results": query_results.total_results,
                    "sources": query_results.get_sources(),
                },
            )

            context = self.context_retriever.retrieve_context_from_results(
                query_results,
                max_context_length=self.config.max_context_length,
            )
            self._record_audit(
                "context_retrieved",
                {
                    "sources": context.sources,
                    "total_tokens": context.total_tokens,
                    "insufficient_context": context.insufficient_context,
                },
            )

            response = self.response_generator.generate_response(user_query, context)
            self._record_audit(
                "response_generated",
                {
                    "sources": response.sources,
                    "confidence_score": response.confidence_score,
                    "context_used": response.context_used,
                },
            )
        except Exception as exc:
            handled = self.error_handler.handle_error(exc, {"operation": "query", "query": user_query})
            self.system_status.record_error(handled["message"])
            self._record_audit("query_failed", handled, level="error")
            response = self._build_insufficient_context_response(
                f"I could not complete the full retrieval pipeline for this query. {handled['message']}",
                generation_time=time.perf_counter() - start_time,
            )
        finally:
            self._finalize_query_metrics(start_time)

        return response

    def query_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[GeneratedResponse]:
        """
        Process multiple queries in batch mode.
        """
        start_time = time.perf_counter()
        self._record_audit("batch_query_started", {"query_count": len(queries)})
        responses = [self.query(query, k=k, threshold=threshold) for query in queries]
        elapsed = time.perf_counter() - start_time
        self._update_metrics(
            last_batch_query_seconds=elapsed,
            batch_queries_processed_total=float(
                self.system_status.performance_metrics.get("batch_queries_processed_total", 0.0)
                + len(queries)
            ),
        )
        self._refresh_system_status()
        self._record_audit(
            "batch_query_completed",
            {"query_count": len(queries), "processing_time": elapsed},
        )
        return responses

    def update_configuration(self, config_or_updates: Union[RAGConfig, Dict[str, Any]]) -> RAGConfig:
        """
        Apply configuration changes and rebuild dependent runtime components.
        """
        self._record_audit("configuration_update_started", {"payload_type": type(config_or_updates).__name__})

        try:
            if isinstance(config_or_updates, RAGConfig):
                new_config = config_or_updates
                new_config.validate()
            elif isinstance(config_or_updates, dict):
                new_config = self.config.update(**config_or_updates)
            else:
                raise ConfigurationError(
                    "Configuration updates must be a RAGConfig or dictionary",
                    ErrorCode.CONFIG_INVALID,
                )
        except ConfigurationError:
            raise
        except ValueError as exc:
            raise ConfigurationError(
                f"Configuration update failed validation: {exc}",
                ErrorCode.CONFIG_VALIDATION_FAILED,
                cause=exc,
            ) from exc

        previous_dimension = self.embedding_generator.provider.dimension
        self.config = new_config
        self.document_processor = self._create_document_processor()
        self.embedding_generator = self._create_embedding_generator()

        if self.embedding_generator.provider.dimension != previous_dimension:
            self.vector_store = self._create_vector_store()
            self._record_audit(
                "vector_store_reset",
                {"reason": "embedding dimension changed during configuration update"},
                level="warning",
            )

        self.query_processor = self._create_query_processor()
        self.context_retriever = self._create_context_retriever()
        self.response_generator = self._create_response_generator()
        self._synchronize_dependencies()
        self._refresh_system_status()
        self._record_audit("configuration_updated", self.config.to_dict())
        return self.config

    def get_system_status(self) -> SystemStatus:
        """
        Return the latest status snapshot for the pipeline.
        """
        self._refresh_system_status()
        return self.system_status

    def run_health_check(self) -> Dict[str, Any]:
        """
        Run a lightweight health check over the pipeline components.
        """
        status = self.get_system_status()
        return {
            "healthy": status.is_healthy(),
            "health": status.health.value,
            "components_status": dict(status.components_status),
            "performance_metrics": dict(status.performance_metrics),
            "error_count": status.error_count,
            "last_error": status.last_error,
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Return a copy of the audit trail recorded for pipeline operations.
        """
        return [dict(entry) for entry in self.audit_trail]

    def _create_document_processor(self) -> DocumentProcessor:
        return DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def _create_embedding_generator(self) -> EmbeddingGenerator:
        return EmbeddingGenerator.from_config(
            self.config,
            provider_kwargs=self.embedding_provider_kwargs,
        )

    def _create_vector_store(self) -> VectorStore:
        return VectorStore(
            dimension=self.embedding_generator.provider.dimension,
            index_type=self.vector_store_index_type,
        )

    def _load_or_create_vector_store(self) -> VectorStore:
        metadata_path = Path(self.config.embeddings_directory) / "vector_store_metadata.json"
        if metadata_path.exists():
            try:
                store = VectorStore.load(self.config.embeddings_directory)
                self._record_audit(
                    "vector_store_loaded",
                    {
                        "directory": self.config.embeddings_directory,
                        "vector_store_size": len(store),
                    },
                )
                return store
            except Exception as exc:
                handled = self.error_handler.handle_error(exc, {"operation": "load_vector_store"})
                self.system_status.record_error(handled["message"])
                self._record_audit("vector_store_load_failed", handled, level="warning")

        store = self._create_vector_store()
        self._record_audit(
            "vector_store_initialized",
            {
                "dimension": store.dimension,
                "index_type": store.index_type,
            },
        )
        return store

    def _create_query_processor(self) -> QueryProcessor:
        return QueryProcessor.from_config(
            self.config,
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
        )

    def _create_context_retriever(self) -> ContextRetriever:
        return ContextRetriever(
            query_processor=self.query_processor,
            max_context_length=self.config.max_context_length,
        )

    def _create_response_generator(self) -> ResponseGenerator:
        return ResponseGenerator.from_config(
            self.config,
            provider_kwargs=self.llm_provider_kwargs,
        )

    def _synchronize_dependencies(self) -> None:
        self.query_processor.embedding_generator = self.embedding_generator
        self.query_processor.vector_store = self.vector_store
        self.context_retriever.query_processor = self.query_processor

    def _refresh_system_status(self) -> None:
        uptime = time.perf_counter() - self.started_at
        self.system_status.timestamp = datetime.now()
        self.system_status.uptime_seconds = uptime

        self.system_status.add_component_status("document_processor", "ready")
        self.system_status.add_component_status(
            "embedding_generator",
            self.embedding_generator.provider.get_config()["provider"],
        )
        self.system_status.add_component_status(
            "vector_store",
            f"{self.vector_store.backend}:{len(self.vector_store)}",
        )
        self.system_status.add_component_status("query_processor", "ready")
        self.system_status.add_component_status("context_retriever", "ready")
        self.system_status.add_component_status(
            "response_generator",
            self.response_generator.provider.get_config()["provider"],
        )

        self._update_metrics(
            vector_store_size=float(len(self.vector_store)),
            audit_event_count=float(len(self.audit_trail)),
            error_count=float(self.system_status.error_count),
        )

        if self.system_status.error_count == 0:
            self.system_status.health = SystemHealth.HEALTHY
        elif self.system_status.error_count < 3:
            self.system_status.health = SystemHealth.DEGRADED
        else:
            self.system_status.health = SystemHealth.UNHEALTHY

    def _update_metrics(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self.system_status.add_performance_metric(name, float(value))

    def _record_audit(self, event: str, details: Dict[str, Any], level: str = "info") -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "level": level,
            "details": dict(details),
        }
        self.audit_trail.append(entry)

        log_method = getattr(self.logger, level, self.logger.info)
        log_method("%s | %s", event, details)

    def _build_insufficient_context_response(self, message: str, generation_time: float) -> GeneratedResponse:
        provider = self.response_generator.provider
        return GeneratedResponse(
            response_text=message,
            sources=[],
            confidence_score=0.0,
            context_used=False,
            token_count=provider.count_tokens(message),
            model_used=provider.model_name,
            generation_time=max(generation_time, 0.0),
        )

    def _finalize_query_metrics(self, start_time: float) -> None:
        elapsed = time.perf_counter() - start_time
        self._update_metrics(
            last_query_seconds=elapsed,
            queries_processed_total=float(
                self.system_status.performance_metrics.get("queries_processed_total", 0.0) + 1
            ),
            vector_store_size=float(len(self.vector_store)),
        )
        self._refresh_system_status()
