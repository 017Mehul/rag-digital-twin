"""
Unit tests for core data models in the RAG Digital Twin system.
"""

import pytest
from datetime import datetime
from hypothesis import given, strategies as st

from src.models.document_chunk import DocumentChunk
from src.models.embedding_metadata import EmbeddingMetadata
from src.models.rag_config import RAGConfig
from src.models.search_results import SearchResults, QueryResults, RetrievedContext, GeneratedResponse
from src.models.system_status import SystemStatus, IngestionResults, SystemHealth
from src.exceptions import ConfigurationError, ErrorCode


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""
    
    def test_document_chunk_creation(self, sample_document_chunk):
        """Test basic DocumentChunk creation."""
        chunk = sample_document_chunk
        assert chunk.content
        assert chunk.source_file
        assert chunk.chunk_id
        assert isinstance(chunk.created_at, datetime)
    
    def test_document_chunk_validation(self):
        """Test DocumentChunk validation."""
        # Test empty content
        with pytest.raises(ValueError, match="content cannot be empty"):
            DocumentChunk(content="", source_file="test.txt")
        
        # Test empty source file
        with pytest.raises(ValueError, match="Source file must be specified"):
            DocumentChunk(content="test content", source_file="")
    
    def test_document_chunk_serialization(self, sample_document_chunk):
        """Test DocumentChunk to_dict and from_dict methods."""
        chunk = sample_document_chunk
        chunk_dict = chunk.to_dict()
        
        assert "content" in chunk_dict
        assert "metadata" in chunk_dict
        assert "source_file" in chunk_dict
        assert "chunk_id" in chunk_dict
        
        # Test round-trip serialization
        restored_chunk = DocumentChunk.from_dict(chunk_dict)
        assert restored_chunk.content == chunk.content
        assert restored_chunk.source_file == chunk.source_file
        assert restored_chunk.chunk_id == chunk.chunk_id
    
    def test_content_preview(self, sample_document_chunk):
        """Test content preview functionality."""
        chunk = sample_document_chunk
        preview = chunk.get_content_preview(50)
        assert len(preview) <= 53  # 50 + "..."
        
        # Test with content shorter than max_length
        short_preview = chunk.get_content_preview(1000)
        assert short_preview == chunk.content


class TestEmbeddingMetadata:
    """Test cases for EmbeddingMetadata model."""
    
    def test_embedding_metadata_creation(self, sample_embedding_metadata):
        """Test basic EmbeddingMetadata creation."""
        metadata = sample_embedding_metadata
        assert metadata.chunk_id
        assert metadata.source_file
        assert metadata.content_preview
        assert metadata.embedding_model
        assert isinstance(metadata.created_at, datetime)
    
    def test_embedding_metadata_validation(self):
        """Test EmbeddingMetadata validation."""
        # Test empty chunk_id
        with pytest.raises(ValueError, match="Chunk ID cannot be empty"):
            EmbeddingMetadata(
                chunk_id="",
                source_file="test.txt",
                content_preview="preview",
                embedding_model="model"
            )
    
    def test_from_document_chunk(self, sample_document_chunk):
        """Test creating EmbeddingMetadata from DocumentChunk."""
        metadata = EmbeddingMetadata.from_document_chunk(
            sample_document_chunk, 
            "text-embedding-ada-002"
        )
        
        assert metadata.chunk_id == sample_document_chunk.chunk_id
        assert metadata.source_file == sample_document_chunk.source_file
        assert metadata.embedding_model == "text-embedding-ada-002"
        assert len(metadata.content_preview) <= 103  # 100 + "..."


class TestRAGConfig:
    """Test cases for RAGConfig model."""
    
    def test_rag_config_creation(self, sample_rag_config):
        """Test basic RAGConfig creation."""
        config = sample_rag_config
        assert config.embedding_provider
        assert config.llm_provider
        assert config.chunk_size > 0
        assert 0.0 <= config.similarity_threshold <= 1.0
    
    def test_rag_config_validation(self):
        """Test RAGConfig validation."""
        # Test invalid provider
        with pytest.raises(ValueError, match="Invalid embedding provider"):
            RAGConfig(embedding_provider="invalid_provider")
        
        # Test invalid chunk size
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            RAGConfig(chunk_size=0)
        
        # Test invalid similarity threshold
        with pytest.raises(ValueError, match="Similarity threshold must be between"):
            RAGConfig(similarity_threshold=1.5)
    
    def test_config_serialization(self, sample_rag_config):
        """Test RAGConfig serialization."""
        config = sample_rag_config
        config_dict = config.to_dict()
        
        # Test round-trip serialization
        restored_config = RAGConfig.from_dict(config_dict)
        assert restored_config.embedding_provider == config.embedding_provider
        assert restored_config.chunk_size == config.chunk_size
        assert restored_config.similarity_threshold == config.similarity_threshold
    
    def test_config_update(self, sample_rag_config):
        """Test RAGConfig update functionality."""
        config = sample_rag_config
        updated_config = config.update(chunk_size=2000, top_k_results=10)
        
        assert updated_config.chunk_size == 2000
        assert updated_config.top_k_results == 10
        assert updated_config.embedding_provider == config.embedding_provider  # Unchanged


class TestSearchResults:
    """Test cases for SearchResults model."""
    
    def test_search_results_creation(self, sample_search_results):
        """Test basic SearchResults creation."""
        results = sample_search_results
        assert len(results.indices) == len(results.distances) == len(results.metadata)
        assert len(results) == 3
        assert not results.is_empty()
    
    def test_search_results_validation(self):
        """Test SearchResults validation."""
        with pytest.raises(ValueError, match="must have the same length"):
            SearchResults(
                indices=[1, 2],
                distances=[0.1],
                metadata=[{"id": "1"}]
            )
    
    def test_get_top_k(self, sample_search_results):
        """Test top-k result limiting."""
        results = sample_search_results
        top_2 = results.get_top_k(2)
        
        assert len(top_2) == 2
        assert top_2.indices == results.indices[:2]
        assert top_2.distances == results.distances[:2]


class TestSystemStatus:
    """Test cases for SystemStatus model."""
    
    def test_system_status_creation(self, sample_system_status):
        """Test basic SystemStatus creation."""
        status = sample_system_status
        assert status.health == SystemHealth.HEALTHY
        assert status.is_healthy()
        assert len(status.components_status) > 0
        assert len(status.performance_metrics) > 0
    
    def test_error_recording(self):
        """Test error recording functionality."""
        status = SystemStatus(health=SystemHealth.HEALTHY)
        status.record_error("Test error message")
        
        assert status.error_count == 1
        assert status.last_error == "Test error message"
        assert status.health == SystemHealth.DEGRADED
    
    def test_status_serialization(self, sample_system_status):
        """Test SystemStatus serialization."""
        status = sample_system_status
        status_dict = status.to_dict()
        
        assert "health" in status_dict
        assert "components_status" in status_dict
        assert "performance_metrics" in status_dict
        assert status_dict["health"] == "healthy"


class TestIngestionResults:
    """Test cases for IngestionResults model."""
    
    def test_ingestion_results_creation(self, sample_ingestion_results):
        """Test basic IngestionResults creation."""
        results = sample_ingestion_results
        assert results.total_documents == 5
        assert results.successful_documents == 3
        assert results.failed_documents == 2
        assert results.has_errors()
        assert not results.is_successful()
    
    def test_success_rate_calculation(self, sample_ingestion_results):
        """Test success rate calculation."""
        results = sample_ingestion_results
        expected_rate = 3 / 5  # 3 successful out of 5 total
        assert results.get_success_rate() == expected_rate
    
    def test_ingestion_results_serialization(self, sample_ingestion_results):
        """Test IngestionResults serialization."""
        results = sample_ingestion_results
        results_dict = results.to_dict()
        
        assert "total_documents" in results_dict
        assert "success_rate" in results_dict
        assert "errors" in results_dict
        assert results_dict["success_rate"] == results.get_success_rate()


# Property-based tests
class TestModelProperties:
    """Property-based tests for data models."""
    
    @given(
        content=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        source_file=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    )
    def test_document_chunk_content_preservation(self, content, source_file):
        """Property: DocumentChunk should preserve content exactly."""
        chunk = DocumentChunk(content=content, source_file=source_file)
        assert chunk.content == content
        assert chunk.source_file == source_file
    
    @given(
        chunk_size=st.integers(min_value=100, max_value=5000),
        chunk_overlap=st.integers(min_value=0, max_value=500)
    )
    def test_rag_config_chunk_overlap_constraint(self, chunk_size, chunk_overlap):
        """Property: Chunk overlap must be less than chunk size."""
        if chunk_overlap >= chunk_size:
            with pytest.raises(ValueError):
                RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            config = RAGConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            assert config.chunk_overlap < config.chunk_size
    
    @given(
        similarity_threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    def test_rag_config_similarity_threshold_bounds(self, similarity_threshold):
        """Property: Similarity threshold must be between 0.0 and 1.0."""
        config = RAGConfig(similarity_threshold=similarity_threshold)
        assert 0.0 <= config.similarity_threshold <= 1.0