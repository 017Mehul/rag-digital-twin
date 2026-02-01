"""
Pytest configuration and shared fixtures for the RAG Digital Twin test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.models.document_chunk import DocumentChunk
from src.models.embedding_metadata import EmbeddingMetadata
from src.models.rag_config import RAGConfig
from src.models.search_results import SearchResults, QueryResults, RetrievedContext, GeneratedResponse
from src.models.system_status import SystemStatus, IngestionResults, SystemHealth


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_document_chunk():
    """Create a sample DocumentChunk for testing."""
    return DocumentChunk(
        content="This is a sample document chunk for testing purposes. It contains enough text to be meaningful.",
        metadata={"title": "Test Document", "author": "Test Author"},
        source_file="test_document.txt",
        chunk_id="test-chunk-001"
    )


@pytest.fixture
def sample_document_chunks():
    """Create multiple sample DocumentChunks for testing."""
    return [
        DocumentChunk(
            content="First chunk of test content with some meaningful text.",
            metadata={"title": "Test Doc 1", "page": 1},
            source_file="doc1.txt",
            chunk_id="chunk-001"
        ),
        DocumentChunk(
            content="Second chunk with different content for testing retrieval.",
            metadata={"title": "Test Doc 2", "page": 1},
            source_file="doc2.txt",
            chunk_id="chunk-002"
        ),
        DocumentChunk(
            content="Third chunk containing additional test information.",
            metadata={"title": "Test Doc 1", "page": 2},
            source_file="doc1.txt",
            chunk_id="chunk-003"
        )
    ]


@pytest.fixture
def sample_embedding_metadata():
    """Create sample EmbeddingMetadata for testing."""
    return EmbeddingMetadata(
        chunk_id="test-chunk-001",
        source_file="test_document.txt",
        content_preview="This is a sample document chunk for testing purposes...",
        embedding_model="text-embedding-ada-002"
    )


@pytest.fixture
def sample_rag_config():
    """Create a sample RAGConfig for testing."""
    return RAGConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-ada-002",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=50,
        top_k_results=3,
        similarity_threshold=0.8
    )


@pytest.fixture
def sample_search_results():
    """Create sample SearchResults for testing."""
    return SearchResults(
        indices=[0, 1, 2],
        distances=[0.1, 0.3, 0.5],
        metadata=[
            {"chunk_id": "chunk-001", "source": "doc1.txt"},
            {"chunk_id": "chunk-002", "source": "doc2.txt"},
            {"chunk_id": "chunk-003", "source": "doc1.txt"}
        ]
    )


@pytest.fixture
def sample_query_results(sample_document_chunks):
    """Create sample QueryResults for testing."""
    return QueryResults(
        query="test query",
        retrieved_chunks=sample_document_chunks,
        relevance_scores=[0.9, 0.7, 0.5],
        total_results=3
    )


@pytest.fixture
def sample_retrieved_context(sample_document_chunks):
    """Create sample RetrievedContext for testing."""
    return RetrievedContext(
        formatted_context="Context: First chunk... Second chunk... Third chunk...",
        source_chunks=sample_document_chunks,
        total_tokens=150,
        sources=["doc1.txt", "doc2.txt"]
    )


@pytest.fixture
def sample_generated_response():
    """Create sample GeneratedResponse for testing."""
    return GeneratedResponse(
        response_text="This is a generated response based on the retrieved context.",
        sources=["doc1.txt", "doc2.txt"],
        confidence_score=0.85,
        context_used=True,
        token_count=25,
        model_used="gpt-3.5-turbo",
        generation_time=1.5
    )


@pytest.fixture
def sample_system_status():
    """Create sample SystemStatus for testing."""
    status = SystemStatus(
        health=SystemHealth.HEALTHY,
        uptime_seconds=3600.0
    )
    status.add_component_status("document_processor", "active")
    status.add_component_status("embedding_generator", "active")
    status.add_component_status("vector_store", "active")
    status.add_performance_metric("avg_query_time", 0.5)
    status.add_performance_metric("documents_processed", 100.0)
    return status


@pytest.fixture
def sample_ingestion_results():
    """Create sample IngestionResults for testing."""
    results = IngestionResults(
        total_documents=5,
        processing_time=10.5
    )
    results.add_successful_document("doc1.txt", 3)
    results.add_successful_document("doc2.txt", 2)
    results.add_successful_document("doc3.txt", 4)
    results.add_failed_document("doc4.txt", "Invalid format")
    results.add_failed_document("doc5.txt", "File not found")
    return results


@pytest.fixture
def sample_text_files(temp_directory):
    """Create sample text files for testing document processing."""
    files = []
    
    # Create sample text files
    for i in range(3):
        file_path = Path(temp_directory) / f"sample_{i}.txt"
        content = f"This is sample document {i}. " * 50  # Create substantial content
        file_path.write_text(content)
        files.append(str(file_path))
    
    return files


# Property-based testing configuration
@pytest.fixture
def hypothesis_settings():
    """Configure hypothesis settings for property-based tests."""
    from hypothesis import settings
    return settings(max_examples=100, deadline=None)


# Test data generators for property-based testing
class TestDataGenerators:
    """Utility class for generating test data with hypothesis."""
    
    @staticmethod
    def valid_chunk_content():
        """Generate valid document chunk content."""
        from hypothesis import strategies as st
        return st.text(min_size=1, max_size=5000).filter(lambda x: x.strip())
    
    @staticmethod
    def valid_file_paths():
        """Generate valid file paths."""
        from hypothesis import strategies as st
        return st.text(min_size=1, max_size=100).filter(
            lambda x: x.strip() and not any(c in x for c in ['<', '>', ':', '"', '|', '?', '*'])
        )
    
    @staticmethod
    def valid_chunk_sizes():
        """Generate valid chunk sizes."""
        from hypothesis import strategies as st
        return st.integers(min_value=100, max_value=5000)
    
    @staticmethod
    def valid_similarity_scores():
        """Generate valid similarity scores."""
        from hypothesis import strategies as st
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    
    @staticmethod
    def valid_embedding_dimensions():
        """Generate valid embedding dimensions."""
        from hypothesis import strategies as st
        return st.integers(min_value=50, max_value=2048)


@pytest.fixture
def test_generators():
    """Provide test data generators for property-based testing."""
    return TestDataGenerators