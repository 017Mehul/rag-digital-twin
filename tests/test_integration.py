"""
Integration tests for the RAG Digital Twin system structure.
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator
from src.exceptions import ErrorCode
from src.models.rag_config import RAGConfig
from src.models.document_chunk import DocumentChunk
from src.models.embedding_metadata import EmbeddingMetadata
from src.models.system_status import IngestionResults
from src.providers import OpenAIEmbeddingProvider
from src.vector_store import VectorStore
from src.utils.config_utils import load_config, create_default_config
from src.utils.file_utils import ensure_directory, is_valid_file
from src.utils.logging_utils import setup_logging, get_logger


class TestSystemIntegration:
    """Integration tests for system components."""
    
    def test_project_structure_exists(self):
        """Test that all required directories exist."""
        required_dirs = [
            "data", "data/processed", "data/raw",
            "embeddings", "src", "src/models", "src/providers", 
            "src/utils", "config", "logs", "tests"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"
    
    def test_core_models_import(self):
        """Test that all core models can be imported."""
        from src.models import (
            DocumentChunk, EmbeddingMetadata, RAGConfig,
            SearchResults, QueryResults, RetrievedContext, 
            GeneratedResponse, SystemStatus, IngestionResults
        )
        
        # Test that classes can be instantiated
        chunk = DocumentChunk(content="test", source_file="test.txt")
        config = RAGConfig()
        
        assert chunk.content == "test"
        assert config.embedding_provider == "openai"
    
    def test_exception_system_import(self):
        """Test that exception system can be imported and used."""
        from src.exceptions import (
            RAGException, ErrorCode, ErrorHandler,
            DocumentProcessingError, ConfigurationError
        )
        
        # Test exception creation
        error = DocumentProcessingError(
            "Test error", 
            ErrorCode.DOCUMENT_NOT_FOUND,
            "test.txt"
        )
        
        # Test error handler
        handler = ErrorHandler()
        response = handler.handle_error(error, {})
        
        assert response["error"] is True
        assert response["error_code"] == "DOC_001"
    
    def test_utility_functions(self):
        """Test that utility functions work correctly."""
        # Test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_subdir"
            ensure_directory(str(test_dir))
            assert test_dir.exists()
            
            # Test file validation
            test_file = test_dir / "test.txt"
            test_file.write_text("test content")
            
            is_valid, error_msg = is_valid_file(str(test_file))
            assert is_valid
            assert error_msg == ""
    
    def test_configuration_system(self):
        """Test configuration loading and validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Create a test configuration
            test_config = RAGConfig(
                embedding_provider="openai",
                chunk_size=500,
                similarity_threshold=0.8
            )
            test_config.to_json_file(str(config_file))
            
            # Load and validate configuration
            loaded_config = RAGConfig.from_json_file(str(config_file))
            assert loaded_config.chunk_size == 500
            assert loaded_config.similarity_threshold == 0.8
    
    def test_logging_system(self):
        """Test logging system setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Setup logging
            logger = setup_logging(
                log_level="INFO",
                log_file=str(log_file)
            )
            
            # Test logging
            logger.info("Test log message")
            
            # Test component logger
            component_logger = get_logger("TestComponent")
            component_logger.info("Component log message")
            
            # Close all handlers to release file locks
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Verify log file was created
            assert log_file.exists()
    
    def test_data_model_integration(self):
        """Test integration between different data models."""
        # Create a document chunk
        chunk = DocumentChunk(
            content="This is test content for integration testing.",
            metadata={"title": "Test Document"},
            source_file="test.txt"
        )
        
        # Create embedding metadata from chunk
        embedding_metadata = EmbeddingMetadata.from_document_chunk(
            chunk, 
            "text-embedding-ada-002"
        )
        
        # Verify relationships
        assert embedding_metadata.chunk_id == chunk.chunk_id
        assert embedding_metadata.source_file == chunk.source_file
        assert embedding_metadata.embedding_model == "text-embedding-ada-002"
        
        # Test serialization round-trip
        chunk_dict = chunk.to_dict()
        restored_chunk = DocumentChunk.from_dict(chunk_dict)
        
        assert restored_chunk.content == chunk.content
        assert restored_chunk.chunk_id == chunk.chunk_id

    def test_document_processor_to_embedding_metadata_flow(self, temp_directory):
        """Test document processing output flowing into embedding generation."""
        processor = DocumentProcessor(chunk_size=60, chunk_overlap=10)
        generator = EmbeddingGenerator(
            provider=OpenAIEmbeddingProvider(
                model_name="text-embedding-3-small",
                dimension=24,
                mock_embeddings=True,
            )
        )
        file_path = Path(temp_directory) / "integration.txt"
        file_path.write_text(
            "This is a document used for integration testing. " * 6,
            encoding="utf-8"
        )

        chunks = processor.process_document(str(file_path), {"title": "Integration Doc"})
        generated = generator.generate_chunk_embeddings(chunks)

        assert chunks
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata["title"] == "Integration Doc" for chunk in chunks)
        assert all(chunk.metadata["file_name"] == "integration.txt" for chunk in chunks)

        assert len(generated) == len(chunks)
        for result, chunk in zip(generated, chunks):
            metadata = result["metadata"]
            assert metadata.chunk_id == chunk.chunk_id
            assert metadata.source_file == chunk.source_file
            assert metadata.embedding_model == "text-embedding-3-small"
            assert len(result["embedding"]) == 24

    def test_document_processor_batch_results_map_to_ingestion_results(self, temp_directory):
        """Test batch processing outcomes can be represented in ingestion results."""
        processor = DocumentProcessor(chunk_size=80, chunk_overlap=20)
        valid_files = []

        for index in range(2):
            file_path = Path(temp_directory) / f"doc_{index}.txt"
            file_path.write_text(
                f"Document {index} content for batch integration testing. " * 4,
                encoding="utf-8"
            )
            valid_files.append(str(file_path))

        invalid_file = Path(temp_directory) / "invalid.bin"
        invalid_file.write_bytes(b"\x00\x01\x02")

        batch_result = processor.process_batch(valid_files + [str(invalid_file)])

        ingestion_results = IngestionResults(total_documents=batch_result["total_documents"], processing_time=0.0)
        for file_path, chunks in batch_result["processed_documents"].items():
            ingestion_results.add_successful_document(file_path, len(chunks))
        for file_path, error_message in batch_result["failed_documents"].items():
            ingestion_results.add_failed_document(file_path, error_message)

        assert ingestion_results.total_documents == 3
        assert ingestion_results.successful_documents == 2
        assert ingestion_results.failed_documents == 1
        assert ingestion_results.total_chunks == sum(
            len(chunks) for chunks in batch_result["processed_documents"].values()
        )
        assert str(invalid_file) in ingestion_results.failed_files
        assert any(str(invalid_file) in error for error in ingestion_results.errors)
        assert any(ErrorCode.DOCUMENT_INVALID_FORMAT.value in error for error in ingestion_results.errors)

    def test_document_embeddings_can_be_indexed_and_retrieved(self, temp_directory):
        """Test the document processor, embedding generator, and vector store together."""
        processor = DocumentProcessor(chunk_size=80, chunk_overlap=20)
        generator = EmbeddingGenerator(
            provider=OpenAIEmbeddingProvider(
                model_name="text-embedding-3-small",
                dimension=32,
                mock_embeddings=True,
            )
        )
        store = VectorStore(dimension=32, index_type="flat")

        file_path = Path(temp_directory) / "knowledge.txt"
        file_path.write_text(
            "Vector databases support semantic retrieval. "
            "FAISS can store and search embeddings efficiently.",
            encoding="utf-8"
        )

        chunks = processor.process_document(str(file_path))
        generated = generator.generate_chunk_embeddings(chunks)
        store.add_documents(generated)

        query_embedding = generator.generate_embedding("semantic retrieval with embeddings")
        results = store.search(query_embedding, top_k=1)

        assert len(results) == 1
        assert results.metadata[0]["chunk"]["source_file"] == str(file_path)
        assert results.metadata[0]["embedding_metadata"]["embedding_model"] == "text-embedding-3-small"
    
    def test_configuration_file_loading(self):
        """Test loading the actual configuration file."""
        config_path = "config/rag_config.yaml"
        
        if Path(config_path).exists():
            # Test that the config file can be loaded
            # Note: This might fail if environment variables are not set
            try:
                config = load_config(config_path)
                assert isinstance(config, RAGConfig)
            except Exception as e:
                # Expected if environment variables are not set
                assert "Environment variable not found" in str(e)
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and contains expected packages."""
        requirements_file = Path("requirements.txt")
        assert requirements_file.exists()
        
        content = requirements_file.read_text()
        
        # Check for key dependencies
        expected_packages = [
            "numpy", "faiss-cpu", "openai", "transformers",
            "PyPDF2", "pytest", "hypothesis"
        ]
        
        for package in expected_packages:
            assert package in content, f"Missing package in requirements.txt: {package}"
    
    def test_pytest_configuration(self):
        """Test that pytest configuration is properly set up."""
        pytest_config = Path("pytest.ini")
        assert pytest_config.exists()
        
        content = pytest_config.read_text()
        
        # Check for key configuration sections
        assert "[tool:pytest]" in content
        assert "testpaths = tests" in content
        assert "hypothesis" in content


class TestPropertyBasedTestingSetup:
    """Test that property-based testing is properly configured."""
    
    def test_hypothesis_import(self):
        """Test that hypothesis can be imported and used."""
        from hypothesis import given, strategies as st
        
        # Test a simple property
        @given(st.integers())
        def test_property(x):
            assert x == x
        
        # Run the property test
        test_property()
    
    def test_hypothesis_configuration(self):
        """Test hypothesis configuration in conftest.py."""
        from tests.conftest import TestDataGenerators
        
        generators = TestDataGenerators()
        
        # Test that generators are available
        assert hasattr(generators, 'valid_chunk_content')
        assert hasattr(generators, 'valid_file_paths')
        assert hasattr(generators, 'valid_chunk_sizes')
    
    def test_test_fixtures_available(self):
        """Test that test fixtures are properly configured."""
        # This would normally be tested within a pytest context
        # Here we just verify the conftest.py file exists and is importable
        import tests.conftest
        
        # Verify key fixtures are defined
        assert hasattr(tests.conftest, 'sample_document_chunk')
        assert hasattr(tests.conftest, 'sample_rag_config')
        assert hasattr(tests.conftest, 'test_generators')
