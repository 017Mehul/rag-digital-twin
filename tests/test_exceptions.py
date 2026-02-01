"""
Unit tests for exception classes and error handling framework.
"""

import pytest
from src.exceptions import (
    RAGException, ErrorCode, ErrorHandler,
    DocumentProcessingError, EmbeddingGenerationError, VectorStoreError,
    QueryProcessingError, ResponseGenerationError, ConfigurationError, SystemError
)


class TestRAGException:
    """Test cases for RAGException base class."""
    
    def test_rag_exception_creation(self):
        """Test basic RAGException creation."""
        exception = RAGException(
            message="Test error message",
            error_code=ErrorCode.SYSTEM_UNKNOWN_ERROR,
            component="TestComponent"
        )
        
        assert exception.message == "Test error message"
        assert exception.error_code == ErrorCode.SYSTEM_UNKNOWN_ERROR
        assert exception.component == "TestComponent"
        assert exception.details == {}
        assert exception.cause is None
    
    def test_rag_exception_with_details(self):
        """Test RAGException with additional details."""
        details = {"file_path": "test.txt", "line_number": 42}
        exception = RAGException(
            message="Test error",
            error_code=ErrorCode.DOCUMENT_INVALID_FORMAT,
            component="DocumentProcessor",
            details=details
        )
        
        assert exception.details == details
    
    def test_rag_exception_with_cause(self):
        """Test RAGException with underlying cause."""
        original_error = ValueError("Original error")
        exception = RAGException(
            message="Wrapped error",
            error_code=ErrorCode.SYSTEM_UNKNOWN_ERROR,
            component="TestComponent",
            cause=original_error
        )
        
        assert exception.cause == original_error
    
    def test_rag_exception_serialization(self):
        """Test RAGException to_dict method."""
        exception = RAGException(
            message="Test error",
            error_code=ErrorCode.CONFIG_INVALID,
            component="Configuration",
            details={"key": "value"}
        )
        
        exception_dict = exception.to_dict()
        
        assert exception_dict["message"] == "Test error"
        assert exception_dict["error_code"] == "CFG_001"
        assert exception_dict["component"] == "Configuration"
        assert exception_dict["details"] == {"key": "value"}
        assert exception_dict["cause"] is None
    
    def test_rag_exception_string_representation(self):
        """Test RAGException string representation."""
        exception = RAGException(
            message="Test error",
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            component="DocumentProcessor"
        )
        
        expected_str = "[DOC_001] DocumentProcessor: Test error"
        assert str(exception) == expected_str


class TestSpecificExceptions:
    """Test cases for specific exception types."""
    
    def test_document_processing_error(self):
        """Test DocumentProcessingError creation."""
        error = DocumentProcessingError(
            message="File not found",
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            file_path="missing.txt"
        )
        
        assert error.component == "DocumentProcessor"
        assert error.details["file_path"] == "missing.txt"
    
    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError creation."""
        error = EmbeddingGenerationError(
            message="Model not available",
            error_code=ErrorCode.EMBEDDING_MODEL_NOT_FOUND,
            model_name="invalid-model"
        )
        
        assert error.component == "EmbeddingGenerator"
        assert error.details["model_name"] == "invalid-model"
    
    def test_vector_store_error(self):
        """Test VectorStoreError creation."""
        error = VectorStoreError(
            message="Index corrupted",
            error_code=ErrorCode.VECTOR_STORE_INDEX_CORRUPTED,
            index_type="faiss"
        )
        
        assert error.component == "VectorStore"
        assert error.details["index_type"] == "faiss"
    
    def test_query_processing_error(self):
        """Test QueryProcessingError creation."""
        error = QueryProcessingError(
            message="Query too long",
            error_code=ErrorCode.QUERY_TOO_LONG,
            query="very long query text" * 100
        )
        
        assert error.component == "QueryProcessor"
        # Query should be truncated in details
        assert len(error.details["query"]) <= 103  # 100 + "..."
    
    def test_response_generation_error(self):
        """Test ResponseGenerationError creation."""
        error = ResponseGenerationError(
            message="API error",
            error_code=ErrorCode.LLM_API_ERROR,
            model_name="gpt-3.5-turbo"
        )
        
        assert error.component == "ResponseGenerator"
        assert error.details["model_name"] == "gpt-3.5-turbo"
    
    def test_configuration_error(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            message="Invalid config",
            error_code=ErrorCode.CONFIG_INVALID,
            config_key="embedding_model"
        )
        
        assert error.component == "Configuration"
        assert error.details["config_key"] == "embedding_model"
    
    def test_system_error(self):
        """Test SystemError creation."""
        error = SystemError(
            message="Resource exhausted",
            error_code=ErrorCode.SYSTEM_RESOURCE_EXHAUSTED,
            system_component="memory"
        )
        
        assert error.component == "System"
        assert error.details["system_component"] == "memory"


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def test_error_handler_creation(self):
        """Test ErrorHandler creation."""
        handler = ErrorHandler()
        assert handler.error_counts == {}
    
    def test_handle_rag_error(self):
        """Test handling of RAG-specific errors."""
        handler = ErrorHandler()
        error = DocumentProcessingError(
            message="File not found",
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            file_path="test.txt"
        )
        
        response = handler.handle_error(error, {"context": "test"})
        
        assert response["error"] is True
        assert response["message"] == "File not found"
        assert response["error_code"] == "DOC_001"
        assert response["component"] == "DocumentProcessor"
        assert "recoverable" in response
        assert "retry_suggested" in response
    
    def test_handle_generic_error(self):
        """Test handling of generic Python errors."""
        handler = ErrorHandler()
        error = ValueError("Generic error")
        
        response = handler.handle_error(error, {"context": "test"})
        
        assert response["error"] is True
        assert response["message"] == "Generic error"
        assert response["error_code"] == "SYS_999"
        assert response["component"] == "Unknown"
        assert response["recoverable"] is False
        assert response["retry_suggested"] is False
    
    def test_error_statistics(self):
        """Test error statistics tracking."""
        handler = ErrorHandler()
        
        # Handle multiple errors
        error1 = DocumentProcessingError("Error 1", ErrorCode.DOCUMENT_NOT_FOUND)
        error2 = DocumentProcessingError("Error 2", ErrorCode.DOCUMENT_NOT_FOUND)
        error3 = EmbeddingGenerationError("Error 3", ErrorCode.EMBEDDING_API_ERROR)
        
        handler.handle_error(error1, {})
        handler.handle_error(error2, {})
        handler.handle_error(error3, {})
        
        stats = handler.get_error_statistics()
        assert stats["DocumentProcessor:DOC_001"] == 2
        assert stats["EmbeddingGenerator:EMB_002"] == 1
    
    def test_reset_error_counts(self):
        """Test resetting error counts."""
        handler = ErrorHandler()
        error = DocumentProcessingError("Error", ErrorCode.DOCUMENT_NOT_FOUND)
        
        handler.handle_error(error, {})
        assert len(handler.get_error_statistics()) > 0
        
        handler.reset_error_counts()
        assert handler.get_error_statistics() == {}
    
    def test_recoverable_error_detection(self):
        """Test detection of recoverable errors."""
        handler = ErrorHandler()
        
        # Test recoverable error
        recoverable_error = EmbeddingGenerationError(
            "Rate limit", 
            ErrorCode.EMBEDDING_RATE_LIMIT
        )
        response = handler.handle_error(recoverable_error, {})
        assert response["recoverable"] is True
        assert response["retry_suggested"] is True
        
        # Test non-recoverable error
        non_recoverable_error = DocumentProcessingError(
            "Invalid format", 
            ErrorCode.DOCUMENT_INVALID_FORMAT
        )
        response = handler.handle_error(non_recoverable_error, {})
        assert response["recoverable"] is False
        assert response["retry_suggested"] is False


class TestErrorCodes:
    """Test cases for ErrorCode enumeration."""
    
    def test_error_code_values(self):
        """Test that error codes have expected values."""
        assert ErrorCode.DOCUMENT_NOT_FOUND.value == "DOC_001"
        assert ErrorCode.EMBEDDING_API_ERROR.value == "EMB_002"
        assert ErrorCode.VECTOR_STORE_INDEX_CORRUPTED.value == "VEC_002"
        assert ErrorCode.QUERY_EMPTY.value == "QRY_001"
        assert ErrorCode.LLM_API_ERROR.value == "LLM_002"
        assert ErrorCode.CONFIG_INVALID.value == "CFG_001"
        assert ErrorCode.SYSTEM_UNKNOWN_ERROR.value == "SYS_999"
    
    def test_error_code_uniqueness(self):
        """Test that all error codes are unique."""
        error_codes = [code.value for code in ErrorCode]
        assert len(error_codes) == len(set(error_codes))