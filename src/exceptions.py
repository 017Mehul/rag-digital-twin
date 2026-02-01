"""
Exception classes and error handling framework for the RAG Digital Twin system.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Standard error codes for the RAG system."""
    # Document Processing Errors
    DOCUMENT_NOT_FOUND = "DOC_001"
    DOCUMENT_INVALID_FORMAT = "DOC_002"
    DOCUMENT_CORRUPTED = "DOC_003"
    DOCUMENT_TOO_LARGE = "DOC_004"
    DOCUMENT_EMPTY = "DOC_005"
    
    # Embedding Generation Errors
    EMBEDDING_MODEL_NOT_FOUND = "EMB_001"
    EMBEDDING_API_ERROR = "EMB_002"
    EMBEDDING_RATE_LIMIT = "EMB_003"
    EMBEDDING_DIMENSION_MISMATCH = "EMB_004"
    EMBEDDING_GENERATION_FAILED = "EMB_005"
    
    # Vector Store Errors
    VECTOR_STORE_NOT_INITIALIZED = "VEC_001"
    VECTOR_STORE_INDEX_CORRUPTED = "VEC_002"
    VECTOR_STORE_SEARCH_FAILED = "VEC_003"
    VECTOR_STORE_SAVE_FAILED = "VEC_004"
    VECTOR_STORE_LOAD_FAILED = "VEC_005"
    
    # Query Processing Errors
    QUERY_EMPTY = "QRY_001"
    QUERY_TOO_LONG = "QRY_002"
    QUERY_PROCESSING_FAILED = "QRY_003"
    QUERY_NO_RESULTS = "QRY_004"
    
    # Response Generation Errors
    LLM_MODEL_NOT_FOUND = "LLM_001"
    LLM_API_ERROR = "LLM_002"
    LLM_RATE_LIMIT = "LLM_003"
    LLM_CONTEXT_TOO_LONG = "LLM_004"
    LLM_GENERATION_FAILED = "LLM_005"
    
    # Configuration Errors
    CONFIG_INVALID = "CFG_001"
    CONFIG_MISSING_REQUIRED = "CFG_002"
    CONFIG_VALIDATION_FAILED = "CFG_003"
    
    # System Errors
    SYSTEM_INITIALIZATION_FAILED = "SYS_001"
    SYSTEM_RESOURCE_EXHAUSTED = "SYS_002"
    SYSTEM_TIMEOUT = "SYS_003"
    SYSTEM_UNKNOWN_ERROR = "SYS_999"


class RAGException(Exception):
    """
    Base exception class for all RAG Digital Twin system errors.
    
    Provides structured error information including error codes,
    component context, and additional metadata for debugging.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode, 
        component: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a RAG exception.
        
        Args:
            message: Human-readable error message
            error_code: Standardized error code
            component: System component where error occurred
            details: Additional error context and metadata
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.component = component
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "component": self.component,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        return f"[{self.error_code.value}] {self.component}: {self.message}"


class DocumentProcessingError(RAGException):
    """Exception raised during document processing operations."""
    
    def __init__(self, message: str, error_code: ErrorCode, file_path: str = "", cause: Optional[Exception] = None):
        details = {"file_path": file_path} if file_path else {}
        super().__init__(message, error_code, "DocumentProcessor", details, cause)


class EmbeddingGenerationError(RAGException):
    """Exception raised during embedding generation operations."""
    
    def __init__(self, message: str, error_code: ErrorCode, model_name: str = "", cause: Optional[Exception] = None):
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, error_code, "EmbeddingGenerator", details, cause)


class VectorStoreError(RAGException):
    """Exception raised during vector store operations."""
    
    def __init__(self, message: str, error_code: ErrorCode, index_type: str = "", cause: Optional[Exception] = None):
        details = {"index_type": index_type} if index_type else {}
        super().__init__(message, error_code, "VectorStore", details, cause)


class QueryProcessingError(RAGException):
    """Exception raised during query processing operations."""
    
    def __init__(self, message: str, error_code: ErrorCode, query: str = "", cause: Optional[Exception] = None):
        details = {"query": query[:100] + "..." if len(query) > 100 else query} if query else {}
        super().__init__(message, error_code, "QueryProcessor", details, cause)


class ResponseGenerationError(RAGException):
    """Exception raised during response generation operations."""
    
    def __init__(self, message: str, error_code: ErrorCode, model_name: str = "", cause: Optional[Exception] = None):
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, error_code, "ResponseGenerator", details, cause)


class ConfigurationError(RAGException):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, config_key: str = "", cause: Optional[Exception] = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, error_code, "Configuration", details, cause)


class SystemError(RAGException):
    """Exception raised for system-level errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, system_component: str = "", cause: Optional[Exception] = None):
        details = {"system_component": system_component} if system_component else {}
        super().__init__(message, error_code, "System", details, cause)


class ErrorHandler:
    """
    Centralized error handling and logging for the RAG system.
    
    Provides consistent error processing, logging, and recovery strategies
    across all system components.
    """
    
    def __init__(self, logger=None):
        """Initialize the error handler with optional logger."""
        self.logger = logger
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an error with appropriate logging and response generation.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Dict containing error response information
        """
        if isinstance(error, RAGException):
            return self._handle_rag_error(error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_rag_error(self, error: RAGException, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG-specific exceptions."""
        error_key = f"{error.component}:{error.error_code.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if self.logger:
            self.logger.error(f"RAG Error: {error}", extra={"context": context, "error_details": error.to_dict()})
        
        return {
            "error": True,
            "message": error.message,
            "error_code": error.error_code.value,
            "component": error.component,
            "recoverable": self._is_recoverable_error(error.error_code),
            "retry_suggested": self._should_retry(error.error_code)
        }
    
    def _handle_generic_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic Python exceptions."""
        error_key = f"Generic:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if self.logger:
            self.logger.error(f"Unexpected error: {error}", extra={"context": context})
        
        return {
            "error": True,
            "message": str(error),
            "error_code": ErrorCode.SYSTEM_UNKNOWN_ERROR.value,
            "component": "Unknown",
            "recoverable": False,
            "retry_suggested": False
        }
    
    def _is_recoverable_error(self, error_code: ErrorCode) -> bool:
        """Determine if an error is recoverable."""
        recoverable_errors = {
            ErrorCode.EMBEDDING_RATE_LIMIT,
            ErrorCode.LLM_RATE_LIMIT,
            ErrorCode.SYSTEM_TIMEOUT,
            ErrorCode.EMBEDDING_API_ERROR,
            ErrorCode.LLM_API_ERROR
        }
        return error_code in recoverable_errors
    
    def _should_retry(self, error_code: ErrorCode) -> bool:
        """Determine if an operation should be retried."""
        retry_errors = {
            ErrorCode.EMBEDDING_RATE_LIMIT,
            ErrorCode.LLM_RATE_LIMIT,
            ErrorCode.SYSTEM_TIMEOUT,
            ErrorCode.EMBEDDING_API_ERROR,
            ErrorCode.LLM_API_ERROR
        }
        return error_code in retry_errors
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return self.error_counts.copy()
    
    def reset_error_counts(self) -> None:
        """Reset error occurrence counters."""
        self.error_counts.clear()