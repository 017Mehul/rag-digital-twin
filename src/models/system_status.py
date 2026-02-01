"""
System status and operational result data models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class SystemHealth(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SystemStatus:
    """
    Current status and health information for the RAG system.
    
    Provides comprehensive information about system components,
    performance metrics, and operational state.
    """
    health: SystemHealth = SystemHealth.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    components_status: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if the system is in a healthy state."""
        return self.health == SystemHealth.HEALTHY
    
    def add_component_status(self, component: str, status: str) -> None:
        """Add or update the status of a system component."""
        self.components_status[component] = status
    
    def add_performance_metric(self, metric: str, value: float) -> None:
        """Add or update a performance metric."""
        self.performance_metrics[metric] = value
    
    def record_error(self, error_message: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error_message
        if self.health == SystemHealth.HEALTHY:
            self.health = SystemHealth.DEGRADED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system status to dictionary for serialization."""
        return {
            "health": self.health.value,
            "timestamp": self.timestamp.isoformat(),
            "components_status": self.components_status,
            "performance_metrics": self.performance_metrics,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class IngestionResults:
    """
    Results from document ingestion operations.
    
    Tracks the success and failure of document processing,
    providing detailed information about the ingestion process.
    """
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate ingestion results after initialization."""
        if self.total_documents < 0:
            raise ValueError("Total documents cannot be negative")
        if self.successful_documents < 0:
            raise ValueError("Successful documents cannot be negative")
        if self.failed_documents < 0:
            raise ValueError("Failed documents cannot be negative")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
    
    def add_successful_document(self, file_path: str, chunks_count: int) -> None:
        """Record a successfully processed document."""
        self.successful_documents += 1
        self.total_chunks += chunks_count
        self.processed_files.append(file_path)
    
    def add_failed_document(self, file_path: str, error_message: str) -> None:
        """Record a failed document processing."""
        self.failed_documents += 1
        self.failed_files.append(file_path)
        self.errors.append(f"{file_path}: {error_message}")
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of document processing."""
        if self.total_documents == 0:
            return 0.0
        return self.successful_documents / self.total_documents
    
    def is_successful(self) -> bool:
        """Check if the ingestion was completely successful."""
        return self.failed_documents == 0 and self.successful_documents > 0
    
    def has_errors(self) -> bool:
        """Check if there were any errors during ingestion."""
        return len(self.errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ingestion results to dictionary for serialization."""
        return {
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
            "processing_time": self.processing_time,
            "success_rate": self.get_success_rate(),
            "errors": self.errors,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files
        }
    
    def __str__(self) -> str:
        """String representation of ingestion results."""
        return (f"IngestionResults(total={self.total_documents}, "
                f"successful={self.successful_documents}, "
                f"failed={self.failed_documents}, "
                f"chunks={self.total_chunks})")