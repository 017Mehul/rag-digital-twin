"""
RAGConfig data model for system configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json


@dataclass
class RAGConfig:
    """
    Configuration settings for the RAG Digital Twin system.
    
    Centralizes all configurable parameters for embedding models,
    LLMs, processing parameters, and system behavior.
    """
    # Model Configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 4000
    
    # Retrieval Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Response Configuration
    max_response_tokens: int = 500
    temperature: float = 0.1
    
    # System Configuration
    batch_size: int = 10
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # File paths
    data_directory: str = "data"
    embeddings_directory: str = "embeddings"
    logs_directory: str = "logs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []
        
        # Validate providers
        valid_providers = ["openai", "huggingface"]
        if self.embedding_provider not in valid_providers:
            errors.append(f"Invalid embedding provider: {self.embedding_provider}")
        if self.llm_provider not in valid_providers:
            errors.append(f"Invalid LLM provider: {self.llm_provider}")
        
        # Validate numeric parameters
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        if self.chunk_overlap < 0:
            errors.append("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        if self.max_context_length <= 0:
            errors.append("Max context length must be positive")
        if self.top_k_results <= 0:
            errors.append("Top-k results must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        if self.max_response_tokens <= 0:
            errors.append("Max response tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            errors.append("Temperature must be between 0.0 and 2.0")
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.max_retries < 0:
            errors.append("Max retries cannot be negative")
        if self.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed: " + "; ".join(errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary for serialization."""
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_context_length": self.max_context_length,
            "top_k_results": self.top_k_results,
            "similarity_threshold": self.similarity_threshold,
            "max_response_tokens": self.max_response_tokens,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "data_directory": self.data_directory,
            "embeddings_directory": self.embeddings_directory,
            "logs_directory": self.logs_directory
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Create RAGConfig from a dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'RAGConfig':
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json_file(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update(self, **kwargs) -> 'RAGConfig':
        """Create a new configuration with updated parameters."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return RAGConfig.from_dict(current_dict)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"RAGConfig(embedding={self.embedding_provider}/{self.embedding_model}, llm={self.llm_provider}/{self.llm_model})"