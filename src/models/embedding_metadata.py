"""
EmbeddingMetadata data model for tracking embedding information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class EmbeddingMetadata:
    """
    Metadata associated with vector embeddings in the system.
    
    Maintains the relationship between embeddings and their source content
    for explainability and debugging purposes.
    """
    chunk_id: str
    source_file: str
    content_preview: str
    embedding_model: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the embedding metadata after initialization."""
        if not self.chunk_id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.source_file:
            raise ValueError("Source file cannot be empty")
        if not self.embedding_model:
            raise ValueError("Embedding model cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the embedding metadata to a dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "content_preview": self.content_preview,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create EmbeddingMetadata from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        return cls(
            chunk_id=data["chunk_id"],
            source_file=data["source_file"],
            content_preview=data["content_preview"],
            embedding_model=data["embedding_model"],
            created_at=created_at
        )
    
    @classmethod
    def from_document_chunk(cls, chunk: 'DocumentChunk', embedding_model: str, preview_length: int = 100) -> 'EmbeddingMetadata':
        """Create EmbeddingMetadata from a DocumentChunk."""
        content_preview = chunk.content[:preview_length]
        if len(chunk.content) > preview_length:
            content_preview += "..."
            
        return cls(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            content_preview=content_preview,
            embedding_model=embedding_model
        )
    
    def __str__(self) -> str:
        """String representation of the embedding metadata."""
        return f"EmbeddingMetadata(chunk_id={self.chunk_id[:8]}, model={self.embedding_model})"