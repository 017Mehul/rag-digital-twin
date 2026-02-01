"""
DocumentChunk data model for representing processed document segments.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class DocumentChunk:
    """
    Represents a chunk of text extracted from a document with associated metadata.
    
    This is the core data structure that flows through the RAG pipeline,
    maintaining the relationship between content and its source.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the document chunk after initialization."""
        if not self.content.strip():
            raise ValueError("Document chunk content cannot be empty")
        if not self.source_file:
            raise ValueError("Source file must be specified")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document chunk to a dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create a DocumentChunk from a dictionary."""
        # Parse datetime if it's a string
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            source_file=data["source_file"],
            chunk_id=data.get("chunk_id", str(uuid.uuid4())),
            embedding_id=data.get("embedding_id"),
            created_at=created_at
        )
    
    def get_content_preview(self, max_length: int = 100) -> str:
        """Get a preview of the content for display purposes."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def __str__(self) -> str:
        """String representation of the document chunk."""
        return f"DocumentChunk(id={self.chunk_id[:8]}, source={self.source_file}, content_length={len(self.content)})"