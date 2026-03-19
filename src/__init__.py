"""
RAG Digital Twin - A sophisticated AI-powered system implementing 
Retrieval-Augmented Generation for domain-specific knowledge bases.
"""

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

__version__ = "1.0.0"
__author__ = "RAG Digital Twin Team"

__all__ = ["DocumentProcessor", "EmbeddingGenerator", "VectorStore"]
