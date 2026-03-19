"""
Context retrieval and formatting utilities for response generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models.document_chunk import DocumentChunk
from .models.search_results import QueryResults, RetrievedContext
from .query_processor import QueryProcessor


class ContextRetriever:
    """
    Convert ranked query results into bounded, deduplicated response context.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        max_context_length: int = 4000,
        insufficient_context_message: str = "Insufficient context to answer this query reliably.",
        min_overlap_chars: int = 20,
    ) -> None:
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if min_overlap_chars < 0:
            raise ValueError("min_overlap_chars cannot be negative")

        self.query_processor = query_processor
        self.max_context_length = max_context_length
        self.insufficient_context_message = insufficient_context_message
        self.min_overlap_chars = min_overlap_chars

    def retrieve_context(
        self,
        query: str,
        max_context_length: Optional[int] = None,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RetrievedContext:
        """
        Retrieve, deduplicate, and format context for a user query.
        """
        query_results = self.query_processor.process_query(query, k=k, threshold=threshold)
        return self.retrieve_context_from_results(query_results, max_context_length=max_context_length)

    def retrieve_context_from_results(
        self,
        query_results: QueryResults,
        max_context_length: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Build a RetrievedContext instance from already-processed query results.
        """
        limit = self.max_context_length if max_context_length is None else max_context_length
        if limit <= 0:
            raise ValueError("max_context_length must be positive")

        if query_results.is_empty():
            message = query_results.message or self.insufficient_context_message
            return RetrievedContext(
                formatted_context=message,
                source_chunks=[],
                total_tokens=self.count_tokens(message),
                sources=[],
                insufficient_context=True,
            )

        deduplicated_chunks = self.deduplicate_chunks(query_results.retrieved_chunks)
        formatted_context, included_chunks = self._build_context(
            deduplicated_chunks,
            max_context_length=limit,
        )

        if not formatted_context.strip():
            message = query_results.message or self.insufficient_context_message
            return RetrievedContext(
                formatted_context=message,
                source_chunks=[],
                total_tokens=self.count_tokens(message),
                sources=[],
                insufficient_context=True,
            )

        sources = self._get_sources(included_chunks)
        return RetrievedContext(
            formatted_context=formatted_context,
            source_chunks=included_chunks,
            total_tokens=self.count_tokens(formatted_context),
            sources=sources,
            insufficient_context=False,
        )

    def format_context(
        self,
        chunks: List[DocumentChunk],
        max_context_length: Optional[int] = None,
    ) -> str:
        """
        Format chunks into a bounded context string with source attribution.
        """
        if not chunks:
            return ""

        limit = self.max_context_length if max_context_length is None else max_context_length
        if limit <= 0:
            raise ValueError("max_context_length must be positive")

        formatted_context, _ = self._build_context(chunks, max_context_length=limit)
        return formatted_context

    def deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Remove duplicate or highly overlapping chunk content while preserving order.
        """
        deduplicated: List[DocumentChunk] = []
        seen_signatures = set()
        last_content_by_source: Dict[str, str] = {}

        for chunk in chunks:
            normalized_content = self._normalize_text(chunk.content)
            if not normalized_content:
                continue
            if normalized_content in seen_signatures:
                continue
            if any(
                normalized_content in existing_signature or existing_signature in normalized_content
                for existing_signature in seen_signatures
            ):
                continue

            trimmed_content = chunk.content.strip()
            previous_content = last_content_by_source.get(chunk.source_file, "")
            overlap_size = self._find_overlap(previous_content, trimmed_content)
            if overlap_size >= self.min_overlap_chars:
                trimmed_content = trimmed_content[overlap_size:].lstrip()

            if not trimmed_content.strip():
                continue

            deduplicated_chunk = DocumentChunk(
                content=trimmed_content,
                metadata=dict(chunk.metadata),
                source_file=chunk.source_file,
                chunk_id=chunk.chunk_id,
                embedding_id=chunk.embedding_id,
                created_at=chunk.created_at,
            )
            deduplicated.append(deduplicated_chunk)
            signature = self._normalize_text(trimmed_content)
            seen_signatures.add(signature)
            last_content_by_source[chunk.source_file] = trimmed_content

        return deduplicated

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    def _build_context(
        self,
        chunks: List[DocumentChunk],
        max_context_length: int,
    ) -> tuple[str, List[DocumentChunk]]:
        sections: List[str] = []
        included_chunks: List[DocumentChunk] = []
        used_length = 0

        for position, chunk in enumerate(chunks, start=1):
            section = self._format_chunk(chunk, position)
            separator_length = 2 if sections else 0
            section_length = len(section)

            if used_length + separator_length + section_length <= max_context_length:
                sections.append(section)
                included_chunks.append(chunk)
                used_length += separator_length + section_length
                continue

            remaining = max_context_length - used_length - separator_length
            if remaining <= 0:
                break

            truncated_section = section[:remaining].rstrip()
            if truncated_section:
                sections.append(truncated_section)
                included_chunks.append(chunk)
            break

        return "\n\n".join(sections).strip(), included_chunks

    def _format_chunk(self, chunk: DocumentChunk, position: int) -> str:
        location = ""
        if "chunk_index" in chunk.metadata:
            location = f", chunk {chunk.metadata['chunk_index']}"
        return f"[Source {position}] {chunk.source_file}{location}\n{chunk.content.strip()}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()

    def _find_overlap(self, left: str, right: str) -> int:
        left_text = left.strip()
        right_text = right.strip()
        max_overlap = min(len(left_text), len(right_text))
        if max_overlap < self.min_overlap_chars:
            return 0

        for size in range(max_overlap, self.min_overlap_chars - 1, -1):
            if left_text[-size:] == right_text[:size]:
                return size

        return 0

    @staticmethod
    def _get_sources(chunks: List[DocumentChunk]) -> List[str]:
        sources: List[str] = []
        for chunk in chunks:
            if chunk.source_file not in sources:
                sources.append(chunk.source_file)
        return sources
