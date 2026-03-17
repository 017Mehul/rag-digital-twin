"""
Document processing utilities for extracting and chunking source documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyPDF2 import PdfReader

from .exceptions import DocumentProcessingError, ErrorCode
from .models.document_chunk import DocumentChunk
from .utils.file_utils import is_supported_file_type, is_valid_file


class DocumentProcessor:
    """
    Process TXT and PDF documents into chunked RAG-ready content.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_file_size_mb: Optional[int] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_mb = max_file_size_mb

    def validate_document(self, file_path: str) -> Path:
        """
        Validate that a document exists, is readable, and has a supported format.
        """
        is_valid, error_message = is_valid_file(
            file_path=file_path,
            max_size_mb=self.max_file_size_mb,
        )
        if not is_valid:
            error_code = ErrorCode.DOCUMENT_NOT_FOUND
            if "too large" in error_message.lower():
                error_code = ErrorCode.DOCUMENT_TOO_LARGE
            raise DocumentProcessingError(error_message, error_code, file_path)

        path = Path(file_path)
        if not is_supported_file_type(file_path):
            raise DocumentProcessingError(
                f"Unsupported document format: {path.suffix or '<none>'}",
                ErrorCode.DOCUMENT_INVALID_FORMAT,
                file_path,
            )

        return path

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a supported document.
        """
        path = self.validate_document(file_path)

        try:
            if path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8")
            elif path.suffix.lower() == ".pdf":
                text = self._extract_pdf_text(path)
            else:
                raise DocumentProcessingError(
                    f"Unsupported document format: {path.suffix or '<none>'}",
                    ErrorCode.DOCUMENT_INVALID_FORMAT,
                    file_path,
                )
        except DocumentProcessingError:
            raise
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(
                "Unable to decode text document",
                ErrorCode.DOCUMENT_CORRUPTED,
                file_path,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise DocumentProcessingError(
                "Failed to extract document text",
                ErrorCode.DOCUMENT_CORRUPTED,
                file_path,
                cause=exc,
            ) from exc

        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise DocumentProcessingError(
                "Document does not contain extractable text",
                ErrorCode.DOCUMENT_EMPTY,
                file_path,
            )

        return normalized_text

    def chunk_text(
        self,
        text: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split extracted text into overlapping chunks while preserving metadata.
        """
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise DocumentProcessingError(
                "Cannot chunk empty document text",
                ErrorCode.DOCUMENT_EMPTY,
                source_file,
            )

        base_metadata = dict(metadata or {})
        step = self.chunk_size - self.chunk_overlap
        chunks: List[DocumentChunk] = []
        start = 0
        chunk_index = 0
        total_length = len(normalized_text)

        while start < total_length:
            target_end = min(start + self.chunk_size, total_length)
            end = self._find_chunk_end(normalized_text, start, target_end)
            chunk_content = normalized_text[start:end].strip()

            if not chunk_content:
                start += max(step, 1)
                continue

            chunk_metadata = {
                **base_metadata,
                "chunk_index": chunk_index,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_size": len(chunk_content),
                "document_length": total_length,
            }
            chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    source_file=source_file,
                )
            )

            if end >= total_length:
                break

            next_start = max(end - self.chunk_overlap, start + 1)
            start = next_start
            chunk_index += 1

        return chunks

    def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Extract and chunk a single document.
        """
        path = self.validate_document(file_path)
        text = self.extract_text(str(path))
        base_metadata = {
            "file_name": path.name,
            "file_stem": path.stem,
            "file_extension": path.suffix.lower(),
        }
        if metadata:
            base_metadata.update(metadata)
        return self.chunk_text(text, str(path), base_metadata)

    def process_batch(
        self,
        file_paths: List[str],
        metadata_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple documents, returning successes and failures together.
        """
        processed_documents: Dict[str, List[DocumentChunk]] = {}
        failed_documents: Dict[str, str] = {}

        for file_path in file_paths:
            try:
                metadata = None
                if metadata_by_file:
                    metadata = metadata_by_file.get(file_path)
                processed_documents[file_path] = self.process_document(file_path, metadata)
            except DocumentProcessingError as exc:
                failed_documents[file_path] = str(exc)

        return {
            "processed_documents": processed_documents,
            "failed_documents": failed_documents,
            "total_documents": len(file_paths),
            "successful_documents": len(processed_documents),
            "failed_count": len(failed_documents),
        }

    def _extract_pdf_text(self, path: Path) -> str:
        """
        Extract text from a PDF file page by page.
        """
        reader = PdfReader(str(path))
        extracted_pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        return "\n".join(extracted_pages)

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        non_empty_lines = [line for line in lines if line]
        return "\n".join(non_empty_lines).strip()

    def _find_chunk_end(self, text: str, start: int, target_end: int) -> int:
        if target_end >= len(text):
            return len(text)

        search_start = min(target_end, len(text) - 1)
        boundary_candidates = [
            text.rfind("\n", start + 1, search_start + 1),
            text.rfind(". ", start + 1, search_start + 1),
            text.rfind(" ", start + 1, search_start + 1),
        ]

        for candidate in boundary_candidates:
            if candidate > start:
                if text[candidate:candidate + 2] == ". ":
                    return candidate + 1
                return candidate

        return target_end
