"""
Tests for the document processing system.
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from src.document_processor import DocumentProcessor
from src.exceptions import DocumentProcessingError, ErrorCode


def _normalize_for_assertion(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(text: str) -> bytes:
    escaped_text = _escape_pdf_text(text)
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            "5 0 obj\n"
            f"<< /Length {len(f'BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET')} >>\n"
            "stream\n"
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET\n"
            "endstream\n"
            "endobj\n"
        ),
    ]

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("utf-8")))
        pdf += obj

    xref_offset = len(pdf.encode("utf-8"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    return pdf.encode("utf-8")


def _chunk_signature(chunks):
    return [
        {
            "content": chunk.content,
            "metadata": chunk.metadata,
            "source_file": chunk.source_file,
        }
        for chunk in chunks
    ]


class TestDocumentProcessor:
    def test_extracts_text_from_txt(self, temp_directory):
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        file_path = Path(temp_directory) / "sample.txt"
        file_path.write_text("First line\n\nSecond line", encoding="utf-8")

        extracted = processor.extract_text(str(file_path))

        assert extracted == "First line\nSecond line"

    def test_rejects_unsupported_document_format(self, temp_directory):
        processor = DocumentProcessor()
        file_path = Path(temp_directory) / "sample.csv"
        file_path.write_text("id,value\n1,test", encoding="utf-8")

        with pytest.raises(DocumentProcessingError) as exc_info:
            processor.extract_text(str(file_path))

        assert exc_info.value.error_code == ErrorCode.DOCUMENT_INVALID_FORMAT


class TestDocumentProcessorProperties:
    @given(
        content=st.text(
            alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-")),
            min_size=1,
            max_size=200,
        ).filter(lambda value: value.strip())
    )
    def test_document_format_support(self, content):
        """
        Property 1: Supported TXT and PDF formats should yield extractable text.
        """
        processor = DocumentProcessor(chunk_size=80, chunk_overlap=20)
        normalized_input = _normalize_for_assertion(content)

        with tempfile.TemporaryDirectory() as temp_directory:
            txt_path = Path(temp_directory) / "property.txt"
            txt_path.write_text(content, encoding="utf-8")

            pdf_path = Path(temp_directory) / "property.pdf"
            pdf_path.write_bytes(_build_simple_pdf(normalized_input))

            txt_text = processor.extract_text(str(txt_path))
            pdf_text = processor.extract_text(str(pdf_path))

            assert txt_text == normalized_input
            assert pdf_text.split() == normalized_input.split()

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
                whitelist_characters=".,!?-\n",
            ),
            min_size=1,
            max_size=3000,
        ).filter(lambda value: _normalize_for_assertion(value)),
        chunk_size=st.integers(min_value=40, max_value=200),
        chunk_overlap=st.integers(min_value=0, max_value=39),
    )
    def test_text_chunking_preservation(self, text, chunk_size, chunk_overlap):
        """
        Property 2: Chunk spans should preserve the full original text.
        """
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        normalized_text = _normalize_for_assertion(text)

        chunks = processor.chunk_text(
            normalized_text,
            "memory.txt",
            metadata={"origin": "property-test"},
        )

        reconstructed = [""] * len(normalized_text)
        for chunk in chunks:
            start = chunk.metadata["chunk_start"]
            end = chunk.metadata["chunk_end"]
            segment = normalized_text[start:end]
            assert chunk.content == segment.strip()
            for index, char in enumerate(segment, start=start):
                reconstructed[index] = char

        assert "".join(reconstructed) == normalized_text

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
                whitelist_characters=".,!?-\n",
            ),
            min_size=1,
            max_size=3000,
        ).filter(lambda value: _normalize_for_assertion(value)),
        chunk_size=st.integers(min_value=40, max_value=200),
        chunk_overlap=st.integers(min_value=0, max_value=39),
        tag=st.text(min_size=1, max_size=20).filter(lambda value: value.strip()),
    )
    def test_metadata_preservation_in_chunks(self, text, chunk_size, chunk_overlap, tag):
        """
        Property 4: Every chunk should preserve source information and metadata.
        """
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        metadata = {"tag": tag, "pipeline": "rag"}

        chunks = processor.chunk_text(_normalize_for_assertion(text), "source.txt", metadata)

        assert chunks
        for index, chunk in enumerate(chunks):
            assert chunk.source_file == "source.txt"
            assert chunk.metadata["tag"] == tag
            assert chunk.metadata["pipeline"] == "rag"
            assert chunk.metadata["chunk_index"] == index
            assert chunk.metadata["chunk_end"] > chunk.metadata["chunk_start"]

    @given(
        valid_contents=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
                    whitelist_characters=".,!?-\n",
                ),
                min_size=1,
                max_size=400,
            ).filter(lambda value: _normalize_for_assertion(value)),
            min_size=1,
            max_size=5,
        )
    )
    def test_batch_processing_resilience_and_equivalence(self, valid_contents):
        """
        Property 3 and 5: Batch processing should isolate failures and match single-file results.
        """
        processor = DocumentProcessor(chunk_size=80, chunk_overlap=20)
        with tempfile.TemporaryDirectory() as temp_directory:
            file_paths = []

            for index, content in enumerate(valid_contents):
                file_path = Path(temp_directory) / f"valid_{index}.txt"
                file_path.write_text(content, encoding="utf-8")
                file_paths.append(str(file_path))

            invalid_path = Path(temp_directory) / "broken.csv"
            invalid_path.write_text("not supported", encoding="utf-8")

            batch_result = processor.process_batch(file_paths + [str(invalid_path)])

            assert batch_result["successful_documents"] == len(file_paths)
            assert batch_result["failed_count"] == 1
            assert str(invalid_path) in batch_result["failed_documents"]

            for file_path in file_paths:
                individual_chunks = processor.process_document(file_path)
                batch_chunks = batch_result["processed_documents"][file_path]

                assert _chunk_signature(batch_chunks) == _chunk_signature(individual_chunks)
