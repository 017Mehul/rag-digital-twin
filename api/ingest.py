"""Document upload and ingestion endpoint."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin Ingestion API", version="1.0.0")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported files: PDF, TXT, MD, DOCX.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File size must be 10 MB or less.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)

    try:
        result = get_pipeline().ingest_documents([str(temporary_path)], persist=False)
        successful = int(getattr(result, "successful_documents", 0))
        failures = list(getattr(result, "errors", []) or [])
        failed_documents = getattr(result, "failed_documents", {}) or {}
        if not successful and failed_documents:
            failures.extend(str(value) for value in failed_documents.values())
        if not successful:
            raise HTTPException(status_code=422, detail=failures or "Document ingestion failed.")
        return {
            "success": True,
            "filename": filename,
            "documents": successful,
            "chunks": int(getattr(result, "total_chunks", 0)),
            "embeddings": int(getattr(result, "total_embeddings", 0)),
            "message": f"{filename} was indexed successfully.",
        }
    finally:
        temporary_path.unlink(missing_ok=True)
