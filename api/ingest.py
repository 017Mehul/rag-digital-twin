"""Document upload and ingestion endpoint."""
from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin Ingestion API", version="1.1.0")

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def _require_ingest_auth(token: str | None) -> None:
    configured = os.getenv("RAG_INGEST_TOKEN")
    if configured:
        if not token or not secrets.compare_digest(token, configured):
            raise HTTPException(status_code=401, detail="Invalid ingestion credentials.")
        return
    if os.getenv("VERCEL") == "1":
        raise HTTPException(
            status_code=503,
            detail="Ingestion is disabled until RAG_INGEST_TOKEN is configured.",
        )


@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_ingest_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_ingest_auth(x_ingest_token)

    durable_store_configured = bool(
        os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_HOST")
    )
    if (
        os.getenv("VERCEL") == "1"
        and not durable_store_configured
        and os.getenv("RAG_ALLOW_EPHEMERAL_INGEST") != "true"
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Runtime ingestion is disabled on Vercel until a durable vector "
                "store is configured. Set PINECONE_API_KEY and "
                "PINECONE_INDEX_HOST, or explicitly enable temporary "
                "RAG_ALLOW_EPHEMERAL_INGEST behavior."
            ),
        )

    filename = Path(file.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported files: PDF, TXT.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File size must be 10 MB or less.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)

    try:
        result = get_pipeline().ingest_documents([str(temporary_path)], persist=True)
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
