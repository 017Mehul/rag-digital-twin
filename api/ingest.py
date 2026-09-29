"""Temporary session-scoped document upload endpoint."""
from __future__ import annotations

import hashlib
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile

from api.runtime import create_session, get_session_pipeline

app = FastAPI(title="RAG Digital Twin Ingestion API", version="1.2.0")

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_BYTES = 10 * 1024 * 1024
SESSION_COOKIE = "rag_session"


def _require_ingest_auth(token: str | None) -> None:
    configured = os.getenv("RAG_INGEST_TOKEN")
    if configured and (not token or not secrets.compare_digest(token, configured)):
        raise HTTPException(status_code=401, detail="Invalid ingestion credentials.")


@app.post("/api/ingest")
async def ingest(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    x_ingest_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_ingest_auth(x_ingest_token)

    filename = Path(file.filename or "document").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported files: PDF, TXT.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File size must be 10 MB or less.")

    session_id = request.cookies.get(SESSION_COOKIE)
    pipeline = get_session_pipeline(session_id or "")
    if pipeline is None:
        session_id, pipeline = create_session()
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            secure=os.getenv("VERCEL") == "1",
            max_age=1800,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)

    try:
        document_id = hashlib.sha256(content).hexdigest()
        metadata = {
            "document_id": document_id,
            "source_file": filename,
            "content_sha256": document_id,
        }
        result = pipeline.ingest_documents(
            [str(temporary_path)],
            metadata_by_file={str(temporary_path): metadata},
            persist=False,
        )
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
            "message": f"{filename} was indexed for this temporary session.",
        }
    finally:
        temporary_path.unlink(missing_ok=True)
