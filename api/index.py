"""Single FastAPI entrypoint for the RAG Digital Twin portfolio demo."""
from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
import hashlib
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from api.runtime import create_session, get_session_pipeline

app = FastAPI(title="RAG Digital Twin API", version="1.3.0")
logger = logging.getLogger("rag_digital_twin.api")

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_BYTES = 10 * 1024 * 1024
SESSION_COOKIE = "rag_session"
SESSION_TTL_SECONDS = int(os.getenv("RAG_SESSION_TTL_SECONDS", "1800"))

_RATE_WINDOW_SECONDS = 60
_DEFAULT_RATE_LIMIT = 30
_requests_by_client: dict[str, deque[float]] = defaultdict(deque)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def _rate_limit(request: Request) -> None:
    try:
        limit = max(1, int(os.getenv("RAG_QUERY_RATE_LIMIT", str(_DEFAULT_RATE_LIMIT))))
    except ValueError:
        limit = _DEFAULT_RATE_LIMIT

    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _requests_by_client[client]
    while bucket and now - bucket[0] >= _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again shortly.",
            headers={"Retry-After": str(_RATE_WINDOW_SECONDS)},
        )
    bucket.append(now)


def _session_pipeline(request: Request) -> Any:
    pipeline = get_session_pipeline(request.cookies.get(SESSION_COOKIE))
    if pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="Your demo session has expired. Upload a document to start a new session.",
        )
    return pipeline


def response_payload(response: Any) -> dict[str, Any]:
    return {
        "response_text": getattr(response, "response_text", ""),
        "sources": list(getattr(response, "sources", []) or []),
        "confidence_score": getattr(response, "confidence_score", None),
        "context_used": getattr(response, "context_used", False),
        "model_used": getattr(response, "model_used", None),
        "generation_time": getattr(response, "generation_time", None),
    }


def build_document_summary(pipeline: Any) -> list[dict[str, Any]]:
    store = pipeline.vector_store
    grouped: dict[str, dict[str, Any]] = {}
    for item in getattr(store, "metadata_store", []) or []:
        chunk = item.get("chunk") or {}
        source = str(item.get("source_file") or chunk.get("source_file") or "Unknown document")
        entry = grouped.setdefault(
            source,
            {
                "name": Path(source).name,
                "path": source,
                "type": Path(source).suffix.lstrip(".").upper() or "OTHER",
                "chunks": 0,
                "date_added": chunk.get("created_at") or item.get("created_at"),
                "status": "Indexed",
            },
        )
        entry["chunks"] += 1
    return sorted(grouped.values(), key=lambda item: str(item.get("date_added") or ""), reverse=True)


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    pipeline = get_session_pipeline(request.cookies.get(SESSION_COOKIE))
    if pipeline is None:
        return {"health": "ready", "healthy": True, "mode": "temporary-session-demo", "documents": 0}
    try:
        status = pipeline.get_system_status()
        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "mode": "temporary-session-demo",
            "documents": len(pipeline.vector_store),
        }
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="RAG service unavailable.") from exc


@app.post("/api/ingest")
async def ingest(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> dict[str, Any]:
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
            max_age=SESSION_TTL_SECONDS,
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


@app.post("/api/query")
def query(request: Request, payload: QueryRequest) -> dict[str, Any]:
    _rate_limit(request)
    pipeline = _session_pipeline(request)
    if len(pipeline.vector_store) <= 0:
        raise HTTPException(
            status_code=400,
            detail="No documents are available. Please add a document to the current session first.",
        )
    try:
        return response_payload(
            pipeline.query(payload.query, k=payload.k, threshold=payload.threshold)
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed.") from exc


@app.get("/api/dashboard")
def dashboard(request: Request) -> dict[str, Any]:
    pipeline = get_session_pipeline(request.cookies.get(SESSION_COOKIE))
    if pipeline is None:
        return {
            "health": "ready",
            "healthy": True,
            "documents": 0,
            "chunks": 0,
            "questions": 0,
            "uptime_seconds": 0,
            "metrics": {},
            "components_status": {},
            "documents_list": [],
            "document_types": {},
        }

    try:
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        documents_list = build_document_summary(pipeline)
        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "documents": len(documents_list),
            "chunks": len(pipeline.vector_store),
            "questions": int(metrics.get("queries_processed_total", 0)),
            "uptime_seconds": round(float(status.uptime_seconds), 2),
            "metrics": metrics,
            "components_status": dict(status.components_status),
            "documents_list": documents_list,
            "document_types": dict(Counter(item["type"] for item in documents_list)),
        }
    except Exception as exc:
        logger.exception("Dashboard failed")
        raise HTTPException(status_code=503, detail="Dashboard unavailable.") from exc
