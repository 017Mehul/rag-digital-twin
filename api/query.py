"""Vercel-compatible query endpoint for the RAG Digital Twin."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin API", version="1.2.0")
logger = logging.getLogger("rag_digital_twin.api")

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


def response_payload(response: Any) -> dict[str, Any]:
    return {
        "response_text": getattr(response, "response_text", ""),
        "sources": list(getattr(response, "sources", []) or []),
        "confidence_score": getattr(response, "confidence_score", None),
        "context_used": getattr(response, "context_used", False),
        "model_used": getattr(response, "model_used", None),
        "generation_time": getattr(response, "generation_time", None),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        return get_pipeline().run_health_check()
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="RAG service unavailable.") from exc


@app.post("/api/query")
def query(request: Request, payload: QueryRequest) -> dict[str, Any]:
    _rate_limit(request)
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        if int(metrics.get("documents_ingested_total", 0)) <= 0 or len(pipeline.vector_store) <= 0:
            raise HTTPException(
                status_code=400,
                detail="No documents are available. Please add a document to the knowledge base first.",
            )
        return response_payload(
            pipeline.query(payload.query, k=payload.k, threshold=payload.threshold)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed.") from exc
