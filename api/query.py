"""Vercel-compatible query endpoint for the RAG Digital Twin."""
from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin API", version="1.1.0")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


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
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {type(exc).__name__}: {exc}") from exc


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        if int(metrics.get("documents_ingested_total", 0)) <= 0 or len(pipeline.vector_store) <= 0:
            raise HTTPException(status_code=400, detail="No documents added yet. Please add a document first from the Ingest Documents section.")
        return response_payload(pipeline.query(request.query, k=request.k, threshold=request.threshold))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}") from exc
