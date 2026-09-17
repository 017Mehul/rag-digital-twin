"""Vercel-compatible API endpoints for the RAG Digital Twin."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="RAG Digital Twin API", version="1.0.0")
_pipeline: Any = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def get_pipeline() -> Any:
    """Initialize the heavy RAG pipeline lazily inside the request lifecycle."""
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        from src.utils.config_utils import load_config

        config_path = os.getenv("RAG_CONFIG_PATH", "config/rag_config.yaml")
        _pipeline = RAGPipeline(load_config(str(ROOT / config_path)))
    return _pipeline


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


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    """Return live metrics from the current RAG pipeline instance."""
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "documents": int(metrics.get("documents_ingested_total", 0)),
            "chunks": int(metrics.get("vector_store_size", 0)),
            "questions": int(metrics.get("queries_processed_total", 0)),
            "uptime_seconds": round(float(status.uptime_seconds), 2),
            "metrics": metrics,
            "components_status": dict(status.components_status),
            "documents_list": [],
            "message": "Metrics are live for this running pipeline instance. Persistent document history requires external storage.",
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard unavailable: {type(exc).__name__}: {exc}") from exc


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        response = get_pipeline().query(request.query, k=request.k, threshold=request.threshold)
        return response_payload(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}") from exc
