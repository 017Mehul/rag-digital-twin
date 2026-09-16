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

from src.rag_pipeline import RAGPipeline
from src.utils.config_utils import load_config

app = FastAPI(title="RAG Digital Twin API", version="1.0.0")
_pipeline: RAGPipeline | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
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
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {exc}") from exc


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        response = get_pipeline().query(request.query, k=request.k, threshold=request.threshold)
        return response_payload(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
