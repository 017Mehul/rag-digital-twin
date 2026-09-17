"""Dashboard metrics endpoint for the RAG Digital Twin."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="RAG Digital Twin Dashboard API", version="1.0.0")
_pipeline: Any = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        from src.utils.config_utils import load_config

        config_path = os.getenv("RAG_CONFIG_PATH", "config/rag_config.yaml")
        _pipeline = RAGPipeline(load_config(str(ROOT / config_path)))
    return _pipeline


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
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
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard unavailable: {type(exc).__name__}: {exc}") from exc
