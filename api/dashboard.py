"""Live dashboard metrics endpoint."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin Dashboard API", version="1.2.0")


def build_document_summary(pipeline: Any) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in getattr(pipeline.vector_store, "metadata_store", []) or []:
        chunk = item.get("chunk") or {}
        source = str(item.get("source_file") or chunk.get("source_file") or "Unknown document")
        entry = grouped.setdefault(source, {
            "name": Path(source).name,
            "path": source,
            "type": Path(source).suffix.lstrip(".").upper() or "OTHER",
            "chunks": 0,
            "date_added": chunk.get("created_at") or item.get("created_at"),
            "status": "Indexed",
        })
        entry["chunks"] += 1
    return sorted(grouped.values(), key=lambda item: str(item.get("date_added") or ""), reverse=True)


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        documents_list = build_document_summary(pipeline)
        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "documents": len(documents_list),
            "chunks": len(getattr(pipeline.vector_store, "metadata_store", []) or []),
            "questions": int(metrics.get("queries_processed_total", 0)),
            "uptime_seconds": round(float(status.uptime_seconds), 2),
            "metrics": metrics,
            "components_status": dict(status.components_status),
            "documents_list": documents_list,
            "document_types": dict(Counter(item["type"] for item in documents_list)),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dashboard unavailable: {type(exc).__name__}: {exc}") from exc
