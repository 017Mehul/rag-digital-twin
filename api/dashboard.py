"""Live dashboard metrics endpoint."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from api.runtime import get_session_pipeline

app = FastAPI(title="RAG Digital Twin Dashboard API", version="1.2.0")


def build_document_summary(pipeline: Any) -> list[dict[str, Any]]:
    store = pipeline.vector_store
    if hasattr(store, "list_documents"):
        try:
            return sorted(
                store.list_documents(),
                key=lambda item: str(item.get("date_added") or ""),
                reverse=True,
            )
        except Exception:
            return []

    grouped: dict[str, dict[str, Any]] = {}
    for item in getattr(store, "metadata_store", []) or []:
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
def dashboard(request: Request) -> dict[str, Any]:
    try:
        pipeline = get_session_pipeline(request.cookies.get("rag_session"))
        if pipeline is None:
            return {"health": "ready", "healthy": True, "documents": 0, "chunks": 0, "questions": 0, "uptime_seconds": 0, "metrics": {}, "components_status": {}, "documents_list": [], "document_types": {}}
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        documents_list = build_document_summary(pipeline)
        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "documents": len(documents_list),
            "chunks": int((pipeline.vector_store.get_stats() if hasattr(pipeline.vector_store, "get_stats") else {"vectors": len(pipeline.vector_store)}).get("vectors", 0)),
            "questions": int(metrics.get("queries_processed_total", 0)),
            "uptime_seconds": round(float(status.uptime_seconds), 2),
            "metrics": metrics,
            "components_status": dict(status.components_status),
            "documents_list": documents_list,
            "document_types": dict(Counter(item["type"] for item in documents_list)),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Dashboard unavailable.") from exc
