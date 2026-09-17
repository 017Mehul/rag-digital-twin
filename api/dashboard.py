"""Live dashboard metrics endpoint for the RAG Digital Twin."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="RAG Digital Twin Dashboard API", version="1.1.0")
_pipeline: Any = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        from src.utils.config_utils import load_config

        config_path = os.getenv("RAG_CONFIG_PATH", "config/rag_config.yaml")
        _pipeline = RAGPipeline(load_config(str(ROOT / config_path)))
    return _pipeline


def build_document_summary(pipeline: Any) -> list[dict[str, Any]]:
    """Build unique document rows from the persisted vector metadata."""
    grouped: dict[str, dict[str, Any]] = {}
    metadata_store = getattr(pipeline.vector_store, "metadata_store", [])

    for item in metadata_store:
        chunk = item.get("chunk") or {}
        source = item.get("source_file") or chunk.get("source_file") or "Unknown document"
        source = str(source)
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
        created_at = chunk.get("created_at") or item.get("created_at")
        if created_at and (not entry.get("date_added") or str(created_at) > str(entry["date_added"])):
            entry["date_added"] = created_at

    documents = list(grouped.values())
    documents.sort(key=lambda item: str(item.get("date_added") or ""), reverse=True)
    return documents


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        status = pipeline.get_system_status()
        metrics = dict(status.performance_metrics)
        documents_list = build_document_summary(pipeline)
        document_types = Counter(item["type"] for item in documents_list)

        # Prefer the actual metadata count over a potentially stale metric.
        document_count = len(documents_list)
        chunk_count = len(getattr(pipeline.vector_store, "metadata_store", []))

        return {
            "health": status.health.value,
            "healthy": status.is_healthy(),
            "documents": document_count,
            "chunks": chunk_count,
            "questions": int(metrics.get("queries_processed_total", 0)),
            "uptime_seconds": round(float(status.uptime_seconds), 2),
            "metrics": metrics,
            "components_status": dict(status.components_status),
            "documents_list": documents_list,
            "document_types": dict(document_types),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dashboard unavailable: {type(exc).__name__}: {exc}",
        ) from exc
