"""Document lifecycle API."""
from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Header, HTTPException

from api.runtime import get_pipeline

app = FastAPI(title="RAG Digital Twin Document API", version="1.0.0")


def _require_auth(token: str | None) -> None:
    configured = os.getenv("RAG_INGEST_TOKEN")
    if configured and (not token or not secrets.compare_digest(token, configured)):
        raise HTTPException(status_code=401, detail="Invalid document management credentials.")
    if os.getenv("VERCEL") == "1" and not configured:
        raise HTTPException(status_code=503, detail="Document management is disabled until RAG_INGEST_TOKEN is configured.")


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    x_ingest_token: str | None = Header(default=None),
) -> dict[str, object]:
    _require_auth(x_ingest_token)
    if len(document_id) != 64:
        raise HTTPException(status_code=400, detail="Invalid document ID.")
    try:
        deleted = get_pipeline().delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"success": True, "document_id": document_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document deletion failed.") from exc
