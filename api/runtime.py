"""Runtime factories for the RAG API.

The public demo uses short-lived, in-memory FAISS sessions. No uploaded document
is intended to become a permanent shared knowledge base.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SESSION_TTL_SECONDS = int(os.getenv("RAG_SESSION_TTL_SECONDS", "1800"))
_sessions: dict[str, tuple[float, Any]] = {}
_sessions_lock = threading.RLock()


def _new_session_pipeline() -> Any:
    from src.rag_pipeline import RAGPipeline
    from src.utils.config_utils import load_config

    config_path = os.getenv("RAG_CONFIG_PATH", "config/rag_config.yaml")
    return RAGPipeline(
        load_config(str(ROOT / config_path)),
        load_persisted_store=False,
    )


def _cleanup_expired_sessions(now: float) -> None:
    expired = [
        session_id
        for session_id, (last_used, _) in _sessions.items()
        if now - last_used > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _sessions.pop(session_id, None)


def create_session() -> tuple[str, Any]:
    session_id = uuid.uuid4().hex
    pipeline = _new_session_pipeline()
    with _sessions_lock:
        _cleanup_expired_sessions(time.time())
        _sessions[session_id] = (time.time(), pipeline)
    return session_id, pipeline


def get_session_pipeline(session_id: str) -> Any | None:
    if not session_id:
        return None
    now = time.time()
    with _sessions_lock:
        _cleanup_expired_sessions(now)
        entry = _sessions.get(session_id)
        if entry is None:
            return None
        _sessions[session_id] = (now, entry[1])
        return entry[1]


def get_pipeline() -> Any:
    """Compatibility factory for CLI/internal callers outside the public demo."""
    return _new_session_pipeline()
