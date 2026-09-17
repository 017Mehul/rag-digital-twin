"""Shared lazy pipeline factory for API handlers."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_pipeline: Any = None


def get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        from src.utils.config_utils import load_config

        config_path = os.getenv("RAG_CONFIG_PATH", "config/rag_config.yaml")
        _pipeline = RAGPipeline(load_config(str(ROOT / config_path)))
    return _pipeline
