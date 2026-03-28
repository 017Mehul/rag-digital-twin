"""
Tests for CLI entry points, config templates, and interactive session handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.cli import ingest_command, query_command
from src.utils.config_utils import create_default_config, load_config


def _write_cli_config(temp_directory: str) -> Path:
    root = Path(temp_directory)
    config_path = root / "cli_config.yaml"
    config_payload = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "provider_config": {
                "mock_embeddings": True,
                "dimension": 32,
            },
            "fallbacks": [],
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "provider_config": {
                "mock_responses": True,
            },
            "fallbacks": [],
        },
        "document_processing": {
            "chunk_size": 120,
            "chunk_overlap": 20,
        },
        "retrieval": {
            "top_k_results": 3,
            "similarity_threshold": 0.0,
            "max_context_length": 400,
        },
        "response": {
            "max_tokens": 200,
            "temperature": 0.1,
        },
        "system": {
            "batch_size": 4,
            "max_retries": 1,
            "timeout_seconds": 15,
        },
        "paths": {
            "data_directory": str(root / "data"),
            "embeddings_directory": str(root / "embeddings"),
            "logs_directory": str(root / "logs"),
        },
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    return config_path


class TestCLIIngestion:
    def test_ingest_command_processes_directory_and_persists_store(self, temp_directory, capsys):
        config_path = _write_cli_config(temp_directory)
        incoming_dir = Path(temp_directory) / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        (incoming_dir / "knowledge.txt").write_text(
            "RAG systems ground answers in retrieved context. " * 6,
            encoding="utf-8",
        )
        (incoming_dir / "notes.txt").write_text(
            "Interactive querying should preserve history across a session. " * 4,
            encoding="utf-8",
        )
        (incoming_dir / "unsupported.bin").write_bytes(b"\x00\x01")

        exit_code = ingest_command(["--config", str(config_path), str(incoming_dir)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Discovered 2 supported document(s)." in captured.out
        assert "Skipped 1 unsupported file(s)." in captured.out
        assert "Ingestion Summary" in captured.out
        assert (Path(temp_directory) / "embeddings" / "vector_store_metadata.json").exists()

    def test_ingest_command_returns_error_when_no_supported_documents_exist(self, temp_directory, capsys):
        config_path = _write_cli_config(temp_directory)
        incoming_dir = Path(temp_directory) / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        (incoming_dir / "unsupported.bin").write_bytes(b"\x00\x01")

        exit_code = ingest_command(["--config", str(config_path), str(incoming_dir)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "No supported documents were found to ingest." in captured.err


class TestCLIQuerying:
    def test_query_command_one_shot_formats_response_and_saves_session(self, temp_directory, capsys):
        config_path = _write_cli_config(temp_directory)
        document_path = Path(temp_directory) / "guide.txt"
        document_path.write_text(
            "Vector stores support semantic retrieval for grounded answering.",
            encoding="utf-8",
        )
        session_path = Path(temp_directory) / "logs" / "sessions" / "oneshot.json"

        ingest_exit_code = ingest_command(["--config", str(config_path), str(document_path)])
        assert ingest_exit_code == 0

        exit_code = query_command(
            [
                "--config",
                str(config_path),
                "--query",
                "How does the system ground answers?",
                "--session-file",
                str(session_path),
            ]
        )
        captured = capsys.readouterr()
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert "Response" in captured.out
        assert "Sources" in captured.out
        assert "Model: gpt-4o-mini" in captured.out
        assert len(session_payload["history"]) == 1
        assert session_payload["history"][0]["query"] == "How does the system ground answers?"

    def test_query_command_interactive_mode_tracks_history_and_status(
        self,
        temp_directory,
        capsys,
        monkeypatch,
    ):
        config_path = _write_cli_config(temp_directory)
        document_path = Path(temp_directory) / "manual.txt"
        document_path.write_text(
            "Session history lets operators review earlier questions and answers.",
            encoding="utf-8",
        )
        session_path = Path(temp_directory) / "logs" / "sessions" / "interactive.json"

        ingest_exit_code = ingest_command(["--config", str(config_path), str(document_path)])
        assert ingest_exit_code == 0

        user_inputs = iter(
            [
                "What does the manual say about sessions?",
                "/history",
                "/status",
                "/exit",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(user_inputs))

        exit_code = query_command(
            [
                "--config",
                str(config_path),
                "--session-file",
                str(session_path),
            ]
        )
        captured = capsys.readouterr()
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))

        assert exit_code == 0
        assert "Interactive query session ready" in captured.out
        assert "Entries shown: 1 of 1" in captured.out
        assert "Health: healthy" in captured.out
        assert len(session_payload["history"]) == 1


class TestConfigTemplates:
    def test_create_default_config_writes_yaml_template(self, temp_directory):
        output_path = Path(temp_directory) / "generated.yaml"

        create_default_config(str(output_path))
        generated = yaml.safe_load(output_path.read_text(encoding="utf-8"))

        assert generated["embedding"]["provider"] == "openai"
        assert generated["embedding"]["provider_config"]["api_key"] == "${OPENAI_API_KEY}"
        assert generated["llm"]["fallbacks"][0]["provider"] == "huggingface"

    def test_load_config_resolves_nested_environment_variables(self, temp_directory, monkeypatch):
        config_path = Path(temp_directory) / "nested_env.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "embedding": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "provider_config": {
                            "api_key": "${OPENAI_API_KEY}",
                            "dimension": 16,
                        },
                        "fallbacks": [],
                    },
                    "llm": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "provider_config": {
                            "api_key": "${OPENAI_API_KEY}",
                            "mock_responses": True,
                        },
                        "fallbacks": [],
                    },
                    "document_processing": {
                        "chunk_size": 100,
                        "chunk_overlap": 10,
                    },
                    "retrieval": {
                        "top_k_results": 3,
                        "similarity_threshold": 0.0,
                        "max_context_length": 200,
                    },
                    "response": {
                        "max_tokens": 100,
                        "temperature": 0.1,
                    },
                    "system": {
                        "batch_size": 2,
                        "max_retries": 1,
                        "timeout_seconds": 10,
                    },
                    "paths": {
                        "data_directory": str(Path(temp_directory) / "data"),
                        "embeddings_directory": str(Path(temp_directory) / "embeddings"),
                        "logs_directory": str(Path(temp_directory) / "logs"),
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = load_config(str(config_path))

        assert config.embedding_provider_config["api_key"] == "test-key"
        assert config.llm_provider_config["api_key"] == "test-key"
