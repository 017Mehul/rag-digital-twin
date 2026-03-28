"""
Command-line interfaces for document ingestion and interactive querying.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

from .exceptions import ConfigurationError, RAGException
from .rag_pipeline import RAGPipeline
from .utils.config_utils import create_default_config, get_default_config_path, load_config
from .utils.file_utils import ensure_directory, is_supported_file_type
from .utils.logging_utils import setup_logging


def build_ingest_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for document ingestion.
    """
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest TXT and PDF documents into the RAG vector store.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Document files or directories to ingest.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=get_default_config_path(),
        help="Path to a YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="Vector store index type to use when creating a new store.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into directories when discovering files.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing the vector store to disk after ingestion.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="CLI log verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--write-config-template",
        help="Write a sample configuration file to this path and exit.",
    )
    return parser


def build_query_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for interactive or one-shot querying.
    """
    parser = argparse.ArgumentParser(
        prog="rag-query",
        description="Query the RAG knowledge base interactively or with a single prompt.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=get_default_config_path(),
        help="Path to a YAML or JSON configuration file.",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Run a single query and exit instead of starting interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override the configured top-k retrieval limit.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the configured similarity threshold.",
    )
    parser.add_argument(
        "--session-file",
        help="Path to a JSON session file for query history.",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Display session history before exiting or starting the prompt.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="Maximum number of session entries to display.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Reset the session history before processing queries.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="Vector store index type to use when creating a new store.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="CLI log verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--write-config-template",
        help="Write a sample configuration file to this path and exit.",
    )
    return parser


def ingest_command(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the document-ingestion CLI.
    """
    parser = build_ingest_parser()
    args = parser.parse_args(argv)

    if args.write_config_template:
        create_default_config(args.write_config_template)
        print(f"Sample configuration written to {Path(args.write_config_template).resolve()}")
        return 0

    if not args.paths:
        parser.error("at least one document path or directory is required")

    try:
        config = _load_runtime_config(args.config, args.log_level, command_name="ingest")
        discovered, unsupported, missing = _discover_documents(
            args.paths,
            recursive=not args.no_recursive,
        )

        _print_discovery_summary(discovered, unsupported, missing, stream=sys.stdout)
        if not discovered:
            print("No supported documents were found to ingest.", file=sys.stderr)
            return 1

        print("[1/4] Configuration loaded")
        print("[2/4] Initializing RAG pipeline")
        pipeline = RAGPipeline(
            config=config,
            vector_store_index_type=args.index_type,
        )

        print(f"[3/4] Ingesting {len(discovered)} document(s)")
        results = pipeline.ingest_documents(discovered, persist=not args.no_persist)

        if args.no_persist:
            print("[4/4] Persistence skipped by flag")
        else:
            print(f"[4/4] Vector store persisted to {Path(config.embeddings_directory).resolve()}")

        _print_ingestion_results(results, stream=sys.stdout)
        return 0 if results.successful_documents > 0 else 1
    except (RAGException, ValueError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        print(f"Unexpected ingestion failure: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_cli_logging()


def query_command(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point for the querying CLI.
    """
    parser = build_query_parser()
    args = parser.parse_args(argv)

    if args.write_config_template:
        create_default_config(args.write_config_template)
        print(f"Sample configuration written to {Path(args.write_config_template).resolve()}")
        return 0

    try:
        config = _load_runtime_config(args.config, args.log_level, command_name="query")
        pipeline = RAGPipeline(
            config=config,
            vector_store_index_type=args.index_type,
        )
        session_path = _resolve_session_path(
            explicit_path=args.session_file,
            logs_directory=config.logs_directory,
            interactive=args.query is None,
        )
        session_state = _load_or_initialize_session(
            session_path=session_path,
            config_path=args.config,
            clear_history=args.clear_history,
        )

        if args.show_history:
            print(_format_history(session_state, limit=args.history_limit))
            if args.query is None:
                return 0

        if args.query is not None:
            response = pipeline.query(args.query, k=args.top_k, threshold=args.threshold)
            _append_session_entry(session_state, args.query, response)
            _save_session(session_state, session_path)
            print(_format_response(response))
            print(f"\nSession file: {session_path}")
            return 0

        return _run_interactive_query_loop(
            pipeline=pipeline,
            session_state=session_state,
            session_path=session_path,
            top_k=args.top_k,
            threshold=args.threshold,
            history_limit=args.history_limit,
            input_fn=input,
            output_stream=sys.stdout,
        )
    except (RAGException, ValueError) as exc:
        print(f"Query session failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        print(f"Unexpected query failure: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_cli_logging()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Support `python -m src.cli ingest ...` and `python -m src.cli query ...`.
    """
    argv = list(argv or sys.argv[1:])
    if not argv:
        print("Usage: python -m src.cli [ingest|query] ...", file=sys.stderr)
        return 1

    command = argv[0]
    sub_args = argv[1:]

    if command == "ingest":
        return ingest_command(sub_args)
    if command == "query":
        return query_command(sub_args)

    print(f"Unknown command: {command}", file=sys.stderr)
    return 1


def _load_runtime_config(config_path: str, log_level: str, command_name: str):
    config = load_config(config_path)
    log_file = Path(config.logs_directory) / f"{command_name}.log"
    setup_logging(log_level=log_level, log_file=str(log_file))
    return config


def _close_cli_logging() -> None:
    logger = logging.getLogger("rag_digital_twin")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _discover_documents(paths: Sequence[str], recursive: bool) -> Tuple[List[str], List[str], List[str]]:
    discovered: List[str] = []
    unsupported: List[str] = []
    missing: List[str] = []
    seen = set()

    for raw_path in paths:
        candidate = Path(raw_path)
        if candidate.is_dir():
            iterator = candidate.rglob("*") if recursive else candidate.glob("*")
            for nested in sorted(iterator):
                if not nested.is_file():
                    continue
                resolved = str(nested.resolve())
                if is_supported_file_type(resolved):
                    if resolved not in seen:
                        discovered.append(resolved)
                        seen.add(resolved)
                else:
                    unsupported.append(resolved)
            continue

        if candidate.is_file():
            resolved = str(candidate.resolve())
            if is_supported_file_type(resolved):
                if resolved not in seen:
                    discovered.append(resolved)
                    seen.add(resolved)
            else:
                unsupported.append(resolved)
            continue

        missing.append(str(candidate))

    return discovered, unsupported, missing


def _print_discovery_summary(
    discovered: List[str],
    unsupported: List[str],
    missing: List[str],
    stream: TextIO,
) -> None:
    print(f"Discovered {len(discovered)} supported document(s).", file=stream)
    for index, file_path in enumerate(discovered, start=1):
        print(f"  {index}. {file_path}", file=stream)

    if unsupported:
        print(f"Skipped {len(unsupported)} unsupported file(s).", file=stream)
    if missing:
        print(f"Skipped {len(missing)} missing path(s).", file=stream)


def _print_ingestion_results(results: Any, stream: TextIO) -> None:
    print("\nIngestion Summary", file=stream)
    print(f"  Total documents: {results.total_documents}", file=stream)
    print(f"  Successful: {results.successful_documents}", file=stream)
    print(f"  Failed: {results.failed_documents}", file=stream)
    print(f"  Chunks created: {results.total_chunks}", file=stream)
    print(f"  Embeddings stored: {results.total_embeddings}", file=stream)
    print(f"  Processing time: {results.processing_time:.2f}s", file=stream)

    if results.errors:
        print("  Errors:", file=stream)
        for error in results.errors:
            print(f"    - {error}", file=stream)


def _resolve_session_path(
    explicit_path: Optional[str],
    logs_directory: str,
    interactive: bool,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    sessions_dir = ensure_directory(str(Path(logs_directory) / "sessions"))
    if interactive:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (sessions_dir / f"session_{timestamp}.json").resolve()
    return (sessions_dir / "query_history.json").resolve()


def _load_or_initialize_session(
    session_path: Path,
    config_path: str,
    clear_history: bool,
) -> Dict[str, Any]:
    if clear_history or not session_path.exists():
        return _new_session_state(session_path, config_path)

    try:
        session_data = json.loads(session_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _new_session_state(session_path, config_path)

    if not isinstance(session_data, dict):
        return _new_session_state(session_path, config_path)

    session_data.setdefault("session_id", session_path.stem)
    session_data.setdefault("created_at", datetime.now().isoformat())
    session_data["updated_at"] = datetime.now().isoformat()
    session_data["config_path"] = str(Path(config_path).resolve())
    session_data.setdefault("history", [])
    return session_data


def _new_session_state(session_path: Path, config_path: str) -> Dict[str, Any]:
    timestamp = datetime.now().isoformat()
    return {
        "session_id": session_path.stem,
        "created_at": timestamp,
        "updated_at": timestamp,
        "config_path": str(Path(config_path).resolve()),
        "history": [],
    }


def _save_session(session_state: Dict[str, Any], session_path: Path) -> None:
    session_state["updated_at"] = datetime.now().isoformat()
    ensure_directory(str(session_path.parent))
    session_path.write_text(json.dumps(session_state, indent=2), encoding="utf-8")


def _append_session_entry(session_state: Dict[str, Any], query: str, response: Any) -> None:
    session_state.setdefault("history", []).append(
        {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_text": response.response_text,
            "sources": list(response.sources),
            "confidence_score": response.confidence_score,
            "model_used": response.model_used,
            "generation_time": response.generation_time,
        }
    )


def _format_response(response: Any) -> str:
    answer_body = response.response_text.split("\n\nSources:\n", 1)[0].strip()
    lines = [
        "Response",
        "--------",
        answer_body,
        "",
        f"Confidence: {response.confidence_score:.3f}",
        f"Model: {response.model_used or 'unknown'}",
        f"Generation time: {response.generation_time:.2f}s",
    ]

    if response.sources:
        lines.extend(["", "Sources", "-------"])
        for index, source in enumerate(response.sources, start=1):
            lines.append(f"{index}. {source}")

    return "\n".join(lines)


def _format_history(session_state: Dict[str, Any], limit: int = 10) -> str:
    history = list(session_state.get("history", []))
    if not history:
        return "No query history recorded yet."

    limit = max(1, limit)
    selected = history[-limit:]
    lines = [
        f"Session: {session_state.get('session_id', 'unknown')}",
        f"Entries shown: {len(selected)} of {len(history)}",
    ]

    start_index = len(history) - len(selected) + 1
    for offset, entry in enumerate(selected, start=start_index):
        timestamp = entry.get("timestamp", "unknown")
        query = entry.get("query", "").strip() or "<empty>"
        lines.append(f"{offset}. {timestamp} | {query}")
        sources = entry.get("sources", [])
        if sources:
            lines.append(f"   Sources: {', '.join(sources)}")

    return "\n".join(lines)


def _format_status(pipeline: RAGPipeline) -> str:
    status = pipeline.get_system_status()
    metrics = status.performance_metrics
    return "\n".join(
        [
            f"Health: {status.health.value}",
            f"Vector store size: {int(metrics.get('vector_store_size', 0.0))}",
            f"Queries processed: {int(metrics.get('queries_processed_total', 0.0))}",
            f"Error count: {status.error_count}",
        ]
    )


def _run_interactive_query_loop(
    pipeline: RAGPipeline,
    session_state: Dict[str, Any],
    session_path: Path,
    top_k: Optional[int],
    threshold: Optional[float],
    history_limit: int,
    input_fn: Any,
    output_stream: TextIO,
) -> int:
    print(f"Interactive query session ready: {session_path}", file=output_stream)
    print("Commands: /help, /history, /status, /session, /clear, /exit", file=output_stream)

    while True:
        try:
            raw_value = input_fn("rag> ")
        except EOFError:
            print("\nSession closed.", file=output_stream)
            _save_session(session_state, session_path)
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted. Session saved.", file=output_stream)
            _save_session(session_state, session_path)
            return 0

        command = raw_value.strip()
        if not command:
            continue

        if command in {"/exit", "/quit"}:
            _save_session(session_state, session_path)
            print("Session saved. Goodbye.", file=output_stream)
            return 0
        if command == "/help":
            print("Enter a natural-language question or one of the commands above.", file=output_stream)
            continue
        if command.startswith("/history"):
            parts = command.split(maxsplit=1)
            limit = history_limit
            if len(parts) == 2 and parts[1].isdigit():
                limit = int(parts[1])
            print(_format_history(session_state, limit=limit), file=output_stream)
            continue
        if command == "/status":
            print(_format_status(pipeline), file=output_stream)
            continue
        if command == "/session":
            print(f"Session file: {session_path}", file=output_stream)
            print(f"Session entries: {len(session_state.get('history', []))}", file=output_stream)
            continue
        if command == "/clear":
            session_state["history"] = []
            _save_session(session_state, session_path)
            print("Session history cleared.", file=output_stream)
            continue

        response = pipeline.query(command, k=top_k, threshold=threshold)
        _append_session_entry(session_state, command, response)
        _save_session(session_state, session_path)
        print(_format_response(response), file=output_stream)
        print("", file=output_stream)


if __name__ == "__main__":  # pragma: no cover - exercised through console entry points
    raise SystemExit(main())
