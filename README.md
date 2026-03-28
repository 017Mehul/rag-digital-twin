# RAG Digital Twin

RAG Digital Twin is a configurable Retrieval-Augmented Generation system for ingesting domain documents, indexing them in a vector store, and answering grounded questions with source attribution.

## What It Includes

- PDF and TXT document ingestion with chunking and validation
- Pluggable embedding and LLM providers with fallback support
- FAISS-backed vector storage with persistence
- Query processing, context retrieval, and grounded response generation
- Monitoring, audit trails, and property-based test coverage
- Command-line workflows for ingestion and interactive querying

## Project Structure

```text
rag-digital-twin/
|-- config/
|   |-- rag_config.yaml
|   `-- rag_config.local.yaml
|-- data/
|   |-- processed/
|   `-- raw/
|-- docs/
|   `-- api.md
|-- embeddings/
|-- logs/
|-- src/
|   |-- models/
|   |-- providers/
|   `-- utils/
`-- tests/
```

## Installation

```bash
git clone <repository-url>
cd rag-digital-twin
pip install -r requirements.txt
pip install -e .
```

For provider-backed runs, copy `.env.example` to `.env` and set the required API keys. For offline/local validation, use the mock-enabled config at `config/rag_config.local.yaml`.

## Configuration Templates

- `config/rag_config.yaml`: production-oriented template with environment-variable API keys and fallback providers
- `config/rag_config.local.yaml`: local mock mode for testing the full CLI flow without external services

You can also generate a sample config file from the CLI:

```bash
rag-ingest --write-config-template config/generated_config.yaml
```

## CLI Usage

### Ingest Documents

Ingest individual files or entire directories:

```bash
rag-ingest --config config/rag_config.local.yaml data/raw
rag-ingest --config config/rag_config.yaml docs/handbook.pdf notes.txt
```

Useful options:

- `--index-type {flat,ivf,hnsw}` to choose the vector-store index when a new store is created
- `--no-recursive` to only inspect the top level of supplied directories
- `--no-persist` to test ingestion without writing the vector store to disk

### Query the Knowledge Base

Run a single query:

```bash
rag-query --config config/rag_config.local.yaml --query "What are the key policies?"
```

Start an interactive session:

```bash
rag-query --config config/rag_config.local.yaml
```

Interactive commands:

- `/help` shows the available commands
- `/history` prints recent query history from the current session
- `/status` shows health and pipeline metrics
- `/session` shows the session file location
- `/clear` clears the current session history
- `/exit` or `/quit` saves the session and closes the prompt

Session history is saved as JSON in `logs/sessions/` by default, or to a custom path with `--session-file`.

### Run Without Installed Entry Points

```bash
python -m src.cli ingest --config config/rag_config.local.yaml data/raw
python -m src.cli query --config config/rag_config.local.yaml --query "What was indexed?"
```

## Python API

```python
from src.rag_pipeline import RAGPipeline
from src.utils.config_utils import load_config

config = load_config("config/rag_config.local.yaml")
pipeline = RAGPipeline(config)

pipeline.ingest_documents(["data/raw/reference.txt"])
response = pipeline.query("What does the reference say about deployment?")

print(response.response_text)
print(response.sources)
```

More detailed integration examples are available in [docs/api.md](docs/api.md).

## Testing

Run the full suite:

```bash
pytest -q
```

Run targeted CLI tests:

```bash
pytest -q tests/test_cli.py
```

## Development Notes

- `load_config()` supports YAML and JSON files.
- Provider-specific settings live under `embedding.provider_config` and `llm.provider_config`.
- Fallback chains are configured with `embedding.fallbacks` and `llm.fallbacks`.
- The CLI uses the same `RAGPipeline` and provider abstractions as the Python API, so scripts and manual runs share one execution path.
