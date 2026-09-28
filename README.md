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
|-- api/
|-- config/
|-- docs/
|-- public/
|-- src/
|-- tests/
|-- requirements.txt
|-- pyproject.toml
|-- setup.py
`-- vercel.json
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

## CLI Usage

### Ingest Documents

```bash
rag-ingest --config config/rag_config.local.yaml data/raw
rag-ingest --config config/rag_config.yaml docs/handbook.pdf notes.txt
```

Useful options:

- `--index-type {flat,ivf,hnsw}` to choose the vector-store index when a new store is created
- `--no-recursive` to only inspect the top level of supplied directories
- `--no-persist` to test ingestion without writing the vector store to disk

### Query the Knowledge Base

```bash
rag-query --config config/rag_config.local.yaml --query "What are the key policies?"
```

## Web deployment

The FastAPI API can run on Vercel, but FAISS files are local filesystem state. Vercel serverless instances do not provide a durable shared filesystem for runtime mutations.

Therefore:

- Runtime document ingestion is disabled on Vercel by default.
- Set `RAG_INGEST_TOKEN` to protect ingestion when it is enabled.
- `RAG_ALLOW_EPHEMERAL_INGEST=true` can enable temporary/demo ingestion, but uploaded data can disappear when the serverless instance is recycled.
- A durable shared vector store must be added before using runtime ingestion as a production feature.
- The query endpoint has an in-process burst limiter. Use platform-level rate limiting for a public production deployment.

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

## Testing

Run the full suite:

```bash
pytest -q
```

Run targeted tests:

```bash
pytest -q tests/test_cli.py
pytest -q tests/test_integration.py
pytest -q tests/test_performance.py
```

## Development Notes

- `load_config()` supports YAML and JSON files.
- Provider-specific settings live under `embedding.provider_config` and `llm.provider_config`.
- Fallback chains are configured with `embedding.fallbacks` and `llm.fallbacks`.
- The CLI uses the same `RAGPipeline` and provider abstractions as the Python API.
