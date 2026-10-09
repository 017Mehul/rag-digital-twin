# RAG Digital Twin

RAG Digital Twin is a configurable Retrieval-Augmented Generation system for ingesting domain documents, indexing them in a vector store, and answering grounded questions with source attribution.

## What It Includes

- PDF and TXT document ingestion with chunking and validation
- NVIDIA embeddings and NVIDIA-hosted LLM generation through an OpenAI-compatible API
- FAISS-backed vector storage with temporary session-scoped storage for the public demo
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
pip install -e .[test]
```

### Production API configuration

The production configuration uses NVIDIA-hosted models through NVIDIA's OpenAI-compatible API.

Set the NVIDIA API credential in the environment variable used by the production configuration:

```bash
OPENAI_API_KEY=<your-nvidia-api-key>
```

The variable name is retained for compatibility with the existing deployment environment; requests are sent to NVIDIA's API endpoint, not OpenAI.

For offline/local validation, use the mock-enabled config at `config/rag_config.local.yaml`.

## Configuration Templates

- `config/rag_config.yaml`: production configuration using NVIDIA embeddings (`nvidia/nemotron-3-embed-1b`) and the NVIDIA-hosted `openai/gpt-oss-20b` LLM through `https://integrate.api.nvidia.com/v1`
- `config/rag_config.local.yaml`: local mock mode for testing the full CLI flow without external services

The production embedding model uses a 2048-dimensional FAISS index. Embedding requests use NVIDIA's query/passage input modes so indexing and retrieval use the model as intended.

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

The FastAPI API can run on Vercel as a portfolio demo. Each browser session gets its own in-memory FAISS knowledge base.

### Temporary demo sessions

- Uploaded documents are kept only in the active server runtime for that session.
- A session expires after 30 minutes of inactivity by default (`RAG_SESSION_TTL_SECONDS`).
- Documents are never written to the repository, Supabase, Pinecone, or another persistent vector database.
- Closing the browser removes the session-only cookie; the server-side session is also cleaned up by the inactivity TTL.
- Because Vercel serverless instances are ephemeral, this mode is intentionally a demo/portfolio architecture rather than a durable multi-user knowledge base.
- PDF and TXT uploads are supported, with a 10 MB per-file limit.
- Each session allows up to 5 unique document uploads.
- Re-uploading the same document replaces its previous indexed copy instead of consuming another session document slot.
- Query and ingestion endpoints have lightweight per-client rate limits.
- API responses are marked `no-store` to avoid caching session data.

`RAG_SESSION_TTL_SECONDS` can be changed for a different demo timeout.

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
- The public production configuration uses NVIDIA-hosted providers; local mock configuration remains available for offline testing.
- The CLI uses the same `RAGPipeline` and provider abstractions as the Python API.
- The public demo intentionally does not use persistent vector storage.
