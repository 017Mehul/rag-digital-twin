# RAG Digital Twin

RAG Digital Twin is a configurable Retrieval-Augmented Generation system for ingesting domain documents, indexing them in a vector store, and answering grounded questions with source attribution.

## What It Includes

- PDF and TXT document ingestion with chunking and validation
- Pluggable embedding and LLM providers with fallback support
- FAISS-backed local vector storage plus optional durable Pinecone storage for serverless deployments
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

The FastAPI API can run on Vercel. Local FAISS remains the default for development, while production runtime ingestion can use Pinecone as the durable shared vector store.

### Durable Vercel storage

Create a Pinecone serverless index whose dimension matches the active embedding model (the production OpenAI `text-embedding-3-small` configuration uses 1536 dimensions), then set these Vercel environment variables:

```text
PINECONE_API_KEY=...
PINECONE_INDEX_HOST=https://...
PINECONE_NAMESPACE=default
RAG_INGEST_TOKEN=...
```

When both Pinecone variables are present, `RAGPipeline` automatically uses the Pinecone backend for ingestion and retrieval. FAISS continues to be used locally when those variables are absent. Pinecone vectors and metadata are stored remotely, so separate Vercel function instances can query the same knowledge base.

Therefore:

- Runtime document ingestion is enabled on Vercel when Pinecone and `RAG_INGEST_TOKEN` are configured.
- Set `RAG_INGEST_TOKEN` to protect ingestion.
- `RAG_ALLOW_EPHEMERAL_INGEST=true` remains available only for temporary/demo FAISS behavior when a durable store is not configured.
- The query endpoint has an in-process burst limiter. Use platform-level rate limiting for a public production deployment.
- Vercel environment-variable changes require a redeploy before the new values are available to the deployment.

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
