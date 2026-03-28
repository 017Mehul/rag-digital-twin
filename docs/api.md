# API Integration Guide

This project exposes both a high-level pipeline and lower-level building blocks so it can be embedded in scripts, services, or notebooks.

## Load Configuration

```python
from src.utils.config_utils import load_config

config = load_config("config/rag_config.local.yaml")
```

`load_config()` supports both YAML and JSON files, resolves environment-variable placeholders such as `${OPENAI_API_KEY}`, and validates provider settings before runtime components are created.

## End-to-End Pipeline

```python
from src.rag_pipeline import RAGPipeline
from src.utils.config_utils import load_config

config = load_config("config/rag_config.local.yaml")
pipeline = RAGPipeline(config)

pipeline.ingest_documents(["data/raw/handbook.txt"])
response = pipeline.query("What does the handbook say about incident response?")

print(response.response_text)
print(response.sources)
```

Use `RAGPipeline` when you want one object that owns ingestion, retrieval, response generation, monitoring, and audit-trail state.

## Component-Level Integration

```python
from src.context_retriever import ContextRetriever
from src.document_processor import DocumentProcessor
from src.embedding_generator import EmbeddingGenerator
from src.query_processor import QueryProcessor
from src.response_generator import ResponseGenerator
from src.utils.config_utils import load_config
from src.vector_store import VectorStore

config = load_config("config/rag_config.local.yaml")

processor = DocumentProcessor(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
)
embedding_generator = EmbeddingGenerator.from_config(config)
vector_store = VectorStore(dimension=embedding_generator.provider.dimension)
query_processor = QueryProcessor.from_config(config, embedding_generator, vector_store)
context_retriever = ContextRetriever(query_processor, max_context_length=config.max_context_length)
response_generator = ResponseGenerator.from_config(config)
```

This composition style is useful when you need custom orchestration around ingestion queues, background workers, or service endpoints.

## Response Contract

`pipeline.query()` returns a `GeneratedResponse` object with:

- `response_text`: grounded answer text
- `sources`: unique source file paths used in the answer
- `confidence_score`: normalized confidence estimate
- `context_used`: whether retrieved context was actually used
- `model_used`: provider model name
- `generation_time`: elapsed generation time in seconds

## Operational Hooks

- `pipeline.get_system_status()` returns component health and performance metrics.
- `pipeline.get_audit_trail()` returns audit entries for ingestion, retrieval, and configuration changes.
- `pipeline.update_configuration({...})` validates changes and rebuilds dependent components when needed.

## CLI Entry Points

The package installs two console scripts:

```bash
rag-ingest --config config/rag_config.local.yaml data/raw
rag-query --config config/rag_config.local.yaml
```

You can also invoke the same functionality without installation:

```bash
python -m src.cli ingest --config config/rag_config.local.yaml data/raw
python -m src.cli query --config config/rag_config.local.yaml --query "What topics are covered?"
```
