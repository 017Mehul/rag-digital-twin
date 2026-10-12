# RAG Digital Twin

> **A production-ready Retrieval-Augmented Generation workspace for turning documents into grounded, traceable answers.**

RAG Digital Twin is a full-stack RAG application that ingests documents, creates semantic embeddings, retrieves relevant passages with FAISS, and generates answers with source attribution. It includes a polished responsive dashboard and a security-hardened FastAPI API.

**Live demo:** https://rag.themglabs.com  
**Repository:** https://github.com/017Mehul/rag-digital-twin

---

## Dashboard Preview

![RAG Digital Twin dashboard](docs/dashboard.webp)

The dashboard provides a single workspace for document ingestion, knowledge-base statistics, queries, retrieval sources, history, system status, and configuration.

---

## ✨ Features

- **PDF & TXT ingestion** with validation and controlled upload limits
- **Semantic embeddings** using NVIDIA Nemotron Embed
- **FAISS vector retrieval** for fast similarity search
- **Grounded generation** using NVIDIA-hosted `openai/gpt-oss-20b`
- **Source attribution** so answers can be traced back to retrieved documents
- **Session-isolated knowledge bases** for the public demo
- **Responsive dashboard** across desktop, tablet, and mobile
- **Operational dashboard** with document, chunk, query, health, and retrieval information
- **Security hardening** including CSRF protection, rate limiting, secure cookies, validation, security headers, and log redaction
- **Automated testing and dependency auditing** for the repository

---

## 🏗️ Architecture

```text
┌──────────────────────────────┐
│ Browser / Responsive UI      │
└──────────────┬───────────────┘
               │ HTTPS + CSRF
┌──────────────▼───────────────┐
│ FastAPI                      │
│ Validation • Rate Limiting   │
│ Session Management           │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ RAG Pipeline                 │
│ Parse → Chunk → Embed        │
│ Retrieve → Generate          │
└───────┬──────────────┬───────┘
        │              │
┌───────▼────────┐ ┌───▼────────────────┐
│ NVIDIA Nemotron│ │ NVIDIA-hosted LLM  │
│ Embeddings     │ │ gpt-oss-20b        │
└───────┬────────┘ └────────────────────┘
        │
┌───────▼────────┐
│ FAISS          │
│ Session-scoped │
└────────────────┘
```

### Request flow

1. Upload a document through the dashboard.
2. Validate filename, type, size, and content.
3. Parse the document and split it into overlapping chunks.
4. Generate NVIDIA passage embeddings.
5. Store embeddings in an in-memory FAISS index.
6. Embed the user's query in query mode.
7. Retrieve the nearest document chunks.
8. Send retrieved context to the NVIDIA-hosted LLM.
9. Return the answer together with source metadata.

---

## 🚀 Production Demo

The public application runs on **Vercel + FastAPI** and intentionally uses temporary, in-memory FAISS sessions rather than a persistent vector database.

### Session model

- Each browser session receives an isolated knowledge base.
- Sessions expire after **30 minutes of inactivity** by default.
- The public demo supports up to **5 unique documents per session**.
- Duplicate content is deduplicated; changed content is treated as a separate document.
- Maximum upload size is **4 MB**.
- Supported formats are **PDF and TXT**.
- Vercel instance recycling can discard in-memory state; this is intentional for the portfolio deployment.
- Query and ingestion endpoints use per-client rate limits.
- API responses use `no-store` caching headers.

> This deployment is designed as a portfolio-quality RAG demonstration rather than durable multi-tenant document storage.

---

## 🔐 Security

Security controls are enforced at the API layer rather than relying only on frontend behavior.

### Implemented controls

- CSRF protection with a double-submit token
- Strict origin validation for browser mutation requests
- Per-client rate limiting
- Filename, extension, size, and content validation
- PDF signature and UTF-8 text validation
- Secure session cookies with HttpOnly, SameSite, and HTTPS settings
- Security headers including:
  - `Strict-Transport-Security`
  - `X-Content-Type-Options`
  - `X-Frame-Options`
  - `Referrer-Policy`
  - `Permissions-Policy`
- No API secrets in frontend code
- Sanitized API errors
- Sensitive-data redaction in logs
- No source maps shipped with the application
- Automated dependency vulnerability auditing

### Deliberate attack-surface boundaries

The current demo does not use a SQL database, password/JWT authentication system, persistent user model, webhook receiver, or user-controlled outbound URL fetcher.

Accordingly, SQL injection, IDOR/BOLA, password/MFA, webhook signature verification, SSRF, and database RLS are outside the current application's attack surface.

---

## 🧠 AI / Retrieval Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Embeddings | NVIDIA Nemotron Embed |
| Embedding model | `nvidia/nemotron-3-embed-1b` |
| Embedding dimension | 2,048 |
| LLM | NVIDIA-hosted `openai/gpt-oss-20b` |
| Vector search | FAISS |
| Document parsing | pypdf |
| Configuration | YAML |
| Runtime | Python 3.12+ |
| Deployment | Vercel |

The application uses NVIDIA's OpenAI-compatible endpoint:

```text
https://integrate.api.nvidia.com/v1
```

The embedding pipeline uses **passage** mode while indexing documents and **query** mode during retrieval.

---

## 📁 Project Structure

```text
rag-digital-twin/
├── api/
│   ├── index.py
│   └── runtime.py
├── config/
│   ├── rag_config.yaml
│   └── rag_config.local.yaml
├── docs/
│   └── dashboard.webp
├── public/
│   ├── index.html
│   ├── app.js
│   ├── sections.js
│   ├── styles.css
│   └── theme.js
├── src/
│   ├── providers/
│   ├── rag_pipeline.py
│   ├── vector_store.py
│   └── document_processor.py
├── tests/
├── requirements.txt
├── pyproject.toml
├── setup.py
└── vercel.json
```

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/017Mehul/rag-digital-twin.git
cd rag-digital-twin
```

### 2. Install dependencies

Requires **Python 3.12+**.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

### 3. Configure the provider

For offline/local development, use:

```text
config/rag_config.local.yaml
```

For NVIDIA-powered generation, set:

```bash
export OPENAI_API_KEY="your-nvidia-api-key"
```

The environment variable name is retained for deployment compatibility; requests are sent to NVIDIA's API endpoint.

**Never commit API keys or other secrets to the repository.**

---

## 🖥️ CLI Usage

### Ingest

```bash
rag-ingest --config config/rag_config.local.yaml data/raw
```

Or with the NVIDIA configuration:

```bash
rag-ingest \
  --config config/rag_config.yaml \
  docs/handbook.pdf \
  notes.txt
```

Useful options:

```text
--index-type {flat,ivf,hnsw}  Choose the FAISS index type
--no-recursive                Do not recursively scan directories
--no-persist                  Run without persisting the vector store
```

### Query

```bash
rag-query \
  --config config/rag_config.local.yaml \
  --query "What are the key policies?"
```

---

## 🐍 Python API

The core pipeline can also be used directly:

```python
from src.rag_pipeline import RAGPipeline
from src.utils.config_utils import load_config

config = load_config("config/rag_config.local.yaml")
pipeline = RAGPipeline(config)

pipeline.ingest_documents(["data/raw/reference.txt"])

response = pipeline.query(
    "What does the reference say about deployment?"
)

print(response.response_text)
print(response.sources)
```

---

## 🧪 Testing

Run the full suite:

```bash
pytest -q
```

Useful targeted suites:

```bash
pytest -q tests/test_cli.py
pytest -q tests/test_integration.py
pytest -q tests/test_performance.py
```

Dependency security audit:

```bash
pip-audit -r requirements.txt
```

Dependency auditing is also configured through GitHub Actions.

---

## 📊 Dashboard & Observability

The dashboard surfaces session-level information such as:

- indexed document count
- vector/chunk count
- query count
- uptime and component health
- provider and model information
- document types
- recent documents
- retrieved sources
- query/history state

Operational logs are sanitized so credentials and sensitive request information are not emitted during normal operation.

---

## 🎯 Design Principles

**Grounded over generic**  
Retrieve relevant document context before generation to reduce unsupported answers.

**Traceable over opaque**  
Keep source metadata attached to responses so retrieved context can be inspected.

**Secure by default**  
Enforce validation, CSRF protection, rate limiting, secure cookies, security headers, and log redaction at the API layer.

**Lightweight deployment**  
Avoid heavyweight local model runtimes and persistent vector infrastructure for the public portfolio demo.

**Provider abstraction**  
Keep embedding and LLM providers behind interfaces so the core pipeline remains testable and extensible.

---

## 🗺️ Roadmap

The portfolio demo is intentionally scoped around the temporary-session architecture. Potential extensions include:

- persistent multi-user vector storage
- authentication and authorization
- background document processing
- larger document collections
- streaming responses
- retrieval-quality evaluation
- production observability and alerting

---

## 👤 Author

**Mehul Gupta**  
Computer Science & Engineering · Manipal University Jaipur

GitHub: https://github.com/017Mehul

---

## 📄 License

No license file is currently included. Reuse of the source is therefore subject to the repository's applicable rights unless a license is added.
