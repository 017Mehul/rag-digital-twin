# RAG Digital Twin

> **A production-ready Retrieval-Augmented Generation (RAG) workspace for turning documents into grounded, traceable answers.**

RAG Digital Twin combines document ingestion, intelligent chunking, NVIDIA-hosted embeddings and LLM generation, FAISS vector retrieval, source attribution, and a responsive web dashboard — without requiring a persistent database for the public demo.

**Live demo:** https://rag.themglabs.com  
**Repository:** https://github.com/017Mehul/rag-digital-twin

---

## ✨ Highlights

- 📄 **PDF & TXT ingestion** with validation and controlled file sizes
- 🧠 **NVIDIA Nemotron embeddings** with 2,048-dimensional vectors
- 💬 **NVIDIA-hosted LLM generation** using `openai/gpt-oss-20b`
- 🔎 **FAISS similarity search** for fast vector retrieval
- 📚 **Source-aware answers** for traceable retrieval
- 🧩 **Session-scoped knowledge bases** for the public demo
- 🛡️ **Security hardening** with CSRF protection, rate limiting, upload validation, secure cookies, security headers, and log redaction
- 📱 **Responsive dashboard** for desktop, tablet, and mobile
- 🧪 **Automated test suite** with integration, CLI, property-based, and performance coverage
- 🔄 **Automated dependency auditing** through GitHub Actions and Dependabot

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
│ Session Management            │
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

1. Upload a document through the web UI.
2. Validate filename, type, size, and content.
3. Parse and split the document into overlapping chunks.
4. Generate NVIDIA passage embeddings.
5. Index embeddings in an in-memory FAISS store.
6. Generate a query embedding for the user's question.
7. Retrieve the nearest document chunks.
8. Pass retrieved context to the NVIDIA-hosted LLM.
9. Return the grounded response with source metadata.

---

## 🚀 Production Demo

The public deployment runs on **Vercel + FastAPI** and intentionally uses **temporary, in-memory FAISS sessions** instead of a persistent vector database.

### Session behavior

- Each browser session gets an isolated knowledge base.
- Sessions expire after **30 minutes of inactivity** by default.
- Closing the browser removes the session-only cookie.
- Documents are not stored in Supabase, Pinecone, or another persistent vector database.
- Vercel instance recycling can discard in-memory state — intentional for this portfolio architecture.
- Up to **5 unique documents** can be uploaded per session.
- Re-uploading the same document replaces its indexed copy without consuming another slot.
- Maximum file size: **10 MB**.
- Supported formats: **PDF and TXT**.
- Query and ingestion endpoints use lightweight per-client rate limits.
- API responses use `no-store` caching headers.

> This architecture is optimized for a public portfolio demonstration, not durable multi-tenant knowledge storage.

---

## 🔐 Security

Security controls are implemented at the API layer rather than relying only on frontend behavior.

### Implemented

- **CSRF protection** using same-site cookies and double-submit tokens
- **Strict origin validation** for browser mutation requests
- **Per-client rate limiting** for ingestion and queries
- **Upload validation** for filename, extension, size, PDF signature, and UTF-8 text
- **Secure session cookies** with HttpOnly, SameSite, and HTTPS settings
- **Security headers**
  - `Strict-Transport-Security`
  - `X-Content-Type-Options`
  - `X-Frame-Options`
  - `Referrer-Policy`
  - `Permissions-Policy`
- **No API secrets in frontend code**
- **Sanitized API errors**
- **Sensitive-data redaction in logs**
- **No source maps shipped with the application**
- **Automated dependency vulnerability auditing**
- **Weekly Dependabot updates**

### Attack-surface review

The current application intentionally has no SQL database, password/JWT authentication system, persistent user object model, webhook receiver, or user-controlled outbound URL fetcher.

Therefore SQL injection, IDOR/BOLA, password/MFA, webhook signature verification, SSRF, and database RLS are not applicable to the current architecture.

---

## 🧠 AI / Retrieval Stack

| Component | Technology |
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

NVIDIA's OpenAI-compatible API endpoint:

```text
https://integrate.api.nvidia.com/v1
```

The embedding pipeline uses NVIDIA's intended **passage** mode during indexing and **query** mode during retrieval.

---

## 📁 Project Structure

```text
rag-digital-twin/
├── api/
│   ├── index.py              # FastAPI entrypoint + API security
│   └── runtime.py            # Temporary session lifecycle
├── config/
│   ├── rag_config.yaml       # NVIDIA production configuration
│   └── rag_config.local.yaml # Offline/mock configuration
├── docs/                     # Project documentation
├── public/
│   ├── index.html            # Web dashboard
│   ├── app.js                # Dashboard behavior
│   ├── ingest.js             # Upload flow
│   ├── sections.js           # Dashboard sections
│   ├── styles.css            # Responsive UI
│   └── theme.js              # Theme handling
├── src/
│   ├── providers/            # Embedding + LLM abstractions
│   ├── rag_pipeline.py       # Core RAG orchestration
│   ├── vector_store.py       # FAISS storage
│   └── document_processor.py # Parsing + chunking
├── tests/                    # Automated test suite
├── requirements.txt
├── pyproject.toml
├── setup.py
└── vercel.json
```

---

## ⚙️ Local Setup

### 1. Clone

```bash
git clone https://github.com/017Mehul/rag-digital-twin.git
cd rag-digital-twin
```

### 2. Install

Requires **Python 3.12+**.

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
```

### 3. Configure

For offline development, use:

```text
config/rag_config.local.yaml
```

For NVIDIA-powered generation, set the NVIDIA credential in your environment:

```bash
export OPENAI_API_KEY="your-nvidia-api-key"
```

> The environment variable name is retained for deployment compatibility. Requests are sent to NVIDIA's API endpoint, not OpenAI.

**Never commit API keys to the repository.**

---

## 🖥️ CLI Usage

### Ingest documents

```bash
rag-ingest --config config/rag_config.local.yaml data/raw
```

Or with NVIDIA:

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

The same pipeline can be used directly from Python:

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

Run the complete suite:

```bash
pytest -q
```

The current project test suite passes **120 tests**.

Targeted examples:

```bash
pytest -q tests/test_cli.py
pytest -q tests/test_integration.py
pytest -q tests/test_performance.py
```

Security dependency auditing:

```bash
pip-audit -r requirements.txt
```

Dependency auditing also runs automatically through GitHub Actions.

---

## 📊 Observability

The dashboard exposes session-level operational information including:

- indexed document count
- vector/chunk count
- query count
- uptime
- provider and model information
- component health
- document types
- recent documents
- retrieval sources

Operational logs are sanitized so sensitive request data and credentials are not emitted into normal application logs.

---

## 🎯 Design Principles

**Grounded over generic**  
Retrieve relevant document context before generation to reduce unsupported responses.

**Traceable over opaque**  
Keep source information attached to responses so retrieved context can be inspected.

**Secure by default**  
Enforce validation, CSRF protection, rate limiting, secure cookies, security headers, and log redaction at the API layer.

**Lightweight deployment**  
Avoid heavyweight local model runtimes and persistent vector infrastructure for the public demo.

**Provider abstraction**  
Keep embedding and LLM providers behind interfaces so the core pipeline remains testable and extensible.

---

## 🗺️ Roadmap

The portfolio demo is complete. Potential production extensions include:

- persistent multi-user vector storage
- authentication and authorization
- background document processing
- larger document collections
- streaming responses
- richer retrieval evaluation
- production observability and alerting

These are intentionally outside the scope of the current temporary-session demo.

---

## 👤 Author

**Mehul Gupta**  
Computer Science & Engineering · Manipal University Jaipur

GitHub: https://github.com/017Mehul

---

## 📄 License

See the repository license and project metadata for licensing information.
