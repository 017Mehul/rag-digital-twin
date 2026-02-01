# RAG Digital Twin

A sophisticated AI-powered system implementing Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses from domain-specific knowledge bases using vector databases and large language models.

## Overview

The RAG Digital Twin follows a 6-step pipeline architecture:

1. **Document Processing**: Extract and chunk text from PDF/TXT documents
2. **Embedding Generation**: Convert text chunks into vector embeddings
3. **Vector Storage**: Store embeddings in FAISS vector database
4. **Query Processing**: Process user queries and perform similarity search
5. **Context Retrieval**: Retrieve and format relevant context
6. **Response Generation**: Generate context-aware responses using LLMs

## Features

- **Multi-format Document Support**: PDF and TXT document processing
- **Flexible Model Support**: OpenAI and Hugging Face embedding/LLM providers
- **Scalable Vector Storage**: FAISS-based vector database with multiple index types
- **Property-Based Testing**: Comprehensive correctness validation using hypothesis
- **Configurable Pipeline**: Extensive configuration options for all components
- **Error Handling**: Robust error handling and recovery mechanisms
- **Monitoring & Logging**: Comprehensive system monitoring and audit trails

## Project Structure

```
rag-digital-twin/
├── data/                    # Input documents
│   ├── processed/          # Processed document chunks
│   └── raw/               # Original documents
├── embeddings/            # Vector store files
├── src/                   # Source code
│   ├── models/           # Data models
│   ├── providers/        # LLM and embedding providers
│   └── utils/           # Utility functions
├── config/              # Configuration files
├── logs/               # System logs
├── tests/              # Test files
└── requirements.txt    # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-digital-twin
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure the system**:
   ```bash
   # Edit config/rag_config.yaml as needed
   ```

## Quick Start

### 1. Document Ingestion

```python
from src.rag_pipeline import RAGPipeline
from src.models.rag_config import RAGConfig

# Load configuration
config = RAGConfig.from_json_file("config/rag_config.yaml")

# Initialize pipeline
pipeline = RAGPipeline(config)

# Ingest documents
results = pipeline.ingest_documents(["data/raw/document1.pdf", "data/raw/document2.txt"])
print(f"Processed {results.successful_documents} documents")
```

### 2. Query Processing

```python
# Query the system
response = pipeline.query("What is the main topic of the documents?")
print(f"Response: {response.response_text}")
print(f"Sources: {response.sources}")
```

## Configuration

The system is configured through `config/rag_config.yaml`. Key configuration sections:

- **Model Configuration**: Embedding and LLM provider settings
- **Processing Configuration**: Document chunking and processing parameters
- **Retrieval Configuration**: Similarity search and context retrieval settings
- **System Configuration**: Performance and operational parameters

## Testing

The project includes comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run only property-based tests
pytest -m "hypothesis"
```

## Core Data Models

### DocumentChunk
Represents processed document segments with metadata and source information.

### EmbeddingMetadata
Tracks embedding information and maintains relationships to source content.

### RAGConfig
Centralized configuration management for all system parameters.

### SearchResults
Results from vector similarity search operations.

### GeneratedResponse
LLM-generated responses with source attribution and confidence metrics.

## Error Handling

The system includes comprehensive error handling with:

- **Structured Exceptions**: RAGException base class with error codes and context
- **Component-Specific Errors**: Specialized exceptions for each system component
- **Error Recovery**: Automatic retry logic and fallback mechanisms
- **Logging Integration**: Detailed error logging for debugging and monitoring

## Development

### Adding New Document Formats

1. Extend the DocumentProcessor class
2. Add format-specific extraction logic
3. Update configuration and tests

### Adding New Model Providers

1. Implement the provider interface
2. Add provider-specific configuration
3. Update the provider factory
4. Add comprehensive tests

### Property-Based Testing

The system uses hypothesis for property-based testing to validate universal correctness properties:

```python
@given(st.text(min_size=1))
def test_document_chunk_content_preservation(content):
    """Property: DocumentChunk should preserve content exactly."""
    chunk = DocumentChunk(content=content, source_file="test.txt")
    assert chunk.content == content
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please refer to the project documentation or create an issue in the repository.