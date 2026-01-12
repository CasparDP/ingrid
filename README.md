# Ingrid

A document processing pipeline for extracting and analyzing scanned letters and newspaper articles.

## Overview

Ingrid processes JPG scans of handwritten and typed documents (letters, newspaper articles) and:

- Extracts text using OCR and handwritten text recognition
- Auto-classifies documents (handwritten/typed, letter/newspaper)
- Detects languages (Dutch, English)
- Generates summaries and metadata using configurable LLM providers
- Stores embeddings in ChromaDB for semantic search
- Enables network-style analysis of topics, people, and relationships

## Status

🚧 **Work in Progress** - This project is in early development.

## Tech Stack

- Python 3.11+ with Poetry
- Docling for document processing
- TrOCR for handwritten text recognition
- ChromaDB for vector embeddings
- SQLite for metadata storage
- Configurable LLM backends (Ollama, Anthropic, Google AI, HuggingFace)

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ingrid.git
cd ingrid

# Install dependencies
poetry install

# Copy configuration templates
cp config.example.yaml config.yaml
cp .env.example .env

# Add your scans to the scans/ directory
# Edit config.yaml and .env with your settings
```

## Usage

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture.

## License

MIT