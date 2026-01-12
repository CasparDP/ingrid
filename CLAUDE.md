# CLAUDE.md - Ingrid Letter Extraction Pipeline

## Current Status

### ✅ Phase 1: Project Foundation (COMPLETED)
- Poetry project initialized with `pyproject.toml`
- Package structure created under `src/ingrid/`
- Configuration system implemented (`config.py`) loading `config.yaml` and `.env`
- Abstract LLM interface created (`src/ingrid/llm/base.py`)
- Ollama provider implemented (`src/ingrid/llm/ollama.py`)
- Ollama Cloud provider implemented (`src/ingrid/llm/ollama_cloud.py`)
  - Supports cloud-hosted models like `qwen3-vl:235b-cloud`
  - Uses `OLLAMA_API_KEY` environment variable
- CLI skeleton with Typer (`src/ingrid/cli.py`)
- `process` command with `--batch` flag

### ✅ Phase 2: OCR/Extraction (COMPLETED)
- **Extraction Module** (`src/ingrid/extraction/`)
  - `models.py`: Core data models (ExtractionResult, PreprocessedImage, ExtractionJob)
  - `preprocessing.py`: Image enhancement (deskew with Hough transform, contrast/sharpness)
  - `ocr.py`: Docling OCR integration for typed documents
  - `htr.py`: TrOCR (microsoft/trocr-large-handwritten) for handwritten text
  - `vision_extract.py`: Vision LLM extraction using existing LLM providers
  - `__init__.py`: ExtractionOrchestrator with parallel execution
- **Configuration Updates**
  - `PreprocessingConfig`: Deskew, contrast, DPI settings
  - Updated `OCRConfig`: Extractor toggles, parallel execution, preprocessing settings
- **CLI Integration**
  - `--verbose` flag for detailed extraction results
  - `process_single()`: Extract text from single file with Rich output
  - `process_batch()`: Process all scans with progress bar
- **Tested & Working**: Successfully extracted 3343 characters from historical scans
- **Performance**: ~60s per document (parallel Docling + TrOCR on CPU/MPS)

### ⏳ Phase 3: Classification & Processing (PLANNED)
### ⏳ Phase 4: Storage & Search (PLANNED)
### ⏳ Phase 5: Analysis & Web GUI (PLANNED)

---

## Project Overview

**Ingrid** is a document processing pipeline that extracts content from scanned letters (handwritten and typed) and newspaper articles, storing the results in a searchable database with embeddings for semantic search and network analysis.

### Primary Goals
1. Extract text from JPG scans of historical letters and newspaper articles
2. Auto-classify documents (handwritten vs typed, letter vs newspaper article)
3. Detect and handle multiple languages (primarily Dutch and English)
4. Generate summaries, metadata, and embeddings
5. Enable network-style analysis based on topics, people, and relationships

---

## Project Structure

```
ingrid/
├── CLAUDE.md                 # This file - project guidelines
├── README.md                 # User-facing documentation
├── pyproject.toml            # Poetry dependencies
├── config.yaml               # Runtime configuration (LLM providers, models, paths)
├── config.example.yaml       # Template configuration (committed to git)
├── .env                      # API keys (gitignored)
├── .env.example              # Template for environment variables
│
├── scans/                    # Input: raw JPG scans (gitignored)
│   └── *.jpg
│
├── output/                   # Output: extracted markdown files
│   └── *.md
│
├── data/                     # Persistent storage
│   ├── ingrid.db             # SQLite database for metadata
│   └── chroma/               # ChromaDB vector store
│
├── src/
│   └── ingrid/
│       ├── __init__.py
│       ├── cli.py            # CLI entry point
│       ├── config.py         # Configuration loader
│       ├── pipeline.py       # Main orchestration
│       │
│       ├── extraction/       # Document processing
│       │   ├── __init__.py
│       │   ├── ocr.py        # OCR with Docling
│       │   ├── htr.py        # Handwritten text recognition
│       │   └── preprocessing.py  # Image enhancement
│       │
│       ├── classification/   # Document classification
│       │   ├── __init__.py
│       │   ├── detector.py   # Handwritten vs typed detection
│       │   ├── doctype.py    # Letter vs newspaper vs other
│       │   └── language.py   # Language detection
│       │
│       ├── llm/              # LLM integration layer
│       │   ├── __init__.py
│       │   ├── base.py       # Abstract LLM interface
│       │   ├── ollama.py     # Ollama local + cloud
│       │   ├── anthropic.py  # Claude API
│       │   ├── google.py     # Google AI (Gemini)
│       │   └── huggingface.py # HuggingFace Inference API
│       │
│       ├── processing/       # Post-extraction processing
│       │   ├── __init__.py
│       │   ├── cleanup.py    # Text correction and formatting
│       │   ├── summarizer.py # Generate summaries
│       │   ├── metadata.py   # Extract metadata (dates, people, etc.)
│       │   └── embeddings.py # Generate embeddings
│       │
│       ├── storage/          # Data persistence
│       │   ├── __init__.py
│       │   ├── database.py   # SQLite operations
│       │   ├── vectorstore.py # ChromaDB operations
│       │   └── models.py     # Data models / schemas
│       │
│       └── analysis/         # Analysis tools (future)
│           ├── __init__.py
│           ├── network.py    # Network analysis
│           └── search.py     # Semantic search
│
├── tests/
│   └── ...
│
└── web/                      # Future: Web GUI for contributors
    └── ...
```

---

## Configuration

### config.yaml Structure

```yaml
# config.yaml
llm:
  # Active provider: ollama, ollama_cloud, anthropic, google, huggingface
  provider: ollama
  
  # Provider-specific settings
  ollama:
    base_url: http://localhost:11434
    model: llama3.2-vision
    embedding_model: nomic-embed-text
  
  ollama_cloud:
    api_key: ${OLLAMA_CLOUD_API_KEY}  # Reference to .env
    model: llama3.2-vision
  
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514
  
  google:
    api_key: ${GOOGLE_AI_API_KEY}
    model: gemini-2.0-flash
  
  huggingface:
    api_key: ${HUGGINGFACE_API_KEY}
    model: meta-llama/Llama-3.2-11B-Vision-Instruct

ocr:
  # OCR engine: docling, tesseract, easyocr
  engine: docling
  
  # Handwritten text recognition model
  htr_model: microsoft/trocr-large-handwritten
  
  # Languages to detect/support
  languages:
    - nl  # Dutch
    - en  # English

classification:
  # Model for document type classification
  auto_detect: true
  confidence_threshold: 0.7

embeddings:
  # Embedding model (can be different from LLM)
  provider: ollama
  model: nomic-embed-text
  dimensions: 768

storage:
  database_path: data/ingrid.db
  chroma_path: data/chroma
  output_path: output

processing:
  batch_size: 10
  generate_summaries: true
  extract_metadata: true
  generate_embeddings: true
```

### Environment Variables (.env)

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_AI_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
OLLAMA_CLOUD_API_KEY=...
```

---

## Data Models

### Document Metadata Schema

```python
class Document:
    id: str                    # UUID
    filename: str              # Original filename
    filepath: str              # Path to source scan
    
    # Classification
    doc_type: str              # letter, newspaper_article, other
    content_type: str          # handwritten, typed, mixed
    languages: list[str]       # Detected languages ['nl', 'en']
    
    # Extracted content
    raw_text: str              # OCR output
    cleaned_text: str          # LLM-cleaned text
    markdown_path: str         # Path to output .md file
    
    # Metadata (extracted by LLM)
    date: date | None          # Date of document (if detected)
    sender: str | None         # For letters
    recipient: str | None      # For letters
    location: str | None       # Place mentioned
    topics: list[str]          # Key topics/themes
    people_mentioned: list[str] # Names mentioned
    
    # Analysis
    summary: str               # LLM-generated summary
    embedding_id: str          # Reference to ChromaDB
    
    # Processing metadata
    confidence_scores: dict    # Per-field confidence
    processing_date: datetime
    manual_tags: list[str]     # User-added tags
    manual_override: dict      # Manual corrections to classification
```

### Output Markdown Format

```markdown
---
id: abc123
filename: PHOTO-2025-12-28-10-15-43.jpg
doc_type: letter
content_type: handwritten
languages: [nl, en]
date: 1943-05-12
sender: Ingrid van der Berg
recipient: Johannes Müller
location: Amsterdam
topics: [family, war, food shortages]
people_mentioned: [Maria, Uncle Hans, Dr. Bakker]
confidence:
  doc_type: 0.92
  content_type: 0.88
  date: 0.75
---

# Letter from Ingrid van der Berg to Johannes Müller

**Date:** May 12, 1943  
**Location:** Amsterdam

## Summary

[LLM-generated summary of the letter's content]

## Transcription

[Full cleaned text of the letter]

## Original OCR Output

<details>
<summary>Raw OCR text (click to expand)</summary>

[Raw OCR output before cleaning]

</details>
```

---

## CLI Interface

```bash
# Process a single file
ingrid process scans/PHOTO-2025-12-28-10-15-43.jpg

# Process all files in scans directory
ingrid process --batch

# Process with manual classification override
ingrid process scans/photo.jpg --doc-type letter --content-type handwritten

# Re-process with different LLM provider
ingrid process --batch --provider anthropic

# Search the database
ingrid search "food shortages during war"

# Export network analysis
ingrid analyze --network --output analysis/network.json

# List all processed documents
ingrid list

# Show document details
ingrid show abc123

# Add manual tags
ingrid tag abc123 --add "verified" --add "important"

# Re-run summarization with different model
ingrid summarize abc123 --provider google

# Database stats
ingrid stats
```

---

## Implementation Guidelines

### Phase 1: Core Pipeline (MVP)

1. **Setup project structure**
   - Initialize Poetry project
   - Create config system with YAML + env vars
   - Set up logging

2. **Basic extraction**
   - Integrate Docling for OCR
   - Implement TrOCR for handwritten text (via HuggingFace)
   - Support Dutch and English

3. **Classification**
   - Handwritten vs typed detector (can use vision model or trained classifier)
   - Document type classification (letter, newspaper, other)
   - Language detection (langdetect or fasttext)

4. **LLM integration**
   - Create abstract LLM interface
   - Implement Ollama provider first (local testing)
   - Add text cleanup prompts
   - Add metadata extraction prompts

5. **Storage**
   - SQLite for document metadata
   - ChromaDB for embeddings
   - Markdown file output

6. **CLI**
   - Basic process command
   - Batch processing
   - List and show commands

### Phase 2: Enhanced Processing

1. **Improved extraction quality**
   - Image preprocessing (deskew, contrast enhancement)
   - Multi-model ensemble for difficult handwriting
   - Confidence scoring

2. **Additional LLM providers**
   - Anthropic Claude
   - Google AI
   - HuggingFace Inference
   - Ollama Cloud

3. **Rich metadata extraction**
   - Date parsing (handle historical date formats)
   - Named entity recognition
   - Topic modeling

4. **Search and analysis**
   - Semantic search via ChromaDB
   - Network analysis (people, topics, locations)

### Phase 3: Web Interface (Future)

1. **Contributor GUI**
   - Document viewer with image + transcription side-by-side
   - Manual correction interface
   - Tagging and verification workflow
   - Search interface

---

## Key Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"

# Document processing
docling = "^2.0"
Pillow = "^10.0"
pdf2image = "^1.16"  # If PDF support needed

# OCR / HTR
transformers = "^4.40"
torch = "^2.0"
# easyocr = "^1.7"  # Alternative OCR

# LLM clients
ollama = "^0.3"
anthropic = "^0.30"
google-generativeai = "^0.7"
huggingface-hub = "^0.23"

# Embeddings and vector store
chromadb = "^0.5"
sentence-transformers = "^3.0"

# Database
sqlalchemy = "^2.0"

# Language detection
langdetect = "^1.0"
# fasttext = "^0.9"  # Alternative, more accurate

# CLI
typer = "^0.12"
rich = "^13.0"

# Configuration
pyyaml = "^6.0"
python-dotenv = "^1.0"
pydantic = "^2.0"
pydantic-settings = "^2.0"

# Utilities
tqdm = "^4.66"
```

---

## LLM Prompts

### Text Cleanup Prompt

```
You are a document transcription specialist. Clean up the following OCR output from a {doc_type} ({content_type}).

The document is in {language}.

OCR Output:
{raw_text}

Instructions:
1. Fix obvious OCR errors while preserving the original meaning
2. Maintain original paragraph structure
3. Keep any dates, names, and places exactly as written
4. Mark unclear sections with [unclear]
5. Do not add any information not present in the original
6. Preserve the original language (do not translate)

Output the cleaned text only, no explanations.
```

### Metadata Extraction Prompt

```
Analyze this {doc_type} and extract metadata.

Text:
{cleaned_text}

Extract the following (respond in JSON):
- date: The date of the document (ISO format if possible, or original format)
- sender: Who wrote/sent this (for letters)
- recipient: Who this was addressed to (for letters)  
- location: Where this was written or references
- topics: Main topics/themes (list of 3-5 keywords)
- people_mentioned: Names of people mentioned
- summary: A 2-3 sentence summary

If a field cannot be determined, use null.
Respond ONLY with valid JSON.
```

### Document Classification Prompt

```
Analyze this image and classify the document.

Determine:
1. doc_type: Is this a "letter", "newspaper_article", or "other"?
2. content_type: Is the text "handwritten", "typed", or "mixed"?
3. languages: What language(s) is the text in? (use ISO codes: en, nl, de, etc.)

Consider:
- Letters typically have a greeting, body, and signature
- Newspaper articles have headlines, columns, and formal structure
- Handwritten text has variable letterforms and connected writing
- Typed text has uniform characters

Respond in JSON format:
{
  "doc_type": "letter|newspaper_article|other",
  "content_type": "handwritten|typed|mixed",
  "languages": ["nl", "en"],
  "confidence": {
    "doc_type": 0.0-1.0,
    "content_type": 0.0-1.0
  }
}
```

---

## Error Handling

1. **OCR failures**: Log error, mark document as failed, continue batch
2. **LLM timeouts**: Retry with exponential backoff (3 attempts)
3. **Classification uncertainty**: If confidence < threshold, flag for manual review
4. **Language detection failure**: Default to Dutch, flag for review

---

## Testing Strategy

1. **Unit tests**: Individual components (OCR wrapper, LLM clients, database operations)
2. **Integration tests**: Full pipeline with sample documents
3. **Golden tests**: Compare output against manually verified transcriptions
4. **Performance tests**: Batch processing timing, memory usage

---

## Quality Metrics

Track these metrics to evaluate pipeline quality:

1. **OCR accuracy**: Character Error Rate (CER), Word Error Rate (WER)
2. **Classification accuracy**: Precision/recall for doc_type and content_type
3. **Metadata extraction**: Accuracy of date, sender, recipient extraction
4. **Processing speed**: Documents per minute
5. **Cost**: API calls per document (when using cloud LLMs)

---

## Notes for Development

- Start with Ollama locally to avoid API costs during development
- Use a small test set (5-10 diverse documents) for rapid iteration
- The `scans/` folder already contains 18 JPG files to work with
- Consider caching LLM responses during development to save time/cost
- ChromaDB can run embedded (no server needed) for simplicity
- The project is named "ingrid" - likely named after a person in the letters

---

## Future Considerations

- **PDF support**: Some archives may provide PDFs instead of JPGs
- **Multi-page documents**: Handle letters that span multiple scans
- **Relationship extraction**: Build knowledge graph of people and their relationships
- **Timeline visualization**: Show documents on a timeline
- **Export formats**: CSV, JSON, QDMX for qualitative data analysis
- **Collaboration**: Multi-user support for distributed transcription efforts