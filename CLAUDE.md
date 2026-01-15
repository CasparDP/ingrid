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

### ✅ Phase 3: Classification (COMPLETED)
- **Classification Module** (`src/ingrid/classification/`)
  - `models.py`: Core data models (ClassificationResult, ClassificationJob, DocType, ClassifierType)
  - `language_detector.py`: Language detection using langdetect (Dutch, English, German, French)
  - `vision_classifier.py`: Vision LLM classifier for image-based classification
  - `heuristic_classifier.py`: Rule-based classifier using text patterns and extraction hints
  - `__init__.py`: ClassificationOrchestrator with fail-safe sequential execution
- **Classification Features**
  - Document type detection (letter, newspaper_article, other)
  - Content type detection (handwritten, typed, mixed)
  - Multi-language support with confidence scores
  - Confidence threshold flagging for manual review
  - Manual override flags (--doc-type, --content-type)
- **Configuration Updates**
  - Extended `ClassificationConfig`: Classifier toggles, parallel execution, review thresholds
  - Added `langdetect` dependency
- **CLI Integration**
  - Classification runs after extraction in both single and batch modes
  - Detailed classification output with confidence scores and reasoning
  - Manual override support for correcting auto-classification
  - Batch processing summary includes classification stats
- **Tested & Working**: Successfully classified documents with fall-back to heuristic classifier
- **Performance**: ~0.3s heuristic, ~10-25s vision LLM per document

### ✅ Phase 4: Text Processing & Metadata (COMPLETED)
- **Processing Module** (`src/ingrid/processing/`)
  - `models.py`: Core data models (ProcessingResult, DocumentMetadata, ProcessorType)
  - `cleanup.py`: LLM-based OCR error correction with temperature=0.3
  - `metadata.py`: JSON-based metadata extraction (date, sender, recipient, location, topics, people)
  - `summarizer.py`: 2-3 sentence summaries in original language with temperature=0.5
  - `markdown.py`: Markdown generator with YAML frontmatter following CLAUDE.md specification
  - `__init__.py`: ProcessingOrchestrator with sequential execution (cleanup → metadata → summary → markdown)
- **Processing Features**
  - Text cleanup: Fix OCR errors while preserving structure, meaning, and original language
  - Metadata extraction: Structured data with confidence scores per field
  - Summarization: Language-aware summaries (Dutch, English, German, French)
  - Markdown output: YAML frontmatter + summary + transcription + collapsible raw OCR
  - Fail-safe design: Partial failures logged, processing continues
- **Configuration Updates**
  - Uses existing `ProcessingConfig`: max_retries, generate_summaries, extract_metadata flags
  - Temperature settings: cleanup=0.3, metadata=0.2, summary=0.5
- **CLI Integration**
  - New flags: `--skip-cleanup`, `--skip-metadata`, `--skip-summary`, `--write-markdown/--no-markdown`
  - Processing runs after classification in both single and batch modes
  - Detailed processing output with Rich formatting
  - Batch processing summary includes processing stats
  - Verbose mode shows full summary in Panel
- **Tested & Working**: Successfully processed documents with text cleanup, summarization, and markdown generation
- **Performance**: ~55s per document (32s cleanup + 16s metadata + 7s summary on Ollama local)
- **Bug Fixes**: Fixed vision_classifier.py AttributeError (total_tokens → tokens_used)

### ✅ Phase 5: Storage & Search (COMPLETED)
- **Storage Module** (`src/ingrid/storage/`)
  - `models.py`: SQLAlchemy Document model with comprehensive metadata tracking
  - `database.py`: DatabaseManager with CRUD operations, tag management, and statistics
  - `vectorstore.py`: VectorStoreManager with ChromaDB + CLIP embeddings
  - `__init__.py`: StorageOrchestrator coordinating database + vector store operations
- **Translation Module** (`src/ingrid/processing/translator.py`)
  - `SummaryTranslator`: Automatic English translation of summaries (temperature=0.3)
  - Integrated into ProcessingOrchestrator as optional 4th step
  - Stores both original and English summaries
- **Database Features** (SQLite + SQLAlchemy)
  - WAL mode for better concurrency
  - Complete document metadata with confidence scores
  - Processing status tracking for all pipeline phases
  - Manual tagging and notes support
  - Comprehensive statistics (by type, language, processing times)
- **Vector Store Features** (ChromaDB + embeddings)
  - **4 Collections**: cleaned_text, summaries, summaries_english, images
  - Text embeddings via configured embedding provider
    - **Recommended**: `nomic-embed-text` (8192 token context, ~28k chars)
    - Run `ollama pull nomic-embed-text` to install
  - Image embeddings via CLIP (openai/clip-vit-base-patch32)
  - Text truncation safety: Long texts truncated to 28,000 chars before embedding
  - MPS/CUDA/CPU device support for CLIP
  - Semantic search across all collections
  - Visual similarity search using CLIP
- **Configuration Updates**
  - `DatabaseConfig`: journal_mode, cache_size, timeout settings
  - `ChromaDBConfig`: similarity_metric configuration
  - `get_embedding_config()` method added to Config class
  - Extended `config.yaml` with database and chromadb sections
- **CLI Integration**
  - `--skip-storage` flag added to `process` command
  - Storage runs after processing in both single and batch modes
  - Batch summary includes storage statistics (stored/failed counts)
  - New `ingrid stats` command: Database and vector collection statistics
  - New `ingrid search <query>` command: Semantic text search with --collection and --top-k flags
- **Markdown Output Updates**
  - YAML frontmatter includes both summaries (original + English)
  - Separate sections for original and English summaries
  - Language code stored for summary language detection
- **Dependencies Added**
  - `sqlalchemy ^2.0`: ORM for database operations
  - `chromadb ^0.5`: Vector store for embeddings
- **Tested & Working**: Full pipeline from extraction → storage → search
- **Performance**: ~10-15s per document for storage (2s CLIP + 3s text embeddings + 1s database)

### ✅ Phase 5 REVISION: Storage Fixes & CLI Commands (COMPLETED)
- **Critical Bug Fixes**
  - Fixed SQLite PRAGMA execution error by adding `text()` wrapper in `database.py`
  - Added `flag_modified()` for JSON field updates (manual tags) in SQLAlchemy
  - Enhanced storage verification logging in `vectorstore.py` and `database.py`
  - Added `verify_storage()` and `verify_database()` methods for health checks
- **Structured LLM Response Parsing** (`src/ingrid/llm/structured.py`) - NEW MODULE
  - Pydantic models: `MetadataResponse`, `ClassificationResponse`, `SummaryResponse`
  - Multi-tier parsing strategy: direct JSON → regex cleaning → manual extraction
  - `extract_with_retry()` function with feedback loop for parse failures
  - Robust handling of malformed JSON from smaller local models
  - Field validators for pronoun filtering and normalization
  - Integrated into `metadata.py` for more reliable extraction
- **Per-Task Model Configuration**
  - Added `task_models` field to `LLMConfig` in `config.py`
  - New `get_provider_for_task()` helper in `llm/__init__.py`
  - Orchestrators updated to use task-specific models:
    - `ProcessingOrchestrator`: Separate models for cleanup, metadata, summarization, translation
    - `ClassificationOrchestrator`: Task-specific model for classification
  - Allows using better models (e.g., qwen3-vl:235b-cloud) for structured JSON tasks
  - Config example in `config.yaml` with commented task_models section
- **New CLI Commands**
  - `ingrid verify-storage`: Health check for database and vector store
  - `ingrid list`: List processed documents with filters (--doc-type, --content-type, --flagged)
  - `ingrid show <id>`: Display detailed document information (supports partial ID match)
  - `ingrid tag <id>`: Manage document tags (--add, --remove, --list)
  - `ingrid search-image <path>`: Visual similarity search using CLIP embeddings
  - `ingrid cleanup`: Clean up database and ChromaDB with options:
    - `--detect-bad`: Scan for documents with bad extractions (LLM instruction leakage)
    - `--delete-bad`: Delete all detected bad extractions
    - `--remove-duplicates`: Remove duplicate documents, keeping most recent
    - `--delete <id>`: Delete a specific document by ID
    - `--dry-run`: Preview what would be deleted without making changes
    - `--delete-markdown`: Also delete markdown files (default: keep them as archive)
    - Automatically deletes ChromaDB embeddings (no manual cleanup needed)
    - Markdown files are preserved by default for archival purposes
- **Bad Extraction Detection** (`cli.py`)
  - Detects documents where OCR failed and LLM returned its own instructions
  - Pattern matching for instruction text (e.g., "corrigeer duidelijke ocr-fouten", "fix obvious ocr errors")
  - Detection of bad topics like "OCR", "correction", "instructions"
  - Short extraction detection (<50 chars)
- **Database Operations**
  - Added `delete_document()` method to `DatabaseManager`
- **Vector Store Operations**
  - Added `delete_embeddings()` method to `VectorStoreManager` for single document cleanup
  - Added `delete_embeddings_batch()` method for bulk cleanup
  - Cleanup command now automatically removes embeddings from all 4 ChromaDB collections
- **Files Changed**
  - Modified: `database.py`, `vectorstore.py`, `metadata.py`, `config.py`, `cli.py`, `processing/__init__.py`, `classification/__init__.py`, `llm/__init__.py`
  - Created: `llm/structured.py` (481 lines)
  - Total: ~1,150 lines of new code
- **Tested & Working**: All storage operations, CLI commands, and per-task model configuration verified

### ✅ Phase 6a: Web Foundation (COMPLETED)
- **Vue 3 Web Application** (`web/`)
  - Vue 3 + Vite + TypeScript + Tailwind CSS project setup
  - Vue Router with 3 views: Dashboard, Documents, Network
  - Responsive layout with navigation bar
  - TypeScript interfaces for Stats, Document, NetworkData types
- **Data Fetching Composables** (`web/src/composables/useData.ts`)
  - `useStats()`: Fetch and display dashboard statistics
  - `useDocuments()`: Fetch and display all documents
  - `useNetwork()`: Fetch network graph data
  - Error handling and loading states for all composables
- **View Components**
  - **DashboardView**: Stats cards (total, letters, newspaper articles, flagged) + Top 5 topics/people/locations
  - **DocumentsView**: Table with filename, type, date, and topics
  - **NetworkView**: Placeholder for D3 visualization (Phase 6d)
- **CLI Export Command** (`ingrid export-web`)
  - Exports 3 JSON files to `web/public/data/`:
    - `stats.json`: Dashboard statistics with aggregated topics/people/locations
    - `documents.json`: All document metadata for web display
    - `network.json`: Nodes (documents) and edges (shared 2+ topics)
  - Uses Counter to aggregate top 20 topics, people, and locations
  - Network edge calculation: Documents sharing 2+ topics are connected
- **GitHub Actions Workflow** (`.github/workflows/deploy-web.yml`)
  - Triggers on push to main when `web/**` changes
  - Builds Vue app with `npm run build` (NODE_ENV=production)
  - Deploys `web/dist/` to GitHub Pages
  - Site URL: `https://CasparDP.github.io/ingrid/`
- **Vite Configuration**
  - Base path: `/ingrid/` for GitHub Pages
  - Path alias: `@/` → `./src/`
  - TypeScript path mapping configured in `tsconfig.app.json`
- **Styling**
  - Tailwind CSS with utility classes
  - Responsive grid layouts (md breakpoints)
  - Custom colors: primary (blue), secondary (purple)
  - Minimal, clean design (polish in Phase 6e)
- **Tested & Working**
  - Vue app builds successfully without errors
  - CLI export command registered and functional
  - All TypeScript paths resolve correctly
  - Ready for deployment to GitHub Pages

### ✅ Phase 6b-e: Web GUI Enhancements (COMPLETED)

**Architecture:**
- Static site hosted on GitHub Pages (no backend needed)
- Vue 3 + Vite + TypeScript + Tailwind CSS
- D3.js for network visualizations
- Data exported as JSON from pipeline (`ingrid export-web`)
- GitHub Actions deploys `web/dist/` to `gh-pages` branch

**Sub-phases:**

#### Phase 6b: Dashboard (COMPLETED ✅)
- **Enhanced Stats Cards**
  - Responsive grid (1 column mobile, 2 on tablet, 4 on desktop)
  - Hover effects with shadow transitions
  - Color-coded metrics with icons (blue, green, purple, yellow)
  - Border-left accent colors for visual hierarchy
  - Large, bold numbers for quick scanning
- **Breakdown Cards** (`web/src/components/BreakdownCard.vue`)
  - Reusable component for category distributions
  - Horizontal progress bars with custom colors
  - Auto-sorted by count (highest first)
  - Custom color mapping per category
  - Three breakdowns: Document Type, Content Type, Language
  - Dark mode support
- **D3 Horizontal Bar Charts** (`web/src/components/HorizontalBarChart.vue`)
  - Reusable D3.js chart component
  - Responsive SVG rendering with auto-resize
  - Hover effects on bars
  - Value labels next to bars
  - Three charts: Top 10 Topics, Top 10 People, Top 10 Locations
  - Data filtering: Removes null/empty entries from people and locations
- **Loading Skeletons**
  - Animated pulse skeletons for stats cards, breakdowns, and charts
  - Dark mode aware skeleton colors
- **Empty States**
  - Icons and messages when no data available
  - Graceful fallbacks for missing topics/people/locations

#### Phase 6c: Document Lookup (COMPLETED ✅)
- **Enhanced DocumentsView** (`web/src/views/DocumentsView.vue`)
  - Full-text search (filename, topics, people, sender, recipient, location)
  - Filter dropdowns: doc_type, content_type, language
  - Flagged-only checkbox filter
  - "Clear filters" button when filters active
  - Results count display
  - Sortable columns (filename, type, date) with sort icons
  - Pagination (15 items per page) with Previous/Next buttons
  - Click row to open document detail modal
  - Loading skeleton states
  - Empty state with helpful message
  - Full dark mode support
- **DocumentDetailModal Component** (`web/src/components/DocumentDetailModal.vue`)
  - Teleported modal with smooth fade/scale transitions
  - Sticky header with filename and close button
  - Type badges (doc_type, content_type, languages, flagged status)
  - Metadata grid (date, sender, recipient, location)
  - Original summary with language indicator
  - English summary (if different from original)
  - Topics list with blue badges
  - People mentioned with purple badges
  - Manual tags display
  - Document ID and processing date footer
  - Full dark mode support

#### Phase 6d: Network Graph (COMPLETED ✅)
- **NetworkGraph Component** (`web/src/components/NetworkGraph.vue`)
  - D3 force-directed simulation
  - Node colors by document type:
    - Blue (#3b82f6) = Letter
    - Green (#22c55e) = Newspaper Article
    - Gray (#9ca3af) = Other
  - Node size scales with connection count (8-25px radius)
  - Edge width based on shared topic count
  - Interactive features:
    - Hover: Node enlarges, edges highlight, tooltip shows details
    - Click: Opens document detail modal
    - Drag: Reposition nodes with physics simulation
    - Zoom: In/Out/Reset buttons + scroll wheel
  - Legend panel explaining node colors
  - Zoom controls (zoom in, zoom out, reset view)
  - Empty state when no connections exist
  - Responsive to container resize
- **Enhanced NetworkView** (`web/src/views/NetworkView.vue`)
  - Stats bar: Documents count, Connections count, Unique topics count
  - Hint text: "Click a node to view document details"
  - "Most Connected Topics" section showing top shared topics
  - Loading skeleton state
  - Integration with DocumentDetailModal
  - Full dark mode support

#### Phase 6e: Polish (COMPLETED ✅)
- **Dark Mode**
  - Toggle button in NavBar (sun/moon icons)
  - System preference detection on first load
  - LocalStorage persistence (`ingrid-dark-mode`)
  - Smooth 200ms transitions between modes
  - Full support across all components and views
- **Enhanced NavBar** (`web/src/components/NavBar.vue`)
  - Logo icon with gradient background
  - Dark mode toggle button
  - Active route highlighting
  - Dark mode styling
- **Consistent Styling**
  - All components use `dark:` Tailwind variants
  - Background: `bg-gray-50` / `dark:bg-gray-900`
  - Cards: `bg-white` / `dark:bg-gray-800`
  - Text: Appropriate gray scale for both modes
  - Borders: `border-gray-200` / `dark:border-gray-700`
  - Form inputs: Dark backgrounds and borders
  - Badges: Semi-transparent dark variants
- **Loading States**
  - Animated pulse skeletons throughout
  - Skeleton colors adapt to dark mode
- **Error States**
  - Red-themed error messages
  - Retry buttons where applicable
- **Empty States**
  - Informative icons and messages
  - Suggestions for user action
- **Tailwind Configuration**
  - `darkMode: 'class'` enabled
  - Custom colors: primary (blue), secondary (purple)

**New/Modified Files:**
```
web/src/
├── App.vue                          # Dark mode state management
├── components/
│   ├── NavBar.vue                   # Logo, dark mode toggle
│   ├── BreakdownCard.vue            # Dark mode support added
│   ├── DocumentDetailModal.vue      # NEW - Modal for document details
│   └── NetworkGraph.vue             # NEW - D3 force-directed graph
├── views/
│   ├── DashboardView.vue            # Loading skeletons, empty states, dark mode
│   ├── DocumentsView.vue            # Search, filter, sort, pagination, dark mode
│   └── NetworkView.vue              # Stats bar, graph integration, dark mode
└── tailwind.config.js               # darkMode: 'class' enabled
```

**Data Export Format:**
```json
// stats.json
{
  "total_documents": 18,
  "by_doc_type": {"letter": 12, "newspaper_article": 5, "other": 1},
  "by_content_type": {"handwritten": 8, "typed": 10},
  "by_language": {"de": 10, "nl": 6, "en": 2},
  "flagged_count": 3,
  "top_topics": [{"name": "Buddhism", "count": 5}, ...],
  "top_people": [{"name": "Chris", "count": 3}, ...],
  "top_locations": [{"name": "Boston", "count": 2}, ...]
}

// documents.json
[{
  "id": "abc123",
  "filename": "PHOTO-2025-12-28-10-15-43.jpg",
  "doc_type": "letter",
  "content_type": "handwritten",
  "languages": ["nl"],
  "date": "1943-05-12",
  "sender": "Ingrid",
  "recipient": "Johannes",
  "location": "Amsterdam",
  "topics": ["family", "war"],
  "people_mentioned": ["Maria", "Hans"],
  "summary": "...",
  "summary_english": "...",
  "flagged_for_review": false,
  "manual_tags": []
}, ...]

// network.json
{
  "nodes": [{"id": "abc123", "label": "PHOTO-2025...", "doc_type": "letter", "topics": [...]}],
  "edges": [{"source": "abc123", "target": "def456", "weight": 2, "shared_topics": [...]}]
}
```

**Site URL:** `https://CasparDP.github.io/ingrid/`

**Build Status:** ✅ Builds successfully (`npm run build`)

### 🔧 Known Issues & Fixes Applied
- **Embedding Model Context Length**: Changed from `tazarov/all-minilm-l6-v2-f32` (256 tokens) to `nomic-embed-text` (8192 tokens) to handle longer documents
- **Text Truncation Safety**: Added truncation at 28,000 characters in `vectorstore.py` to prevent context overflow (chunking planned for future if needed)
- **SQLite PRAGMA Error**: Fixed by wrapping raw SQL with `sqlalchemy.text()` in `database.py`
- **Anti-Hallucination Controls**: Enhanced metadata extraction prompts with strict grounding rules:
  - LLM instructed to extract ONLY explicitly written information
  - Null values required for uncertain/missing fields (not "Unknown" or guesses)
  - Low confidence scores (0.1-0.3) for ambiguous information
  - High confidence (0.7-1.0) only for clearly stated facts
- **Pydantic Validation for Metadata** (`llm/structured.py`):
  - `MetadataResponse` model with field descriptions enforcing extraction rules
  - `field_validator` to filter pronouns (ich, du, I, you, hij, zij, etc.) from `people_mentioned`
  - Automatic normalization of "unknown", "n/a", "null" → `None` for sender/recipient
  - Pronoun filter includes German, Dutch, and English pronouns plus common non-name words

### 📋 Planned Improvements (Future)
- Text chunking for very long documents (currently truncated at 28k chars)
- Person-based network connections (documents mentioning same person)
- Temporal connections (documents close in time)
- Timeline view for documents
- Multi-page document support (letters spanning multiple scans)

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
│       │   ├── structured.py # Pydantic models & JSON parsing
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
├── web/                      # Vue 3 web application (Phase 6)
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue               # Root component with dark mode management
│   │   ├── router/
│   │   │   └── index.ts
│   │   ├── views/
│   │   │   ├── DashboardView.vue   # Stats cards, breakdowns, D3 bar charts
│   │   │   ├── DocumentsView.vue   # Search, filter, sort, pagination, detail modal
│   │   │   └── NetworkView.vue     # D3 force-directed network graph
│   │   ├── components/
│   │   │   ├── NavBar.vue              # Navigation with dark mode toggle
│   │   │   ├── BreakdownCard.vue       # Category distribution progress bars
│   │   │   ├── HorizontalBarChart.vue  # D3 horizontal bar charts
│   │   │   ├── DocumentDetailModal.vue # Document detail modal (NEW)
│   │   │   └── NetworkGraph.vue        # D3 force-directed graph (NEW)
│   │   ├── composables/
│   │   │   └── useData.ts          # Data fetching composables
│   │   ├── types/
│   │   │   └── index.ts            # TypeScript interfaces
│   │   └── assets/
│   │       └── main.css            # Tailwind imports
│   ├── public/
│   │   └── data/                   # JSON data (generated by `ingrid export-web`)
│   │       ├── stats.json
│   │       ├── documents.json
│   │       └── network.json
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js          # Dark mode enabled
│   ├── postcss.config.js
│   └── tsconfig.json
│
└── .github/
    └── workflows/
        └── deploy-web.yml          # GitHub Actions: build & deploy to gh-pages
```

---

## Configuration

### config.yaml Structure

```yaml
# config.yaml
llm:
  # Active provider: ollama, ollama_cloud, anthropic, google, huggingface
  provider: ollama

  # Optional: Override model for specific tasks (NEW in Phase 5)
  # Useful for using better models for structured JSON output
  task_models:
    metadata_extraction: qwen3-vl:235b-cloud  # Better at structured JSON
    classification: qwen3-vl:235b-cloud       # Better at structured JSON
    summarization: gemma3:4b                  # Good for text generation
    cleanup: gemma3:4b                        # Good for text correction
    translation: gemma3:4b                    # Good for translation

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
# ============= Processing =============
# Process a single file
ingrid process scans/PHOTO-2025-12-28-10-15-43.jpg

# Process all files in scans directory
ingrid process --batch

# Process with manual classification override
ingrid process scans/photo.jpg --doc-type letter --content-type handwritten

# Skip specific processing steps
ingrid process --skip-cleanup --skip-metadata --skip-storage

# ============= Storage & Database =============
# Verify database and vector store health (NEW in Phase 5)
ingrid verify-storage

# Database statistics
ingrid stats

# ============= Document Management =============
# List all processed documents (NEW in Phase 5)
ingrid list

# List with filters
ingrid list --doc-type letter --limit 20
ingrid list --flagged --content-type handwritten

# Show detailed document info (NEW in Phase 5)
ingrid show abc123
ingrid show PHOTO-2025-12-28-10-15-43.jpg
ingrid show abc123 --json

# ============= Tagging =============
# Manage document tags (NEW in Phase 5)
ingrid tag abc123 --add "verified" --add "important"
ingrid tag abc123 --remove "needs-review"
ingrid tag abc123 --list

# ============= Search =============
# Semantic text search
ingrid search "food shortages during war"
ingrid search "meditation" --collection summaries --top-k 5

# Visual similarity search (NEW in Phase 5)
ingrid search-image scans/letter.jpg
ingrid search-image scans/letter.jpg --top-k 5

# ============= Cleanup =============
# Scan for bad extractions (LLM instruction leakage)
ingrid cleanup --detect-bad

# Preview duplicate removal
ingrid cleanup --remove-duplicates --dry-run

# Delete bad extractions and duplicates (keeps markdown files as archive)
ingrid cleanup --delete-bad --remove-duplicates

# Delete bad extractions AND their markdown files
ingrid cleanup --delete-bad --delete-markdown

# Delete a specific document by ID
ingrid cleanup --delete abc123

# Full cleanup workflow:
# 1. Preview: ingrid cleanup --delete-bad --remove-duplicates --dry-run
# 2. Execute: ingrid cleanup --delete-bad --remove-duplicates
#    (ChromaDB embeddings deleted automatically, markdown files preserved)
# 3. Re-process if needed: ingrid process --batch
# 4. Re-export: ingrid export-web

# ============= Web Export =============
# Export data for web dashboard
ingrid export-web
ingrid export-web --output web/public/data

# ============= Future Commands (Phase 6) =============
# Export network analysis
# ingrid analyze --network --output analysis/network.json

# Re-run processing steps
# ingrid summarize abc123 --provider google
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
