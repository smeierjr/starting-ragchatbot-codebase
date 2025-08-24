# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables - required before running
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality Tools
```bash
# Format code with black and isort
./scripts/format.sh

# Run linting with flake8
./scripts/lint.sh

# Run type checking with mypy
./scripts/typecheck.sh

# Run all quality checks
./scripts/quality.sh
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials using a **tool-based AI architecture**. The system processes course documents, stores them as vector embeddings, and uses Claude AI with search tools to answer questions.

### Core Architecture Pattern

The system follows a **tool-based RAG pattern** where Claude autonomously decides when to search the knowledge base:

1. **User Query** → Frontend (HTML/JS)
2. **API Layer** → FastAPI endpoints (`app.py`)
3. **RAG Orchestrator** → Main coordination logic (`rag_system.py`)
4. **AI Generator** → Claude integration with tool calling (`ai_generator.py`)
5. **Tool Manager** → Search tool execution (`search_tools.py`)
6. **Vector Store** → ChromaDB semantic search (`vector_store.py`)

### Key Components

**Backend (`backend/` directory):**
- `app.py` - FastAPI application with CORS, serves frontend + API endpoints
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Anthropic Claude integration with tool calling support
- `search_tools.py` - Tool-based search system for AI (`CourseSearchTool`)
- `vector_store.py` - ChromaDB vector storage with semantic course name resolution
- `document_processor.py` - Processes course documents into 800-char chunks with overlap
- `session_manager.py` - Conversation history management (max 2 exchanges)
- `models.py` - Pydantic data models (`Course`, `Lesson`, `CourseChunk`)
- `config.py` - Configuration settings loaded from environment

**Frontend:**
- Static HTML/CSS/JS served from `frontend/` directory
- Single-page chat interface with course statistics sidebar

**Data:**
- `docs/` - Course documents (PDF/DOCX/TXT) auto-loaded on startup
- ChromaDB storage in `./chroma_db` directory

### Document Processing Flow

1. **File Reading** - UTF-8 with fallback error handling
2. **Metadata Extraction** - Course title, instructor, links from first 3 lines
3. **Lesson Parsing** - Detects "Lesson N:" markers and optional lesson links
4. **Text Chunking** - Sentence-based chunking (800 chars, 100 char overlap)
5. **Context Enhancement** - Adds course/lesson context to each chunk
6. **Vector Storage** - Stores in ChromaDB with metadata for filtering

Expected document format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional]
[content...]

Lesson 1: Next Topic
[content...]
```

### Query Processing Flow

1. **Frontend** sends query to `/api/query`
2. **RAG System** retrieves conversation history, calls AI Generator
3. **Claude** decides whether to use search tool or answer directly
4. **Search Tool** (if used) performs semantic search with course name resolution
5. **Vector Store** queries ChromaDB collections (course_catalog + course_content)
6. **AI Generator** synthesizes final response using search results
7. **Session Manager** updates conversation history
8. **Frontend** displays response with collapsible sources

### Configuration

Key settings in `config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation exchanges to remember
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - AI model used

### Tool-Based Search System

The system uses **Anthropic's tool calling** where Claude autonomously decides when to search:

- **CourseSearchTool** - Semantic search with course name resolution and lesson filtering
- **ToolManager** - Registers tools and handles execution
- **Course Name Resolution** - Uses vector similarity to match partial course names
- **Source Tracking** - Collects sources from searches for UI display

### Development Notes

- **Code Quality Tools** - Black (formatting), isort (imports), flake8 (linting), mypy (typing)
- **Test Framework** - pytest with asyncio and mock support
- **uv package manager** for Python dependency management
- **FastAPI with auto-reload** for development
- **Static file serving** with no-cache headers for development
- **CORS enabled** for all origins (development configuration)
- **Environment variables** required: `ANTHROPIC_API_KEY`

### Session Management

- Each chat session maintains conversation history
- History limited to last 2 exchanges (configurable)
- Session IDs generated incrementally (`session_1`, `session_2`, etc.)
- History provides context for follow-up questions

### Vector Storage Schema

**Collections:**
- `course_catalog` - Course metadata (title, instructor, lessons) for name resolution
- `course_content` - Text chunks with course/lesson metadata for content search

**Metadata Fields:**
- `course_title` - Exact course title for filtering
- `lesson_number` - Lesson number for filtering (optional)
- `chunk_index` - Sequential chunk position in document
- make sure to use uv to manage all dependencies.
- don't run the server using ./run.sh.  I will do it myself