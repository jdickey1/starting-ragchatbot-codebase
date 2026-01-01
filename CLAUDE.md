# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using semantic search and Claude AI. Built with FastAPI backend, vanilla JS frontend, ChromaDB for vector storage.

## Commands

This project uses `uv` for dependency management and running Python files.

```bash
# Install dependencies
uv sync

# Run the application (starts uvicorn on port 8000)
./run.sh

# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any Python file
uv run python <file.py>
```

Access points when running:
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

## Architecture

```
Frontend (HTML/JS)  →  FastAPI API  →  RAGSystem  →  Claude API + ChromaDB
```

### Key Design Pattern: Tool-Based RAG

Unlike traditional RAG that always retrieves before generating, this system gives Claude a `search_course_content` tool. Claude decides when to search based on query type:
- General questions → direct answer (no search)
- Course-specific questions → uses tool, then synthesizes answer

### Backend Components (`backend/`)

| Component | Purpose |
|-----------|---------|
| `app.py` | FastAPI server, `/api/query` and `/api/courses` endpoints |
| `rag_system.py` | Orchestrates all components, main `query()` method |
| `ai_generator.py` | Claude API calls with tool-use handling |
| `search_tools.py` | Tool definitions (`CourseSearchTool`) and `ToolManager` |
| `vector_store.py` | ChromaDB wrapper, dual collections (catalog + content) |
| `document_processor.py` | Parses course docs, chunks text (800 chars, 100 overlap) |
| `session_manager.py` | Conversation history per session (max 10 messages) |
| `config.py` | Settings loaded from environment |

### Vector Store Collections

- `course_catalog` - Course metadata for fuzzy name matching
- `course_content` - Text chunks with course/lesson filters

### Course Document Format

Documents in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
[content...]
```

## Configuration

Environment variables (`.env`):
- `ANTHROPIC_API_KEY` - Required

Defaults in `config.py`:
- Model: `claude-sonnet-4-20250514`
- Embeddings: `all-MiniLM-L6-v2`
- Chunk size: 800 chars, 100 char overlap
- Max search results: 5
