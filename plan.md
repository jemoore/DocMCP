# Plan: Implement DocMCP Service

## TL;DR
Implement a Python MCP server (using fastmcp + ChromaDB + sentence-transformers) that provides semantic search over local PDF, Markdown, and HTML documents. The server runs in Docker, persists its index across restarts, and exposes 5 MCP tools via Streamable HTTP. Build in 5 phases: project scaffolding → document parsers → indexing engine → MCP tools/server → Docker packaging.

---

## Phase 1: Project Scaffolding & Configuration

**Goal:** Establish the project structure, dependency management, and configuration loading.

1. Create `pyproject.toml` with project metadata and all dependencies:
   - `fastmcp`, `chromadb`, `sentence-transformers`, `pymupdf`, `markdown`, `beautifulsoup4`
   - Dev dependencies: `pytest`, `pytest-asyncio`
   - Build system and entry point (`docmcp` CLI or module)

2. Create the directory structure per spec section 8:
   - `src/docmcp/__init__.py`
   - `src/docmcp/config.py`
   - `src/docmcp/models.py`
   - `src/docmcp/server.py`
   - `src/docmcp/indexer.py`
   - `src/docmcp/search.py`
   - `src/docmcp/parsers/__init__.py`
   - `src/docmcp/parsers/pdf.py`
   - `src/docmcp/parsers/markdown.py`
   - `src/docmcp/parsers/html.py`
   - `tests/__init__.py`

3. Implement `src/docmcp/config.py`:
   - Dataclass `Config` with all fields from spec section 6
   - Load from environment variables with defaults
   - Parse `DOCMCP_DOC_DIRS` as comma-separated list
   - Validate directories exist (warn if not, don't crash)

4. Implement `src/docmcp/models.py`:
   - Pydantic models (or dataclasses) for tool inputs/outputs:
     - `SearchResult`, `DocumentInfo`, `IndexStatus`, `ReindexSummary`

**Files created:**
- `pyproject.toml`
- `src/docmcp/__init__.py`
- `src/docmcp/config.py`
- `src/docmcp/models.py`
- All other empty `__init__.py` / stub files

**Verification:**
- `uv sync` installs all dependencies without errors
- `python -c "from docmcp.config import Config; print(Config())"` prints defaults

---

## Phase 2: Document Parsers

**Goal:** Implement text extraction for each supported document type.

5. Implement `src/docmcp/parsers/pdf.py`:
   - Function `extract_text(path: Path) -> list[tuple[str, dict]]` returning list of (text, metadata) per page
   - Use `pymupdf.open(path)`, iterate pages with `page.get_text()`
   - Metadata includes `page_number` for each page's text

6. Implement `src/docmcp/parsers/markdown.py`:
   - Function `extract_text(path: Path) -> list[tuple[str, dict]]`
   - Read file, use `markdown` lib to convert to plain text (strip tags), or just use raw markdown text
   - Return as single entry (no page concept)

7. Implement `src/docmcp/parsers/html.py`:
   - Function `extract_text(path: Path) -> list[tuple[str, dict]]`
   - Use `BeautifulSoup` with `html.parser`, extract text via `.get_text()`
   - Strip scripts/styles before extraction

8. Implement `src/docmcp/parsers/__init__.py`:
   - Registry mapping extensions to parser functions
   - Function `parse_file(path: Path) -> list[tuple[str, dict]]` that dispatches to the correct parser based on file extension

9. Write `tests/test_parsers.py`:
   - Unit tests for each parser with small sample files (inline or fixture)
   - Test the dispatch function with each file type

**Files created/modified:**
- `src/docmcp/parsers/pdf.py`
- `src/docmcp/parsers/markdown.py`
- `src/docmcp/parsers/html.py`
- `src/docmcp/parsers/__init__.py`
- `tests/test_parsers.py`

**Verification:**
- `uv run pytest tests/test_parsers.py` — all tests pass

---

## Phase 3: Indexing Engine

**Goal:** Implement document scanning, chunking, hashing, and ChromaDB storage.

10. Implement chunking logic in `src/docmcp/indexer.py`:
    - Function `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`
    - Recursive character splitter: split on `\n\n`, then `\n`, then `. `, then space
    - Respect chunk_size and overlap from config

11. Implement document scanning in `src/docmcp/indexer.py`:
    - Function `scan_documents(doc_dirs: list[Path]) -> list[Path]`
    - Recursively walk directories, filter by supported extensions
    - Function `compute_hash(path: Path) -> str` — SHA-256 of file content

12. Implement the `Indexer` class in `src/docmcp/indexer.py`:
    - Constructor takes `Config`, initializes `chromadb.PersistentClient` and gets/creates collection
    - Use `chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction` as the collection's embedding function (avoids managing sentence-transformers directly)
    - Store document metadata (hash, path, indexed_at) in a separate ChromaDB metadata collection or a simple JSON sidecar file in the index directory
    - `index_incremental()`: scan → hash compare → parse/chunk/add new, update changed, delete removed → return `ReindexSummary`
    - `index_full()`: clear collection → reprocess everything → return `ReindexSummary`
    - `get_status() -> IndexStatus`: query collection count, compute index dir size, return stats
    - `get_document(name: str)`: retrieve all chunks for a document, concatenate text
    - `list_documents(filter: str | None)`: query distinct document metadata

13. Write `tests/test_indexer.py`:
    - Test `chunk_text` with various inputs (short text, long text, overlap behavior)
    - Test `scan_documents` with a temp directory structure
    - Test `Indexer` methods with mocked ChromaDB client

**Files created/modified:**
- `src/docmcp/indexer.py`
- `tests/test_indexer.py`

**Verification:**
- `uv run pytest tests/test_indexer.py` — all tests pass

---

## Phase 4: Search & MCP Server

**Goal:** Implement search logic and wire up all 5 MCP tools.

14. Implement `src/docmcp/search.py`:
    - Function `search(collection, query: str, limit: int) -> list[SearchResult]`
    - Use `collection.query(query_texts=[query], n_results=limit)`
    - Map ChromaDB results to `SearchResult` models
    - Normalize distances to 0–1 scores (ChromaDB returns distances; convert with `1 / (1 + distance)` or similar)

15. Implement `src/docmcp/server.py`: *depends on steps 12, 14*
    - Create `FastMCP("DocMCP")` instance
    - On module load: instantiate `Config`, create `Indexer`
    - Use a lifespan or startup hook to:
      - Load existing ChromaDB index (fast)
      - Launch background thread for incremental indexing (`threading.Thread(target=indexer.index_incremental, daemon=True).start()`)
    - Register 5 tools with `@mcp.tool`:
      - `search_documents(query: str, limit: int = 10)` → calls `search()`
      - `get_document(document_name: str)` → calls `indexer.get_document()`
      - `list_documents(filter: str = "")` → calls `indexer.list_documents()`
      - `reindex(full: bool = False)` → calls `indexer.index_full()` or `indexer.index_incremental()`
      - `index_status()` → calls `indexer.get_status()`
    - Main block: `mcp.run(transport="http", host=config.host, port=config.port)`

16. Write `tests/test_search.py`:
    - Test score normalization
    - Test result mapping from ChromaDB response format to `SearchResult`

**Files created/modified:**
- `src/docmcp/search.py`
- `src/docmcp/server.py`
- `tests/test_search.py`

**Verification:**
- `uv run pytest` — all tests pass
- Manual test: `uv run python -m docmcp.server` starts and listens on port 8808
- Manual test: use `fastmcp` client or `curl` to call tools against a small test corpus

---

## Phase 5: Docker Packaging

**Goal:** Containerize the service with proper volume mounts.

17. Create `Dockerfile`:
    - `FROM python:3.12-slim`
    - Install `uv` via pip or copy binary
    - Copy `pyproject.toml` and `uv.lock`, run `uv sync --no-dev`
    - Copy `src/`
    - Create `/data/docs` and `/data/index` directories
    - `EXPOSE 8808`
    - `CMD ["uv", "run", "python", "-m", "docmcp.server"]`

18. Create `docker-compose.yml`:
    - Per spec section 7 example
    - Service `docmcp` with build context, port mapping, volume mounts, environment

19. Update `README.md`:
    - Quick start with Docker Compose
    - Configuration reference
    - MCP client connection example

**Files created/modified:**
- `Dockerfile`
- `docker-compose.yml`
- `README.md`

**Verification:**
- `docker compose build` completes without errors
- `docker compose up` starts the server, logs show index loading
- From another machine on the network, connect an MCP client to `http://<host>:8808` and run `search_documents`

---

## Key Technical Decisions

- **ChromaDB's built-in `SentenceTransformerEmbeddingFunction`** handles embedding generation, so we don't need to manage `sentence-transformers` directly for indexing/querying. ChromaDB auto-embeds documents on `add()` and queries on `query()`.
- **Background thread** (not asyncio) for startup re-indexing, since ChromaDB operations are synchronous and blocking.
- **fastmcp transport is `"http"`** (not `"streamable-http"`) — the library maps this to the Streamable HTTP MCP protocol.
- **Document metadata tracking** via ChromaDB collection metadata on each chunk (source_path, content_hash, indexed_at) — avoids a separate metadata store.
- **`uv`** for dependency management and running the application.
- **Unit tests only** with pytest, mocking ChromaDB and file I/O where needed.
