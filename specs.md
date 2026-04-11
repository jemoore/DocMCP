# DocMCP — MCP Server for Document Search

An MCP server that provides semantic search capabilities over a local corpus of documents. Clients on the local network connect via MCP and use tools to search, retrieve, and manage indexed documents.

---

## 1. Technology Stack

| Component          | Choice                                              |
| ------------------ | --------------------------------------------------- |
| Language           | Python 3.12+                                        |
| MCP framework      | `fastmcp`
| MCP transport      | Streamable HTTP                                     |
| Embedding model    | `sentence-transformers` (local, no external API)    |
| Vector store       | ChromaDB (persistent, local storage)                |
| Document parsing   | `pymupdf` (PDF), `markdown` (Markdown), `beautifulsoup4` (HTML) |
| Packaging          | Docker container                                    |

---

## 2. Supported Document Types

| Format   | Extensions        |
| -------- | ----------------- |
| PDF      | `.pdf`            |
| Markdown | `.md`, `.markdown`|
| HTML     | `.html`, `.htm`   |

---

## 3. Architecture Overview

```
┌──────────────┐       Streamable HTTP        ┌───────────────────────────────┐
│  MCP Client  │  ◄──────────────────────►    │  DocMCP Server (container)    │
│  (LLM agent) │                              │                               │
└──────────────┘                              │  ┌─────────────────────────┐  │
                                              │  │  MCP Tool Layer         │  │
                                              │  └────────┬────────────────┘  │
                                              │           │                   │
                                              │  ┌────────▼────────────────┐  │
                                              │  │  Search / Index Service │  │
                                              │  └────────┬────────────────┘  │
                                              │           │                   │
                                              │  ┌────────▼────────────────┐  │
                                              │  │  ChromaDB (persistent)  │  │
                                              │  └────────────────────────-┘  │
                                              │                               │
                                              │  /data/docs ← volume mount   │
                                              │  /data/index ← volume mount  │
                                              └───────────────────────────────┘
```

---

## 4. MCP Tools

### 4.1 `search_documents`

Perform semantic search across all indexed documents.

| Parameter    | Type   | Required | Description                                      |
| ------------ | ------ | -------- | ------------------------------------------------ |
| `query`      | string | yes      | Natural-language search query                    |
| `limit`      | int    | no       | Max results to return (default: 10, max: 50)     |

**Returns:** A list of result objects, each containing:
- `document_name` — filename or relative path of the source document
- `chunk_text` — the matching text chunk
- `score` — relevance score (0–1, higher is better)
- `metadata` — object with `source_path`, `page_number` (if applicable), `chunk_index`

### 4.2 `get_document`

Retrieve the full text content of a specific indexed document.

| Parameter       | Type   | Required | Description                              |
| --------------- | ------ | -------- | ---------------------------------------- |
| `document_name` | string | yes      | Filename or relative path of the document|

**Returns:** The full extracted text content of the document plus metadata (file size, page count if PDF, last-modified timestamp).

### 4.3 `list_documents`

List all documents currently in the index.

| Parameter | Type   | Required | Description                                |
| --------- | ------ | -------- | ------------------------------------------ |
| `filter`  | string | no       | Glob pattern to filter results (e.g. `*.pdf`) |

**Returns:** A list of document objects, each containing:
- `document_name` — relative path
- `type` — file type (pdf, markdown, html)
- `size_bytes` — file size
- `last_modified` — ISO 8601 timestamp
- `chunk_count` — number of chunks in the index
- `indexed_at` — ISO 8601 timestamp of when it was last indexed

### 4.4 `reindex`

Trigger re-indexing of the document corpus.

| Parameter | Type   | Required | Description                                               |
| --------- | ------ | -------- | --------------------------------------------------------- |
| `full`    | bool   | no       | If `true`, rebuild entire index. Default `false` (incremental — only new/changed files). |

**Returns:** Summary object with counts: `added`, `updated`, `removed`, `unchanged`, `errors` (list of filenames that failed with reason).

### 4.5 `index_status`

Report the current state of the index.

**Parameters:** None.

**Returns:**
- `total_documents` — number of indexed documents
- `total_chunks` — total chunks across all documents
- `index_size_bytes` — size of the persistent index on disk
- `last_indexed_at` — ISO 8601 timestamp of last indexing run
- `embedding_model` — name of the sentence-transformer model in use
- `source_directories` — list of configured document directories

---

## 5. Document Processing Pipeline

### 5.1 Ingestion

1. Recursively scan all configured source directories for files with supported extensions.
2. For each file, compute a content hash (SHA-256) and compare against the stored hash in the index metadata.
3. Skip files whose hash has not changed (incremental indexing).
4. Extract text content using the appropriate parser (pymupdf for PDF, markdown lib for Markdown, BeautifulSoup for HTML).

### 5.2 Chunking

- Split extracted text into chunks using a recursive character splitter.
- Target chunk size: **1000 characters** with **200-character overlap**.
- Preserve logical boundaries where possible (paragraphs, headings, page breaks).
- Store metadata with each chunk: source file path, chunk index, page number (PDFs).

### 5.3 Embedding & Storage

- Generate embeddings for each chunk using the configured `sentence-transformers` model (default: `all-MiniLM-L6-v2`).
- Store embeddings + chunk text + metadata in ChromaDB.
- ChromaDB data is persisted to `/data/index` inside the container (volume-mounted to the host).

### 5.4 Startup Behavior

- On startup, load the existing ChromaDB index from the persistent volume.
- Run an incremental index pass: add new files, update changed files, remove entries for deleted files.
- The server should begin accepting MCP connections as soon as the existing index is loaded, even if re-indexing of new/changed files is still in progress.

---

## 6. Configuration

Configuration via environment variables (with sensible defaults):

| Variable                | Default                  | Description                                        |
| ----------------------- | ------------------------ | -------------------------------------------------- |
| `DOCMCP_HOST`           | `0.0.0.0`               | Host to bind the server to                         |
| `DOCMCP_PORT`           | `8808`                   | Port to listen on                                  |
| `DOCMCP_DOC_DIRS`       | `/data/docs`             | Comma-separated list of document directories       |
| `DOCMCP_INDEX_DIR`      | `/data/index`            | Directory for persistent ChromaDB storage          |
| `DOCMCP_EMBEDDING_MODEL`| `all-MiniLM-L6-v2`      | Sentence-transformer model name                    |
| `DOCMCP_CHUNK_SIZE`     | `1000`                   | Target chunk size in characters                    |
| `DOCMCP_CHUNK_OVERLAP`  | `200`                    | Chunk overlap in characters                        |
| `DOCMCP_LOG_LEVEL`      | `INFO`                   | Logging level (DEBUG, INFO, WARNING, ERROR)        |

---

## 7. Deployment

### Docker

Provide a `Dockerfile` and `docker-compose.yml`.

**Dockerfile:**
- Base image: `python:3.12-slim`
- Install dependencies via `requirements.txt` (or `pyproject.toml`)
- Expose the configured port
- Entrypoint runs the MCP server

**docker-compose.yml:**
- Volume-mount host document directories into `/data/docs` (read-only)
- Volume-mount a host directory into `/data/index` for persistent index storage
- Expose the port to the local network
- Environment variable overrides

Example usage:
```yaml
services:
  docmcp:
    build: .
    ports:
      - "8808:8808"
    volumes:
      - /home/user/documents:/data/docs/documents:ro
      - /home/user/notes:/data/docs/notes:ro
      - docmcp-index:/data/index
    environment:
      DOCMCP_LOG_LEVEL: INFO

volumes:
  docmcp-index:
```

---

## 8. Project Structure

```
DocMCP/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── specs.md
├── src/
│   └── docmcp/
│       ├── __init__.py
│       ├── server.py          # MCP server setup, tool registration
│       ├── config.py          # Configuration loading from env vars
│       ├── indexer.py         # Document scanning, hashing, chunking, embedding
│       ├── search.py          # Search logic (query embedding, ChromaDB query)
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── pdf.py         # PDF text extraction
│       │   ├── markdown.py    # Markdown text extraction
│       │   └── html.py        # HTML text extraction
│       └── models.py          # Data models / schemas for tool inputs/outputs
└── tests/
    ├── __init__.py
    ├── test_indexer.py
    ├── test_search.py
    └── test_parsers.py
```

---

## 9. Acceptance Criteria

1. **Network accessible:** The MCP server listens on `0.0.0.0` and is reachable from any machine on the local network.
2. **Semantic search works:** `search_documents` returns relevant chunks for natural-language queries against a mixed corpus of PDF, Markdown, and HTML files.
3. **Full document retrieval:** `get_document` returns the complete extracted text of any indexed document.
4. **Document listing:** `list_documents` returns all indexed documents with accurate metadata.
5. **Incremental indexing:** On startup or `reindex(full=false)`, only new or modified files are processed. Deleted files are removed from the index.
6. **Full reindex:** `reindex(full=true)` rebuilds the entire index from scratch.
7. **Index persistence:** The ChromaDB index survives container restarts via volume mount. Startup with a warm index is fast (seconds, not minutes).
8. **Index status:** `index_status` returns accurate statistics about the current index.
9. **Docker deployment:** The server runs in a Docker container with document directories mounted as read-only volumes.
10. **Startup availability:** The server accepts MCP connections as soon as the existing index is loaded, before any incremental re-indexing completes.
