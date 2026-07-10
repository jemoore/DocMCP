# DocMCP — Issues & Improvements

Findings from code review (2026-07-08), oriented around the primary use case:
semantic Q&A over a library of hundreds of PDF books, with answers grounded in
the books and synthesized by the connected LLM client.

## Bugs / blocking issues

- [x] **1. Server is unusable while indexing.** `index_incremental()` /
  `index_full()` hold `self._lock` for the entire run, and `search`,
  `get_document`, `list_documents`, and `index_status` all take the same lock.
  First-time indexing of hundreds of books takes hours, during which every tool
  call (including `index_status`) blocks and times out. Lock per-file instead of
  per-run; for full rebuilds, build into a temp collection and swap so the live
  index stays queryable.
- [x] **2. Files that extract no text are silently re-processed forever.** When
  `parse_file` returns no segments (scanned/image-only PDFs), `_index_file`
  returns early without writing a `_doc_meta` entry, so every incremental pass
  re-hashes and re-parses the whole file — and counts it as "added" each time.
  Record empty files in meta (`chunk_count: 0`) and surface them so books that
  need OCR are visible.
- [x] **3. Large books can fail to index outright.** `_index_file` pushes all
  chunks in a single `collection.add()`; ChromaDB's SQLite backend has a max
  batch size (~5,461 records) and a long book can exceed it, failing the whole
  file. Batch the adds (e.g. 1,000 at a time).
- [x] **4. `get_document` returns the entire book text.** A full book is
  0.5–2 MB of text returned as one tool result, which blows up any LLM context.
  Add page-range and offset/max-chars parameters plus a truncation flag.
- [x] **5. `reindex` runs synchronously in the tool call.** A full rebuild of a
  large corpus takes hours; the MCP client times out long before it returns.
  Run reindexing in a background thread, return immediately, and expose
  progress (in-progress flag, N of M files, last-run summary) via
  `index_status`.
- [x] **6. Relevance scores are misleading with the default distance metric.**
  The collection is created without a configured space, so Chroma defaults to
  L2, but `normalize_score` treats distance like a bounded value. Create the
  collection with `metadata={"hnsw:space": "cosine"}` (requires a one-time
  rebuild).
- [x] **7. Every incremental pass re-hashes every file.** `compute_hash` reads
  every byte of every book on each pass. Check `(size, mtime)` first and only
  hash when those changed.

## Improvements for the query-books workflow

- [x] **Document filter on `search_documents`** — e.g. a glob on
  `document_name` passed as a Chroma `where` clause, so searches can be scoped
  to one book/series out of hundreds.
- [x] **Return neighboring context** — 1,000-char chunks are thin evidence for
  synthesis. Added a `get_context` tool that merges the chunks around a search
  hit into one passage with the chunk overlap removed.
- [x] **Embedding model token limit** — `all-MiniLM-L6-v2` truncates at 256
  tokens (≈1,000 chars); chunks are right at the edge. The server now warns at
  startup when the configured chunk size exceeds the model's window, and the
  README documents model selection (e.g. `BAAI/bge-small-en-v1.5` for longer
  windows).
- [ ] **Hybrid retrieval (BM25 + vector)** — pure semantic search is weak on
  exact names, code identifiers, and rare terms common in technical books.
  Larger feature; split out from the token-limit item above.
- [x] **Book page noise & cross-page chunking** — repeating headers/footers/
  page numbers (normalized for digits) are stripped when they recur on ≥60%
  of pages; page texts are concatenated before chunking (with per-chunk page
  attribution by start position) so paragraphs spanning page breaks stay
  together.
- [x] **Persist the HuggingFace model cache in Docker** — `HF_HOME` now points
  at `/data/hf-cache`, backed by the `docmcp-hf-cache` volume.
- [x] **Add `.txt` parser** (trivial) and **`.epub`** (pymupdf reads EPUB
  natively) for the occasional non-PDF docs.
- [x] **Startup indexing error visibility** — the lifespan indexing thread only
  logs; if it dies, `index_status` gives no hint. Track last-run
  outcome/errors in status. (Addressed together with items 1/5: `index_status`
  now reports live progress and the last run's summary including errors.)

## Minor

- [ ] `_remove_document`'s bare `except Exception: pass` can hide real Chroma
  failures, leaving stale chunks while meta says the doc is gone.
- [ ] `list_documents` parameter `filter` shadows the Python builtin.
- [ ] `__main__.py` docstring says it enables `python -m docmcp.server`, but it
  actually enables `python -m docmcp`.
