from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResultMetadata:
    source_path: str
    page_number: int | None = None
    chunk_index: int = 0


@dataclass
class SearchResult:
    document_name: str
    chunk_text: str
    score: float
    metadata: SearchResultMetadata


@dataclass
class ChunkContext:
    """A window of consecutive chunks around a search hit, merged into one
    continuous passage (chunk overlap removed)."""

    document_name: str
    chunk_index: int
    first_chunk_index: int
    last_chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str


@dataclass
class DocumentInfo:
    document_name: str
    type: str
    size_bytes: int
    last_modified: str
    chunk_count: int
    indexed_at: str


@dataclass
class DocumentContent:
    document_name: str
    content: str
    size_bytes: int
    page_count: int | None
    last_modified: str
    total_chars: int = 0
    offset: int = 0
    truncated: bool = False


@dataclass
class IndexProgress:
    """Live state of an indexing run, exposed via index_status."""

    in_progress: bool = False
    phase: str | None = None  # "incremental" or "full"
    files_total: int = 0
    files_processed: int = 0
    current_file: str | None = None
    started_at: str | None = None


@dataclass
class IndexStatus:
    total_documents: int
    total_chunks: int
    index_size_bytes: int
    last_indexed_at: str | None
    embedding_model: str
    source_directories: list[str]
    indexing: IndexProgress = field(default_factory=IndexProgress)
    last_run: "ReindexSummary | None" = None


@dataclass
class ReindexError:
    filename: str
    reason: str


@dataclass
class ReindexSummary:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    empty: int = 0
    errors: list[ReindexError] = field(default_factory=list)
