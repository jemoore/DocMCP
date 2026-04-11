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


@dataclass
class IndexStatus:
    total_documents: int
    total_chunks: int
    index_size_bytes: int
    last_indexed_at: str | None
    embedding_model: str
    source_directories: list[str]


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
    errors: list[ReindexError] = field(default_factory=list)
