from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from docmcp.config import Config
from docmcp.models import (
    DocumentContent,
    DocumentInfo,
    IndexStatus,
    ReindexError,
    ReindexSummary,
)
from docmcp.parsers import SUPPORTED_EXTENSIONS, parse_file

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
META_FILENAME = "doc_meta.json"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks using a recursive character splitter."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Recursively split text trying each separator in order."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current)
                    # Start new chunk with overlap from previous
                    if overlap > 0 and current:
                        overlap_text = current[-overlap:]
                        current = overlap_text + sep + part
                    else:
                        current = part

            if current.strip():
                chunks.append(current)

            if len(chunks) > 1:
                return chunks

    # Fallback: hard split by chunk_size
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
    return chunks


def scan_documents(doc_dirs: list[Path]) -> list[Path]:
    """Recursively scan directories for supported document files."""
    files: list[Path] = []
    for doc_dir in doc_dirs:
        if not doc_dir.exists():
            logger.warning("Directory does not exist, skipping: %s", doc_dir)
            continue
        for root, _, filenames in os.walk(doc_dir):
            for filename in filenames:
                path = Path(root) / filename
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(path)
    return sorted(files)


def compute_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Indexer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.index_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(config.index_dir))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        self._meta_path = config.index_dir / META_FILENAME
        self._doc_meta = self._load_meta()
        self._last_indexed_at: str | None = self._doc_meta.get("_last_indexed_at")

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_meta(self) -> None:
        self._meta_path.write_text(json.dumps(self._doc_meta, indent=2))

    def _relative_name(self, path: Path) -> str:
        """Get a relative document name from the absolute path."""
        for doc_dir in self.config.doc_dirs:
            try:
                return str(path.relative_to(doc_dir))
            except ValueError:
                continue
        return path.name

    def index_incremental(self) -> ReindexSummary:
        """Index only new or changed files; remove deleted ones."""
        logger.info("Starting incremental indexing...")
        summary = ReindexSummary()
        current_files = scan_documents(self.config.doc_dirs)
        current_paths = {str(p) for p in current_files}

        # Remove entries for deleted files
        for stored_path in list(self._doc_meta.keys()):
            if stored_path.startswith("_"):
                continue
            if stored_path not in current_paths:
                self._remove_document(stored_path)
                summary.removed += 1

        # Process current files
        for path in current_files:
            path_str = str(path)
            file_hash = compute_hash(path)
            stored = self._doc_meta.get(path_str)

            if stored and stored.get("hash") == file_hash:
                summary.unchanged += 1
                continue

            try:
                is_update = stored is not None
                self._index_file(path, file_hash)
                if is_update:
                    summary.updated += 1
                else:
                    summary.added += 1
            except Exception as e:
                logger.error("Failed to index %s: %s", path, e)
                summary.errors.append(
                    ReindexError(filename=str(path), reason=str(e))
                )

        now = datetime.now(timezone.utc).isoformat()
        self._last_indexed_at = now
        self._doc_meta["_last_indexed_at"] = now
        self._save_meta()

        logger.info(
            "Incremental indexing complete: added=%d, updated=%d, removed=%d, unchanged=%d, errors=%d",
            summary.added,
            summary.updated,
            summary.removed,
            summary.unchanged,
            len(summary.errors),
        )
        return summary

    def index_full(self) -> ReindexSummary:
        """Rebuild the entire index from scratch."""
        logger.info("Starting full reindex...")

        # Clear everything
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        self._doc_meta = {}

        summary = ReindexSummary()
        current_files = scan_documents(self.config.doc_dirs)

        for path in current_files:
            try:
                file_hash = compute_hash(path)
                self._index_file(path, file_hash)
                summary.added += 1
            except Exception as e:
                logger.error("Failed to index %s: %s", path, e)
                summary.errors.append(
                    ReindexError(filename=str(path), reason=str(e))
                )

        now = datetime.now(timezone.utc).isoformat()
        self._last_indexed_at = now
        self._doc_meta["_last_indexed_at"] = now
        self._save_meta()

        logger.info(
            "Full reindex complete: added=%d, errors=%d",
            summary.added,
            len(summary.errors),
        )
        return summary

    def _index_file(self, path: Path, file_hash: str) -> None:
        """Parse, chunk, embed, and store a single file."""
        path_str = str(path)
        doc_name = self._relative_name(path)

        # Remove old chunks if updating
        self._remove_document(path_str)

        # Parse
        segments = parse_file(path)
        if not segments:
            return

        # Combine all text segments with their metadata
        all_chunks: list[tuple[str, dict]] = []
        for text, meta in segments:
            chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            for chunk in chunks:
                all_chunks.append((chunk, meta))

        if not all_chunks:
            return

        # Build ChromaDB entries
        now = datetime.now(timezone.utc).isoformat()
        ids = [f"{path_str}::chunk_{i}" for i in range(len(all_chunks))]
        documents = [c[0] for c in all_chunks]
        metadatas = [
            {
                "source_path": path_str,
                "document_name": doc_name,
                "chunk_index": i,
                "page_number": c[1].get("page_number", -1),
                "content_hash": file_hash,
                "indexed_at": now,
            }
            for i, c in enumerate(all_chunks)
        ]

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

        # Update file metadata
        stat = path.stat()
        self._doc_meta[path_str] = {
            "hash": file_hash,
            "document_name": doc_name,
            "type": path.suffix.lstrip(".").lower(),
            "size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "chunk_count": len(all_chunks),
            "indexed_at": now,
        }

    def _remove_document(self, path_str: str) -> None:
        """Remove all chunks for a document from the collection."""
        try:
            results = self._collection.get(
                where={"source_path": path_str},
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception:
            pass  # Collection may be empty or doc not found
        self._doc_meta.pop(path_str, None)

    def get_document(self, document_name: str) -> DocumentContent:
        """Retrieve full text of a document by name."""
        results = self._collection.get(
            where={"document_name": document_name},
        )

        if not results["ids"]:
            raise ValueError(f"Document not found: {document_name}")

        # Sort by chunk_index to reconstruct in order
        pairs = list(zip(results["documents"], results["metadatas"]))
        pairs.sort(key=lambda p: p[1].get("chunk_index", 0))

        full_text = "\n".join(doc for doc, _ in pairs)
        meta = pairs[0][1]

        # Find the file metadata
        file_meta = None
        for key, val in self._doc_meta.items():
            if key.startswith("_"):
                continue
            if val.get("document_name") == document_name:
                file_meta = val
                break

        page_count = None
        if file_meta and file_meta.get("type") == "pdf":
            page_numbers = [p[1].get("page_number", -1) for p in pairs]
            valid = [pn for pn in page_numbers if pn > 0]
            if valid:
                page_count = max(valid)

        return DocumentContent(
            document_name=document_name,
            content=full_text,
            size_bytes=file_meta["size_bytes"] if file_meta else 0,
            page_count=page_count,
            last_modified=file_meta["last_modified"] if file_meta else "",
        )

    def list_documents(self, filter_pattern: str | None = None) -> list[DocumentInfo]:
        """List all indexed documents, optionally filtered by glob pattern."""
        docs: list[DocumentInfo] = []
        for key, val in self._doc_meta.items():
            if key.startswith("_"):
                continue
            doc_name = val["document_name"]
            if filter_pattern and not fnmatch.fnmatch(doc_name, filter_pattern):
                continue
            docs.append(
                DocumentInfo(
                    document_name=doc_name,
                    type=val["type"],
                    size_bytes=val["size_bytes"],
                    last_modified=val["last_modified"],
                    chunk_count=val["chunk_count"],
                    indexed_at=val["indexed_at"],
                )
            )
        return sorted(docs, key=lambda d: d.document_name)

    def get_status(self) -> IndexStatus:
        """Return current index statistics."""
        total_docs = sum(1 for k in self._doc_meta if not k.startswith("_"))
        total_chunks = self._collection.count()

        # Compute index directory size
        index_size = 0
        if self.config.index_dir.exists():
            for root, _, files in os.walk(self.config.index_dir):
                for f in files:
                    index_size += (Path(root) / f).stat().st_size

        return IndexStatus(
            total_documents=total_docs,
            total_chunks=total_chunks,
            index_size_bytes=index_size,
            last_indexed_at=self._last_indexed_at,
            embedding_model=self.config.embedding_model,
            source_directories=[str(d) for d in self.config.doc_dirs],
        )
