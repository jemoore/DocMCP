from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import threading
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
from docmcp.search import search as _search

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
META_FILENAME = "doc_meta.json"
FULLTEXT_DIRNAME = "fulltext"


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters with the
    requested overlap, using a recursive character splitter."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = _recursive_split(text, SEPARATORS, chunk_size, overlap)
    return [c for c in chunks if c.strip()]


def _recursive_split(
    text: str, separators: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Split `text` on the first applicable separator, recursing with finer
    separators into any piece that is still larger than `chunk_size`, then
    merge adjacent pieces back up to the target size."""
    if len(text) <= chunk_size:
        return [text]

    # Choose the first separator that occurs in the text (the empty string
    # separator, which splits between characters, is always applicable).
    separator = ""
    remaining = separators[separators.index("") + 1 :]
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            remaining = separators[i + 1 :]
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    pieces = list(text) if separator == "" else text.split(separator)

    final_chunks: list[str] = []
    good_splits: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            good_splits.append(piece)
            continue
        # Flush the small pieces gathered so far, then recurse into the
        # oversized piece with the remaining (finer) separators.
        if good_splits:
            final_chunks.extend(
                _merge_splits(good_splits, separator, chunk_size, overlap)
            )
            good_splits = []
        if remaining:
            final_chunks.extend(
                _recursive_split(piece, remaining, chunk_size, overlap)
            )
        else:
            final_chunks.append(piece)

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, overlap))
    return final_chunks


def _merge_splits(
    splits: list[str], separator: str, chunk_size: int, overlap: int
) -> list[str]:
    """Greedily merge small pieces into chunks up to `chunk_size`, carrying up
    to `overlap` characters of trailing context into the start of each new
    chunk. Mirrors the merge step of a LangChain-style recursive splitter."""
    sep_len = len(separator)
    chunks: list[str] = []
    current: list[str] = []
    total = 0

    for piece in splits:
        piece_len = len(piece)
        if total + piece_len + (sep_len if current else 0) > chunk_size and current:
            chunks.append(separator.join(current))
            # Trim from the front until the carried-over context fits within
            # the overlap budget (and the incoming piece will fit).
            while current and (
                total > overlap
                or (
                    total + piece_len + (sep_len if current else 0) > chunk_size
                    and total > 0
                )
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current = current[1:]
        current.append(piece)
        total += piece_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(separator.join(current))
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

        # Serializes all index mutations (and the collection swap in
        # index_full) against reads so concurrent tool calls and the startup
        # background thread cannot corrupt shared state.
        self._lock = threading.RLock()

        self._client = chromadb.PersistentClient(path=str(config.index_dir))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        self._meta_path = config.index_dir / META_FILENAME
        self._fulltext_dir = config.index_dir / FULLTEXT_DIRNAME
        self._fulltext_dir.mkdir(parents=True, exist_ok=True)
        self._doc_meta = self._load_meta()
        self._last_indexed_at: str | None = self._doc_meta.get("_last_indexed_at")

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_meta(self) -> None:
        # Write atomically so a crash mid-write cannot corrupt the metadata.
        tmp = self._meta_path.with_suffix(self._meta_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._doc_meta, indent=2))
        tmp.replace(self._meta_path)

    def _fulltext_path(self, path_str: str) -> Path:
        """Return the on-disk blob path holding a document's full extracted text."""
        key = hashlib.sha256(path_str.encode("utf-8")).hexdigest()
        return self._fulltext_dir / f"{key}.txt"

    def _relative_name(self, path: Path) -> str:
        """Get a relative document name from the absolute path.

        When more than one source directory is configured, the name is
        namespaced with the directory's basename so that identically-named
        files in different roots do not collide into one document.
        """
        multi = len(self.config.doc_dirs) > 1
        for doc_dir in self.config.doc_dirs:
            try:
                rel = str(path.relative_to(doc_dir))
            except ValueError:
                continue
            return f"{doc_dir.name}/{rel}" if multi else rel
        return path.name

    def index_incremental(self) -> ReindexSummary:
        """Index only new or changed files; remove deleted ones."""
        with self._lock:
            return self._index_incremental_locked()

    def _index_incremental_locked(self) -> ReindexSummary:
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
        with self._lock:
            return self._index_full_locked()

    def _index_full_locked(self) -> ReindexSummary:
        logger.info("Starting full reindex...")

        # Clear everything
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
        )
        self._doc_meta = {}
        # Drop stale full-text blobs from documents that may no longer exist.
        for blob in self._fulltext_dir.glob("*.txt"):
            blob.unlink(missing_ok=True)

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

        # Persist the full extracted text verbatim so get_document can return
        # it without reconstructing from overlapping chunks.
        full_text = "\n\n".join(text for text, _ in segments)
        fulltext_path = self._fulltext_path(path_str)
        fulltext_path.write_text(full_text, encoding="utf-8")

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
            "full_text_path": str(fulltext_path),
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
        self._fulltext_path(path_str).unlink(missing_ok=True)
        self._doc_meta.pop(path_str, None)

    def search(self, query: str, limit: int = 10):
        """Run a semantic search, holding the lock so the collection cannot be
        swapped out (by a full reindex) mid-query."""
        with self._lock:
            return _search(self._collection, query, limit)

    def get_document(self, document_name: str) -> DocumentContent:
        """Retrieve full text of a document by name."""
        with self._lock:
            results = self._collection.get(
                where={"document_name": document_name},
            )

            if not results["ids"]:
                raise ValueError(f"Document not found: {document_name}")

            # Guard against distinct files that resolved to the same name: if
            # the matched chunks span multiple source paths, refuse to merge
            # them and report the conflict instead of returning garbled text.
            source_paths = {m.get("source_path") for m in results["metadatas"]}
            if len(source_paths) > 1:
                conflicting = ", ".join(sorted(str(p) for p in source_paths))
                raise ValueError(
                    f"Ambiguous document name '{document_name}' matches multiple "
                    f"files: {conflicting}"
                )

            # Find the file metadata
            file_meta = None
            for key, val in self._doc_meta.items():
                if key.startswith("_"):
                    continue
                if val.get("document_name") == document_name:
                    file_meta = val
                    break

            # Return the verbatim extracted text from its blob. Fall back to a
            # chunk reconstruction only if the blob is missing (older index).
            content = None
            if file_meta and file_meta.get("full_text_path"):
                blob = Path(file_meta["full_text_path"])
                if blob.exists():
                    content = blob.read_text(encoding="utf-8")

            pairs = list(zip(results["documents"], results["metadatas"]))
            pairs.sort(key=lambda p: p[1].get("chunk_index", 0))
            if content is None:
                content = "\n".join(doc for doc, _ in pairs)

            page_count = None
            if file_meta and file_meta.get("type") == "pdf":
                valid = [
                    p[1].get("page_number", -1)
                    for p in pairs
                    if p[1].get("page_number", -1) > 0
                ]
                if valid:
                    page_count = max(valid)

            return DocumentContent(
                document_name=document_name,
                content=content,
                size_bytes=file_meta["size_bytes"] if file_meta else 0,
                page_count=page_count,
                last_modified=file_meta["last_modified"] if file_meta else "",
            )

    def list_documents(self, filter_pattern: str | None = None) -> list[DocumentInfo]:
        """List all indexed documents, optionally filtered by glob pattern."""
        with self._lock:
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
        with self._lock:
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
