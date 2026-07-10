from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from docmcp.config import Config
from docmcp.models import (
    ChunkContext,
    DocumentContent,
    DocumentInfo,
    IndexProgress,
    IndexStatus,
    ReindexError,
    ReindexSummary,
)
from docmcp.parsers import SUPPORTED_EXTENSIONS, parse_file
from docmcp.search import search as _search

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
REBUILD_COLLECTION_NAME = "documents_rebuild"
META_FILENAME = "doc_meta.json"
FULLTEXT_DIRNAME = "fulltext"

# ChromaDB's SQLite backend rejects add() calls above its max batch size
# (~5,461 records), which a single long book can exceed. Adding in batches
# also bounds the size of each embedding call.
ADD_BATCH_SIZE = 1000

# Cosine distance is bounded (0..2), which normalize_score relies on to
# produce meaningful 0-1 relevance scores; Chroma's default is unbounded L2.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


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


def chunk_segments(
    segments: list[tuple[str, dict]], chunk_size: int, overlap: int
) -> list[tuple[str, int | None]]:
    """Chunk the concatenation of all segment texts, so chunks (and their
    overlap) can span segment/page boundaries instead of splitting paragraphs
    at every page break. Each chunk is assigned the page number of the segment
    containing its first character."""
    boundaries: list[tuple[int, int | None]] = []
    parts: list[str] = []
    offset = 0
    for text, meta in segments:
        boundaries.append((offset, meta.get("page_number")))
        parts.append(text)
        offset += len(text) + 2  # account for the "\n\n" joiner
    combined = "\n\n".join(parts)

    result: list[tuple[str, int | None]] = []
    cursor = 0
    for chunk in chunk_text(combined, chunk_size, overlap):
        # Chunks are exact substrings of the combined text with strictly
        # increasing start positions, so an advancing find() locates each one.
        pos = combined.find(chunk, cursor)
        if pos == -1:
            pos = combined.find(chunk)
        page: int | None = None
        for start, seg_page in boundaries:
            if start <= pos:
                page = seg_page
            else:
                break
        result.append((chunk, page))
        cursor = pos + 1
    return result


def merge_overlapping_texts(texts: list[str]) -> str:
    """Join consecutive chunk texts into one passage, dropping the overlap the
    chunker carried from each chunk into the start of the next."""
    merged = texts[0]
    for text in texts[1:]:
        k = _overlap_length(merged, text)
        merged += text[k:] if k else "\n" + text
    return merged


def _overlap_length(a: str, b: str) -> int:
    """Length of the longest suffix of `a` that is a prefix of `b`."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0


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
    """Indexes documents into ChromaDB and serves queries over them.

    Locking model: `_run_lock` serializes whole indexing runs (only one
    incremental/full run at a time), while `_state_lock` protects the shared
    mutable state (`_doc_meta`, the `_collection` reference, and progress
    counters) and is only ever held for short, non-blocking operations. Slow
    work — parsing, embedding, `collection.add` — happens outside any lock so
    searches and status calls stay responsive during multi-hour index runs.
    A full rebuild indexes into a temporary collection and atomically swaps it
    in at the end, so the live index remains queryable throughout.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.index_dir.mkdir(parents=True, exist_ok=True)

        self._run_lock = threading.Lock()
        self._state_lock = threading.RLock()

        self._client = chromadb.PersistentClient(path=str(config.index_dir))
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata=COLLECTION_METADATA,
        )
        self._warn_if_wrong_distance_metric()
        self._warn_if_chunks_exceed_model_window()
        self._meta_path = config.index_dir / META_FILENAME
        self._fulltext_dir = config.index_dir / FULLTEXT_DIRNAME
        self._fulltext_dir.mkdir(parents=True, exist_ok=True)
        self._doc_meta = self._load_meta()
        self._last_indexed_at: str | None = self._doc_meta.get("_last_indexed_at")
        self._progress = IndexProgress()
        self._last_run: ReindexSummary | None = None

    def _warn_if_wrong_distance_metric(self) -> None:
        """Warn when an existing collection was built with a different
        distance metric — get_or_create_collection does not update the metric
        of a pre-existing collection, so only a full rebuild migrates it."""
        try:
            space = self._collection.configuration_json["hnsw"]["space"]
        except Exception:
            return
        if space != "cosine":
            logger.warning(
                "Existing index uses '%s' distance but scores assume cosine; "
                "run reindex(full=True) to rebuild with the correct metric.",
                space,
            )

    def _warn_if_chunks_exceed_model_window(self) -> None:
        """Warn when chunks are longer than the embedding model can encode —
        text past the model's max sequence length is silently truncated at
        embedding time, i.e. invisible to search."""
        try:
            max_tokens = int(self._embedding_fn._model.max_seq_length)
        except Exception:
            return
        # ~4 characters per English token is the usual rule of thumb.
        approx_chars = max_tokens * 4
        if self.config.chunk_size > approx_chars:
            logger.warning(
                "DOCMCP_CHUNK_SIZE=%d exceeds what %s can encode (%d tokens "
                "≈ %d chars); chunk tails beyond that are invisible to "
                "search. Reduce the chunk size or configure a model with a "
                "longer sequence length.",
                self.config.chunk_size,
                self.config.embedding_model,
                max_tokens,
                approx_chars,
            )

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        return {}

    def _save_meta(self) -> None:
        # Write atomically so a crash mid-write cannot corrupt the metadata.
        with self._state_lock:
            payload = json.dumps(self._doc_meta, indent=2)
        tmp = self._meta_path.with_suffix(self._meta_path.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(self._meta_path)

    def _fulltext_path(self, path_str: str) -> Path:
        """Return the on-disk blob path holding a document's extracted text
        segments (JSON list of {text, page_number})."""
        key = hashlib.sha256(path_str.encode("utf-8")).hexdigest()
        return self._fulltext_dir / f"{key}.json"

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

    # ------------------------------------------------------------------
    # Progress tracking

    def _progress_start(self, phase: str, files_total: int) -> None:
        with self._state_lock:
            self._progress = IndexProgress(
                in_progress=True,
                phase=phase,
                files_total=files_total,
                files_processed=0,
                current_file=None,
                started_at=datetime.now(timezone.utc).isoformat(),
            )

    def _progress_step(self, current_file: str) -> None:
        with self._state_lock:
            self._progress.current_file = current_file
            self._progress.files_processed += 1

    def _progress_finish(self, summary: ReindexSummary) -> None:
        with self._state_lock:
            self._progress.in_progress = False
            self._progress.current_file = None
            self._last_run = summary

    # ------------------------------------------------------------------
    # Indexing

    def index_incremental(self) -> ReindexSummary:
        """Index only new or changed files; remove deleted ones. Blocks until
        the run completes (see start_reindex for the background variant)."""
        with self._run_lock:
            return self._run_incremental()

    def index_full(self) -> ReindexSummary:
        """Rebuild the entire index from scratch. Blocks until the run
        completes (see start_reindex for the background variant)."""
        with self._run_lock:
            return self._run_full()

    def start_reindex(self, full: bool = False) -> bool:
        """Kick off an indexing run in a background thread.

        Returns False (without starting anything) if a run is already in
        progress. Progress and the final summary are exposed via get_status().
        """
        if not self._run_lock.acquire(blocking=False):
            return False

        def run() -> None:
            try:
                if full:
                    self._run_full()
                else:
                    self._run_incremental()
            except Exception:
                logger.exception(
                    "Background %s indexing run failed",
                    "full" if full else "incremental",
                )
            finally:
                self._run_lock.release()

        threading.Thread(target=run, daemon=True, name="docmcp-reindex").start()
        return True

    def _run_incremental(self) -> ReindexSummary:
        logger.info("Starting incremental indexing...")
        summary = ReindexSummary()
        current_files = scan_documents(self.config.doc_dirs)
        current_paths = {str(p) for p in current_files}
        self._progress_start("incremental", len(current_files))

        try:
            # Remove entries for deleted files
            with self._state_lock:
                stored_paths = [
                    k for k in self._doc_meta if not k.startswith("_")
                ]
            for stored_path in stored_paths:
                if stored_path not in current_paths:
                    with self._state_lock:
                        self._remove_document(
                            stored_path, self._collection, self._doc_meta
                        )
                    summary.removed += 1

            # Process current files
            for path in current_files:
                path_str = str(path)
                self._progress_step(path_str)
                with self._state_lock:
                    stored = self._doc_meta.get(path_str)

                # Quick check: same size and mtime means unchanged without
                # reading the file — hashing every byte of a large library on
                # every pass is minutes of pure I/O.
                stat = path.stat()
                if (
                    stored
                    and stored.get("size_bytes") == stat.st_size
                    and stored.get("mtime_ns") == stat.st_mtime_ns
                ):
                    summary.unchanged += 1
                    continue

                file_hash = compute_hash(path)
                if stored and stored.get("hash") == file_hash:
                    # Content unchanged despite a new mtime (e.g. touched or
                    # re-copied file); refresh the stat info so the next pass
                    # can skip the hash again.
                    with self._state_lock:
                        stored["size_bytes"] = stat.st_size
                        stored["mtime_ns"] = stat.st_mtime_ns
                    summary.unchanged += 1
                    continue

                try:
                    is_update = stored is not None
                    chunk_count = self._index_file(
                        path, file_hash, self._collection, self._doc_meta
                    )
                    if chunk_count == 0:
                        summary.empty += 1
                    elif is_update:
                        summary.updated += 1
                    else:
                        summary.added += 1
                except Exception as e:
                    logger.error("Failed to index %s: %s", path, e)
                    summary.errors.append(
                        ReindexError(filename=str(path), reason=str(e))
                    )

            now = datetime.now(timezone.utc).isoformat()
            with self._state_lock:
                self._last_indexed_at = now
                self._doc_meta["_last_indexed_at"] = now
            self._save_meta()
        finally:
            self._progress_finish(summary)

        logger.info(
            "Incremental indexing complete: added=%d, updated=%d, removed=%d, "
            "unchanged=%d, empty=%d, errors=%d",
            summary.added,
            summary.updated,
            summary.removed,
            summary.unchanged,
            summary.empty,
            len(summary.errors),
        )
        return summary

    def _run_full(self) -> ReindexSummary:
        logger.info("Starting full reindex...")
        summary = ReindexSummary()
        current_files = scan_documents(self.config.doc_dirs)
        self._progress_start("full", len(current_files))

        try:
            # Build into a temporary collection so the live index stays
            # queryable for the whole (potentially hours-long) rebuild, then
            # swap it in atomically at the end.
            try:
                self._client.delete_collection(REBUILD_COLLECTION_NAME)
            except Exception:
                pass  # no leftover rebuild collection
            new_collection = self._client.create_collection(
                name=REBUILD_COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata=COLLECTION_METADATA,
            )
            new_meta: dict = {}

            for path in current_files:
                self._progress_step(str(path))
                try:
                    file_hash = compute_hash(path)
                    chunk_count = self._index_file(
                        path, file_hash, new_collection, new_meta
                    )
                    if chunk_count == 0:
                        summary.empty += 1
                    else:
                        summary.added += 1
                except Exception as e:
                    logger.error("Failed to index %s: %s", path, e)
                    summary.errors.append(
                        ReindexError(filename=str(path), reason=str(e))
                    )

            now = datetime.now(timezone.utc).isoformat()
            new_meta["_last_indexed_at"] = now

            # Swap the freshly built collection and metadata in.
            with self._state_lock:
                self._client.delete_collection(COLLECTION_NAME)
                new_collection.modify(name=COLLECTION_NAME)
                self._collection = new_collection
                self._doc_meta = new_meta
                self._last_indexed_at = now
            self._save_meta()
            self._cleanup_stale_fulltext()
        finally:
            self._progress_finish(summary)

        logger.info(
            "Full reindex complete: added=%d, empty=%d, errors=%d",
            summary.added,
            summary.empty,
            len(summary.errors),
        )
        return summary

    def _cleanup_stale_fulltext(self) -> None:
        """Delete full-text blobs no longer referenced by any document."""
        with self._state_lock:
            keep = {
                val["full_text_path"]
                for key, val in self._doc_meta.items()
                if not key.startswith("_") and val.get("full_text_path")
            }
        for blob in self._fulltext_dir.iterdir():
            if str(blob) not in keep:
                blob.unlink(missing_ok=True)

    def _index_file(
        self, path: Path, file_hash: str, collection, meta: dict
    ) -> int:
        """Parse, chunk, embed, and store a single file into `collection`,
        recording it in `meta`. Returns the number of chunks stored.

        Files that yield no extractable text (e.g. scanned/image-only PDFs)
        are still recorded with chunk_count 0 so they are not re-parsed on
        every subsequent incremental run and can be surfaced to the user.
        """
        path_str = str(path)
        doc_name = self._relative_name(path)

        # Remove old chunks if updating
        with self._state_lock:
            self._remove_document(path_str, collection, meta)

        # Parse and chunk (slow; done outside any lock). Chunking spans page
        # boundaries so paragraphs split across pages stay together.
        segments = parse_file(path)
        all_chunks = chunk_segments(
            segments, self.config.chunk_size, self.config.chunk_overlap
        )

        now = datetime.now(timezone.utc).isoformat()
        stat = path.stat()
        entry = {
            "hash": file_hash,
            "document_name": doc_name,
            "type": path.suffix.lstrip(".").lower(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "last_modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "chunk_count": len(all_chunks),
            "indexed_at": now,
        }

        if all_chunks:
            # Persist the extracted text segments verbatim (with page numbers)
            # so get_document can return exact text and slice by page without
            # reconstructing from overlapping chunks.
            fulltext_path = self._fulltext_path(path_str)
            payload = json.dumps(
                [
                    {"text": text, "page_number": seg_meta.get("page_number")}
                    for text, seg_meta in segments
                ]
            )
            tmp = fulltext_path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(fulltext_path)
            entry["full_text_path"] = str(fulltext_path)

            # Build ChromaDB entries (embedding happens inside add(); slow,
            # done outside any lock)
            ids = [f"{path_str}::chunk_{i}" for i in range(len(all_chunks))]
            documents = [chunk for chunk, _ in all_chunks]
            metadatas = [
                {
                    "source_path": path_str,
                    "document_name": doc_name,
                    "chunk_index": i,
                    "page_number": page if page is not None else -1,
                    "content_hash": file_hash,
                    "indexed_at": now,
                }
                for i, (_, page) in enumerate(all_chunks)
            ]
            for start in range(0, len(all_chunks), ADD_BATCH_SIZE):
                end = start + ADD_BATCH_SIZE
                collection.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
        else:
            logger.warning(
                "No extractable text in %s — recording as empty "
                "(scanned/image-only PDF?)",
                path,
            )

        with self._state_lock:
            meta[path_str] = entry
        if meta is self._doc_meta:
            # Persist per file so a crash mid-run doesn't lose hours of work.
            self._save_meta()
        return len(all_chunks)

    def _remove_document(self, path_str: str, collection, meta: dict) -> None:
        """Remove all chunks and stored text for a document.

        Callers must hold `_state_lock` when `meta` is the live doc meta.
        """
        try:
            results = collection.get(
                where={"source_path": path_str},
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
        except Exception:
            pass  # Collection may be empty or doc not found
        entry = meta.pop(path_str, None)
        if entry and entry.get("full_text_path"):
            # Covers legacy .txt blobs as well as current .json ones.
            Path(entry["full_text_path"]).unlink(missing_ok=True)
        self._fulltext_path(path_str).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Queries

    def search(
        self, query: str, limit: int = 10, document_filter: str | None = None
    ):
        """Run a semantic search against the live collection, optionally
        restricted to documents whose name matches a glob pattern.

        The collection reference is snapshotted under the state lock, but the
        query itself runs unlocked so searches are not blocked by indexing. If
        a full rebuild swaps the collection out mid-query, retry once against
        the fresh reference.
        """
        where = None
        if document_filter:
            with self._state_lock:
                names = sorted(
                    {
                        val["document_name"]
                        for key, val in self._doc_meta.items()
                        if not key.startswith("_")
                        and fnmatch.fnmatch(val["document_name"], document_filter)
                    }
                )
            if not names:
                return []
            where = (
                {"document_name": names[0]}
                if len(names) == 1
                else {"document_name": {"$in": names}}
            )

        for attempt in range(2):
            with self._state_lock:
                collection = self._collection
            try:
                return _search(collection, query, limit, where=where)
            except Exception:
                if attempt == 1:
                    raise
                logger.debug("Search failed, retrying with fresh collection")

    def get_context(
        self,
        document_name: str,
        chunk_index: int,
        before: int = 2,
        after: int = 2,
    ) -> ChunkContext:
        """Return the passage surrounding a chunk: the chunks from
        `chunk_index - before` through `chunk_index + after`, merged into one
        continuous text with the chunk overlap removed."""
        before = max(0, before)
        after = max(0, after)
        lo = max(0, chunk_index - before)
        hi = chunk_index + after

        with self._state_lock:
            collection = self._collection
        results = collection.get(
            where={
                "$and": [
                    {"document_name": document_name},
                    {"chunk_index": {"$gte": lo}},
                    {"chunk_index": {"$lte": hi}},
                ]
            }
        )

        if not results["ids"]:
            raise ValueError(
                f"No chunks found for document '{document_name}' around "
                f"chunk {chunk_index}"
            )

        source_paths = {m.get("source_path") for m in results["metadatas"]}
        if len(source_paths) > 1:
            conflicting = ", ".join(sorted(str(p) for p in source_paths))
            raise ValueError(
                f"Ambiguous document name '{document_name}' matches multiple "
                f"files: {conflicting}"
            )

        pairs = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda p: p[1].get("chunk_index", 0),
        )
        indices = [m.get("chunk_index", 0) for _, m in pairs]
        if chunk_index not in indices:
            raise ValueError(
                f"Chunk {chunk_index} not found in '{document_name}'"
            )

        pages = [
            m["page_number"]
            for _, m in pairs
            if m.get("page_number", -1) > 0
        ]
        return ChunkContext(
            document_name=document_name,
            chunk_index=chunk_index,
            first_chunk_index=indices[0],
            last_chunk_index=indices[-1],
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            text=merge_overlapping_texts([doc for doc, _ in pairs]),
        )

    def _find_file_meta(self, document_name: str) -> dict | None:
        with self._state_lock:
            for key, val in self._doc_meta.items():
                if key.startswith("_"):
                    continue
                if val.get("document_name") == document_name:
                    return dict(val)
        return None

    def _load_segments(self, file_meta: dict | None) -> list[tuple[str, int | None]] | None:
        """Load the stored text segments for a document as (text, page) pairs.

        Understands both the current JSON blob format and the legacy plain-text
        format (single segment, no page info). Returns None if no usable blob
        exists.
        """
        if not file_meta or not file_meta.get("full_text_path"):
            return None
        blob = Path(file_meta["full_text_path"])
        if not blob.exists():
            return None
        raw = blob.read_text(encoding="utf-8")
        if blob.suffix == ".json":
            try:
                data = json.loads(raw)
                return [(s["text"], s.get("page_number")) for s in data]
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Corrupt full-text blob: %s", blob)
                return None
        return [(raw, None)]

    def get_document(
        self,
        document_name: str,
        page_start: int | None = None,
        page_end: int | None = None,
        offset: int = 0,
        max_chars: int = 50_000,
    ) -> DocumentContent:
        """Retrieve the text of a document by name.

        Books can run to megabytes of text, so the result is windowed: an
        optional page range (PDFs only), then a character offset/limit within
        the selected text. `truncated` is set when more text exists beyond the
        returned slice; `max_chars <= 0` disables the character limit.
        """
        with self._state_lock:
            collection = self._collection
        file_meta = self._find_file_meta(document_name)

        results = collection.get(
            where={"document_name": document_name},
        )

        if not results["ids"] and file_meta is None:
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

        if file_meta is not None and file_meta.get("chunk_count", 0) == 0:
            raise ValueError(
                f"Document '{document_name}' contains no extractable text. "
                "It may be a scanned/image-only PDF that needs OCR."
            )

        # Prefer the verbatim stored segments; fall back to a chunk
        # reconstruction only if the blob is missing (older index).
        segments = self._load_segments(file_meta)
        if segments is None:
            pairs = list(zip(results["documents"], results["metadatas"]))
            pairs.sort(key=lambda p: p[1].get("chunk_index", 0))
            segments = [
                (
                    doc,
                    m["page_number"] if m.get("page_number", -1) > 0 else None,
                )
                for doc, m in pairs
            ]

        page_numbers = [p for _, p in segments if p is not None]
        page_count = max(page_numbers) if page_numbers else None

        if page_start is not None or page_end is not None:
            if not page_numbers:
                raise ValueError(
                    f"Document '{document_name}' has no page information; "
                    "page ranges only apply to PDFs."
                )
            lo = page_start if page_start is not None else 1
            hi = page_end if page_end is not None else page_count
            segments = [
                (text, page)
                for text, page in segments
                if page is not None and lo <= page <= hi
            ]

        full_text = "\n\n".join(text for text, _ in segments)
        total_chars = len(full_text)
        offset = max(0, offset)
        content = full_text[offset:]
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]
        truncated = offset + len(content) < total_chars

        return DocumentContent(
            document_name=document_name,
            content=content,
            size_bytes=file_meta["size_bytes"] if file_meta else 0,
            page_count=page_count,
            last_modified=file_meta["last_modified"] if file_meta else "",
            total_chars=total_chars,
            offset=offset,
            truncated=truncated,
        )

    def list_documents(self, filter_pattern: str | None = None) -> list[DocumentInfo]:
        """List all indexed documents, optionally filtered by glob pattern."""
        with self._state_lock:
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
        """Return current index statistics, including live indexing progress
        and the summary of the most recent run."""
        with self._state_lock:
            collection = self._collection
            total_docs = sum(1 for k in self._doc_meta if not k.startswith("_"))
            progress = replace(self._progress)
            last_run = self._last_run
            last_indexed_at = self._last_indexed_at

        total_chunks = collection.count()

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
            last_indexed_at=last_indexed_at,
            embedding_model=self.config.embedding_model,
            source_directories=[str(d) for d in self.config.doc_dirs],
            indexing=progress,
            last_run=last_run,
        )
