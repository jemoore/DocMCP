from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from docmcp.config import Config
from docmcp.indexer import (
    Indexer,
    chunk_segments,
    chunk_text,
    compute_hash,
    merge_overlapping_texts,
    scan_documents,
)


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        text = "Hello world"
        result = chunk_text(text, chunk_size=1000, overlap=200)
        assert result == ["Hello world"]

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text("", chunk_size=1000, overlap=200) == []
        assert chunk_text("   ", chunk_size=1000, overlap=200) == []

    def test_splits_on_double_newline(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = chunk_text(text, chunk_size=30, overlap=0)
        assert len(result) >= 2
        assert "Paragraph one." in result[0]

    def test_splits_long_text(self) -> None:
        text = " ".join(f"word{i}" for i in range(200))
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert len(result) > 1
        # All text should be represented
        combined = " ".join(result)
        assert "word0" in combined
        assert "word199" in combined

    def test_overlap_produces_overlapping_chunks(self) -> None:
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        result = chunk_text(text, chunk_size=20, overlap=5)
        assert len(result) > 1

    def test_no_chunk_exceeds_chunk_size(self) -> None:
        # Dense text whose only breaks are single newlines (no blank lines).
        # The splitter must recurse with finer separators so no chunk exceeds
        # chunk_size — previously oversized pieces slipped through and were
        # silently truncated at embedding time.
        para = ("lorem ipsum " * 40).strip()
        text = "\n".join([para, para, para])
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_recurses_to_character_level_when_no_separators(self) -> None:
        text = "x" * 1000  # contains none of the configured separators
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert all(len(c) <= 100 for c in chunks)
        # With no overlap the character-level split covers the text exactly.
        assert "".join(chunks) == text

    def test_overlap_increases_total_length(self) -> None:
        words = " ".join(f"w{i}" for i in range(200))
        without = chunk_text(words, chunk_size=100, overlap=0)
        with_overlap = chunk_text(words, chunk_size=100, overlap=40)
        assert sum(len(c) for c in with_overlap) > sum(len(c) for c in without)


class TestChunkSegments:
    def test_single_segment_no_pages(self) -> None:
        segments = [("hello world content", {})]
        result = chunk_segments(segments, chunk_size=1000, overlap=0)
        assert result == [("hello world content", None)]

    def test_chunks_span_page_boundaries(self) -> None:
        # Two short pages must merge into one chunk instead of one chunk per
        # page — previously each page was chunked in isolation.
        segments = [
            ("first page text.", {"page_number": 1}),
            ("second page text.", {"page_number": 2}),
        ]
        result = chunk_segments(segments, chunk_size=1000, overlap=0)
        assert len(result) == 1
        chunk, page = result[0]
        assert "first page text." in chunk
        assert "second page text." in chunk
        assert page == 1  # page of the chunk's first character

    def test_chunk_page_assignment_follows_start_position(self) -> None:
        page1 = "alpha " * 30  # 180 chars
        page2 = "omega " * 30
        segments = [
            (page1.strip(), {"page_number": 1}),
            (page2.strip(), {"page_number": 2}),
        ]
        result = chunk_segments(segments, chunk_size=200, overlap=0)
        assert len(result) >= 2
        assert result[0][1] == 1
        assert result[-1][1] == 2
        # All text is represented across the chunks.
        combined = " ".join(chunk for chunk, _ in result)
        assert "alpha" in combined and "omega" in combined


class TestMergeOverlappingTexts:
    def test_removes_carried_overlap(self) -> None:
        # Second chunk starts with the tail of the first (the chunker's
        # overlap); the merge must not duplicate it.
        merged = merge_overlapping_texts(
            ["one two three four", "three four five six"]
        )
        assert merged == "one two three four five six"

    def test_no_overlap_joins_with_newline(self) -> None:
        merged = merge_overlapping_texts(["first part", "second part"])
        assert merged == "first part\nsecond part"

    def test_single_text_unchanged(self) -> None:
        assert merge_overlapping_texts(["only text"]) == "only text"


class TestScanDocuments:
    def test_finds_supported_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "readme.md").touch()
        (tmp_path / "page.html").touch()
        (tmp_path / "data.txt").touch()
        (tmp_path / "book.epub").touch()
        (tmp_path / "report.docx").touch()  # should be excluded
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.htm").touch()

        result = scan_documents([tmp_path])
        names = [p.name for p in result]
        assert "doc.pdf" in names
        assert "readme.md" in names
        assert "page.html" in names
        assert "nested.htm" in names
        assert "data.txt" in names
        assert "book.epub" in names
        assert "report.docx" not in names

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        result = scan_documents([tmp_path / "nope"])
        assert result == []

    def test_multiple_dirs(self, tmp_path: Path) -> None:
        dir1 = tmp_path / "a"
        dir2 = tmp_path / "b"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "one.md").touch()
        (dir2 / "two.html").touch()

        result = scan_documents([dir1, dir2])
        names = [p.name for p in result]
        assert "one.md" in names
        assert "two.html" in names


class TestComputeHash:
    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_hash(f)
        h2 = compute_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_hash(f1) != compute_hash(f2)


class TestRelativeName:
    """Document naming — exercised without constructing a full Indexer (which
    would load the embedding model) via __new__ + a stub config."""

    def _indexer(self, doc_dirs: list[Path]) -> Indexer:
        ix = Indexer.__new__(Indexer)
        ix.config = Config(doc_dirs=doc_dirs)
        return ix

    def test_single_dir_uses_plain_relative_path(self) -> None:
        ix = self._indexer([Path("/docs")])
        assert ix._relative_name(Path("/docs/sub/readme.md")) == "sub/readme.md"

    def test_multi_dir_namespaces_by_basename(self) -> None:
        ix = self._indexer([Path("/a"), Path("/b")])
        assert ix._relative_name(Path("/a/readme.md")) == "a/readme.md"
        assert ix._relative_name(Path("/b/readme.md")) == "b/readme.md"

    def test_path_outside_all_dirs_falls_back_to_basename(self) -> None:
        ix = self._indexer([Path("/a")])
        assert ix._relative_name(Path("/elsewhere/x.md")) == "x.md"
