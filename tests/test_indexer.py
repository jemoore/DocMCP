from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from docmcp.indexer import chunk_text, compute_hash, scan_documents


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


class TestScanDocuments:
    def test_finds_supported_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "readme.md").touch()
        (tmp_path / "page.html").touch()
        (tmp_path / "data.txt").touch()  # should be excluded
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.htm").touch()

        result = scan_documents([tmp_path])
        names = [p.name for p in result]
        assert "doc.pdf" in names
        assert "readme.md" in names
        assert "page.html" in names
        assert "nested.htm" in names
        assert "data.txt" not in names

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
