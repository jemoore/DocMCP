from __future__ import annotations

from pathlib import Path

from docmcp.parsers import read_text


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a plain-text file."""
    text = read_text(path)
    if not text.strip():
        return []
    return [(text, {})]
