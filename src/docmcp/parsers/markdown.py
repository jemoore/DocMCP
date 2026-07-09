from __future__ import annotations

import re
from pathlib import Path

import markdown

from docmcp.parsers import read_text


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a Markdown file, stripping markup to plain text."""
    raw = read_text(path)
    html = markdown.markdown(raw)
    # Strip HTML tags to get plain text
    text = re.sub(r"<[^>]+>", "", html)
    if not text.strip():
        return []
    return [(text, {})]
