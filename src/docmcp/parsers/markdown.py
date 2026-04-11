from __future__ import annotations

import re
from pathlib import Path

import markdown


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a Markdown file, stripping markup to plain text."""
    raw = path.read_text(encoding="utf-8")
    html = markdown.markdown(raw)
    # Strip HTML tags to get plain text
    text = re.sub(r"<[^>]+>", "", html)
    if not text.strip():
        return []
    return [(text, {})]
