from __future__ import annotations

from pathlib import Path

import pymupdf


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a PDF file, returning (text, metadata) per page."""
    results: list[tuple[str, dict]] = []
    doc = pymupdf.open(path)
    try:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                results.append((text, {"page_number": page_num + 1}))
    finally:
        doc.close()
    return results
