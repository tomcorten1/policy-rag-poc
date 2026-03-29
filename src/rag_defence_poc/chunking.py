from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    page_start: int | None
    page_end: int | None


def chunk_text(
    text: str,
    *,
    max_chars: int = 2400,
    overlap_ratio: float = 0.12,
    page: int | None = None,
) -> list[TextChunk]:
    """
    Character-based windowing (~600 tokens default) with overlap.
    When `page` is set, all chunks get that page in metadata (per-page ingestion).
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    overlap = int(max_chars * overlap_ratio)
    step = max(1, max_chars - overlap)
    chunks: list[TextChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(
                    text=piece,
                    page_start=page,
                    page_end=page,
                )
            )
        if end >= len(cleaned):
            break
        start += step
    return chunks


def chunk_pages(pages: list[tuple[int, str]], *, max_chars: int = 2400) -> list[TextChunk]:
    """Chunk each PDF page separately (stable page citations)."""
    out: list[TextChunk] = []
    for page_num, raw in pages:
        out.extend(chunk_text(raw, max_chars=max_chars, page=page_num))
    return out
