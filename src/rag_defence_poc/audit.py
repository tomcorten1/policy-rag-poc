from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path


def append_audit(
    path: Path,
    *,
    question: str,
    chunk_ids: list[str],
    scores: list[float],
    refused: bool,
    model: str,
    latency_ms: float,
    answer_preview: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "question": question,
        "top_k_chunk_ids": chunk_ids,
        "scores": scores,
        "refused": refused,
        "model": model,
        "latency_ms": round(latency_ms, 2),
    }
    if answer_preview is not None:
        record["answer_preview"] = answer_preview[:500]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit_timestamp() -> str:
    return datetime.now(UTC).isoformat()
