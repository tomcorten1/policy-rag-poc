from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from chromadb.api.models.Collection import Collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    text: str
    page: str | None
    doc_type: str
    similarity: float


def cosine_similarity_from_distance(distance: float) -> float:
    """Chroma cosine space: distance = 1 - cosine_similarity for L2-normalized vectors."""
    return float(max(0.0, min(1.0, 1.0 - distance)))


def retrieve(
    collection: Collection,
    query_embedding: np.ndarray,
    *,
    top_k: int,
    min_similarity: float,
) -> tuple[list[RetrievedChunk], bool]:
    """
    Returns (chunks, refused_early).
    refused_early True if best chunk is below min_similarity.
    """
    q = query_embedding.astype(np.float32).tolist()
    res = collection.query(
        query_embeddings=[q],
        n_results=top_k,
        include=["distances", "documents", "metadatas"],
    )
    ids = res.get("ids") or [[]]
    dists = res.get("distances") or [[]]
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]

    row_ids = ids[0] if ids else []
    row_dists = dists[0] if dists else []
    row_docs = docs[0] if docs else []
    row_metas = metas[0] if metas else []

    if not row_ids:
        return [], True

    chunks: list[RetrievedChunk] = []
    n = min(len(row_ids), len(row_dists), len(row_docs), len(row_metas))
    for i in range(n):
        cid = row_ids[i]
        dist = row_dists[i]
        text = row_docs[i]
        meta = row_metas[i] or {}
        sim = cosine_similarity_from_distance(float(dist))
        page = meta.get("page") or None
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=str(meta.get("doc_id", "")),
                title=str(meta.get("title", "")),
                source_url=str(meta.get("source_url", "")),
                text=text or "",
                page=page if page else None,
                doc_type=str(meta.get("doc_type", "")),
                similarity=sim,
            )
        )

    best = chunks[0].similarity if chunks else 0.0
    if best < min_similarity:
        return chunks, True
    return chunks, False
