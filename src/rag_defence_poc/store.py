from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection


def get_collection(chroma_path: Path, collection_name: str, reset: bool = False) -> Collection:
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
