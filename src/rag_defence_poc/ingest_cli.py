from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rag_defence_poc.chunking import chunk_pages, chunk_text
from rag_defence_poc.config import settings
from rag_defence_poc.embedder import Embedder
from rag_defence_poc.store import get_collection


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages: list[tuple[int, str]] = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages.append((i + 1, text))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest manifest into Chroma index")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.manifest_path,
        help="Path to manifest YAML",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingest",
    )
    args = parser.parse_args()

    root = Path.cwd()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    raw_yaml = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    docs = raw_yaml.get("documents") or []
    if not docs:
        raise SystemExit("No documents: in manifest")

    embedder = Embedder(settings.embedding_model)
    collection = get_collection(
        settings.chroma_path,
        settings.collection_name,
        reset=args.reset,
    )

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    chunk_counter: dict[str, int] = {}

    for entry in docs:
        doc_id = str(entry["doc_id"])
        rel = Path(entry["path"])
        file_path = rel if rel.is_absolute() else root / rel
        if not file_path.is_file():
            raise SystemExit(f"Missing file for {doc_id}: {file_path}")

        title = str(entry.get("title", doc_id))
        source_url = str(entry.get("source_url", ""))
        doc_type = str(entry.get("doc_type", "other"))
        suffix = file_path.suffix.lower()

        chunks_text: list[tuple[str, str | None]] = []
        if suffix == ".pdf":
            pages = load_pdf_pages(file_path)
            for ch in chunk_pages(pages):
                page_str = str(ch.page_start) if ch.page_start is not None else ""
                chunks_text.append((ch.text, page_str or None))
        elif suffix in (".md", ".txt"):
            body = file_path.read_text(encoding="utf-8")
            for ch in chunk_text(body, page=None):
                chunks_text.append((ch.text, None))
        else:
            raise SystemExit(
                f"Unsupported type {suffix} for {doc_id}. "
                "Supported: .pdf, .md, .txt"
            )

        for text, page in chunks_text:
            idx = chunk_counter.get(doc_id, 0)
            chunk_id = f"{doc_id}_{idx}"
            chunk_counter[doc_id] = idx + 1
            all_ids.append(chunk_id)
            all_docs.append(text)
            meta = {
                "doc_id": doc_id,
                "title": title,
                "source_url": source_url,
                "doc_type": doc_type,
            }
            if page:
                meta["page"] = page
            all_metas.append(meta)

    if not all_ids:
        raise SystemExit("No chunks produced")

    embeddings = embedder.encode(all_docs)
    embs_list = [e.tolist() for e in embeddings]

    batch = 64
    for i in range(0, len(all_ids), batch):
        j = i + batch
        collection.add(
            ids=all_ids[i:j],
            embeddings=embs_list[i:j],
            documents=all_docs[i:j],
            metadatas=all_metas[i:j],
        )

    print(f"Ingested {len(all_ids)} chunks into {settings.chroma_path}")


if __name__ == "__main__":
    main()
