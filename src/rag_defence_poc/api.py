from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_defence_poc.audit import append_audit
from rag_defence_poc.config import settings
from rag_defence_poc.embedder import Embedder
from rag_defence_poc.llm import ollama_chat, openai_chat
from rag_defence_poc.prompts import (
    build_context_blocks,
    refusal_message,
    system_prompt,
    user_prompt,
)
from rag_defence_poc.retrieve import RetrievedChunk, retrieve
from rag_defence_poc.store import get_collection

app = FastAPI(title="Citation-locked policy RAG", version="0.1.0")

_embedder: Embedder | None = None
_collection = None


def _get_deps():
    global _embedder, _collection
    if _embedder is None:
        _embedder = Embedder(settings.embedding_model)
    if _collection is None:
        _collection = get_collection(
            settings.chroma_path,
            settings.collection_name,
            reset=False,
        )
    return _embedder, _collection


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    language: str = "nl"


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    page: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    retrieval_scores: list[float] = []


def _citations_for_response(chunks: list[RetrievedChunk]) -> list[CitationOut]:
    out: list[CitationOut] = []
    for c in chunks:
        out.append(
            CitationOut(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                url=c.source_url,
                page=c.page,
            )
        )
    return out


def _llm_complete(system: str, user: str) -> tuple[str, str]:
    backend = settings.llm_backend.lower()
    if backend == "openai":
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY not set",
            )
        text = openai_chat(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            system=system,
            user=user,
        )
        return text, f"openai:{settings.openai_model}"
    text = ollama_chat(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        system=system,
        user=user,
    )
    return text, f"ollama:{settings.ollama_model}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    embedder, collection = _get_deps()
    q_emb = embedder.encode_query(req.question)
    chunks, refused = retrieve(
        collection,
        q_emb,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
    )

    scores = [c.similarity for c in chunks]

    if refused or not chunks:
        msg = refusal_message(req.language)
        latency = (time.perf_counter() - t0) * 1000
        append_audit(
            settings.audit_log_path,
            question=req.question,
            chunk_ids=[c.chunk_id for c in chunks],
            scores=scores,
            refused=True,
            model="none",
            latency_ms=latency,
            answer_preview=msg,
        )
        return AskResponse(
            answer=msg,
            citations=[],
            refused=True,
            retrieval_scores=scores,
        )

    context, _cite_map = build_context_blocks(chunks)
    sys_p = system_prompt(req.language)
    usr_p = user_prompt(req.question, context, req.language)

    try:
        answer, model_label = _llm_complete(sys_p, usr_p)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    latency = (time.perf_counter() - t0) * 1000
    append_audit(
        settings.audit_log_path,
        question=req.question,
        chunk_ids=[c.chunk_id for c in chunks],
        scores=scores,
        refused=False,
        model=model_label,
        latency_ms=latency,
        answer_preview=answer,
    )

    return AskResponse(
        answer=answer,
        citations=_citations_for_response(chunks),
        refused=False,
        retrieval_scores=scores,
    )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "rag_defence_poc.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
