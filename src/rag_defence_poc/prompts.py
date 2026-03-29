from __future__ import annotations

from rag_defence_poc.retrieve import RetrievedChunk


def build_context_blocks(chunks: list[RetrievedChunk]) -> tuple[str, dict[int, str]]:
    """
    Build numbered context for the LLM. Returns (context_string, citation_map).
    citation_map: bracket number -> chunk_id
    """
    lines: list[str] = []
    cite_map: dict[int, str] = {}
    for i, c in enumerate(chunks, start=1):
        cite_map[i] = c.chunk_id
        page = f"p.{c.page}" if c.page else "n/a"
        lines.append(
            f"[{i}] doc_id={c.doc_id} | {page} | title={c.title}\n{c.text}"
        )
    return "\n\n".join(lines), cite_map


def system_prompt(language: str) -> str:
    if language.lower().startswith("nl"):
        return (
            "Je bent een beleidsassistent. Je antwoordt uitsluitend op basis van de "
            "meegegeven fragmenten. Gebruik geen eigen algemene kennis buiten wat "
            "letterlijk of redelijkerwijs uit de fragmenten volgt. "
            "Als de fragmenten onvoldoende zijn, zeg dat expliciet. "
            "Als fragmenten elkaar tegenspreken, noem beide standpunten en verwijs naar "
            "de bronnen. "
            "Voeg aan relevante zinnen bronverwijzingen toe in de vorm [1], [2], ... "
            "die verwijzen naar de genummerde fragmenten. "
            "Antwoord in het Nederlands."
        )
    return (
        "You are a policy assistant. Answer only from the provided passages. "
        "Do not use outside general knowledge beyond what the passages support. "
        "If passages are insufficient, say so explicitly. "
        "If passages conflict, describe both and cite sources. "
        "Add inline citations [1], [2], ... referring to the numbered passages. "
        "Answer in English."
    )


def user_prompt(question: str, context: str, language: str) -> str:
    if language.lower().startswith("nl"):
        return (
            f"Vraag:\n{question}\n\n"
            f"Fragmenten:\n{context}\n\n"
            "Geef een beknopt antwoord met bronverwijzingen [n] zoals hierboven."
        )
    return (
        f"Question:\n{question}\n\n"
        f"Passages:\n{context}\n\n"
        "Give a concise answer with [n] citations as above."
    )


def refusal_message(language: str) -> str:
    if language.lower().startswith("nl"):
        return (
            "Geen voldoende relevante fragmenten in de corpus om deze vraag "
            "betrouwbaar te beantwoorden. Probeer een specifiekere vraag of voeg "
            "documenten toe via de ingest-stap."
        )
    return (
        "Not enough relevant passages in the corpus to answer reliably. "
        "Try a narrower question or add documents via ingest."
    )
