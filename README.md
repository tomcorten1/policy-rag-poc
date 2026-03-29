# Citation-locked policy RAG (hobby POC)

Small **Python** stack: manifest-driven ingest → **Chroma** (cosine) + **sentence-transformers** embeddings → **FastAPI** `/ask` with **citation-only** prompts and a **similarity gate**. Optional **Ollama** or **OpenAI** for generation.

This is a **portfolio / interview** demo: answers are grounded in ingested open publications. The default manifest uses two **Open Overheid PDF’s** (Defensie DS/AI-strategie + handreiking generatieve AI); see [`corpus/README.md`](corpus/README.md). Source files live under **`corpus/raw/`** (gitignored); fetch them with:

```bash
./scripts/download_corpus.sh
```

Optional: `corpus/sample_documents/policy_demo_nl.md` is a tiny **fictitious** file for offline tests only (not listed in the default manifest).

## Quick start

```bash
cd rag-defence-poc
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[openai]"
./scripts/download_corpus.sh
```

Build the index (from the project root):

```bash
rag-ingest --reset
```

Run the API:

```bash
rag-serve
# or: uvicorn rag_defence_poc.api:app --host 127.0.0.1 --port 8000
```

**Browser UI:** open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — a minimal page posts to the same `POST /ask` endpoint (no extra build step). You still need a built index (`rag-ingest --reset`) and a working LLM backend (Ollama or OpenAI), same as below.

Ask a question:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Welke vier defensiebrede doelstellingen noemt de DS/AI-strategie?","language":"nl"}'
```

### LLM backend

- **Ollama (default):** install [Ollama](https://ollama.com/), then **pull** the model once (required; otherwise `/api/chat` returns 404):

  ```bash
  ollama pull llama3.2
  ```

  Optional environment variables:

  - `OLLAMA_BASE_URL=http://127.0.0.1:11434`
  - `OLLAMA_MODEL=llama3.2`

- **OpenAI:** set `LLM_BACKEND=openai` and `OPENAI_API_KEY=...` (optional extra: `OPENAI_MODEL=gpt-4o-mini`).

### Configuration (environment)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHROMA_PATH` | `data/chroma` | Persistent vector store directory |
| `COLLECTION_NAME` | `policy_corpus` | Chroma collection name |
| `MANIFEST_PATH` | `corpus/manifest.yaml` | Ingest manifest |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `TOP_K` | `8` | Retrieved chunks |
| `MIN_SIMILARITY` | `0.28` | Refuse if best cosine similarity is below this |
| `LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `AUDIT_LOG_PATH` | `logs/queries.jsonl` | Append-only query log |

## Adding documents

1. Download open **PDF** (or plain **`.md` / `.txt`**) files, e.g. from [Open Overheid](https://open.overheid.nl/) or [EUR-Lex](https://eur-lex.europa.eu/) (browser download if curl is blocked).
2. Place files under `corpus/raw/` (gitignored by default).
3. Add entries to `corpus/manifest.yaml` with `path`, `doc_id`, `title`, `source_url`, `doc_type`.
4. Run `rag-ingest --reset`.

Supported extensions: **`.pdf`**, **`.md`**, **`.txt`**.

## Audit log

Each `/ask` call appends one JSON line to `logs/queries.jsonl` (create-on-write): question, chunk ids, scores, refused flag, model id, latency. Do not log personal data in questions on shared systems.

## Manual evaluation (smoke)

After `./scripts/download_corpus.sh` and `rag-ingest --reset`:

| # | Question (NL) | Expected source |
|---|-----------------|-----------------|
| 1 | Welke vier defensiebrede doelstellingen noemt de DS/AI-strategie? | Defensie-strategie PDF |
| 2 | Welke vijf toepassingsgebieden worden genoemd? | idem |
| 3 | Wat wil Defensie in 2027 bereiken volgens de strategie? | idem |
| 4 | Welke randvoorwaarden noemt de strategie voor data governance? | idem |
| 5 | Welke thema’s behandelt de handreiking voor generatieve AI? | Handreiking-PDF |

Voeg zelf extra PDF’s toe (bv. AI Act van EUR-Lex) voor EU-juridische detailvragen; zie [`corpus/README.md`](corpus/README.md).

**Pass criteria:** cited chunks come from the expected document; no unsupported factual claims; refusal on out-of-corpus questions is acceptable.

## Licence

MIT (code). Third-party documents remain under their respective terms; do not commit copyrighted PDFs unless your licence allows it.
