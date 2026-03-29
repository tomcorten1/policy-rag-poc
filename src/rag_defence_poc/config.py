from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths (relative to cwd or absolute)
    chroma_path: Path = Path("data/chroma")
    collection_name: str = "policy_corpus"
    manifest_path: Path = Path("corpus/manifest.yaml")

    # Embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Retrieval
    top_k: int = 8
    min_similarity: float = 0.28  # cosine similarity; tune per corpus (0–1)

    # LLM
    llm_backend: str = "ollama"  # ollama | openai
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Audit
    audit_log_path: Path = Path("logs/queries.jsonl")


settings = Settings()
