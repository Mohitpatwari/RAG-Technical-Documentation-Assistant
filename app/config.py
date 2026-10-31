"""
app/config.py
Centralised settings loaded from environment variables / .env (start the app from the project root).
All other modules import `settings` from here.
"""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ── LLM ────────────────────────────────────────────────────────────────────
    google_api_key: str = Field(default="", description="Google AI Studio API key.")
    # gemini-1.5-* is retired. If you get a 404 "model not found", check the current
    # Flash model name in Google AI Studio and set LLM_MODEL in .env.
    llm_model: str = Field(default="gemini-2.5-flash", description="LLM model identifier.")
    llm_temperature: float = Field(default=0.0)
    llm_timeout: float = Field(default=30.0, description="Per-request timeout (seconds).")
    llm_max_retries: int = Field(default=3, description="Attempts for transient LLM errors.")

    # ── Embeddings / vector store ──────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="rag_docs")

    # ── Retrieval / graph ──────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5)
    max_retries: int = Field(default=2, description="Max query-rewrite/retrieval retries.")

    # ── Bonus features ─────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Leave empty to disable web search.")
    enable_hallucination_check: bool = Field(default=False)

    # ── Storage ────────────────────────────────────────────────────────────────
    feedback_file: str = Field(default="./data/feedback.jsonl")
    queries_file: str = Field(default="./data/queries.jsonl")
    corpus_dir: str = Field(default="./data/corpus")

    # ── Server ─────────────────────────────────────────────────────────────────
    # Plain string on purpose: pydantic-settings JSON-decodes List[str] env values
    # before validators run, which breaks comma-separated values in .env.
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:7860")
    log_level: str = Field(default="INFO")

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
