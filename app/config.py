from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "farmagraph123"
    neo4j_database: str = "neo4j"

    ollama_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: int = Field(default=180, gt=0)
    embedding_dimension: int = Field(default=768, gt=0)

    chunk_size: int = Field(default=900, gt=100)
    vector_top_k: int = Field(default=5, gt=0, le=20)
    max_context_facts: int = Field(default=28, gt=0)
    max_hops: int = Field(default=2, ge=1, le=3)

    data_dir: Path = ROOT / "data"
    corpus_dir: Path = ROOT / "corpus"


@lru_cache
def get_settings() -> Settings:
    return Settings()
