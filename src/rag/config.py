"""Configuração central, carregada de variáveis de ambiente (.env).

Mantém as escolhas plugáveis (modelo de embedding, reranker, modelo de geração) num só lugar,
para que trocar de modelo seja uma mudança de config — não de código — e alimente os
experimentos de avaliação.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    # Postgres
    pg_user: str = os.getenv("POSTGRES_USER", "rag")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "rag")
    pg_db: str = os.getenv("POSTGRES_DB", "rag")
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = int(os.getenv("POSTGRES_PORT", "5432"))

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Modelos (plugáveis)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


settings = Settings()
