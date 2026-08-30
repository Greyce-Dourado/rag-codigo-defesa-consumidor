"""Smoke test mínimo: garante que o pacote e a config carregam."""

from rag.config import settings


def test_config_loads():
    assert settings.embedding_model
    assert settings.pg_dsn.startswith("postgresql://")
