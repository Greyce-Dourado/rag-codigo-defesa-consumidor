"""Geração de embeddings densos (bge-m3 por padrão), atrás de uma interface plugável."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rag.config import settings

# Singleton preguiçoso: o modelo (~2 GB) é carregado UMA vez, na primeira chamada de get_model(),
# e reutilizado depois. Nunca no import.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str], batch_size: int = 16, show_progress: bool = False) -> list[list[float]]:
    model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return vecs.tolist()
