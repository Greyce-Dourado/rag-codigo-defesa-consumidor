"""Recuperação densa: pergunta -> top-k artigos por similaridade de cosseno (pgvector)."""

from __future__ import annotations

from rag.embeddings.encoder import embed
from rag.store.db import connect

# <=> é o operador de DISTÂNCIA de cosseno do pgvector (0 = idêntico, 2 = oposto).
# ORDER BY distância ASC + LIMIT k => os k artigos mais próximos. É aqui que o índice HNSW entra.
_SEARCH = """
SELECT id, artigo, capitulo, secao, texto, embedding <=> %s::vector AS distancia
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


def dense_search(query: str, k: int = 5) -> list[dict]:
    qv = embed([query])[0]  # mesmo modelo dos documentos: pergunta e artigos no mesmo espaço
    with connect() as conn, conn.cursor() as cur:
        cur.execute(_SEARCH, (qv, qv, k))
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def dense_then_rerank(query: str, k: int = 5, candidates: int = 20) -> list[dict]:
    """Duas etapas: bi-encoder pega `candidates` candidatos (rápido), cross-encoder reordena
    e devolve os `k` melhores (preciso). Import tardio pra não carregar o reranker sem uso."""
    from rag.retrieval.rerank import rerank

    cands = dense_search(query, k=candidates)
    return rerank(query, cands, top_k=k)


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]) or "posso desistir de uma compra feita pela internet?"
    print(f"pergunta: {pergunta}\n")
    for r in dense_search(pergunta, k=5):
        print(f"  {r['id']:>8}  dist={r['distancia']:.3f}  {r['texto'][:90]}")
