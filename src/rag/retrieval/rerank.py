"""Re-ranking com cross-encoder (bge-reranker-v2-m3).

Diferença para o embedding (bge-m3):
- bge-m3 é BI-ENCODER: um vetor por texto, isolado. Rápido/escalável, mas nunca vê pergunta e
  documento juntos.
- o reranker é CROSS-ENCODER: recebe o PAR (pergunta, documento) e devolve uma nota de
  relevância. Muito mais preciso, porém caro — por isso só reordena candidatos já filtrados.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.config import settings

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Reordena os candidatos pela nota do cross-encoder e devolve os top_k (com a nota)."""
    pairs = [(query, c["texto"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    ordenados = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ordenados[:top_k]]


if __name__ == "__main__":
    # Smoke test: o cross-encoder deve dar nota ALTA ao artigo relevante e BAIXA ao irrelevante.
    m = get_reranker()
    q = "posso desistir de uma compra feita pela internet?"
    demo = [
        ("relevante  (art_49)", "Art. 49. O consumidor pode desistir do contrato no prazo de 7 dias..."),
        ("irrelevante(art_119)", "Art. 119. Revogam-se as disposições em contrário."),
    ]
    for label, texto in demo:
        score = float(m.predict([(q, texto)])[0])
        print(f"{label}: score={score:.3f}")
