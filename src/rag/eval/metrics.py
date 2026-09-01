"""Métricas de avaliação de recuperação (escritas à mão, com relevância binária).

Convenção: `ranked` é a lista de ids recuperados EM ORDEM (posição 0 = mais relevante);
`relevant` é o conjunto de ids do ground truth para a pergunta.
"""

from __future__ import annotations

import math


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / posição do PRIMEIRO relevante. 0 se nenhum aparece. Base do MRR."""
    for i, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 se ALGUM relevante está no top-k, senão 0.0 (a métrica mais 'grossa')."""
    return 1.0 if any(rid in relevant for rid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fração dos relevantes que apareceu no top-k. Com 1 relevante, é igual ao hit@k."""
    if not relevant:
        return 0.0
    hits = sum(1 for rid in ranked[:k] if rid in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    # ganho 1 para relevante, descontado por 1/log2(posição+1): rank 1 vale 1.0, rank 2 ~0.63...
    return sum(
        1.0 / math.log2(i + 1)
        for i, rid in enumerate(ranked[:k], start=1)
        if rid in relevant
    )


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """DCG normalizado pelo DCG ideal (todos os relevantes no topo). Penaliza ranquear o
    relevante em posição baixa — mais sensível à ORDEM que o recall."""
    dcg = dcg_at_k(ranked, relevant, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
