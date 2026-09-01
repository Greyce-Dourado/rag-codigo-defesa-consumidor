"""Etapa 4/5 — avalia e COMPARA estratégias de recuperação contra o evalset.

Estratégias:
  - densa         : bi-encoder (bge-m3) + cosseno, top-10.
  - densa+rerank  : bi-encoder pega 20 candidatos, cross-encoder reordena para top-10.

Mostra a tabela de métricas lado a lado e a posição do 1º artigo relevante por pergunta em
cada estratégia (pra ver onde o rerank ajudou ou atrapalhou).

Uso:  ./.venv/bin/python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.retrieval.search import dense_search, dense_then_rerank

ROOT = Path(__file__).resolve().parents[1]
EVALSET = ROOT / "evalset" / "questions.jsonl"

FINAL = 10        # tamanho do ranking avaliado
CANDIDATES = 20   # candidatos do 1º estágio para o rerank

console = Console()

STRATEGIES = {
    "densa": lambda q: [r["id"] for r in dense_search(q, k=FINAL)],
    "densa+rerank": lambda q: [r["id"] for r in dense_then_rerank(q, k=FINAL, candidates=CANDIDATES)],
}


def run(strategy_fn, questions):
    return [
        (q, strategy_fn(q["pergunta"]), set(q["artigos_relevantes"]))
        for q in questions
    ]


def primeiro_rank(ranked, rel):
    return next((i for i, rid in enumerate(ranked, 1) if rid in rel), None)


def main() -> int:
    questions = [json.loads(line) for line in EVALSET.open(encoding="utf-8")]
    n = len(questions)

    console.print(f"Rodando {len(STRATEGIES)} estratégias sobre {n} perguntas…\n")
    resultados = {nome: run(fn, questions) for nome, fn in STRATEGIES.items()}

    # Tabela comparativa das métricas.
    tabela = Table(title=f"Comparação de estratégias — N={n}")
    tabela.add_column("métrica")
    for nome in STRATEGIES:
        tabela.add_column(nome, justify="right")

    linhas = [
        ("Hit@1", lambda pq: sum(hit_at_k(r, rel, 1) for _, r, rel in pq) / n),
        ("Hit@3", lambda pq: sum(hit_at_k(r, rel, 3) for _, r, rel in pq) / n),
        ("Recall@5", lambda pq: sum(recall_at_k(r, rel, 5) for _, r, rel in pq) / n),
        ("nDCG@10", lambda pq: sum(ndcg_at_k(r, rel, 10) for _, r, rel in pq) / n),
        ("MRR", lambda pq: sum(reciprocal_rank(r, rel) for _, r, rel in pq) / n),
    ]
    for nome_metrica, calc in linhas:
        tabela.add_row(nome_metrica, *[f"{calc(resultados[s]):.3f}" for s in STRATEGIES])
    console.print(tabela)

    # Posição do 1º relevante por pergunta, lado a lado.
    console.print("\n[bold]Posição do 1º artigo relevante (por pergunta):[/]")
    porq = Table()
    porq.add_column("pergunta")
    porq.add_column("esperado")
    for nome in STRATEGIES:
        porq.add_column(nome, justify="right")
    base = resultados["densa"]
    for idx, (q, _, rel) in enumerate(base):
        row = [q["id"], sorted(rel)[0]]
        for nome in STRATEGIES:
            _, ranked, _ = resultados[nome][idx]
            pos = primeiro_rank(ranked, rel)
            row.append(str(pos) if pos else "—")
        porq.add_row(*row)
    console.print(porq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
