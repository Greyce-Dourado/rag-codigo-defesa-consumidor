"""Etapa 4 — avalia a recuperação densa contra o evalset (ground truth por artigo).

Roda cada pergunta pela busca densa, calcula Hit/Recall/nDCG @k e MRR, e mostra a posição do
1º artigo relevante por pergunta (pra investigar acertos e erros).

Uso:  ./.venv/bin/python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.eval.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rag.retrieval.search import dense_search

ROOT = Path(__file__).resolve().parents[1]
EVALSET = ROOT / "evalset" / "questions.jsonl"

KS = [1, 3, 5, 10]
RETRIEVE_K = 10

console = Console()


def main() -> int:
    questions = [json.loads(line) for line in EVALSET.open(encoding="utf-8")]

    # Recupera uma vez por pergunta (top-RETRIEVE_K) e guarda o ranking de ids.
    resultados = []
    for q in questions:
        ranked = [r["id"] for r in dense_search(q["pergunta"], k=RETRIEVE_K)]
        resultados.append((q, ranked, set(q["artigos_relevantes"])))

    n = len(resultados)
    table = Table(title=f"Recuperação densa (bge-m3) — N={n}")
    table.add_column("métrica")
    for k in KS:
        table.add_column(f"@{k}", justify="right")

    for nome, fn in [("Hit rate", hit_at_k), ("Recall", recall_at_k), ("nDCG", ndcg_at_k)]:
        row = [nome]
        for k in KS:
            media = sum(fn(ranked, rel, k) for _, ranked, rel in resultados) / n
            row.append(f"{media:.3f}")
        table.add_row(*row)

    console.print(table)
    mrr = sum(reciprocal_rank(ranked, rel) for _, ranked, rel in resultados) / n
    console.print(f"[bold]MRR:[/] {mrr:.3f}\n")

    console.print("[bold]Posição do 1º artigo relevante por pergunta:[/]")
    for q, ranked, rel in resultados:
        pos = next((i for i, rid in enumerate(ranked, 1) if rid in rel), None)
        marca = f"rank {pos}" if pos else "[red]FORA do top-10[/]"
        console.print(f"  {q['id']}  (esperado {sorted(rel)[0]}): {marca}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
