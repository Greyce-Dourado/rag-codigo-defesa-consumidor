"""Etapa 6 — avaliação da geração: acurácia de citação (objetiva) + groundedness (LLM-judge).

Para cada pergunta do evalset: roda o RAG completo (retrieval+rerank -> Gemini), checa se citou
o artigo correto e pede ao Gemini-juiz uma nota de fidelidade ao contexto.

Uso:  ./.venv/bin/python scripts/evaluate_generation.py
Requer GEMINI_API_KEY. Faz ~2 chamadas ao Gemini por pergunta.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from google.genai import errors
from rich.console import Console
from rich.table import Table

from rag.eval.judge import citou_correto, judge_groundedness
from rag.generation.answer import answer

ROOT = Path(__file__).resolve().parents[1]
EVALSET = ROOT / "evalset" / "questions.jsonl"

console = Console()


def _contexto(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Art. {c['artigo']}] {c['texto']}" for c in chunks)


def with_retry(fn, *, tries: int = 4):
    """Repete `fn` em erros transitórios do Gemini. 429 (rate limit) espera ~35s — a janela que
    o free tier pede; 503 (sobrecarga) usa backoff exponencial. Outros erros sobem na hora."""
    for i in range(tries):
        try:
            return fn()
        except errors.APIError as e:
            if e.code == 429 and i < tries - 1:
                espera = 35.0
            elif e.code == 503 and i < tries - 1:
                espera = 2.0 * (2 ** i)
            else:
                raise
            console.print(f"  [yellow]{e.code} — aguardando {espera:.0f}s…[/]")
            time.sleep(espera)


def main() -> int:
    questions = [json.loads(line) for line in EVALSET.open(encoding="utf-8")]
    # Amostra por causa da cota do free tier (padrão 6; passe outro número no argv).
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    questions = questions[:limite]
    n = len(questions)
    console.print(f"[dim]avaliando {n} perguntas (amostra; free tier ~20 req)[/]")

    tabela = Table(title=f"Avaliação da geração — N={n}")
    tabela.add_column("pergunta")
    tabela.add_column("esperado")
    tabela.add_column("citou?", justify="center")
    tabela.add_column("grounded", justify="right")

    linhas = []  # guarda os resultados para persistir em arquivo
    for i, q in enumerate(questions, 1):
        console.print(f"  [dim]({i}/{n}) {q['id']}…[/]")
        resp, chunks = with_retry(lambda: answer(q["pergunta"], k=5))
        rel = set(q["artigos_relevantes"])

        cit_ok = citou_correto(resp.artigos_citados, rel)
        julg = with_retry(lambda: judge_groundedness(q["pergunta"], _contexto(chunks), resp.resposta))

        linhas.append({
            "id": q["id"], "esperado": sorted(rel)[0], "citou": cit_ok,
            "citados": resp.artigos_citados, "groundedness": julg.groundedness,
            "justificativa": julg.justificativa,
        })
        tabela.add_row(
            q["id"], sorted(rel)[0],
            "[green]sim[/]" if cit_ok else "[red]não[/]",
            f"{julg.groundedness:.2f}",
        )
        time.sleep(1)  # respeita o rate limit do free tier

    acertos_cit = sum(r["citou"] for r in linhas)
    ground_media = sum(r["groundedness"] for r in linhas) / n

    console.print(tabela)
    console.print(f"\n[bold]Acurácia de citação:[/] {acertos_cit}/{n} = {acertos_cit / n:.3f}")
    console.print(f"[bold]Groundedness média:[/] {ground_media:.3f}")

    # Persiste os resultados (não some se o terminal fechar; serve de artefato pro README).
    out = ROOT / "results" / "generation_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"n": n, "acuracia_citacao": acertos_cit / n, "groundedness_media": ground_media,
             "por_pergunta": linhas},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[dim]resultados salvos em {out.relative_to(ROOT)}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
