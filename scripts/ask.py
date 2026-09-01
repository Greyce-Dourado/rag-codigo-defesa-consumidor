"""Etapa 3 — pergunta -> recuperação densa -> resposta do Gemini com citação.

Uso:  ./.venv/bin/python scripts/ask.py "sua pergunta aqui"
Requer GEMINI_API_KEY no .env.
"""

from __future__ import annotations

import sys

from rich.console import Console

from rag.generation.answer import answer

console = Console()


def main() -> int:
    pergunta = " ".join(sys.argv[1:]) or "posso desistir de uma compra feita pela internet?"
    resp, chunks = answer(pergunta, k=5)

    console.print(f"[bold cyan]Pergunta:[/] {pergunta}\n")
    console.print(f"[bold green]Resposta:[/] {resp.resposta}\n")
    console.print(f"[bold]Artigos citados:[/] {', '.join(resp.artigos_citados) or '—'}")
    console.print(f"[dim]Recuperados (top-5): {', '.join(c['id'] for c in chunks)}[/]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
