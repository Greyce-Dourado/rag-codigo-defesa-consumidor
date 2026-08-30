"""Etapa 1 — Ingestão + chunking estrutural do CDC.

Fluxo: download (cacheado) -> parsing estrutural -> validação -> chunks -> data/processed/.
Imprime um relatório de validação (não só "funcionou").

Uso:
    python scripts/ingest.py            # usa cache se existir
    python scripts/ingest.py --force    # rebaixa o HTML
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.chunking.article import build_chunks
from rag.ingest.parser import parse_cdc, validar
from rag.ingest.planalto import download_cdc

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cdc.html"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"

console = Console()


def main(force: bool = False) -> int:
    console.print("[bold]1/3[/] baixando CDC do Planalto…")
    download_cdc(RAW, force=force)

    console.print("[bold]2/3[/] parsing estrutural…")
    artigos = parse_cdc(RAW)
    rel = validar(artigos)

    t = Table(title="Validação do parsing")
    t.add_column("métrica"); t.add_column("valor", justify="right")
    t.add_row("artigos extraídos", f"{rel['n_artigos']} / {rel['esperados']}")
    t.add_row("faltantes", str(rel["faltantes"]) or "—")
    t.add_row("inesperados", str(rel["inesperados"]) or "—")
    t.add_row("vetados (excluídos do corpus)", f"{len(rel['vetados'])} {rel['vetados']}")
    t.add_row("com parágrafos", str(rel["com_paragrafos"]))
    t.add_row("com incisos", str(rel["com_incisos"]))
    t.add_row("com alterações (proveniência)", str(rel["com_alteracoes"]))
    console.print(t)

    if rel["faltantes"]:
        console.print(f"[yellow]aviso:[/] artigos faltantes: {rel['faltantes']}")

    console.print("[bold]3/3[/] gerando chunks…")
    chunks = build_chunks(artigos)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    tam = [len(c.texto) for c in chunks]
    console.print(
        f"[green]ok[/] {len(chunks)} chunks -> {OUT.relative_to(ROOT)} "
        f"(texto: min={min(tam)}, méd={sum(tam)//len(tam)}, max={max(tam)} chars)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
