"""Etapa 2 — gera embeddings dos chunks e indexa no pgvector.

Pré-requisitos: `python scripts/ingest.py` já rodou (data/processed/chunks.jsonl existe) e o
Postgres está no ar com o schema criado (db/schema.sql).

Uso:  ./.venv/bin/python scripts/index.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from rag.chunking.article import Chunk
from rag.embeddings.encoder import embed
from rag.store.db import upsert_chunks

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"

console = Console()


def main() -> int:
    chunks = [Chunk(**json.loads(line)) for line in CHUNKS.open(encoding="utf-8")]

    console.print(f"[bold]1/2[/] gerando embeddings de {len(chunks)} chunks…")
    # texto_para_embedding prefixa Capítulo/Seção ao texto (contexto ajuda a busca).
    vecs = embed([c.texto_para_embedding for c in chunks], show_progress=True)

    console.print("[bold]2/2[/] inserindo no pgvector…")
    total = upsert_chunks(chunks, vecs)

    console.print(f"[green]ok[/] {total} chunks no banco")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
