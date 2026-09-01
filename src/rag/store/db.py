"""Persistência no Postgres + pgvector: conexão e upsert dos chunks com seus embeddings."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from rag.chunking.article import Chunk
from rag.config import settings


def connect() -> psycopg.Connection:
    """Abre a conexão e registra o adaptador do pgvector (ensina o psycopg a mandar/receber
    o tipo `vector` do Postgres como lista Python)."""
    conn = psycopg.connect(settings.pg_dsn)
    register_vector(conn)
    return conn


# ON CONFLICT (id) DO UPDATE = "upsert": rodar de novo atualiza em vez de duplicar/quebrar.
_UPSERT = """
INSERT INTO chunks (id, artigo, sufixo, lei, titulo, capitulo, secao, texto,
                    n_paragrafos, n_incisos, alteracoes, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    texto      = EXCLUDED.texto,
    alteracoes = EXCLUDED.alteracoes,
    embedding  = EXCLUDED.embedding
"""


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """Insere (ou atualiza) os chunks com seus vetores. Retorna o total de linhas na tabela."""
    with connect() as conn, conn.cursor() as cur:
        for c, emb in zip(chunks, embeddings):
            cur.execute(
                _UPSERT,
                (
                    c.id, c.artigo, c.sufixo, c.lei, c.titulo, c.capitulo, c.secao,
                    c.texto, c.n_paragrafos, c.n_incisos, Jsonb(c.alteracoes), emb,
                ),
            )
        conn.commit()
        cur.execute("SELECT count(*) FROM chunks")
        return cur.fetchone()[0]
