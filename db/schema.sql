-- Schema do corpus vetorizado do CDC.
-- Executar com:  psql -d rag -f db/schema.sql
-- Pré-requisito: extensão pgvector habilitada (CREATE EXTENSION vector).

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,   -- ex.: "art_49" — casa com o chunk e o evalset
    artigo        INT  NOT NULL,      -- número do artigo (para citação)
    sufixo        TEXT NOT NULL DEFAULT '',  -- ex.: "A" em 45-A (genérico; vazio no CDC)
    lei           TEXT NOT NULL,      -- fonte normativa
    titulo        TEXT NOT NULL DEFAULT '',   -- hierarquia: Título
    capitulo      TEXT NOT NULL DEFAULT '',   -- hierarquia: Capítulo
    secao         TEXT NOT NULL DEFAULT '',   -- hierarquia: Seção
    texto         TEXT NOT NULL,      -- texto normativo (citação + base do Full-Text Search)
    n_paragrafos  INT  NOT NULL DEFAULT 0,
    n_incisos     INT  NOT NULL DEFAULT 0,
    alteracoes    JSONB NOT NULL DEFAULT '[]',  -- proveniência das alterações legislativas

    -- Vetor denso do bge-m3. A dimensão é FIXA em 1024: um INSERT com tamanho diferente falha.
    embedding     VECTOR(1024),

    -- Coluna gerada para a busca LEXICAL da recuperação híbrida (Etapa 5).
    -- STORED = materializada em disco e mantida em sincronia com `texto` automaticamente.
    -- Config 'portuguese' aplica stemming/stopwords em PT-BR.
    texto_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('portuguese', texto)) STORED
);

-- Índice vetorial HNSW.
-- vector_cosine_ops: a operator class TEM que casar com o operador da query (<=> cosine),
-- senão o planner ignora o índice. m / ef_construction são os botões de recall x custo de build.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Índice GIN para o Full-Text Search (busca lexical rápida sobre o tsvector).
CREATE INDEX IF NOT EXISTS chunks_texto_tsv_gin
    ON chunks USING gin (texto_tsv);
