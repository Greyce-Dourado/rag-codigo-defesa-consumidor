-- Executado automaticamente na primeira subida do container (docker-entrypoint-initdb.d).
-- Habilita a extensão pgvector. As tabelas/índices são criados pela aplicação (src/rag/store),
-- para que o schema fique versionado em código e não escondido aqui.
CREATE EXTENSION IF NOT EXISTS vector;
