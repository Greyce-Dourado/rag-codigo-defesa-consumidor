# RAG sobre Legislação Brasileira (CDC)

RAG (Retrieval-Augmented Generation) canônico e de baixo custo sobre o **Código de Defesa do
Consumidor** (Lei 8.078/1990), construído para responder perguntas em linguagem natural
**citando o artigo** que fundamenta a resposta — e para **medir**, com métricas objetivas, se a
recuperação e a geração estão boas.

> Este repositório é tanto um projeto de estudo de RAG do zero quanto uma peça de portfólio. O
> foco não é "funcionar", é **entender e justificar cada decisão de arquitetura**. As seções de
> decisão abaixo são a parte mais importante do repo.
>
> _README em português; uma versão em inglês pode ser adicionada depois._

## Status

🚧 Em construção. Etapa atual: **esqueleto do projeto**. Ver [Roadmap](#roadmap).

---

## Por que este projeto existe

RAG canônico — chunking, embeddings, busca vetorial, recuperação seletiva top-k, re-ranking e
**avaliação da qualidade da recuperação** — em vez de *full context injection* (carregar tudo no
prompt). O objetivo é dominar a etapa de **busca semântica seletiva** e, principalmente, tratar a
**avaliação como cidadã de primeira classe**, não como um apêndice.

## Por que legislação brasileira

Texto jurídico tem **estrutura hierárquica nativa** (lei → artigo → parágrafo → inciso). Isso dá
duas vantagens que definem a arquitetura:

1. **Chunking não é chute.** A unidade semântica do domínio é o artigo — então o chunk é o
   artigo, não uma janela fixa de N tokens.
2. **Avaliação de recuperação é objetiva.** Cada artigo é uma unidade com ID, então o ground
   truth de "qual artigo responde a esta pergunta" é factual e mensurável.

---

## Arquitetura

```mermaid
flowchart LR
    A[CDC - Planalto/LexML] --> B[Parsing estrutural]
    B --> C[Chunking por artigo + metadados]
    C --> D[Embedding bge-m3]
    D --> E[(Postgres + pgvector<br/>HNSW + FTS)]
    Q[Pergunta] --> R[Retrieval<br/>dense / hybrid RRF]
    E --> R
    R --> K[Re-rank bge-reranker-v2-m3<br/>top-20 → top-5]
    K --> G[Geração Gemini<br/>resposta + citação]
    G --> ANS[Resposta com artigos citados]
    R -.->|ground truth| EV[Avaliação<br/>recall@k / MRR / nDCG]
    G -.->|LLM-as-judge| EV
```

### Decisões de arquitetura (o núcleo do portfólio)

| Camada | Escolha | Racional resumido |
|---|---|---|
| **Chunking** | Estrutural por artigo | A unidade semântica do domínio é o artigo; evita cortar no meio de uma norma. Metadados: lei, nº do artigo, §, inciso. |
| **Embedding** | `BAAI/bge-m3` (dense, 1024-d) | Melhor encoder denso multilíngue open-source, PT-BR forte, contexto longo (8k) para artigos extensos. Treinado para *retrieval*, não classificação. Plugável. |
| **Vector store** | Postgres + `pgvector` (HNSW) | Vetor como "mais uma coluna"; zero custo; um container; habilita hybrid sem serviço extra. |
| **Retrieval** | dense (baseline) → hybrid dense+lexical (RRF) | Lexical via Full-Text Search nativo do Postgres (`portuguese`); termos jurídicos exatos importam. Hybrid entra como **experimento medido**, não como suposição. |
| **Re-ranking** | `BAAI/bge-reranker-v2-m3` (cross-encoder) | Rerankeia top-20 → top-5. Valor **provado por avaliação**, não afirmado. |
| **Geração** | Gemini (free tier), system instructions + saída estruturada | Resposta ancorada nos artigos recuperados, com citação explícita. |
| **Avaliação** | recall@k / MRR / nDCG à mão + Gemini-as-judge | Métricas de recuperação escritas na mão (entender > importar). Qualidade da resposta: groundedness + acurácia de citação. RAGAS como cross-check opcional. |

Detalhamento e trade-offs completos (incluindo *quando eu NÃO usaria estas escolhas*): ver
[`docs/decisoes.md`](docs/decisoes.md) _(a escrever)_.

---

## Custo

Roda inteiro em **free tier / zero gasto**: embeddings e reranker open-source em CPU local,
Postgres em Docker, geração no free tier do Gemini. `docker compose up` + uma chave grátis do
Gemini e funciona — sem billing.

## Setup

```bash
# 1. Subir o Postgres com pgvector
docker compose up -d

# 2. Ambiente Python
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Configurar variáveis
cp .env.example .env   # e preencher GEMINI_API_KEY
```

## Uso _(a implementar por etapa)_

```bash
python scripts/ingest.py      # baixa e faz parsing estrutural do CDC
python scripts/index.py       # gera embeddings e indexa no pgvector
python scripts/ask.py "..."   # pergunta → resposta com citação
python scripts/evaluate.py    # roda a suíte de avaliação
```

## Roadmap

- [x] Esqueleto do projeto (estrutura, docker, config, README)
- [x] **Etapa 1** — Ingestão + chunking estrutural do CDC (108 artigos; 11 vetados excluídos)
- [ ] **Etapa 2** — Embeddings + indexação no pgvector (HNSW + FTS)
- [ ] **Etapa 3** — Retrieval (dense) + geração com citação (Gemini)
- [ ] **Etapa 4** — Suíte de avaliação de recuperação (recall@k / MRR / nDCG) + eval-set
- [ ] **Etapa 5** — Re-ranking + experimento comparativo (com/sem rerank; dense vs hybrid)
- [ ] **Etapa 6** — Avaliação de geração (groundedness, citação) via Gemini-as-judge
- [ ] Polimento: `docs/decisoes.md`, testes, README final

## Licença

MIT _(a definir)_.
