"""Chunking estrutural: 1 artigo = 1 chunk.

O texto embeddado é o artigo normativo limpo (caput + §§ + incisos, sem anotações de
alteração). Título/Capítulo/Seção e a proveniência das alterações viram metadados — úteis
para filtro, citação e futuras variações de avaliação.
"""

from __future__ import annotations

from pydantic import BaseModel

from rag.ingest.parser import Artigo

LEI = "Lei 8.078/1990 (CDC)"


class Chunk(BaseModel):
    id: str  # ex.: "art_49" — casa com o schema do evalset
    lei: str
    artigo: int
    sufixo: str
    titulo: str
    capitulo: str
    secao: str
    texto: str
    n_paragrafos: int
    n_incisos: int
    alteracoes: list[str]

    @property
    def texto_para_embedding(self) -> str:
        """Prefixamos o contexto hierárquico ao texto: ajuda o modelo a desambiguar artigos
        parecidos de capítulos diferentes e melhora o match lexical na busca híbrida."""
        contexto = " > ".join(p for p in (self.capitulo, self.secao) if p)
        return f"{contexto}\n{self.texto}" if contexto else self.texto


def build_chunks(artigos: list[Artigo]) -> list[Chunk]:
    """Converte artigos em chunks. Artigos vetados são excluídos do corpus: não têm conteúdo
    normativo e só poluiriam a recuperação."""
    return [
        Chunk(
            id=a.id,
            lei=LEI,
            artigo=a.numero,
            sufixo=a.sufixo,
            titulo=a.titulo,
            capitulo=a.capitulo,
            secao=a.secao,
            texto=a.texto,
            n_paragrafos=a.n_paragrafos,
            n_incisos=a.n_incisos,
            alteracoes=a.alteracoes,
        )
        for a in artigos
        if not a.vetado
    ]
