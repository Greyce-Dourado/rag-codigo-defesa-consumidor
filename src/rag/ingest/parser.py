"""Parser estrutural do HTML do CDC (Planalto).

Estratégia: como é um documento único e estável, um parser determinístico ancorado em
regex nos marcadores legais ("Art. N", "§", incisos, TÍTULO/CAPÍTULO/SEÇÃO) é mais legível
e defensável do que um parser genérico. Passos:

1. decode latin-1 + unescape de entidades HTML;
2. converte tags de bloco (<p>, <br>, </div>...) em quebras de linha e remove o resto;
3. normaliza espaços intra-linha (tabs/CRLF) preservando a estrutura de linhas;
4. captura e REMOVE as anotações de alteração legislativa (guardadas como proveniência);
5. varre as linhas mantendo o contexto Título/Capítulo/Seção e agrega o corpo de cada artigo;
6. valida a sequência 1..119 e sinaliza faltantes/duplicados.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from pydantic import BaseModel

# Anotações de alteração legislativa: "(Redação dada pela Lei...)", "(Vide ...)", "(Vetado)",
# "(Incluído...)", "(Revogado...)". Podem conter espaços/tabs mas, após a normalização de
# whitespace, ficam em linha única — então [^)]* basta.
_ANNOT_RE = re.compile(
    r"\(\s*(?:Reda[çc][aã]o dada|Vide|Vetado|Inclu[íi]d[oa]|Revogad[oa]|Mensagem de veto|"
    r"Regulamento|Vig[êe]ncia)[^)]*\)",
    re.IGNORECASE,
)
_ART_RE = re.compile(r"^Art\.\s*(\d+)\s*[º°.]?\s*(?:[-–]\s*([A-Z]))?", re.IGNORECASE)
_TITULO_RE = re.compile(r"^T[ÍI]TULO\s+([IVXLC]+)", re.IGNORECASE)
_CAPITULO_RE = re.compile(r"^CAP[ÍI]TULO\s+([IVXLC]+)", re.IGNORECASE)
_SECAO_RE = re.compile(r"^SE[ÇC][ÃA]O\s+([IVXLC]+)", re.IGNORECASE)
_PARAGRAFO_RE = re.compile(r"^(§\s*\d+|Par[áa]grafo [úu]nico)", re.IGNORECASE)
_INCISO_RE = re.compile(r"^[IVXLC]+\s*[-–]")

CDC_TOTAL_ARTIGOS = 119


class Artigo(BaseModel):
    numero: int
    sufixo: str = ""  # ex.: "A" em "Art. 45-A" (o CDC não usa, mas o parser é genérico)
    titulo: str = ""
    capitulo: str = ""
    secao: str = ""
    texto: str  # texto normativo limpo (caput + §§ + incisos), sem anotações
    n_paragrafos: int = 0
    n_incisos: int = 0
    vetado: bool = False  # artigo que existe na numeração mas foi vetado (sem conteúdo)
    alteracoes: list[str] = []  # proveniência: anotações removidas do texto

    @property
    def id(self) -> str:
        return f"art_{self.numero}{self.sufixo}"


def _to_lines(raw_bytes: bytes) -> list[str]:
    """HTML cru -> lista de linhas de texto limpas (sem tags, whitespace normalizado)."""
    text = raw_bytes.decode("latin-1")
    # tags de bloco viram quebra de linha; o resto some
    text = re.sub(r"<\s*(br|/p|/div|/tr|/h[1-6]|/li)\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\xa0]+", " ", line).strip()  # colapsa espaços, tabs e NBSP
        if line:
            lines.append(line)
    return lines


def parse_cdc(html_path: Path) -> list[Artigo]:
    lines = _to_lines(html_path.read_bytes())

    artigos: dict[int, Artigo] = {}
    ctx_titulo = ctx_capitulo = ctx_secao = ""
    cur: Artigo | None = None
    buf: list[str] = []

    def flush():
        nonlocal cur, buf
        if cur is None:
            return
        body = " ".join(buf)
        alteracoes = [re.sub(r"\s+", " ", m).strip() for m in _ANNOT_RE.findall(body)]
        clean = _ANNOT_RE.sub("", body)
        clean = re.sub(r"\s+", " ", clean).strip()
        cur.texto = clean
        cur.alteracoes = alteracoes
        # sem texto normativo depois de remover o marcador "Art. N" => artigo vetado
        corpo = _ART_RE.sub("", clean, count=1).strip(" .-–")
        cur.vetado = len(corpo) < 3
        # contagem de sub-estrutura a partir dos marcadores no corpo
        cur.n_paragrafos = len(re.findall(r"§\s*\d+|Par[áa]grafo [úu]nico", clean, re.IGNORECASE))
        cur.n_incisos = len(re.findall(r"(?:^|\s)[IVXLC]+\s*[-–]\s", clean))
        # mantém o artigo mais longo em caso de duplicata (evita falsos positivos curtos)
        prev = artigos.get(cur.numero)
        if prev is None or len(clean) > len(prev.texto):
            artigos[cur.numero] = cur
        cur, buf = None, []

    for line in lines:
        if _TITULO_RE.match(line):
            flush(); ctx_titulo, ctx_capitulo, ctx_secao = line, "", ""; continue
        if _CAPITULO_RE.match(line):
            flush(); ctx_capitulo, ctx_secao = line, ""; continue
        if _SECAO_RE.match(line):
            flush(); ctx_secao = line; continue

        m = _ART_RE.match(line)
        if m:
            flush()
            cur = Artigo(
                numero=int(m.group(1)),
                sufixo=(m.group(2) or "").upper(),
                titulo=ctx_titulo,
                capitulo=ctx_capitulo,
                secao=ctx_secao,
                texto="",
            )
            buf = [line]
        elif cur is not None:
            buf.append(line)
    flush()

    return [artigos[n] for n in sorted(artigos)]


def validar(artigos: list[Artigo]) -> dict:
    """Relatório de validação: nunca escondemos falhas de parsing."""
    numeros = {a.numero for a in artigos}
    esperados = set(range(1, CDC_TOTAL_ARTIGOS + 1))
    return {
        "n_artigos": len(artigos),
        "esperados": CDC_TOTAL_ARTIGOS,
        "faltantes": sorted(esperados - numeros),
        "inesperados": sorted(numeros - esperados),
        "vetados": sorted(a.numero for a in artigos if a.vetado),
        "com_paragrafos": sum(1 for a in artigos if a.n_paragrafos),
        "com_incisos": sum(1 for a in artigos if a.n_incisos),
        "com_alteracoes": sum(1 for a in artigos if a.alteracoes),
    }
