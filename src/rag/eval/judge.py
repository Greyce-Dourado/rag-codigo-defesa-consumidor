"""Avaliação da geração.

- Acurácia de citação: OBJETIVA (compara artigos citados x ground truth).
- Groundedness: LLM-as-judge (Gemini), pois exige julgamento semântico.
"""

from __future__ import annotations

import re

from google import genai
from google.genai import types
from pydantic import BaseModel

from rag.config import settings

# --- Citação: checagem objetiva ------------------------------------------------------------

def _normaliza_citacao(texto: str) -> set[str]:
    """'Art. 49' / 'art. 54-G' -> {'art_49'} / {'art_54G'}, para comparar com o evalset."""
    ids = set()
    for m in re.finditer(r"[Aa]rt\.?\s*(\d+)\s*-?\s*([A-Za-z])?", texto):
        sufixo = (m.group(2) or "").upper()
        ids.add(f"art_{m.group(1)}{sufixo}")
    return ids


def citou_correto(artigos_citados: list[str], relevantes: set[str]) -> bool:
    """True se ALGUM artigo citado bate com o ground truth."""
    citados = set()
    for a in artigos_citados:
        citados |= _normaliza_citacao(a)
    return bool(citados & relevantes)


# --- Groundedness: LLM-as-judge ------------------------------------------------------------

_JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso e imparcial de respostas de um sistema de RAG jurídico. "
    "Recebe uma PERGUNTA, um CONTEXTO (artigos do CDC recuperados) e uma RESPOSTA gerada. "
    "Avalie APENAS a fidelidade da RESPOSTA ao CONTEXTO (groundedness): "
    "1.0 = toda afirmação da resposta é sustentada pelo contexto; "
    "0.5 = parcialmente sustentada; "
    "0.0 = a resposta afirma coisas que NÃO estão no contexto (alucinação). "
    "Ignore se a resposta é 'boa' em geral — só importa se está ancorada no contexto fornecido. "
    "Justifique em uma frase."
)


class Julgamento(BaseModel):
    groundedness: float
    justificativa: str


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada — preencha o .env.")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def judge_groundedness(pergunta: str, contexto: str, resposta: str) -> Julgamento:
    prompt = f"PERGUNTA:\n{pergunta}\n\nCONTEXTO:\n{contexto}\n\nRESPOSTA:\n{resposta}"
    resp = _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_JUDGE_SYSTEM,
            response_mime_type="application/json",
            response_schema=Julgamento,
            temperature=0,
        ),
    )
    return resp.parsed
