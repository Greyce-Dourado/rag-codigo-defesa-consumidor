"""Geração da resposta com Gemini, ancorada nos artigos recuperados (com citação)."""

from __future__ import annotations

from google import genai
from google.genai import types
from pydantic import BaseModel

from rag.config import settings
from rag.retrieval.search import dense_search

# System instruction: prende o modelo ao contexto recuperado. É a principal defesa contra
# alucinação — o modelo só pode responder com base nos artigos que passamos.
_SYSTEM = (
    "Você é um assistente jurídico especializado no Código de Defesa do Consumidor (CDC). "
    "Responda à pergunta USANDO EXCLUSIVAMENTE os artigos fornecidos no contexto. "
    "Cite explicitamente os artigos que embasam a resposta (ex.: 'Art. 49'). "
    "Se os artigos fornecidos não responderem à pergunta, diga claramente que não encontrou "
    "base no CDC fornecido. Nunca invente artigos nem use conhecimento externo."
)


class Resposta(BaseModel):
    """Saída estruturada: separa o texto da resposta das citações — isso torna a citação
    verificável por máquina, o que a avaliação (Etapa 6) vai usar."""

    resposta: str
    artigos_citados: list[str]


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY não configurada — preencha o .env.")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Art. {c['artigo']}] {c['texto']}" for c in chunks)


def answer(query: str, k: int = 5) -> tuple[Resposta, list[dict]]:
    chunks = dense_search(query, k=k)
    prompt = (
        f"Contexto (artigos do CDC recuperados):\n{_format_context(chunks)}\n\n"
        f"Pergunta: {query}"
    )
    resp = _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
            response_schema=Resposta,
            temperature=0,  # determinismo: mesma pergunta -> mesma resposta (reprodutível/avaliável)
        ),
    )
    return resp.parsed, chunks
