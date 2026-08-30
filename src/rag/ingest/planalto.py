"""Download do texto oficial do CDC no Planalto.

O Planalto reseta conexões de clientes sem User-Agent de browser, então enviamos um UA
explícito. O HTML é cacheado em data/raw/ para tornar a ingestão reprodutível sem depender
da rede a cada execução.
"""

from __future__ import annotations

from pathlib import Path

import httpx

CDC_URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078.htm"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)


def download_cdc(dest: Path, *, force: bool = False) -> Path:
    """Baixa o HTML do CDC para `dest` (cacheado). Retorna o caminho do arquivo.

    O conteúdo do Planalto é servido em latin-1; guardamos os bytes crus e deixamos o
    parser decidir o decode, para não perder informação de encoding.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    resp = httpx.get(CDC_URL, headers={"User-Agent": _UA}, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
