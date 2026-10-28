"""MPPG: paginação conservadora Scopus/WoS/SciELO, exclusivamente opt-in.

Não executa solicitações por conta própria; os adaptadores preexistentes fazem
as solicitações quando a busca externa for explicitamente executada. Sem
modificação do protocolo ou das decisões de triagem.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SUPPORTED = ("scopus", "wos", "scielo")
MARKER = "MPPG_LEGACY_PAGING_V2"

_WOS_LAST_REQUEST_AT: float | None = None


def _safe_log_url(url: str) -> str:
    """Nunca registra credenciais de URLs institucionais ou personalizadas."""
    parts = urlsplit(url)
    blocked = {"api_key", "apikey", "key", "token", "insttoken",
               "access_token", "auth", "authorization", "password",
               "email", "mailto", "x-api-key", "x-els-apikey"}
    pairs = [(key, "REDACTED" if key.casefold() in blocked else value)
             for key, value in parse_qsl(parts.query, keep_blank_values=True)]
    # Elimina também possíveis credenciais user:password@host.
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port is not None else "")
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(pairs), ""))


def _wait_wos_request() -> None:
    """Respeita o plano Starter gratuito (uma requisicao por segundo)."""
    global _WOS_LAST_REQUEST_AT
    now = time.monotonic()
    if _WOS_LAST_REQUEST_AT is not None:
        remaining = 1.05 - (now - _WOS_LAST_REQUEST_AT)
        if remaining > 0:
            time.sleep(remaining)
    _WOS_LAST_REQUEST_AT = time.monotonic()


def _count(value: Any) -> int | None:
    try:
        number = int(value)
    except (ValueError, TypeError):
        return None
    return number if number >= 0 else None


def _meta(base: str, payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    if base == "scopus":
        results = payload.get("search-results") or {}
        if isinstance(results, dict):
            return _count(results.get("opensearch:totalResults"))
    if base == "wos":
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            return _count(metadata.get("total"))
    return None


def _rewrite(url: str, base: str, page: int, size: int) -> str:
    parts = urlsplit(url)
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k not in ("start", "page", "from", "count", "limit", "db")]
    if base == "scopus":
        pairs.extend((("start", str(page * size)), ("count", str(size))))
    elif base == "wos":
        pairs.extend((("db", "WOS"), ("page", str(page + 1)), ("limit", str(size))))
    elif base == "scielo":
        # Sem restrições antecipadas à coleção brasileira e ao tipo artigo.
        pairs = [(k, v) for k, v in pairs
                 if k not in ("filter[in][]", "filter[type][]")]
        pairs.extend((("from", str(page * size)), ("page", str(page + 1)),
                      ("count", str(size))))
    else:
        raise ValueError("Fonte sem paginação habilitada.")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), parts.fragment))


def _matches(url: str, base: str) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.lower()
    if base == "scopus":
        return host == "api.elsevier.com" and path.endswith("/content/search/scopus")
    if base == "wos":
        return host == "api.clarivate.com" and path.endswith("/apis/wos-starter/v1/documents")
    if base == "scielo":
        # Permite rota institucional configurada, restringindo a troca a uma
        # requisição com a assinatura completa do adaptador SciELO original.
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        return (parsed.scheme == "https" and "q" in params
                and "filter[in][]" in params and "filter[type][]" in params
                and params.get("format") == "summary")
    return False


@contextmanager
def _page_request(ns: dict, base: str, page: int, size: int, observation: dict):
    name = "_request_text" if base == "scielo" else "_request_json"
    original = ns[name]

    def intercept(url: str, *args: Any, **kwargs: Any):
        if not _matches(url, base):
            return original(url, *args, **kwargs)
        effective = _rewrite(url, base, page, size)
        observation["intercepted"] = True
        observation["query_url"] = _safe_log_url(effective)
        if base == "wos":
            _wait_wos_request()
        value = original(effective, *args, **kwargs)
        observation["total"] = _meta(base, value)
        return value

    ns[name] = intercept
    try:
        yield
    finally:
        ns[name] = original


def _identity(row: dict, base: str) -> str:
    doi = str(row.get("doi") or "").casefold().strip()
    url = str(row.get("url") or "").casefold().strip()
    rid = str(row.get("id_registro") or "").casefold().strip()
    title = str(row.get("titulo") or "").casefold().strip()
    if doi:
        return "doi:" + doi
    if base == "scielo" and url:
        return "url:" + url
    if rid:
        return "id:" + rid
    if title:
        return "title:" + title
    return ""


def _pages(ns: dict, base: str, query: str, config: dict, original, calls: list):
    settings = config["__mppg_cobertura__"]
    variants = config.get("__mppg_variants__") or {}
    effective = variants.get(query, {}).get(base, query)
    limit = min(int(settings["limite_por_consulta"]), 10000)
    size = min(int(settings["tamanho_pagina"]), limit,
               min(25, int(config.get("scopus_per_query") or 10))
               if base == "scopus" else 50)
    max_pages = min(int(settings["max_paginas"]), 200)
    if limit <= 0 or size <= 0 or max_pages <= 0:
        raise ValueError("Limites de paginação inválidos.")

    seen: set[str] = set()
    records: list[dict] = []
    warnings: list[str] = []
    total: int | None = None
    page_count = 0
    complete = False
    last_log: dict = {}
    last_page_count = 0
    for page in range(max_pages):
        if len(records) >= limit:
            break
        if base == "scopus" and (page + 1) * size > 5000:
            warnings.append("Scopus: limite de 5000 registros com paginação por offset; necessário cursor ou particionamento.")
            break
        page_cfg = dict(config)
        page_cfg.pop("__mppg_cobertura__", None)
        page_cfg["per_base"] = size
        if base == "scopus":
            page_cfg["scopus_per_query"] = size
        observation: dict = {}
        if page:
            time.sleep(0.50)
        try:
            with _page_request(ns, base, page, size, observation):
                page_rows, last_log = original(effective, page_cfg)
        except Exception as exc:
            # Um bloqueio anti-automacao precisa interromper a FONTE inteira;
            # nao basta encerrar a pagina e continuar com a proxima consulta.
            unavailable = ns.get("ProviderUnavailableError")
            if isinstance(unavailable, type) and isinstance(exc, unavailable):
                raise
            if not page_count:
                raise
            warnings.append(f"Falha na página {page + 1}: {type(exc).__name__}; resultado parcial.")
            break
        page_count += 1
        if not observation.get("intercepted"):
            warnings.append("Endpoint não interceptado; paginação não comprovada.")
            break
        if not isinstance(page_rows, list):
            warnings.append("Resposta não apresentou lista de registros.")
            break
        last_page_count = len(page_rows)
        if observation.get("total") is not None:
            total = observation["total"]
        count_new = 0
        for idx, row in enumerate(page_rows, start=1):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if base == "scielo":
                # O parser original numera cada página a partir de 1.
                # Evita IDs artificiais repetidos entre páginas.
                identity_seed = (str(item.get("doi") or "").strip()
                                 or str(item.get("url") or "").strip()
                                 or (str(item.get("titulo") or "").strip() + "|" +
                                     str(item.get("ano") or "").strip()).strip("|")
                                 or f"{effective}|{page}|{idx}")
                item["id_registro"] = "scielo:" + sha256(
                    identity_seed.encode("utf-8")
                ).hexdigest()[:24]
            key = _identity(item, base)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            if len(records) >= limit:
                break
            records.append(item)
            count_new += 1
        if total is not None and len(records) >= total:
            complete = True
            break
        if not page_rows:
            if total == 0:
                complete = True
            elif total is None:
                warnings.append("Total não informado; esgotamento não comprovado.")
            else:
                warnings.append("Página vazia antes de recuperar o total informado.")
            break
        if len(page_rows) < size:
            if total is None:
                warnings.append("Página parcial sem total informado; esgotamento não comprovado.")
            else:
                warnings.append("Página menor que o tamanho requisitado e total não alcançado.")
            break
        if count_new == 0:
            warnings.append("Página sem novos identificadores; interrompida.")
            break

    if not complete:
        if total is None:
            warnings.append("Cobertura não demonstrada: provedor não informou total verificável.")
        elif len(records) < total:
            warnings.append("Cobertura truncada: recuperados menos registros que o total do provedor.")

    log = dict(last_log)
    if observation.get("query_url"):
        # Chave Scopus/WoS trafega exclusivamente em cabeçalho.
        log["url"] = observation["query_url"]
    log.update({"provider": base, "retrieved": len(records),
                "query_original": query, "query_effective": effective,
                "source_total_available": total, "pages_fetched": page_count,
                "page_size": size, "cap_per_query": limit,
                "coverage_exhausted": complete, "coverage_truncated": not complete,
                "coverage_warnings": list(dict.fromkeys(warnings))})
    calls.append({"fonte": base, "consulta_original": query,
                  "consulta_efetiva": effective, "total_disponivel": total,
                  "recuperados": len(records), "paginas": page_count,
                  "limite": limit, "esgotada": complete,
                  "truncada": not complete, "avisos": log["coverage_warnings"]})
    return records, log


def install_hooks(ns: dict) -> None:
    """Ativa paginação de bases legadas somente se cobertura estiver habilitada."""
    if ns.get("_mppg_legacy_paging_v2_instalada"):
        return
    if not ns.get("_mppg_cobertura_v16_instalada"):
        raise RuntimeError("Instale cobertura PRISMA antes da paginação legada.")
    try:
        from . import prisma_cobertura as coverage
    except ImportError:
        import prisma_cobertura as coverage
    calls = coverage.CALLS
    for base in SUPPORTED:
        old = ns.get("_" + base)
        if not callable(old):
            raise RuntimeError(f"Adaptador não localizado: {base}")

        def wrap(query, config, *, _base=base, _old=old):
            settings = config.get("__mppg_cobertura__") if isinstance(config, dict) else None
            if not settings or not settings.get("habilitada"):
                return _old(query, config)
            return _pages(ns, _base, query, config, _old, calls)

        wrap.__name__ = "_" + base
        ns["_" + base] = wrap
    ns["_mppg_legacy_paging_v2_instalada"] = True
