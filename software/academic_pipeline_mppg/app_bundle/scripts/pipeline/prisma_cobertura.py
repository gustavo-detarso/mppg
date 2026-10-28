#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPPG PRISMA: paginação, estratégias específicas por base e auditoria técnica.

Hooks opt-in, carregados junto ao módulo nativo, sem dependências extras. Por
padrão a busca existente permanece exatamente inalterada. Não triamos estudos.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import hashlib
import json
import os
import re
import tempfile
import tomllib

MARKER = 'MPPG_COBERTURA_V1_6'
SUPPORTED = ('openalex', 'crossref', 'pubmed', 'europe_pmc')
LEGACY = ('semantic_scholar', 'scopus', 'wos', 'scielo', 'core')
KNOWN = SUPPORTED + LEGACY
CALLS: list[dict[str, Any]] = []


def _integer(value: Any, default: int, low: int, high: int) -> int:
    try:
        x = int(value)
    except (ValueError, TypeError):
        x = default
    return max(low, min(high, x))


def _flag(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).casefold() in {'true', '1', 'sim', 'yes'} if value is not None else default


def _get_section(cfg: dict, key: str) -> dict:
    value = cfg.get(key)
    return value if isinstance(value, dict) else {}


def _search_settings(cfg: dict) -> tuple[dict, dict]:
    search = _get_section(cfg, 'busca_prisma')
    coverage = _get_section(search, 'cobertura')
    # Explicit opt-in, fail closed on unconfigured projects.
    enabled = _flag(coverage.get('habilitada'), False)
    return search, {
        'habilitada': enabled,
        'limite_por_consulta': _integer(coverage.get('limite_por_consulta'), 200, 1, 10000),
        'tamanho_pagina': _integer(coverage.get('tamanho_pagina'), 50, 1, 100),
        'max_paginas': _integer(coverage.get('max_paginas'), 10, 1, 200),
        'auditoria': _flag(coverage.get('auditoria'), True),
    }


def _strategy_from_profile(cfg: dict) -> dict:
    """Carrega o arquivo da estratégia configurada, se presente; não o edita."""
    search = _get_section(cfg, 'busca_prisma')
    raw = search.get('estrategia_busca_path')
    if not isinstance(raw, str) or not raw.strip():
        return search
    base = Path(str(cfg.get('__config_dir__') or os.getcwd()))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f'Estratégia PRISMA configurada não encontrada: {path}')
    parsed = tomllib.loads(path.read_text(encoding='utf-8'))
    ext = _get_section(parsed, 'busca_prisma')
    return {**search, **ext}


def _strategy_variants(cfg: dict) -> dict[str, dict[str, str]]:
    """Mapeia texto original -> {base: texto específico}, sem substituir outros eixos.

    Estrutura TOML sugerida:
    [[busca_prisma.estrategias]]
    id='...'
    consultas=['genérica 1', 'genérica 2']
    [busca_prisma.estrategias.consultas_por_base]
    pubmed=['termo[tiab]', '...']
    """
    section = _strategy_from_profile(cfg)
    strategies = section.get('estrategias') or []
    if not isinstance(strategies, list):
        raise ValueError('busca_prisma.estrategias deve ser uma lista.')
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for s in strategies:
        if not isinstance(s, dict):
            continue
        queries = s.get('consultas') or []
        variants = s.get('consultas_por_base') or {}
        if not isinstance(queries, list) or not isinstance(variants, dict):
            raise ValueError('consultas e consultas_por_base precisam ser lista e tabela.')
        for base, values in variants.items():
            if base not in KNOWN:
                raise ValueError(f'Base de estratégia não suportada: {base}')
            if not isinstance(values, list) or len(values) != len(queries):
                raise ValueError(f'{base}: consultas_por_base exige uma consulta por índice da lista original.')
            for generic, specific in zip(queries, values):
                orig, adapted = str(generic).strip(), str(specific).strip()
                if not orig or not adapted:
                    raise ValueError(f'{base}: consultas vazias são proibidas.')
                if base in out[orig] and out[orig][base] != adapted:
                    raise ValueError(f'{base}: consulta genérica duplicada com traduções conflitantes: {orig}')
                out[orig][base] = adapted
    # Formato simples, acrescentável ao fim de um TOML já existente:
    # [[busca_prisma.consultas_especificas]]
    # original = 'query genérica'
    # pubmed = '(..)[tiab]'
    extra = section.get('consultas_especificas') or []
    if not isinstance(extra, list):
        raise ValueError('busca_prisma.consultas_especificas deve ser uma lista.')
    for item in extra:
        if not isinstance(item, dict):
            raise ValueError('Consulta específica deve ser uma tabela.')
        orig = str(item.get('original') or '').strip()
        if not orig:
            raise ValueError('consulta_especifica.original é obrigatório.')
        for base, specific in item.items():
            if base == 'original':
                continue
            if base not in KNOWN:
                raise ValueError(f'Base de consulta_especifica não reconhecida: {base}')
            adapted = str(specific or '').strip()
            if not adapted:
                raise ValueError(f'Consulta específica vazia: {base}')
            if base in out[orig] and out[orig][base] != adapted:
                raise ValueError(f'Conflito entre estratégias da base {base}: {orig}')
            out[orig][base] = adapted
    return dict(out)


def _rewrite(url: str, **edits: Any) -> str:
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in edits.items():
        if value is None:
            params.pop(key, None)
        else:
            params[key] = str(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def _provider_url(base: str, url: str) -> bool:
    u = urlsplit(url)
    host, path = u.hostname or '', u.path.lower()
    return (
        (base == 'crossref' and host == 'api.crossref.org' and path.endswith('/works')) or
        (base == 'openalex' and host == 'api.openalex.org' and path.endswith('/works')) or
        (base == 'pubmed' and host == 'eutils.ncbi.nlm.nih.gov' and path.endswith('/esearch.fcgi')) or
        (base == 'europe_pmc' and host.endswith('ebi.ac.uk') and '/search' in path)
    )


def _response_meta(base: str, payload: Any) -> tuple[int | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    if base == 'openalex':
        meta = _get_section(payload, 'meta')
        return _as_count(meta.get('count')), None
    if base == 'crossref':
        msg = _get_section(payload, 'message')
        return _as_count(msg.get('total-results')), str(msg.get('next-cursor') or '') or None
    if base == 'pubmed':
        return _as_count(_get_section(payload, 'esearchresult').get('count')), None
    if base == 'europe_pmc':
        return _as_count(payload.get('hitCount')), str(payload.get('nextCursorMark') or '') or None
    return None, None


def _as_count(value: Any) -> int | None:
    try:
        n = int(value)
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


@contextmanager
def _paged_request(ns: dict, base: str, *, page: int, size: int, cursor: str | None,
                   observation: dict[str, Any]):
    original = ns['_request_json']
    def intercept(url: str, *args: Any, **kwargs: Any) -> Any:
        if not _provider_url(base, url):
            return original(url, *args, **kwargs)
        if base == 'openalex':
            effective = _rewrite(url, per_page=size, page=page + 1, cursor=None, **{'per-page': None})
        elif base == 'crossref':
            effective = _rewrite(url, rows=size, cursor=cursor or '*', offset=None)
        elif base == 'pubmed':
            effective = _rewrite(url, retmax=size, retstart=page * size)
        else:
            effective = _rewrite(url, pageSize=size, cursorMark=cursor or '*', page=None)
        payload = original(effective, *args, **kwargs)
        total, next_cursor = _response_meta(base, payload)
        observation['total'] = total
        observation['next_cursor'] = next_cursor
        observation['url_consulta'] = _masked(effective)
        observation['consulta_paginada'] = True
        return payload
    ns['_request_json'] = intercept
    try:
        yield
    finally:
        ns['_request_json'] = original


def _masked(url: str) -> str:
    parts = urlsplit(url)
    tokens = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key in ('api_key', 'apiKey', 'key', 'token', 'email', 'mailto'):
        if key in tokens:
            tokens[key] = '[REDACTED]'
    # cursors are opaque and long; their raw values are unnecessary in audit.
    if 'cursor' in tokens and tokens['cursor'] != '*':
        tokens['cursor'] = '[OPAQUE]'
    if 'cursorMark' in tokens and tokens['cursorMark'] != '*':
        tokens['cursorMark'] = '[OPAQUE]'
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(tokens), ''))


def _record_identity(item: dict) -> str:
    doi = str(item.get('doi') or '').lower().strip()
    return doi or str(item.get('id_registro') or '').strip() or str(item.get('titulo') or '').strip()


def _fetch_pages(ns: dict, base: str, original: Any, query: str, cfg: dict, settings: dict) -> tuple[list, dict]:
    variants = cfg.get('__mppg_variants__') or {}
    effective_query = variants.get(query, {}).get(base, query)
    limit = settings['limite_por_consulta']
    page_size = min(settings['tamanho_pagina'], limit)
    max_pages = settings['max_paginas']
    records: list[dict] = []
    seen: set[str] = set()
    count: int | None = None
    cursor: str | None = None
    warnings: list[str] = []
    pages = 0
    exhausted = False
    for page in range(max_pages):
        remaining = limit - len(records)
        if remaining <= 0:
            break
        # Tamanho constante mantém offsets corretos em PubMed/OpenAlex.
        size = page_size
        if base == 'pubmed' and page * page_size >= 10000:
            warnings.append('PubMed: ESearch limita acesso aos primeiros 10000 PMIDs; particione a consulta por data.')
            break
        if base == 'openalex' and (page + 1) * page_size > 10000:
            warnings.append('OpenAlex: paginação page excederia 10000; consulte por recorte ou use cursor com API key.')
            break
        observation: dict[str, Any] = {}
        page_cfg = {**cfg, 'per_base': size}
        try:
            with _paged_request(ns, base, page=page, size=size, cursor=cursor, observation=observation):
                page_records, provider_log = original(effective_query, page_cfg)
        except Exception as exc:
            if not pages:
                raise
            warnings.append(f'Página {page + 1} não recuperada: {type(exc).__name__} (consulta parcial).')
            break
        pages += 1
        if observation.get('total') is not None:
            count = observation['total']
        next_cursor = observation.get('next_cursor')
        new_items = 0
        for item in page_records:
            key = _record_identity(item)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            if len(records) >= limit:
                break
            records.append(item)
            new_items += 1
        if provider_log.get('status_detail'):
            warnings.append(str(provider_log['status_detail']))
        if not observation.get('consulta_paginada'):
            warnings.append('Endpoint esperado não foi interceptado: paginação não verificável.')
            break
        if count is not None and len(records) >= count:
            exhausted = True
            break
        if not page_records or len(page_records) < size:
            exhausted = count is None or len(records) >= count
            if not exhausted:
                warnings.append('Página menor que o solicitado, apesar de haver mais resultados conforme o total informado.')
            break
        if base in ('crossref', 'europe_pmc'):
            if not next_cursor or next_cursor == cursor:
                # If total confirms more results, this is incomplete, not exhaustion.
                exhausted = count is not None and len(records) >= count
                if not exhausted:
                    warnings.append('Cursor ausente/repetido com resultados possivelmente restantes.')
                break
            cursor = next_cursor
        if new_items == 0:
            warnings.append('Página sem novos identificadores; interrompida para evitar loop.')
            break
    # The default if provider does not return total is unknown, never claim exhaustion
    complete = exhausted and (count is None or len(records) >= count or len(page_records) < size)
    truncated = (count is not None and len(records) < count) or (not complete)
    log = dict(provider_log) if pages else {'provider': base, 'retrieved': 0, 'error': 'sem páginas'}
    log.update({
        'retrieved': len(records), 'query_original': query, 'query_effective': effective_query,
        'source_total_available': count, 'pages_fetched': pages,
        'page_size': page_size, 'cap_per_query': limit,
        'coverage_exhausted': bool(complete), 'coverage_truncated': bool(truncated),
        'coverage_warnings': warnings,
    })
    CALLS.append({'fonte': base, 'consulta_original': query, 'consulta_efetiva': effective_query,
                  'total_disponivel': count, 'recuperados': len(records), 'paginas': pages,
                  'limite': limit, 'esgotada': bool(complete), 'truncada': bool(truncated), 'avisos': warnings})
    return records, log


def _resolve_paths(cfg: dict) -> tuple[Path, str]:
    paths = _get_section(cfg, 'paths')
    base = Path(str(cfg.get('__config_dir__') or os.getcwd()))
    output = Path(str(paths.get('research_output_dir') or '../../output/pesquisa')).expanduser()
    if not output.is_absolute():
        output = base / output
    prefix = str(paths.get('research_prefix') or '').strip()
    if not prefix or '/' in prefix or '\\' in prefix:
        raise ValueError('paths.research_prefix ausente/inválido; auditoria não criada.')
    if _flag(paths.get('create_research_subdir'), True):
        output = output / prefix
    return output, prefix


def _audit_records(records: list[dict]) -> dict:
    source_counts: Counter[str] = Counter()
    abstract_counts: Counter[str] = Counter()
    no_source = 0
    no_query = 0
    title_counts = Counter()
    doi_counts = Counter()
    for row in records:
        sources = row.get('fontes') or row.get('fonte') or []
        sources = sources if isinstance(sources, list) else [sources]
        sources = [str(v).strip().lower() for v in sources if str(v).strip()]
        if not sources:
            no_source += 1
        if not row.get('consultas_busca'):
            no_query += 1
        for source in set(sources):
            source_counts[source] += 1
            if str(row.get('resumo') or '').strip():
                abstract_counts[source] += 1
        title = re.sub(r'\W+', ' ', str(row.get('titulo') or '').casefold()).strip()
        if title:
            title_counts[title] += 1
        doi = str(row.get('doi') or '').strip().casefold().removeprefix('https://doi.org/')
        if doi:
            doi_counts[doi] += 1
    return {'registros_unicos': len(records),
            'com_resumo': sum(bool(str(row.get('resumo') or '').strip()) for row in records),
            'sem_fonte': no_source, 'sem_consulta': no_query,
            'por_fonte': dict(sorted(source_counts.items())),
            'resumos_por_fonte': dict(sorted(abstract_counts.items())),
            'repeticoes_de_titulo': sum(v - 1 for v in title_counts.values() if v > 1),
            'repeticoes_de_doi': sum(v - 1 for v in doi_counts.values() if v > 1)}


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.prisma_cobertura_', delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_audit(cfg: dict, calls: list[dict], run_result: Any) -> tuple[Path, Path] | None:
    output, prefix = _resolve_paths(cfg)
    target = output / f'{prefix}.candidatos_deduplicados.json'
    if not target.is_file():
        # Never invent data; caller should know that no audit was produced.
        return None
    raw = target.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise TypeError('JSON deduplicado não contém lista.')
    stats = _audit_records(records)
    alerts = []
    for call in calls:
        if call['truncada']:
            alerts.append({'tipo': 'RECUPERACAO_INCOMPLETA', 'fonte': call['fonte'],
                           'consulta': call['consulta_original'], 'recuperados': call['recuperados'],
                           'total_disponivel': call['total_disponivel']})
        if call['recuperados'] == 0:
            alerts.append({'tipo': 'CONSULTA_SEM_RETORNO', 'fonte': call['fonte'],
                           'consulta': call['consulta_original']})
    if stats['sem_fonte']:
        alerts.append({'tipo': 'PROVENIENCIA_AUSENTE', 'total': stats['sem_fonte']})
    if stats['sem_consulta']:
        alerts.append({'tipo': 'CONSULTA_AUSENTE', 'total': stats['sem_consulta']})
    if stats['repeticoes_de_titulo']:
        alerts.append({'tipo': 'TITULOS_COINCIDENTES_PARA_REVISAO', 'total': stats['repeticoes_de_titulo']})
    if stats['registros_unicos'] and stats['com_resumo'] * 2 < stats['registros_unicos']:
        alerts.append({'tipo': 'MENOS_DE_METADE_COM_RESUMO', 'total': stats['com_resumo']})
    for provider, n in stats['por_fonte'].items():
        has = stats['resumos_por_fonte'].get(provider, 0)
        if n >= 5 and has * 2 < n:
            alerts.append({'tipo': 'POUCOS_RESUMOS_NA_FONTE', 'fonte': provider, 'com_resumo': has, 'total': n})
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    data = {'schema_version': '1.0', 'sistema': MARKER, 'momento_utc': now,
            'arquivo': str(target), 'arquivo_sha256': hashlib.sha256(raw).hexdigest(),
            'contagens_prisma_modificadas': False, 'triagem_humana_modificada': False,
            'estatisticas': stats, 'consultas': calls, 'alertas': alerts}
    summary = ['# Auditoria técnica da cobertura PRISMA', '',
               f'- Execução UTC: {now}', f'- Registros únicos: **{stats["registros_unicos"]}**',
               f'- Com resumo: **{stats["com_resumo"]}**',
               f'- Consultas executadas: **{len(calls)}**',
               f'- Consultas com recuperação incompleta: **{sum(bool(c["truncada"]) for c in calls)}**',
               '- Este relatório não substitui a triagem humana e não estima recall.', '',
               '## Cobertura por consulta', '',
               '| Fonte | Recuperados | Total informado | Páginas | Completa? |',
               '|---|---:|---:|---:|---|']
    for c in calls:
        total = 'não informado' if c['total_disponivel'] is None else str(c['total_disponivel'])
        summary.append(f'| {c["fonte"]} | {c["recuperados"]} | {total} | {c["paginas"]} | {"não" if c["truncada"] else "sim"} |')
    summary += ['', '## Alertas', '']
    if not alerts:
        summary.append('Nenhum alerta produzido pelas verificações estruturais.')
    for alert in alerts:
        summary.append(f'- **{alert["tipo"]}**: {json.dumps(alert, ensure_ascii=False)}')
    summary.append('')
    # Writes only novel, timestamped artifacts; does not edit corpus or historical reports.
    out_json = output / f'{prefix}.auditoria_cobertura_{now}.json'
    out_md = output / f'{prefix}.auditoria_cobertura_{now}.md'
    if out_json.exists() or out_md.exists():
        raise FileExistsError('Colisão do identificador de auditoria; tente novamente.')
    _atomic_write(out_json, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())
    _atomic_write(out_md, ('\n'.join(summary) + '\n').encode())
    return out_json, out_md


def install_hooks(ns: dict) -> None:
    """Instala a integração no módulo nativo, chamada uma vez após definições."""
    if ns.get('_mppg_cobertura_v16_instalada'):
        return
    mandatory = ('_query_config', '_request_json', 'run_external_prisma_search') + SUPPORTED
    for name in mandatory:
        fn_name = '_' + name if name in SUPPORTED else name
        if not callable(ns.get(fn_name)):
            raise RuntimeError(f'Função de integração ausente: {fn_name}')
    original_config = ns['_query_config']
    original_run = ns['run_external_prisma_search']
    native = {source: ns['_' + source] for source in SUPPORTED}
    legacy = {base: ns['_' + base] for base in LEGACY if callable(ns.get('_' + base))}

    def configured(cfg: dict) -> dict:
        config = original_config(cfg)
        search, settings = _search_settings(cfg)
        if not settings['habilitada']:
            return config
        config = dict(config)
        config['__mppg_cobertura__'] = settings
        config['__mppg_variants__'] = _strategy_variants(cfg)
        return config

    def make_adapter(base: str, original: Any):
        def adapter(query: str, config: dict):
            settings = config.get('__mppg_cobertura__') if isinstance(config, dict) else None
            if not settings or not settings.get('habilitada'):
                return original(query, config)
            return _fetch_pages(ns, base, original, query, config, settings)
        adapter.__name__ = f'_{base}'
        return adapter

    def make_legacy_adapter(base: str, original: Any):
        def adapter(query: str, config: dict):
            settings = config.get('__mppg_cobertura__') if isinstance(config, dict) else None
            if not settings or not settings.get('habilitada'):
                return original(query, config)
            effective = (config.get('__mppg_variants__') or {}).get(query, {}).get(base, query)
            result, log = original(effective, config)
            warning = 'Fonte sem paginação auditável nesta versão; limite e total desconhecidos.'
            log = {**log, 'query_original': query, 'query_effective': effective,
                   'coverage_truncated': True, 'coverage_exhausted': False,
                   'source_total_available': None, 'coverage_warnings': [warning]}
            CALLS.append({'fonte': base, 'consulta_original': query, 'consulta_efetiva': effective,
                          'total_disponivel': None, 'recuperados': len(result),
                          'paginas': 1, 'limite': config.get('per_base'), 'esgotada': False,
                          'truncada': True, 'avisos': [warning]})
            return result, log
        adapter.__name__ = f'_{base}'
        return adapter

    def run(*args: Any, **kwargs: Any) -> Any:
        cfg = args[0] if args else (kwargs.get('cfg') or kwargs.get('config'))
        enabled = isinstance(cfg, dict) and _search_settings(cfg)[1]['habilitada']
        if not enabled:
            return original_run(*args, **kwargs)
        start = len(CALLS)
        result = original_run(*args, **kwargs)
        calls = CALLS[start:]
        if _search_settings(cfg)[1]['auditoria']:
            try:
                files = _write_audit(cfg, calls, result)
                if files:
                    print('AUDITORIA_COBERTURA_PRISMA_JSON:', files[0])
                    print('AUDITORIA_COBERTURA_PRISMA_MD:', files[1])
                else:
                    print('AVISO_COBERTURA: arquivo deduplicado não localizado; relatório técnico não criado.')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                print(f'AVISO_COBERTURA: relatório técnico não pôde ser gerado: {type(exc).__name__}: {exc}')
        return result

    ns['_query_config'] = configured
    for base, fn in native.items():
        ns['_' + base] = make_adapter(base, fn)
    for base, fn in legacy.items():
        ns['_' + base] = make_legacy_adapter(base, fn)
    ns['run_external_prisma_search'] = run
    ns['_mppg_cobertura_v16_instalada'] = True
