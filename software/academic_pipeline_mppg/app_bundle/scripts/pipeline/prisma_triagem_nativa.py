#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ponte conservadora para importação PRISMA integral pelo importador nativo.

--dry-run: verifica fontes e simula import_manual_prisma_triage em TemporaryDirectory.
--prepare: mesmo teste e cria cópia independente, sem publicar relatórios finais.
--importar: importa decisões já registradas na cópia, para diretório separado.

Não consulta APIs; não edita corpus, triagem anterior, TOMLs nem relatórios da busca.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import uuid
import sys
import tempfile
import tomllib
from typing import Any

STAGE_NAME = 'triagem_nativa_v1'
HUMAN_FIELDS = ('decisao_titulo_resumo', 'motivo_exclusao_titulo_resumo',
                'texto_completo_local', 'decisao_texto_completo',
                'motivo_exclusao_texto_completo', 'incluir_final', 'observacoes')
REQUIRED = {'id_registro', 'titulo', 'fontes', 'consultas_busca',
            'decisao_titulo_resumo', 'decisao_texto_completo', 'incluir_final'}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        heads = list(reader.fieldnames or [])
        if not heads or len(set(heads)) != len(heads) or not REQUIRED <= set(heads):
            raise ValueError('CSV sem cabeçalhos íntegros/obrigatórios.')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('CSV possui campos a mais/a menos; possível corrupção.')
    ids = [r['id_registro'].strip() for r in rows]
    if not ids or any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('CSV contém IDs vazios ou repetidos.')
    return heads, rows


def provenance(value: Any) -> list[str]:
    if isinstance(value, list):
        out = value
    elif isinstance(value, str):
        v = value.strip()
        if not v:
            out = []
        elif v.startswith('['):
            out = json.loads(v)
            if not isinstance(out, list):
                raise ValueError('Representação de proveniência JSON não é lista.')
        else:
            out = [x.strip() for x in v.split(';')]
    else:
        out = []
    if not all(isinstance(s, str) and s.strip() for s in out):
        raise ValueError('Lista de proveniência contém itens inválidos.')
    return sorted(set(s.strip() for s in out))


def inspect_source(csv_path: Path, manifesto: Path, *, prepared: bool = False) -> tuple[str, list[str], list[dict], dict, dict]:
    man = json.loads(manifesto.read_text(encoding='utf-8'))
    if not isinstance(man, dict):
        raise ValueError('Manifesto não é objeto JSON.')
    if not prepared and man.get('novo_csv_sha256') != sha(csv_path):
        raise RuntimeError('Hash do CSV não coincide com o manifesto; operação recusada.')
    candidates = Path(str(man['candidatos'])).expanduser().resolve(strict=True)
    if man.get('candidatos_sha256') != sha(candidates):
        raise RuntimeError('JSON dos candidatos foi alterado em relação ao manifesto.')
    initial_old = Path(str(man['triagem_anterior'])).expanduser().resolve(strict=True)
    if man.get('triagem_anterior_sha256') != sha(initial_old):
        raise RuntimeError('Triagem anterior foi alterada em relação ao manifesto.')
    prefix = candidates.name.removesuffix('.candidatos_deduplicados.json')
    if candidates.name == prefix:
        raise ValueError('Nome dos candidatos não é do formato esperado.')
    if not prepared and csv_path.parent != candidates.parent:
        raise ValueError('O CSV deve estar na mesma pasta do JSON da busca.')
    initial_report = candidates.parent / (prefix + '.prisma_report.json')
    initial = json.loads(initial_report.read_text(encoding='utf-8'))
    if not isinstance(initial, dict) or not isinstance(initial.get('contagens'), dict):
        raise ValueError('Relatório PRISMA inicial sem contagens.')
    heads, rows = read_csv(csv_path)
    records = json.loads(candidates.read_text(encoding='utf-8'))
    if not isinstance(records, list) or len(records) != len(rows):
        raise ValueError('O número de linhas não corresponde ao JSON dos candidatos.')
    index = {str(x.get('id_registro','')).strip(): x for x in records}
    if len(index) != len(records) or set(index) != {r['id_registro'].strip() for r in rows}:
        raise ValueError('Os IDs do CSV não correspondem exatamente ao corpus.')
    if len(rows) != int(initial['contagens']['registros_apos_deduplicacao']):
        raise ValueError('Total de registros diverge da contagem PRISMA deduplicada.')
    n0 = int(initial['contagens'].get('registros_enviados_para_triagem') or 0)
    if n0 > len(rows):
        raise ValueError('Contagem de triagem inicial superior ao total de candidatos.')
    normal: list[dict[str, str]] = []
    for row in rows:
        rid = row['id_registro'].strip()
        candidate = index[rid]
        x = dict(row)
        for field in ('fontes','consultas_busca'):
            stored = provenance(candidate.get(field))
            received = provenance(row.get(field))
            if not stored or not received or set(received) - set(stored):
                raise ValueError(f'{rid}: campo {field} vazio ou diferente dos candidatos.')
            # Canoniza a partir do JSON de descoberta, preservando toda a proveniência.
            x[field] = json.dumps(stored, ensure_ascii=False, separators=(',', ':'))
        normal.append(x)
    missing_title = [r['id_registro'] for r in normal if not r['titulo'].strip()]
    return prefix, heads, normal, man, {'candidatos':len(rows),
        'ja_na_triagem':n0, 'adicionados':len(rows)-n0,
        'sem_titulo':missing_title,
        'decisoes_titulo_resumo':sum(r['decisao_titulo_resumo'].strip().upper() in ('INCLUIR','EXCLUIR') for r in normal),
        'inclusoes_finais':sum(r['incluir_final'].strip().upper() == 'INCLUIR' for r in normal)}


def write_csv(path: Path, heads: list[str], rows: list[dict[str,str]]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=heads, extrasaction='raise')
        w.writeheader()
        w.writerows(rows)


def load_native(root: Path):
    p = root / 'app_bundle/scripts/pipeline'
    if not (p / 'prisma_busca_externa.py').is_file():
        raise FileNotFoundError(f'Coletor nativo não encontrado: {p}')
    sys.path.insert(0, str(p))
    # Import já instalado pelo MPPG, incluindo hooks de cobertura.
    mod = importlib.import_module('prisma_busca_externa')
    if Path(mod.__file__).resolve() != (p / 'prisma_busca_externa.py').resolve():
        raise RuntimeError('Importou outro coletor, não o da instalação selecionada.')
    for name in ('_read_triage', '_final_markdown', '_query_config', 'import_manual_prisma_triage'):
        if not callable(getattr(mod, name, None)):
            raise RuntimeError(f'Função nativa ausente: {name}')
    return mod


def load_profile(config: Path) -> dict:
    with config.open('rb') as f:
        cfg = tomllib.load(f)
    cfg['__config_dir__'] = str(config.parent)
    return cfg


def reconcile(native: Any, cfg: dict, report: dict, prefix: str, source: Path, target: Path,
              stats: dict, *, final: bool) -> dict:
    counts = dict(report.get('contagens') or {})
    if int(counts.get('registros_na_planilha') or -1) != stats['candidatos']:
        raise RuntimeError('Importação nativa alterou a quantidade de registros.')
    counts['registros_enviados_para_triagem_inicial'] = stats['ja_na_triagem']
    counts['registros_adicionados_a_triagem_integral'] = stats['adicionados']
    counts['registros_enviados_para_triagem'] = stats['candidatos']
    report['contagens'] = counts
    report['origem_triagem_integral'] = {
        'procedimento': 'MPPG_PRISMA_TRIAGEM_NATIVA_V1',
        'busca_original_preservada': True,
        'triagem_inicial': stats['ja_na_triagem'],
        'ampliacao_sem_nova_busca': stats['adicionados'],
        'arquivo_original': str(source),
        'integridade_candidatos_verificada': True,
        'registros_sem_titulo_para_conferencia': stats['sem_titulo'],
    }
    total = counts['registros_na_planilha']
    screened = counts.get('registros_com_triagem_titulo_resumo_concluida', 0)
    report['status'] = 'triagem_concluida_importada' if screened == total else 'triagem_parcial_importada'
    report['triagem_humana_pendente'] = total - screened
    if final:
        native._write_json(target / f'{prefix}.prisma_report_final.json', report)
        # Reconstrói markdown APENAS do relatório novo, com contagens atualizadas.
        content = native._final_markdown(native._query_config(cfg), source, counts, report.get('estudos_incluidos') or [])
        content += ('\n\n## Rastreabilidade da triagem integral\n\n'
                    f'- Encaminhados na coleta inicial: **{stats["ja_na_triagem"]}**.\n'
                    f'- Adicionados a partir do JSON já coletado: **{stats["adicionados"]}**.\n'
                    f'- Total na planilha integral: **{total}**.\n'
                    f'- Triagem título/resumo pendente: **{total-screened}**.\n'
                    '- O relatório inicial da busca foi preservado; não houve busca adicional.\n')
        (target / f'{prefix}.relatorio_prisma_final.md').write_text(content, encoding='utf-8')
    return report


def native_preview(root: Path, cfg: dict, prefix: str, heads: list[str], rows: list[dict], initial: Path, stats: dict):
    native = load_native(root)
    with tempfile.TemporaryDirectory(prefix='mppg_importacao_prisma_') as tmp:
        target = Path(tmp)
        source = target / f'{prefix}.triagem_integral_normalizada.csv'
        write_csv(source, heads, rows)
        shutil.copy2(initial, target / f'{prefix}.prisma_report.json')
        report = native.import_manual_prisma_triage(cfg, target, prefix, source)
        return reconcile(native, cfg, report, prefix, source, target, stats, final=True)


def cli(args: argparse.Namespace) -> dict:
    root = args.root.expanduser().resolve(strict=True)
    profile = args.config.expanduser().resolve(strict=True)
    csv_path = args.csv.expanduser().resolve(strict=True)
    manifest_path = args.manifesto.expanduser().resolve(strict=True)
    prefix, heads, rows, man, stats = inspect_source(csv_path, manifest_path)
    cfg = load_profile(profile)
    candidate_root = Path(str(man['candidatos'])).resolve().parent
    initial = candidate_root / f'{prefix}.prisma_report.json'
    report = native_preview(root, cfg, prefix, heads, rows, initial, stats)
    if int(report['contagens'].get('registros_com_triagem_titulo_resumo_concluida', -1)) != stats['decisoes_titulo_resumo']:
        raise RuntimeError('Contagem de decisões do importador não coincide com a planilha.')
    result = {'status':'SIMULACAO_NATIVA_APROVADA' if args.dry_run else 'TRIAGEM_NATIVA_PREPARADA',
              'registros':stats['candidatos'], 'ja_na_triagem':stats['ja_na_triagem'],
              'adicionados':stats['adicionados'], 'sem_titulo':stats['sem_titulo'],
              'triagem_humana_concluida':stats['decisoes_titulo_resumo'],
              'importador_nativo_executado_em_area_temporaria':True,
              'nenhum_relatorio_final_publicado':True,
              'originais_modificados':False,
              'busca_executada':False}
    if args.dry_run:
        return result
    stage = candidate_root / STAGE_NAME
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f'Já existe área de preparo: {stage}. Nenhuma sobrescrita permitida.')
    stage.mkdir(mode=0o750)
    try:
        prepared_csv = stage / f'{prefix}.triagem_nativa_v1.csv'
        write_csv(prepared_csv, heads, rows)
        validation = stage / f'{prefix}.triagem_nativa_v1.validacao.json'
        result['arquivo_csv_preparado'] = str(prepared_csv)
        result['sha256_csv_preparado'] = sha(prepared_csv)
        result['arquivo_manifesto_preparo'] = str(validation)
        result['csv_origem_sha256'] = sha(csv_path)
        validation.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    except Exception:
        shutil.rmtree(stage)
        raise
    return result


def import_decisions(args: argparse.Namespace) -> dict:
    """Importa APENAS decisões efetivamente iniciadas para uma NOVA pasta.

    Fonte original e manifesto permanecem íntegros. O CSV preparado pode ter
    mudanças posteriores em decisões, observações ou metadados, mas deve manter
    todos os IDs e a proveniência do corpus canônico.
    """
    root = args.root.expanduser().resolve(strict=True)
    profile = args.config.expanduser().resolve(strict=True)
    csv_orig = args.csv.expanduser().resolve(strict=True)
    manifest = args.manifesto.expanduser().resolve(strict=True)
    prefix, heads_orig, original, man, stats = inspect_source(csv_orig, manifest)
    stage = Path(str(man['candidatos'])).resolve().parent / STAGE_NAME
    prepared = stage / f'{prefix}.triagem_nativa_v1.csv'
    validation_path = stage / f'{prefix}.triagem_nativa_v1.validacao.json'
    if not stage.is_dir() or stage.is_symlink() or not prepared.is_file() or not validation_path.is_file():
        raise FileNotFoundError('Triagem nativa não preparada; execute --prepare primeiro.')
    validity = json.loads(validation_path.read_text(encoding='utf-8'))
    if validity.get('csv_origem_sha256') != sha(csv_orig):
        raise RuntimeError('CSV de origem divergiu desde a preparação.')
    heads, rows = read_csv(prepared)
    if heads != heads_orig or len(rows) != len(original):
        raise ValueError('Arquivo preparado mudou quantidade de linhas ou cabeçalhos.')
    initial_ids = [r['id_registro'] for r in original]
    if [r['id_registro'] for r in rows] != initial_ids:
        raise ValueError('Ordem ou IDs da triagem preparada foram alterados.')
    # IDs e fontes são imutáveis, mas os campos de decisão e o texto de títulos
    # podem ser corrigidos pelo revisor humano, sem perder rastreabilidade.
    for old, new in zip(original, rows):
        for name in ('fontes', 'consultas_busca'):
            if provenance(new[name]) != provenance(old[name]):
                raise ValueError(f'{new["id_registro"]}: proveniência {name} alterada.')
        for name, allowed in {
            'decisao_titulo_resumo': {'PENDENTE', 'INCLUIR', 'EXCLUIR'},
            'decisao_texto_completo': {'PENDENTE', 'NAO_INICIADO', 'INCLUIR', 'EXCLUIR'},
            'incluir_final': {'PENDENTE', 'INCLUIR', 'EXCLUIR'},
        }.items():
            if new[name].strip().upper() not in allowed:
                raise ValueError(f'{new["id_registro"]}: valor inválido em {name}.')
        if new['incluir_final'].strip().upper() == 'INCLUIR' and not new['titulo'].strip():
            raise ValueError(f'{new["id_registro"]}: inclusão final sem título.')
    screened = sum(r['decisao_titulo_resumo'].strip().upper() != 'PENDENTE' for r in rows)
    if screened == 0:
        raise ValueError('Nenhuma decisão humana foi preenchida; não publicar relatório final prematuro.')
    cfg = load_profile(profile)
    native = load_native(root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = stage / f'importacao_{stamp}'
    target.mkdir(mode=0o750)
    try:
        initial_path = Path(str(man['candidatos'])).resolve().parent / f'{prefix}.prisma_report.json'
        shutil.copy2(initial_path, target / f'{prefix}.prisma_report.json')
        result_native = native.import_manual_prisma_triage(cfg, target, prefix, prepared)
        report = reconcile(native, cfg, result_native, prefix, prepared, target, stats, final=True)
        if report['contagens']['registros_com_triagem_titulo_resumo_concluida'] != screened:
            raise RuntimeError('Divergência na contagem de decisões humanas.')
        out = {'status': report['status'], 'total': len(rows),
               'decisoes_titulo_resumo': screened, 'pendentes':len(rows)-screened,
               'csv_triagem_sha256':sha(prepared), 'originais_modificados':False,
               'relatorio_nativo_isolado':str(target / f'{prefix}.prisma_report_final.json')}
        (target / 'manifesto_importacao_integral.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return out
    except Exception:
        # Só remove o diretório exclusivo desta tentativa, nunca os originais.
        shutil.rmtree(target)
        raise

def main() -> int:
    ap = argparse.ArgumentParser(description='MPPG: integração conservadora da triagem integral com o importador nativo')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--manifesto', type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--importar-decisoes', action='store_true')
    args = ap.parse_args()
    try:
        result = import_decisions(args) if args.importar_decisoes else cli(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f'ERRO_INTEGRACAO_NATIVA: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
