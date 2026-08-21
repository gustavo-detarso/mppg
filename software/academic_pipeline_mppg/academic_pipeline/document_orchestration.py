from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class DocumentStageResult:
    terminal: bool
    value: Any
    values: dict[str, Any]

def load_config_impl(runtime, path):
    return runtime['_refs_apply_runtime_policy'](runtime['_refs_original_load_config'](path))

def output_paths_impl(runtime, cfg):
    """Resolve a saída final do documento pela seção [paths]."""
    config_dir = runtime['Path'](str(cfg.get('__config_dir__') or runtime['Path'].cwd())).resolve()
    paths = runtime['_section'](cfg, 'paths')
    projeto = runtime['_section'](cfg, 'projeto')
    prefix = str(paths.get('document_prefix') or projeto.get('nome') or 'documento').strip() or 'documento'
    out_base = runtime['resolve_path'](paths.get('document_output_dir') or '../../output/documento', config_dir) or config_dir / 'output/documento'
    out_dir = out_base / prefix if bool(paths.get('create_document_subdir', True)) else out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    return (out_dir, prefix)

def apply_cli_path_overrides_impl(runtime, cfg, args):
    """Aplica overrides de caminhos informados na linha de comando.

    A prioridade fica: CLI > TOML. Os caminhos continuam sendo resolvidos
    posteriormente em relação ao diretório do TOML, salvo quando absolutos.
    """
    paths = cfg.setdefault('paths', {})
    if not isinstance(paths, dict):
        paths = {}
        cfg['paths'] = paths
    if getattr(args, 'output_dir', ''):
        paths['document_output_dir'] = args.output_dir
    if getattr(args, 'work_dir', ''):
        paths['work_dir'] = args.work_dir
    if getattr(args, 'cache_dir', ''):
        paths['cache_dir'] = args.cache_dir
    if getattr(args, 'research_output_dir', ''):
        paths['research_output_dir'] = args.research_output_dir
    if getattr(args, 'output_prefix', ''):
        paths['document_prefix'] = args.output_prefix
    if getattr(args, 'no_output_subdir', False):
        paths['create_document_subdir'] = False
    doc = cfg.setdefault('documento', {})
    if not isinstance(doc, dict):
        doc = {}
        cfg['documento'] = doc
    if getattr(args, 'layout', ''):
        doc['layout'] = args.layout
    if getattr(args, 'tipo_conteudo', ''):
        doc['tipo_conteudo'] = args.tipo_conteudo
    if getattr(args, 'genero_academico', ''):
        doc['genero_academico'] = args.genero_academico
    return cfg

def load_existing_document_json_impl(runtime, path):
    return runtime['AcademicDocument'].model_validate_json(path.read_text(encoding='utf-8'))

def resolve_bib_for_existing_document_impl(runtime, document, document_json_path, out_dir, prefix):
    """Resolve o .bib em modo --somente-renderizar sem exigir que ele já esteja no output_dir."""
    raw = str(document.bibliography.bib_path or f'{prefix}.bib').strip()
    candidates: list[runtime['Path']] = []
    if raw:
        p = runtime['Path'](raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([document_json_path.parent / p, out_dir / p])
    candidates.extend([document_json_path.with_suffix('.bib'), document_json_path.with_name(prefix + '.bib'), out_dir / f'{prefix}.bib'])
    found = next((c.resolve() for c in candidates if c.exists()), out_dir / f'{prefix}.bib')
    target = out_dir / found.name
    if found.exists() and found.resolve() != target.resolve():
        import shutil
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)
        found = target
    keys = list(document.bibliography.entries_used or [])
    if found.exists() and (not keys):
        import re
        text = found.read_text(encoding='utf-8', errors='ignore')
        keys = [m.group(1).strip() for m in re.finditer('@[^{}]+\\{\\s*([^,]+)\\s*,', text)]
    return (found, keys)

def _resolve_latex_paths_for_recompile_impl(runtime, args, cfg):
    base = runtime['Path'](str((cfg or {}).get('__config_dir__') or runtime['Path'].cwd())).resolve()
    latex_cfg = (cfg or {}).get('latex', {}) if isinstance((cfg or {}).get('latex', {}), dict) else {}
    academic_writing = runtime['resolve_path'](args.academic_writing or latex_cfg.get('org_latex_class_init'), base)
    latex_extra = runtime['resolve_path'](args.latex_extra_path or latex_cfg.get('latex_extra_path'), base)
    pdf_engine = str(args.pdf_engine or latex_cfg.get('pdf_engine') or 'lualatex')
    return (academic_writing, latex_extra, pdf_engine)

def run_recompile_impl(runtime, args, cfg):
    if not args.org:
        raise RuntimeError('Use --recompile --org caminho/arquivo.org')
    org_path = runtime['Path'](args.org).expanduser().resolve()
    if not org_path.exists():
        raise FileNotFoundError(f'ORG não encontrado: {org_path}')
    academic_writing, latex_extra, pdf_engine = runtime['_resolve_latex_paths_for_recompile'](args, cfg)
    removed = [] if args.no_clean else runtime['clean_aux_files'](org_path)
    pdf = runtime['run_compile_sequence'](org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
    out_dir = org_path.parent
    prefix = org_path.stem
    outputs = {'org': str(org_path), 'pdf': str(pdf), 'removed_aux': removed}
    runtime['write_outputs_manifest'](out_dir / f'{prefix}.outputs.txt', outputs)
    runtime['stage']('Gerando run_report e manifestos')
    report = runtime['make_run_report'](cfg=cfg or {'__config_dir__': str(runtime['Path'].cwd())}, config_path=runtime['Path'](str((cfg or {}).get('__config_path__'))) if cfg and cfg.get('__config_path__') else None, out_dir=out_dir, prefix=prefix, model=None, outputs=outputs, warnings=[], extra={'mode': 'recompile'})
    runtime['write_json'](out_dir / f'{prefix}.run_report.json', report)
    runtime['print_outputs'](outputs, title='Recompilação concluída')
    return 0

def render_additional_language_versions_impl(runtime, *, client, model, cfg, document, bib_path, bib_keys, out_dir, prefix, doc_cfg, latex_cfg, config_dir, abstract_bundle):
    """Traduz e renderiza versões adicionais a partir do document.json canônico.

    Cada versão recebe diretório próprio dentro de ``idiomas/<codigo>`` e
    compartilha a bibliografia original, copiada sem tradução. A função nunca
    consulta novamente o corpus nem gera uma segunda análise acadêmica.
    """
    result: dict[str, runtime['Any']] = {}
    warnings: list[str] = []
    languages = runtime['requested_translation_languages'](cfg)
    if not languages:
        return (result, warnings)
    base_dir = out_dir / 'idiomas'
    max_chars = runtime['translation_batch_size'](cfg)
    docx_cfg = cfg.get('docx', {}) if isinstance(cfg.get('docx'), dict) else {}
    reference_docx = runtime['resolve_path'](docx_cfg.get('reference_docx') or doc_cfg.get('docx_reference'), config_dir)
    academic_writing = runtime['resolve_path'](latex_cfg.get('org_latex_class_init'), config_dir)
    latex_extra = runtime['resolve_path'](latex_cfg.get('latex_extra_path'), config_dir)
    pdf_engine = str(latex_cfg.get('pdf_engine') or 'lualatex')
    for language_code, language_label in languages:
        runtime['stage'](f'Traduzindo paper para {language_label}')
        translated_document, audit = runtime['translate_document_model'](client, model, document, language_code, max_chars=max_chars)
        language_dir = base_dir / language_code
        language_dir.mkdir(parents=True, exist_ok=True)
        language_prefix = f'{prefix}_{language_code}'
        language_bib = language_dir / bib_path.name
        if bib_path.exists() and bib_path.resolve() != language_bib.resolve():
            runtime['shutil'].copy2(bib_path, language_bib)
        document_json = language_dir / f'{language_prefix}.document.json'
        runtime['write_json'](document_json, translated_document.model_dump())
        audit_path = language_dir / f'{language_prefix}.translation_audit.json'
        runtime['write_json'](audit_path, audit)
        runtime['stage'](f'Renderizando ORG traduzido ({language_label})')
        org_path = language_dir / f'{language_prefix}.org'
        org_text = runtime['render_org_latex'](translated_document, org_path, language_bib.name, cfg=cfg, bib_keys=bib_keys)
        if abstract_bundle:
            org_text = runtime['inject_paper_abstracts_into_org'](org_path, abstract_bundle, [language_code])
        runtime['raise_if_errors'](runtime['validate_org_text'](org_text, bib_keys), f'Validação do ORG traduzido falhou ({language_label})')
        pdf_path: runtime['Path'] | None = None
        if bool(doc_cfg.get('exportar_pdf', True)):
            runtime['stage'](f'Compilando PDF traduzido ({language_label})')
            pdf_path = runtime['run_compile_sequence'](org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
        docx_path: runtime['Path'] | None = None
        if bool(doc_cfg.get('exportar_docx', True)):
            runtime['stage'](f'Renderizando DOCX traduzido ({language_label})')
            docx_path = runtime['render_docx'](translated_document, language_dir / f'{language_prefix}.docx', bib_path=language_bib, reference_docx=reference_docx, cfg=cfg)
            if abstract_bundle:
                runtime['inject_paper_abstracts_into_docx'](docx_path, abstract_bundle, [language_code])
            validation = runtime['validate_docx_file'](docx_path, expected_title=translated_document.metadata.titulo, require_references=bool(translated_document.bibliography.entries_used))
            if validation and validation.get('warnings'):
                warnings.extend([f"DOCX {language_code}: {item}" for item in validation.get('warnings', [])])
        quality = runtime['build_quality_report'](translated_document, org_path=org_path, bib_keys=bib_keys)
        quality_path = language_dir / f'{language_prefix}.quality_report.md'
        runtime['write_quality_report'](quality, quality_path)
        if quality.get('warnings'):
            warnings.extend([f"QUALIDADE {language_code}: {item}" for item in quality.get('warnings', [])])
        result[language_code] = {'idioma': language_label, 'output_dir': str(language_dir), 'document_json': str(document_json), 'translation_audit': str(audit_path), 'org': str(org_path), 'pdf': str(pdf_path) if pdf_path else None, 'docx': str(docx_path) if docx_path else None, 'bib': str(language_bib), 'quality_report': str(quality_path)}
    return (result, warnings)

def _refs_disabled_impl(runtime, cfg):
    if not isinstance(cfg, dict):
        return False
    bibliography = cfg.get('bibliografia', {}) if isinstance(cfg.get('bibliografia'), dict) else {}
    document = cfg.get('documento', {}) if isinstance(cfg.get('documento'), dict) else {}
    local = cfg.get('documentos_locais', {}) if isinstance(cfg.get('documentos_locais'), dict) else {}
    if 'ativo' in bibliography:
        return not bool(bibliography.get('ativo'))
    if 'referencias_formais' in document:
        return not bool(document.get('referencias_formais'))
    return local.get('auto_detect_bib') is False and local.get('gerar_bib_revisado_ia') is False and (document.get('usar_citacoes_latex_diretas') is False)

def _refs_apply_runtime_policy_impl(runtime, cfg):
    if not runtime['_refs_disabled'](cfg):
        return cfg
    bibliography = cfg.setdefault('bibliografia', {})
    if not isinstance(bibliography, dict):
        bibliography = {}
        cfg['bibliografia'] = bibliography
    bibliography['ativo'] = False
    bibliography['gerar_arquivo_bib'] = False
    bibliography['buscar_metadados_por_doi'] = False
    bibliography['enriquecer_metadados_buscadores'] = False
    document = cfg.setdefault('documento', {})
    if not isinstance(document, dict):
        document = {}
        cfg['documento'] = document
    document['referencias_formais'] = False
    document['usar_citacoes_latex_diretas'] = False
    local = cfg.setdefault('documentos_locais', {})
    if isinstance(local, dict):
        local['auto_detect_bib'] = False
        local['gerar_bib_revisado_ia'] = False
        local['enriquecer_metadados_buscadores'] = False
        local['extrair_doi_dos_pdfs'] = False
        local['buscar_metadados_por_doi'] = False
    orientations = cfg.setdefault('orientacoes', {})
    if not isinstance(orientations, dict):
        orientations = {}
        cfg['orientacoes'] = orientations
    instruction = 'Não inclua citações no corpo do texto, notas bibliográficas, seção Referências, lista bibliográfica ou arquivo .bib. Use exclusivamente o corpus local e não invente fontes.'
    current = str(orientations.get('inline') or '').strip()
    if instruction not in current:
        orientations['inline'] = (current + '\n\n' + instruction).strip()
    return cfg

def build_bibliography_impl(runtime, cfg, docs, out_dir, prefix, client, model):
    if runtime['_refs_disabled'](cfg):
        from types import SimpleNamespace
        return SimpleNamespace(bib_path=runtime['Path'](out_dir) / f'{prefix}.bib', keys=[])
    return runtime['_refs_original_build_bibliography'](cfg, docs, out_dir, prefix, client, model)

def _refs_clear_document_bibliography_impl(runtime, document):
    bibliography = getattr(document, 'bibliography', None)
    if bibliography is not None:
        try:
            bibliography.entries_used = []
        except Exception:
            pass
        try:
            bibliography.bib_path = ''
        except Exception:
            pass
    return document

def _refs_strip_org_impl(runtime, text):
    import re as _re
    text = _re.sub('(?im)^.*(?:#\\+(?:print_)?bibliography|\\\\addbibresource|\\\\printbibliography).*(?:\\n|$)', '', text)
    text = _re.sub('(?is)\\[cite(?:/[\\w-]+)?\\s*:[^\\]]*\\]', '', text)
    text = _re.sub('(?is)\\[@[A-Za-z0-9_:.+/\\-]+(?:;\\s*@[A-Za-z0-9_:.+/\\-]+)*\\]', '', text)
    text = _re.sub('(?is)\\\\(?:auto|text|para|smart|foot|super)?cite(?:\\[[^\\]]*\\])?(?:\\[[^\\]]*\\])?\\{[^}]*\\}', '', text)
    text = _re.sub('(?ims)^\\*+\\s*(?:refer[eê]ncias|bibliografia)\\s*$.*?(?=^\\*+\\s+|\\Z)', '', text)
    text = _re.sub('(?is)\\\\(?:section|section\\*|chapter|chapter\\*)\\{\\s*(?:refer[eê]ncias|bibliografia)\\s*\\}.*?(?=\\\\(?:section|chapter)\\{|\\\\end\\{document\\}|\\Z)', '', text)
    return _re.sub('\\n{3,}', '\n\n', text).strip() + '\n'

def render_org_latex_impl(runtime, document, org_path, bib_filename, *, cfg, bib_keys):
    if not runtime['_refs_disabled'](cfg):
        return runtime['_refs_original_render_org_latex'](document, org_path, bib_filename, cfg=cfg, bib_keys=bib_keys)
    document = runtime['_refs_clear_document_bibliography'](document)
    rendered = runtime['_refs_original_render_org_latex'](document, org_path, bib_filename, cfg=cfg, bib_keys=[])
    clean = runtime['_refs_strip_org'](rendered)
    runtime['Path'](org_path).write_text(clean, encoding='utf-8')
    return clean

def run_document_stage_001(args, runtime):
    if args.quality_report:
        if not args.document_json:
            raise RuntimeError('--quality-report exige --document-json caminho/document.json')
        document_json = runtime['Path'](args.document_json).expanduser().resolve()
        document = runtime['load_existing_document_json'](document_json)
        org = runtime['Path'](args.org).expanduser().resolve() if args.org else None
        bib_keys: list[str] = []
        if args.bib:
            if __package__:
                from .bibliography_manager import split_bib_entries, bib_entry_key
            else:
                from bibliography_manager import split_bib_entries, bib_entry_key
            bib_path = runtime['Path'](args.bib).expanduser().resolve()
            if bib_path.exists():
                bib_keys = [runtime['k'] for e in split_bib_entries(bib_path.read_text(encoding='utf-8', errors='ignore')) if (k := bib_entry_key(e))]
        report = runtime['build_quality_report'](document, org_path=org, bib_keys=bib_keys or list(document.bibliography.entries_used or []))
        out = document_json.with_suffix('.quality_report.md')
        runtime['write_quality_report'](report, out)
        print(f'Relatório de qualidade: {out}')
        return DocumentStageResult(True, 0 if report.get('ok') else 1, {})
    _ap003d_values = {}
    for _ap003d_name in ('bib_entry_key', 'bib_keys', 'bib_path', 'document', 'document_json', 'e', 'k', 'org', 'out', 'report', 'split_bib_entries'):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_002(args, runtime):
    doc_cfg = runtime['cfg'].get('documento', {}) if isinstance(runtime['cfg'].get('documento'), dict) else {}
    _ap003d_values = {}
    for _ap003d_name in ('doc_cfg',):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_003(args, runtime):
    if args.somente_mapa_mental:
        if not runtime['document_json_path'].exists():
            raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {runtime['document_json_path']}")
        if not runtime['should_generate_mindmap'](runtime['cfg']):
            raise RuntimeError('[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.')
        runtime['stage']('Carregando document.json existente')
        document = runtime['load_existing_document_json'](runtime['document_json_path'])
        removed_mindmap_files: list[str] = []
        if args.forcar_regeneracao_mapa_mental:
            runtime['stage']('Removendo mapa mental existente')
            removed_mindmap_files = runtime['delete_existing_mindmap_outputs'](runtime['cfg'], runtime['out_dir'])
        mm_diag = None
        if args.reusar_mapa_mental:
            runtime['stage']('Tentando reutilizar mapa mental existente')
            mm_diag = runtime['attach_existing_mindmap_if_available'](document, runtime['cfg'], runtime['out_dir'])
            if not runtime['mm_diag']:
                runtime['warnings'].append('Mapa mental existente não encontrado; gerando novo mapa mental.')
        if not runtime['mm_diag']:
            runtime['stage']('Inicializando cliente OpenAI')
            client, model = runtime['make_client'](runtime['model'])
            runtime['stage']('Gerando/renderizando apenas o mapa mental')
            mm_diag = runtime['generate_and_attach_mindmap'](client, runtime['model'], runtime['cfg'], document, runtime['out_dir'])
        if removed_mindmap_files:
            mm_diag = dict(runtime['mm_diag'] or {})
            runtime['mm_diag']['removed_before_regeneration'] = removed_mindmap_files
        document.diagnostics.mindmap_json = runtime['json'].dumps(runtime['mm_diag'], ensure_ascii=False)
        runtime['stage']('Salvando document.json atualizado')
        runtime['write_json'](runtime['document_json_path'], document.model_dump())
        outputs = {'output_dir': str(runtime['out_dir']), 'document_json': str(runtime['document_json_path']), 'mindmap_puml': (runtime['mm_diag'] or {}).get('puml_path') if runtime['mm_diag'] else None, 'mindmap_image': (runtime['mm_diag'] or {}).get('image_path') if runtime['mm_diag'] else None, 'mindmap_reused': bool((runtime['mm_diag'] or {}).get('reused')), 'mindmap_removed': removed_mindmap_files}
        report = runtime['make_run_report'](cfg=runtime['cfg'], config_path=runtime['Path'](str(runtime['cfg'].get('__config_path__'))), out_dir=runtime['out_dir'], prefix=runtime['prefix'], model=runtime['model'], outputs=outputs, warnings=runtime['warnings'], extra={'mode': 'somente_mapa_mental'})
        runtime['write_json'](runtime['out_dir'] / f"{runtime['prefix']}.run_report.json", report)
        runtime['write_outputs_manifest'](runtime['out_dir'] / f"{runtime['prefix']}.outputs.txt", outputs)
        runtime['print_outputs'](outputs, title=f"academic_pipeline {runtime['PIPELINE_VERSION']} — mapa mental renderizado")
        if runtime['warnings']:
            print('Avisos:')
            for w in runtime['warnings']:
                print(f'- {w}')
        return DocumentStageResult(True, 0, {})
    _ap003d_values = {}
    for _ap003d_name in ('client', 'document', 'mm_diag', 'model', 'outputs', 'removed_mindmap_files', 'report', 'w'):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_004(args, runtime):
    runtime['stage']('Renderizando ORG/LaTeX')
    _ap003d_values = {}
    for _ap003d_name in ():
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_005(args, runtime):
    org_text = runtime['render_org_latex'](runtime['document'], runtime['org_path'], runtime['bib_path'].name if 'bib_path' in locals() else f"{runtime['prefix']}.bib", cfg=runtime['cfg'], bib_keys=runtime['bib_keys'] if 'bib_keys' in locals() else None)
    if runtime['paper_abstract_bundle']:
        runtime['stage']('Inserindo resumo e palavras-chave no ORG')
        org_text = runtime['inject_paper_abstracts_into_org'](runtime['org_path'], runtime['paper_abstract_bundle'], runtime['main_document_abstract_languages'](runtime['cfg']))
    runtime['stage']('Validando ORG renderizado')
    runtime['raise_if_errors'](runtime['validate_org_text'](org_text, runtime['bib_keys']), 'Validação do ORG renderizado falhou')
    _ap003d_values = {}
    for _ap003d_name in ('org_text',):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_006(args, runtime):
    if bool(runtime['doc_cfg'].get('exportar_pdf', True)):
        academic_writing = runtime['resolve_path'](runtime['latex_cfg'].get('org_latex_class_init'), runtime['config_dir'])
        latex_extra = runtime['resolve_path'](runtime['latex_cfg'].get('latex_extra_path'), runtime['config_dir'])
        pdf_engine = str(runtime['latex_cfg'].get('pdf_engine') or 'lualatex')
        runtime['stage']('Compilando PDF via Emacs/LaTeX')
        pdf_path = runtime['run_compile_sequence'](runtime['org_path'], academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
    _ap003d_values = {}
    for _ap003d_name in ('academic_writing', 'latex_extra', 'pdf_engine', 'pdf_path'):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_007(args, runtime):
    if bool(runtime['doc_cfg'].get('exportar_docx', True)):
        docx_cfg = runtime['cfg'].get('docx', {}) if isinstance(runtime['cfg'].get('docx'), dict) else {}
        ref = runtime['resolve_path'](docx_cfg.get('reference_docx') or runtime['doc_cfg'].get('docx_reference'), runtime['config_dir'])
        runtime['stage']('Renderizando DOCX')
        docx_path = runtime['render_docx'](runtime['document'], runtime['out_dir'] / f"{runtime['prefix']}.docx", bib_path=runtime['bib_path'], reference_docx=ref, cfg=runtime['cfg'])
        if runtime['paper_abstract_bundle']:
            runtime['stage']('Inserindo resumo e palavras-chave no DOCX')
            runtime['inject_paper_abstracts_into_docx'](docx_path, runtime['paper_abstract_bundle'], runtime['main_document_abstract_languages'](runtime['cfg']))
        docx_validation = runtime['validate_docx_file'](docx_path, expected_title=runtime['document'].metadata.titulo, require_references=bool(runtime['document'].bibliography.entries_used))
        if docx_validation and docx_validation.get('warnings'):
            runtime['warnings'].extend([f"DOCX: {w}" for w in docx_validation.get('warnings', [])])
    _ap003d_values = {}
    for _ap003d_name in ('docx_cfg', 'docx_path', 'docx_validation', 'ref', 'w'):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_008(args, runtime):
    if args.somente_renderizar:
        if runtime['requested_translation_languages'](runtime['cfg']):
            runtime['warnings'].append('Versões adicionais por IA não foram atualizadas no modo --somente-renderizar. Execute a geração completa para traduzir o document.json canônico.')
    elif runtime['requested_translation_languages'](runtime['cfg']):
        try:
            translated_outputs, translation_warnings = runtime['render_additional_language_versions'](client=runtime['client'], model=runtime['model'], cfg=runtime['cfg'], document=runtime['document'], bib_path=runtime['bib_path'], bib_keys=runtime['bib_keys'], out_dir=runtime['out_dir'], prefix=runtime['prefix'], doc_cfg=runtime['doc_cfg'], latex_cfg=runtime['latex_cfg'], config_dir=runtime['config_dir'], abstract_bundle=runtime['paper_abstract_bundle'] or None)
            runtime['warnings'].extend(translation_warnings)
        except runtime['TranslationError'] as exc:
            runtime['warnings'].append(f'TRADUÇÃO: {exc}')
    _ap003d_values = {}
    for _ap003d_name in ('exc', 'translated_outputs', 'translation_warnings'):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_009(args, runtime):
    compliance_report = runtime['run_institution_compliance'](runtime['cfg'], org_path=runtime['org_path'], bib_path=runtime['bib_path'], docx_path=runtime['docx_path'], pdf_path=runtime['pdf_path'])
    _ap003d_values = {}
    for _ap003d_name in ('compliance_report',):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_010(args, runtime):
    runtime['outputs']['compliance_report'] = str(runtime['compliance_md'])
    _ap003d_values = {}
    for _ap003d_name in ():
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_011(args, runtime):
    quality = runtime['build_quality_report'](runtime['document'], org_path=runtime['org_path'], bib_keys=runtime['bib_keys'])
    _ap003d_values = {}
    for _ap003d_name in ('quality',):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)

def run_document_stage_012(args, runtime):
    runtime['outputs']['quality_report'] = str(runtime['quality_path'])
    report = runtime['make_run_report'](cfg=runtime['cfg'], config_path=runtime['Path'](str(runtime['cfg'].get('__config_path__'))), out_dir=runtime['out_dir'], prefix=runtime['prefix'], model=runtime['model'], outputs=runtime['outputs'], warnings=runtime['warnings'], extra={'mode': 'somente_renderizar' if args.somente_renderizar else 'full', 'work_dir': str(runtime['work_dir']), 'cache_dir': str(runtime['cache_dir']), 'precheck': runtime['precheck'], 'docx_validation': runtime['docx_validation']})
    _ap003d_values = {}
    for _ap003d_name in ('report',):
        if _ap003d_name in locals():
            _ap003d_values[_ap003d_name] = locals()[_ap003d_name]
    return DocumentStageResult(False, None, _ap003d_values)
__all__ = ['DocumentStageResult', 'load_config_impl', 'output_paths_impl', 'apply_cli_path_overrides_impl', 'load_existing_document_json_impl', 'resolve_bib_for_existing_document_impl', '_resolve_latex_paths_for_recompile_impl', 'run_recompile_impl', 'render_additional_language_versions_impl', '_refs_disabled_impl', '_refs_apply_runtime_policy_impl', 'build_bibliography_impl', '_refs_clear_document_bibliography_impl', '_refs_strip_org_impl', 'render_org_latex_impl', 'run_document_stage_001', 'run_document_stage_002', 'run_document_stage_003', 'run_document_stage_004', 'run_document_stage_005', 'run_document_stage_006', 'run_document_stage_007', 'run_document_stage_008', 'run_document_stage_009', 'run_document_stage_010', 'run_document_stage_011', 'run_document_stage_012']
