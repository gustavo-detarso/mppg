from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DispatchResult:
    handled: bool
    value: Any = None


def _not_handled() -> DispatchResult:
    return DispatchResult(False, None)


def dispatch_stage_001(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:914-920
    if args.gui:
        if __package__:
            from app_bundle.scripts.pipeline.academic_pipeline_gui import run_gui
        else:
            from app_bundle.scripts.pipeline.academic_pipeline_gui import run_gui
        return DispatchResult(True, run_gui())
    return _not_handled()


def dispatch_stage_002(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:922-928
    if args.tui:
        if __package__:
            from app_bundle.scripts.pipeline.academic_pipeline_tui import run_tui
        else:
            from app_bundle.scripts.pipeline.academic_pipeline_tui import run_tui
        return DispatchResult(True, run_tui(no_clear=bool(args.no_clear)))
    return _not_handled()


def dispatch_stage_003(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:930-937
    if args.list_toml_profiles:
        if __package__:
            from app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo import print_profiles
        else:
            from app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo import print_profiles
        print_profiles()
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_004(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:939-946
    if args.init_toml:
        if __package__:
            from app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo import generate_interactive
        else:
            from app_bundle.scripts.pipeline.academic_pipeline_toml_generator_interativo import generate_interactive
        generate_interactive(non_interactive_profile=args.toml_profile or None, no_clear=bool(args.no_clear))
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_005(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:948-950
    if args.list_institutions:
        print(runtime['describe_institution_profiles']())
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_006(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:952-967
    if args.list_layouts:
        if not args.config:
            raise RuntimeError('--list-layouts exige --config caminho.toml')
        cfg_layouts = runtime['load_config'](runtime['Path'](args.config).expanduser().resolve())
        layouts = runtime['available_layouts'](cfg_layouts)
        if not layouts:
            print('Nenhum layout declarado no perfil institucional.')
        else:
            print('Layouts disponíveis:')
            for layout_id, spec in layouts.items():
                desc = str(spec.get('description') or spec.get('descricao') or '').strip()
                genero = str(spec.get('genero_academico') or '').strip()
                print(f'- {layout_id}' + (f' ({genero})' if genero else '') + (f': {desc}' if desc else ''))
            resolved = runtime['resolve_layout_spec'](cfg_layouts)
            print(f'Layout resolvido para este TOML: {resolved.id}')
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_007(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:969-971
    if args.explain_profile:
        print(runtime['explain_profile'](args.explain_profile))
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_008(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:973-978
    if args.show_prompts:
        if not args.config:
            raise RuntimeError('--show-prompts exige --config caminho.toml')
        cfg_preview = runtime['load_config'](runtime['Path'](args.config).expanduser().resolve())
        print(runtime['json'].dumps(runtime['prompt_report_for_cfg'](cfg_preview), ensure_ascii=False, indent=2))
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_009(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:980-990
    if args.init_project:
        base_dir = runtime['Path'](args.base_dir).expanduser().resolve() if args.base_dir else None
        result = runtime['init_project'](args.init_project, project_type=args.project_type, base_dir=base_dir, overwrite=bool(args.overwrite_project), institution=args.institution)
        print('Projeto criado:')
        print(f'- Diretório: {result.project_dir}')
        print(f'- TOML: {result.config_path}')
        print(f'- DOI manifest: {result.doi_manifest_path}')
        print(f'- Documentos ZIP: {result.documentos_zip_path}')
        print(f'- Orientações ZIP: {result.orientacoes_zip_path}')
        print(f'- README: {result.readme_path}')
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_010(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:992-1009
    if args.make_doi_manifest:
        input_zip = runtime['Path'](args.input_zip).expanduser().resolve() if args.input_zip else None
        input_dir = runtime['Path'](args.input_dir).expanduser().resolve() if args.input_dir else None
        if args.output:
            output = runtime['Path'](args.output).expanduser().resolve()
        elif input_zip:
            output = input_zip.parent / 'doi_manifest.csv'
        elif input_dir:
            output = input_dir / 'doi_manifest.csv'
        else:
            raise RuntimeError('Use --make-doi-manifest com --input-zip ou --input-dir.')
        result = runtime['make_doi_manifest'](input_zip, input_dir, output, overwrite=True)
        print('DOI manifest gerado:')
        print(f"- Fonte: {result['source']}")
        print(f"- Saída: {result['output']}")
        print(f"- Arquivos listados: {result['total_files']}")
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_011(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1011-1017
    if args.inspect_bib:
        bib = runtime['Path'](args.inspect_bib).expanduser().resolve()
        prefix = bib.with_name(bib.name + '_inspection')
        report = runtime['inspect_bib'](bib, output_prefix=prefix)
        print(runtime['render_bib_inspection_markdown'](report))
        print(f'Relatórios: {str(prefix)}.md e {str(prefix)}.json')
        return DispatchResult(True, 0 if report.get('ok') else 1)
    return _not_handled()


def dispatch_stage_012(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1045-1046
    if args.somente_renderizar and args.somente_mapa_mental:
        raise RuntimeError('Use apenas um entre --somente-renderizar e --somente-mapa-mental.')
    return _not_handled()


def dispatch_stage_013(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1047-1048
    if args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental:
        raise RuntimeError('Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.')
    return _not_handled()


def dispatch_stage_014(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1050-1060
    if args.write_prompt_lock:
        if not runtime['cfg']:
            raise RuntimeError('--write-prompt-lock exige --config caminho.toml')
        out_dir, prefix = runtime['research_output_paths'](runtime['cfg']) if runtime['external_search_enabled'](runtime['cfg']) else runtime['output_paths'](runtime['cfg'])
        lock_path = out_dir / f'{prefix}.prompt_lock.json'
        lock_md = out_dir / f'{prefix}.prompt_lock.md'
        lock = runtime['write_prompt_lock'](runtime['cfg'], lock_path)
        runtime['write_prompt_lock_markdown'](lock, lock_md)
        print(f'Prompt lock gerado: {lock_path}')
        print(f'Prompt lock markdown: {lock_md}')
        return DispatchResult(True, 0)
    return _not_handled()


def dispatch_stage_015(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1062-1074
    if args.check_institution_compliance:
        if not runtime['cfg']:
            raise RuntimeError('--check-institution-compliance exige --config caminho.toml')
        out_dir, prefix = runtime['output_paths'](runtime['cfg'])
        org = runtime['Path'](args.org).expanduser().resolve() if args.org else out_dir / f'{prefix}.org'
        bib = runtime['Path'](args.bib).expanduser().resolve() if args.bib else out_dir / f'{prefix}.bib'
        docx = runtime['Path'](args.docx).expanduser().resolve() if args.docx else out_dir / f'{prefix}.docx'
        pdf = runtime['Path'](args.pdf).expanduser().resolve() if args.pdf else out_dir / f'{prefix}.pdf'
        report = runtime['run_institution_compliance'](runtime['cfg'], org_path=org, bib_path=bib, docx_path=docx, pdf_path=pdf)
        md_path, json_path = runtime['write_compliance_reports'](report, out_dir / prefix)
        print(runtime['render_compliance_markdown'](report))
        print(f'Relatórios: {md_path} e {json_path}')
        return DispatchResult(True, 0 if report.get('ok') else 2)
    return _not_handled()


def dispatch_stage_016(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1076-1082
    if args.doctor:
        report = runtime['run_doctor'](runtime['cfg'])
        runtime['print_doctor_report'](report)
        if runtime['cfg']:
            out_dir, prefix = runtime['research_output_paths'](runtime['cfg']) if runtime['external_search_enabled'](runtime['cfg']) else runtime['output_paths'](runtime['cfg'])
            runtime['write_json'](out_dir / f'{prefix}.doctor_report.json', report)
        return DispatchResult(True, 0 if report.get('ok') else 2)
    return _not_handled()


def dispatch_stage_017(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1084-1091
    if args.check_config:
        if not runtime['cfg']:
            raise RuntimeError('--check-config exige --config caminho.toml')
        report = runtime['check_config'](runtime['cfg'])
        runtime['print_check_config_report'](report)
        out_dir, prefix = runtime['research_output_paths'](runtime['cfg']) if runtime['external_search_enabled'](runtime['cfg']) else runtime['output_paths'](runtime['cfg'])
        runtime['write_json'](out_dir / f'{prefix}.check_config_report.json', report)
        return DispatchResult(True, 0 if report.get('ok') else 2)
    return _not_handled()


def dispatch_stage_018(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1093-1094
    if args.recompile:
        return DispatchResult(True, runtime['run_recompile'](args, runtime['cfg']))
    return _not_handled()


def dispatch_stage_019(
    args: Any,
    runtime: Mapping[str, Any],
) -> DispatchResult:
    # Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1099-1139
    if args.prisma_importar_triagem:
        if not runtime['external_search_enabled'](runtime['cfg']):
            raise RuntimeError('--prisma-importar-triagem exige um TOML do perfil relatorio_prisma_busca_orientada_fgv.')
        out_dir, prefix = runtime['research_output_paths'](runtime['cfg'])
        runtime['stage']('Importando planilha de triagem PRISMA preenchida')
        prisma_outputs = runtime['import_manual_prisma_triage'](runtime['cfg'], out_dir, prefix, runtime['Path'](args.prisma_importar_triagem))
        org_path, pdf_path = runtime['render_external_prisma_outputs'](runtime['cfg'], out_dir, prefix, prisma_outputs, phase='final')
        artifacts = prisma_outputs.setdefault('artefatos', {}) if isinstance(prisma_outputs, dict) else {}
        if org_path:
            artifacts['relatorio_org'] = str(org_path)
        if pdf_path:
            artifacts['relatorio_pdf'] = str(pdf_path)
        report_json_path = artifacts.get('prisma_report_json') if isinstance(artifacts, dict) else ''
        if report_json_path:
            runtime['write_json'](runtime['Path'](str(report_json_path)), prisma_outputs)
        outputs = {'output_dir': str(out_dir), 'org': str(org_path) if org_path else None, 'pdf': str(pdf_path) if pdf_path else None, 'relatorio_pesquisa': prisma_outputs}
        report = runtime['make_run_report'](cfg=runtime['cfg'], config_path=runtime['Path'](str(runtime['cfg'].get('__config_path__'))), out_dir=out_dir, prefix=prefix, model=None, outputs=outputs, warnings=[], extra={'mode': 'prisma_importar_triagem'})
        runtime['write_json'](out_dir / f'{prefix}.run_report.json', report)
        runtime['write_outputs_manifest'](out_dir / f'{prefix}.outputs.txt', outputs)
        runtime['print_outputs'](outputs, title=f"academic_pipeline {runtime['PIPELINE_VERSION']} — triagem PRISMA consolidada")
        return DispatchResult(True, 0)
    return _not_handled()


__all__ = ['DispatchResult', 'dispatch_stage_001', 'dispatch_stage_002', 'dispatch_stage_003', 'dispatch_stage_004', 'dispatch_stage_005', 'dispatch_stage_006', 'dispatch_stage_007', 'dispatch_stage_008', 'dispatch_stage_009', 'dispatch_stage_010', 'dispatch_stage_011', 'dispatch_stage_012', 'dispatch_stage_013', 'dispatch_stage_014', 'dispatch_stage_015', 'dispatch_stage_016', 'dispatch_stage_017', 'dispatch_stage_018', 'dispatch_stage_019']
