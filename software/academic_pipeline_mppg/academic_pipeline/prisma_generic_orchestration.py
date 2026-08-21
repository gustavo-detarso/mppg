from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
_MISSING = object()
_PROTECTED_RUNTIME_NAMES = frozenset(('PrismaStageResult', '_MISSING', '_PROTECTED_RUNTIME_NAMES', '__all__', '_ap003e_body__json_or_none_1', '_ap003e_body__prisma_artigo_generico_get_arg_1', '_ap003e_body__prisma_artigo_generico_out_dir_1', '_ap003e_body__prisma_artigo_generico_run_export_1', '_ap003e_body__prisma_artigo_generico_run_freeze_1', '_ap003e_body__prisma_artigo_generico_strip_1', '_ap003e_body__prisma_curadoria_arg_1', '_ap003e_body__prisma_curadoria_build_cmd_1', '_ap003e_body__prisma_curadoria_config_from_args_1', '_ap003e_body__prisma_curadoria_default_config_1', '_ap003e_body__prisma_curadoria_default_out_dir_1', '_ap003e_body__prisma_curadoria_default_prompt_1', '_ap003e_body__prisma_curadoria_dispatch_1', '_ap003e_body__prisma_curadoria_fluxo_completo_1', '_ap003e_body__prisma_curadoria_importar_no_pipeline_1', '_ap003e_body__prisma_curadoria_input_from_args_1', '_ap003e_body__prisma_curadoria_menu_1', '_ap003e_body__prisma_curadoria_mostrar_caminhos_1', '_ap003e_body__prisma_curadoria_out_from_args_1', '_ap003e_body__prisma_curadoria_pipeline_supports_flag_1', '_ap003e_body__prisma_curadoria_prompt_from_args_1', '_ap003e_body__prisma_curadoria_reexportar_xlsx_1', '_ap003e_body__prisma_curadoria_run_command_1', '_ap003e_body__prisma_curadoria_run_ia_1', '_ap003e_body__prisma_curadoria_script_path_1', '_ap003e_body__section_1', '_ap003e_body_main_2', '_ap003e_body_make_client_1', '_ap003e_body_render_external_prisma_outputs_1', '_ap003e_body_research_output_paths_1', '_ap003e_body_stage_1', '_invoke_with_runtime', '_json_or_none_impl_001', '_prisma_artigo_generico_get_arg_impl_001', '_prisma_artigo_generico_out_dir_impl_001', '_prisma_artigo_generico_run_export_impl_001', '_prisma_artigo_generico_run_freeze_impl_001', '_prisma_artigo_generico_strip_impl_001', '_prisma_curadoria_arg_impl_001', '_prisma_curadoria_build_cmd_impl_001', '_prisma_curadoria_config_from_args_impl_001', '_prisma_curadoria_default_config_impl_001', '_prisma_curadoria_default_out_dir_impl_001', '_prisma_curadoria_default_prompt_impl_001', '_prisma_curadoria_dispatch_impl_001', '_prisma_curadoria_fluxo_completo_impl_001', '_prisma_curadoria_importar_no_pipeline_impl_001', '_prisma_curadoria_input_from_args_impl_001', '_prisma_curadoria_menu_impl_001', '_prisma_curadoria_mostrar_caminhos_impl_001', '_prisma_curadoria_out_from_args_impl_001', '_prisma_curadoria_pipeline_supports_flag_impl_001', '_prisma_curadoria_prompt_from_args_impl_001', '_prisma_curadoria_reexportar_xlsx_impl_001', '_prisma_curadoria_run_command_impl_001', '_prisma_curadoria_run_ia_impl_001', '_prisma_curadoria_script_path_impl_001', '_section_impl_001', 'make_client_impl_001', 'render_external_prisma_outputs_impl_001', 'research_output_paths_impl_001', 'run_prisma_generic_entrypoint', 'run_prisma_stage_001', 'run_prisma_stage_002', 'run_prisma_stage_003', 'run_prisma_stage_004', 'run_prisma_stage_005', 'run_prisma_stage_006', 'run_prisma_stage_007', 'run_prisma_stage_008', 'stage_impl_001', 'stage_with_runtime', 'json_or_none_with_runtime', 'make_client_with_runtime', 'section_with_runtime', 'research_output_paths_with_runtime', 'render_external_prisma_outputs_with_runtime', 'prisma_curadoria_default_config_with_runtime', 'prisma_curadoria_default_out_dir_with_runtime', 'prisma_curadoria_default_prompt_with_runtime', 'prisma_curadoria_script_path_with_runtime', 'prisma_curadoria_arg_with_runtime', 'prisma_curadoria_config_from_args_with_runtime', 'prisma_curadoria_out_from_args_with_runtime', 'prisma_curadoria_prompt_from_args_with_runtime', 'prisma_curadoria_input_from_args_with_runtime', 'prisma_curadoria_run_command_with_runtime', 'prisma_curadoria_build_cmd_with_runtime', 'prisma_curadoria_run_ia_with_runtime', 'prisma_curadoria_reexportar_xlsx_with_runtime', 'prisma_curadoria_pipeline_supports_flag_with_runtime', 'prisma_curadoria_importar_no_pipeline_with_runtime', 'prisma_curadoria_fluxo_completo_with_runtime', 'prisma_curadoria_mostrar_caminhos_with_runtime', 'prisma_curadoria_menu_with_runtime', 'prisma_curadoria_dispatch_with_runtime', 'prisma_artigo_generico_get_arg_with_runtime', 'prisma_artigo_generico_strip_with_runtime', 'prisma_artigo_generico_out_dir_with_runtime', 'prisma_artigo_generico_run_export_with_runtime', 'prisma_artigo_generico_run_freeze_with_runtime', 'run_prisma_generic_with_runtime'))

@dataclass(frozen=True, slots=True)
class PrismaStageResult:
    terminal: bool
    value: Any
    values: dict[str, Any]

def _invoke_with_runtime(function: Any, runtime: Mapping[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    namespace = function.__globals__
    previous: dict[str, Any] = {}
    added: list[str] = []
    for name, value in runtime.items():
        if name in _PROTECTED_RUNTIME_NAMES:
            continue
        if name in namespace:
            previous[name] = namespace[name]
        else:
            added.append(name)
        namespace[name] = value
    try:
        return function(*args, **kwargs)
    finally:
        for name in added:
            namespace.pop(name, None)
        for name, value in previous.items():
            namespace[name] = value

def _ap003e_body_stage_1(message):
    """Mostra etapas de execução em tempo real."""
    print(f'[ETAPA] {message}', flush=True)

def stage_with_runtime(runtime, /, message):
    return _invoke_with_runtime(_ap003e_body_stage_1, runtime, (message,), {})

def stage_impl_001(runtime, /, message):
    return stage_with_runtime(runtime, message)

def _ap003e_body__json_or_none_1(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value

def json_or_none_with_runtime(runtime, /, value):
    return _invoke_with_runtime(_ap003e_body__json_or_none_1, runtime, (value,), {})

def _json_or_none_impl_001(runtime, /, value):
    return json_or_none_with_runtime(runtime, value)

def _ap003e_body_make_client_1(model_override):
    from openai import OpenAI
    load_dotenv(override=False)
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY não encontrado no ambiente/.env.')
    return (OpenAI(api_key=os.getenv('OPENAI_API_KEY')), model_override or os.getenv('OPENAI_MODEL') or DEFAULT_MODEL)

def make_client_with_runtime(runtime, /, model_override):
    return _invoke_with_runtime(_ap003e_body_make_client_1, runtime, (model_override,), {})

def make_client_impl_001(runtime, /, model_override):
    return make_client_with_runtime(runtime, model_override)

def _ap003e_body__section_1(cfg, name):
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}

def section_with_runtime(runtime, /, cfg, name):
    return _invoke_with_runtime(_ap003e_body__section_1, runtime, (cfg, name), {})

def _section_impl_001(runtime, /, cfg, name):
    return section_with_runtime(runtime, cfg, name)

def _ap003e_body_research_output_paths_1(cfg):
    """Resolve a saída canônica da busca e consolidação PRISMA.

    A pesquisa bibliográfica não deve compartilhar ``document_output_dir`` com
    documentos acadêmicos. ``research_output_dir`` e ``research_prefix`` são
    resolvidos em relação ao TOML; na ausência deles, usa-se uma pasta
    ``output_pesquisa`` dentro do próprio projeto.
    """
    config_dir = Path(str(cfg.get('__config_dir__') or Path.cwd())).resolve()
    paths = _section(cfg, 'paths')
    projeto = _section(cfg, 'projeto')
    project_name = str(projeto.get('nome') or '').strip()
    default_prefix = f'relatorio_prisma_{project_name}' if project_name else 'relatorio_prisma'
    prefix = str(paths.get('research_prefix') or default_prefix).strip() or default_prefix
    out_base = resolve_path(paths.get('research_output_dir') or 'output_pesquisa', config_dir) or config_dir / 'output_pesquisa'
    out_dir = out_base / prefix if bool(paths.get('create_research_subdir', True)) else out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    return (out_dir, prefix)

def research_output_paths_with_runtime(runtime, /, cfg):
    return _invoke_with_runtime(_ap003e_body_research_output_paths_1, runtime, (cfg,), {})

def research_output_paths_impl_001(runtime, /, cfg):
    return research_output_paths_with_runtime(runtime, cfg)

def _ap003e_body_render_external_prisma_outputs_1(cfg, out_dir, prefix, prisma_payload, *, phase):
    """Renderiza relatório PRISMA externo em ORG e, quando solicitado, em PDF.

    O perfil de busca não constrói ``document.json``. Por isso, esta rotina
    usa o relatório estruturado da busca/triagem e preserva o layout e a engine
    de LaTeX definidos no TOML, como os demais perfis que exportam PDF.
    """
    report_cfg = cfg.get('relatorio_pesquisa', {}) if isinstance(cfg.get('relatorio_pesquisa'), dict) else {}
    org_requested = bool(report_cfg.get('exportar_org', True))
    pdf_requested = bool(report_cfg.get('exportar_pdf', False))
    if not org_requested and (not pdf_requested):
        return (None, None)
    stage(f'Renderizando relatório PRISMA {phase} em ORG')
    org_path = render_external_prisma_org_report(cfg, out_dir, prefix, prisma_payload, phase=phase)
    try:
        if __package__:
            from .prisma_diagrama_fluxo import ensure_prisma_flow_diagram
        else:
            from prisma_diagrama_fluxo import ensure_prisma_flow_diagram
        ensure_prisma_flow_diagram(cfg=cfg, out_dir=out_dir, prefix=prefix, org_path=org_path, prisma_payload=prisma_payload, phase=phase)
    except Exception as exc:
        print(f'[WARN] Não foi possível gerar/inserir o diagrama PRISMA: {exc}')
    pdf_path: Path | None = None
    if pdf_requested:
        latex_cfg = cfg.get('latex', {}) if isinstance(cfg.get('latex'), dict) else {}
        config_dir = Path(str(cfg.get('__config_dir__') or Path.cwd())).resolve()
        academic_writing = resolve_path(latex_cfg.get('org_latex_class_init'), config_dir)
        latex_extra = resolve_path(latex_cfg.get('latex_extra_path'), config_dir)
        pdf_engine = str(latex_cfg.get('pdf_engine') or 'lualatex')
        stage(f'Compilando PDF PRISMA {phase} via {pdf_engine}')
        pdf_path = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
    return (org_path, pdf_path)

def render_external_prisma_outputs_with_runtime(runtime, /, cfg, out_dir, prefix, prisma_payload, *, phase):
    return _invoke_with_runtime(_ap003e_body_render_external_prisma_outputs_1, runtime, (cfg, out_dir, prefix, prisma_payload), {'phase': phase})

def render_external_prisma_outputs_impl_001(runtime, /, cfg, out_dir, prefix, prisma_payload, *, phase):
    return render_external_prisma_outputs_with_runtime(runtime, cfg, out_dir, prefix, prisma_payload, phase=phase)

def _ap003e_body__prisma_curadoria_default_config_1():
    return 'app_bundle/projetos/prisma_fluxo_pmf/prisma_fluxo_pmf.toml'

def prisma_curadoria_default_config_with_runtime(runtime, /):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_default_config_1, runtime, (), {})

def _prisma_curadoria_default_config_impl_001(runtime, /):
    return prisma_curadoria_default_config_with_runtime(runtime)

def _ap003e_body__prisma_curadoria_default_out_dir_1():
    return 'app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf'

def prisma_curadoria_default_out_dir_with_runtime(runtime, /):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_default_out_dir_1, runtime, (), {})

def _prisma_curadoria_default_out_dir_impl_001(runtime, /):
    return prisma_curadoria_default_out_dir_with_runtime(runtime)

def _ap003e_body__prisma_curadoria_default_prompt_1():
    return ''

def prisma_curadoria_default_prompt_with_runtime(runtime, /):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_default_prompt_1, runtime, (), {})

def _prisma_curadoria_default_prompt_impl_001(runtime, /):
    return prisma_curadoria_default_prompt_with_runtime(runtime)

def _ap003e_body__prisma_curadoria_script_path_1():
    return 'app_bundle/scripts/pipeline/prisma_curadoria_ia_referencias.py'

def prisma_curadoria_script_path_with_runtime(runtime, /):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_script_path_1, runtime, (), {})

def _prisma_curadoria_script_path_impl_001(runtime, /):
    return prisma_curadoria_script_path_with_runtime(runtime)

def _ap003e_body__prisma_curadoria_arg_1(args, name, default):
    return getattr(args, name, default)

def prisma_curadoria_arg_with_runtime(runtime, /, args, name, default):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_arg_1, runtime, (args, name, default), {})

def _prisma_curadoria_arg_impl_001(runtime, /, args, name, default):
    return prisma_curadoria_arg_with_runtime(runtime, args, name, default)

def _ap003e_body__prisma_curadoria_config_from_args_1(args):
    return _prisma_curadoria_arg(args, 'config', None) or _prisma_curadoria_arg(args, 'cfg', None) or _prisma_curadoria_default_config()

def prisma_curadoria_config_from_args_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_config_from_args_1, runtime, (args,), {})

def _prisma_curadoria_config_from_args_impl_001(runtime, /, args):
    return prisma_curadoria_config_from_args_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_out_from_args_1(args):
    return _prisma_curadoria_arg(args, 'prisma_curadoria_out_dir', None) or _prisma_curadoria_default_out_dir()

def prisma_curadoria_out_from_args_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_out_from_args_1, runtime, (args,), {})

def _prisma_curadoria_out_from_args_impl_001(runtime, /, args):
    return prisma_curadoria_out_from_args_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_prompt_from_args_1(args):
    return _prisma_curadoria_arg(args, 'prisma_curadoria_prompt', None) or _prisma_curadoria_default_prompt()

def prisma_curadoria_prompt_from_args_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_prompt_from_args_1, runtime, (args,), {})

def _prisma_curadoria_prompt_from_args_impl_001(runtime, /, args):
    return prisma_curadoria_prompt_from_args_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_input_from_args_1(args, *, default_xlsx):
    explicit = _prisma_curadoria_arg(args, 'prisma_curadoria_input', None)
    if explicit:
        return explicit
    if default_xlsx:
        from pathlib import Path
        return str(Path(_prisma_curadoria_out_from_args(args)) / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_referencias.xlsx')
    return ''

def prisma_curadoria_input_from_args_with_runtime(runtime, /, args, *, default_xlsx):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_input_from_args_1, runtime, (args,), {'default_xlsx': default_xlsx})

def _prisma_curadoria_input_from_args_impl_001(runtime, /, args, *, default_xlsx):
    return prisma_curadoria_input_from_args_with_runtime(runtime, args, default_xlsx=default_xlsx)

def _ap003e_body__prisma_curadoria_run_command_1(cmd):
    import subprocess
    print()
    print('[ETAPA] Executando:')
    print(' '.join(cmd))
    print()
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print('[OK] Etapa concluída.')
    else:
        print(f'[ERRO] Etapa falhou com código {proc.returncode}.')
    return proc.returncode

def prisma_curadoria_run_command_with_runtime(runtime, /, cmd):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_run_command_1, runtime, (cmd,), {})

def _prisma_curadoria_run_command_impl_001(runtime, /, cmd):
    return prisma_curadoria_run_command_with_runtime(runtime, cmd)

def _ap003e_body__prisma_curadoria_build_cmd_1(args, *, usar_ia, reexportar_xlsx):
    import sys
    from pathlib import Path
    script = Path(_prisma_curadoria_script_path())
    if not script.exists():
        raise SystemExit(f'Script de curadoria IA não encontrado: {script}. Rode o aplicador da curadoria IA v2 antes.')
    cmd = [sys.executable, str(script), '--config', _prisma_curadoria_config_from_args(args), '--out-dir', _prisma_curadoria_out_from_args(args)]
    prompt = _prisma_curadoria_prompt_from_args(args)
    if prompt:
        cmd += ['--prompt-curadoria', prompt]
    if reexportar_xlsx:
        input_path = _prisma_curadoria_input_from_args(args, default_xlsx=True)
        cmd += ['--input', input_path, '--reexportar-xlsx']
    else:
        input_path = _prisma_curadoria_input_from_args(args)
        if input_path:
            cmd += ['--input', input_path]
        if usar_ia:
            cmd += ['--usar-ia']
    max_incluir = _prisma_curadoria_arg(args, 'prisma_curadoria_max_incluir', None)
    if max_incluir:
        cmd += ['--max-incluir', str(max_incluir)]
    top_n = _prisma_curadoria_arg(args, 'prisma_curadoria_top_n_candidatos', None)
    if top_n:
        cmd += ['--top-n-candidatos', str(top_n)]
    limiar = _prisma_curadoria_arg(args, 'prisma_curadoria_limiar_minimo', None)
    if limiar:
        cmd += ['--limiar-minimo-inclusao', str(limiar)]
    return cmd

def prisma_curadoria_build_cmd_with_runtime(runtime, /, args, *, usar_ia, reexportar_xlsx):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_build_cmd_1, runtime, (args,), {'usar_ia': usar_ia, 'reexportar_xlsx': reexportar_xlsx})

def _prisma_curadoria_build_cmd_impl_001(runtime, /, args, *, usar_ia, reexportar_xlsx):
    return prisma_curadoria_build_cmd_with_runtime(runtime, args, usar_ia=usar_ia, reexportar_xlsx=reexportar_xlsx)

def _ap003e_body__prisma_curadoria_run_ia_1(args, *, usar_ia):
    cmd = _prisma_curadoria_build_cmd(args, usar_ia=usar_ia, reexportar_xlsx=False)
    return _prisma_curadoria_run_command(cmd)

def prisma_curadoria_run_ia_with_runtime(runtime, /, args, *, usar_ia):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_run_ia_1, runtime, (args,), {'usar_ia': usar_ia})

def _prisma_curadoria_run_ia_impl_001(runtime, /, args, *, usar_ia):
    return prisma_curadoria_run_ia_with_runtime(runtime, args, usar_ia=usar_ia)

def _ap003e_body__prisma_curadoria_reexportar_xlsx_1(args):
    cmd = _prisma_curadoria_build_cmd(args, usar_ia=False, reexportar_xlsx=True)
    return _prisma_curadoria_run_command(cmd)

def prisma_curadoria_reexportar_xlsx_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_reexportar_xlsx_1, runtime, (args,), {})

def _prisma_curadoria_reexportar_xlsx_impl_001(runtime, /, args):
    return prisma_curadoria_reexportar_xlsx_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_pipeline_supports_flag_1(flag):
    import subprocess
    import sys
    try:
        proc = subprocess.run([sys.executable, '-m', 'academic_pipeline', '--help'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    except Exception:
        return False
    return flag in (proc.stdout or '')

def prisma_curadoria_pipeline_supports_flag_with_runtime(runtime, /, flag):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_pipeline_supports_flag_1, runtime, (flag,), {})

def _prisma_curadoria_pipeline_supports_flag_impl_001(runtime, /, flag):
    return prisma_curadoria_pipeline_supports_flag_with_runtime(runtime, flag)

def _ap003e_body__prisma_curadoria_importar_no_pipeline_1(args):
    import sys
    from pathlib import Path
    cfg = _prisma_curadoria_config_from_args(args)
    out_dir = Path(_prisma_curadoria_out_from_args(args))
    triagem = out_dir / 'relatorio_prisma_prisma_fluxo_pmf.triagem_humana.csv'
    if not triagem.exists():
        print(f'[ERRO] CSV de triagem humana não encontrado: {triagem}')
        print('[INFO] Rode primeiro a curadoria IA ou a reexportação do XLSX.')
        return 1
    if _prisma_curadoria_pipeline_supports_flag('--prisma-importar-triagem'):
        cmd = [sys.executable, '-m', 'academic_pipeline', '--config', cfg, '--prisma-importar-triagem', str(triagem)]
    else:
        print('[WARN] --prisma-importar-triagem não apareceu no --help.')
        print('[WARN] O CSV ficará no OUT e o pipeline será executado normalmente.')
        cmd = [sys.executable, '-m', 'academic_pipeline', '--config', cfg]
    return _prisma_curadoria_run_command(cmd)

def prisma_curadoria_importar_no_pipeline_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_importar_no_pipeline_1, runtime, (args,), {})

def _prisma_curadoria_importar_no_pipeline_impl_001(runtime, /, args):
    return prisma_curadoria_importar_no_pipeline_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_fluxo_completo_1(args):
    rc = _prisma_curadoria_run_ia(args, usar_ia=not bool(_prisma_curadoria_arg(args, 'prisma_curadoria_sem_ia', False)))
    if rc:
        return rc
    return _prisma_curadoria_importar_no_pipeline(args)

def prisma_curadoria_fluxo_completo_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_fluxo_completo_1, runtime, (args,), {})

def _prisma_curadoria_fluxo_completo_impl_001(runtime, /, args):
    return prisma_curadoria_fluxo_completo_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_mostrar_caminhos_1(args):
    from pathlib import Path
    out_dir = Path(_prisma_curadoria_out_from_args(args))
    print()
    print('Caminhos da curadoria PRISMA')
    print('=' * 72)
    print(f'Config TOML:        {_prisma_curadoria_config_from_args(args)}')
    print(f'Prompt curadoria:  {_prisma_curadoria_prompt_from_args(args)}')
    print(f'Output PRISMA:     {out_dir}')
    print(f'Script curadoria:  {_prisma_curadoria_script_path()}')
    print()
    print('Arquivos esperados/gerados:')
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.xlsx'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_referencias.xlsx'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.triagem_humana.csv'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_seminario.csv'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_resumo.txt'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_log.json'}")
    print()

def prisma_curadoria_mostrar_caminhos_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_mostrar_caminhos_1, runtime, (args,), {})

def _prisma_curadoria_mostrar_caminhos_impl_001(runtime, /, args):
    return prisma_curadoria_mostrar_caminhos_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_menu_1(args):
    while True:
        print()
        print('PRISMA — Curadoria IA de referências')
        print('=' * 72)
        print('1. Rodar curadoria IA v2 com prompt estruturado')
        print('2. Rodar curadoria sem IA, por heurística local')
        print('3. Reexportar XLSX revisado para triagem_humana.csv')
        print('4. Importar triagem_humana.csv e gerar PRISMA final')
        print('5. Fluxo completo: curadoria + importação/geração PRISMA')
        print('6. Mostrar caminhos/arquivos da curadoria')
        print('0. Sair')
        print()
        escolha = input('Escolha uma opção: ').strip()
        if escolha == '1':
            rc = _prisma_curadoria_run_ia(args, usar_ia=True)
        elif escolha == '2':
            rc = _prisma_curadoria_run_ia(args, usar_ia=False)
        elif escolha == '3':
            rc = _prisma_curadoria_reexportar_xlsx(args)
        elif escolha == '4':
            rc = _prisma_curadoria_importar_no_pipeline(args)
        elif escolha == '5':
            rc = _prisma_curadoria_fluxo_completo(args)
        elif escolha == '6':
            _prisma_curadoria_mostrar_caminhos(args)
            rc = 0
        elif escolha in {'0', 'q', 'Q', 'sair', 'Sair'}:
            return 0
        else:
            print('[WARN] Opção inválida.')
            continue
        if rc:
            return rc

def prisma_curadoria_menu_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_menu_1, runtime, (args,), {})

def _prisma_curadoria_menu_impl_001(runtime, /, args):
    return prisma_curadoria_menu_with_runtime(runtime, args)

def _ap003e_body__prisma_curadoria_dispatch_1(args):
    if _prisma_curadoria_arg(args, 'prisma_curadoria_menu', False):
        return _prisma_curadoria_menu(args)
    if _prisma_curadoria_arg(args, 'prisma_curadoria_reexportar_xlsx', False):
        return _prisma_curadoria_reexportar_xlsx(args)
    if _prisma_curadoria_arg(args, 'prisma_curadoria_fluxo_completo', False):
        return _prisma_curadoria_fluxo_completo(args)
    if _prisma_curadoria_arg(args, 'prisma_curadoria_importar', False):
        return _prisma_curadoria_importar_no_pipeline(args)
    if _prisma_curadoria_arg(args, 'prisma_curadoria_ia', False):
        usar_ia = not bool(_prisma_curadoria_arg(args, 'prisma_curadoria_sem_ia', False))
        return _prisma_curadoria_run_ia(args, usar_ia=usar_ia)
    return 0

def prisma_curadoria_dispatch_with_runtime(runtime, /, args):
    return _invoke_with_runtime(_ap003e_body__prisma_curadoria_dispatch_1, runtime, (args,), {})

def _prisma_curadoria_dispatch_impl_001(runtime, /, args):
    return prisma_curadoria_dispatch_with_runtime(runtime, args)

def _ap003e_body__prisma_artigo_generico_get_arg_1(argv, name):
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + '='):
            return item.split('=', 1)[1]
    return None

def prisma_artigo_generico_get_arg_with_runtime(runtime, /, argv, name):
    return _invoke_with_runtime(_ap003e_body__prisma_artigo_generico_get_arg_1, runtime, (argv, name), {})

def _prisma_artigo_generico_get_arg_impl_001(runtime, /, argv, name):
    return prisma_artigo_generico_get_arg_with_runtime(runtime, argv, name)

def _ap003e_body__prisma_artigo_generico_strip_1(argv):
    bool_flags = {'--prisma-exportar-bib', '--prisma-congelar-artigo', '--prisma-gerar-toml-artigo', '--prisma-gerar-artigo-final'}
    value_flags = {'--prisma-bib-input', '--prisma-bib-output', '--prisma-artigo-dir', '--prisma-congelamento-dir', '--prisma-artigo-toml-output', '--prisma-csl-path', '--prisma-dados-pesquisa-path', '--prisma-artigo-prefix', '--prisma-autor-artigo', '--prisma-professor-artigo', '--prisma-openai-model-artigo'}
    result = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in bool_flags:
            i += 1
            continue
        if item in value_flags:
            i += 2
            continue
        if any((item.startswith(flag + '=') for flag in value_flags)):
            i += 1
            continue
        result.append(item)
        i += 1
    return result

def prisma_artigo_generico_strip_with_runtime(runtime, /, argv):
    return _invoke_with_runtime(_ap003e_body__prisma_artigo_generico_strip_1, runtime, (argv,), {})

def _prisma_artigo_generico_strip_impl_001(runtime, /, argv):
    return prisma_artigo_generico_strip_with_runtime(runtime, argv)

def _ap003e_body__prisma_artigo_generico_out_dir_1(argv):
    from pathlib import Path
    out_arg = _prisma_artigo_generico_get_arg(argv, '--prisma-curadoria-out-dir') or _prisma_artigo_generico_get_arg(argv, '--prisma-out-dir')
    if out_arg:
        return Path(out_arg)
    cfg = _prisma_artigo_generico_get_arg(argv, '--config')
    if cfg:
        cfg_path = Path(cfg)
        if cfg_path.exists():
            return cfg_path.resolve().parent / 'output_pesquisa' / f'relatorio_prisma_{cfg_path.stem}'
    return Path('app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf')

def prisma_artigo_generico_out_dir_with_runtime(runtime, /, argv):
    return _invoke_with_runtime(_ap003e_body__prisma_artigo_generico_out_dir_1, runtime, (argv,), {})

def _prisma_artigo_generico_out_dir_impl_001(runtime, /, argv):
    return prisma_artigo_generico_out_dir_with_runtime(runtime, argv)

def _ap003e_body__prisma_artigo_generico_run_export_1(argv, silent):
    import subprocess, sys
    out_dir = _prisma_artigo_generico_out_dir(argv)
    prefix = out_dir.name if out_dir.name.startswith('relatorio_prisma_') else 'relatorio_prisma_prisma_fluxo_pmf'
    cmd = [sys.executable, '-m', 'app_bundle.scripts.pipeline.prisma_exportar_bib', '--out-dir', str(out_dir), '--prefix', prefix]
    val = _prisma_artigo_generico_get_arg(argv, '--prisma-bib-input')
    if val:
        cmd += ['--input', val]
    val = _prisma_artigo_generico_get_arg(argv, '--prisma-bib-output')
    if val:
        cmd += ['--output', val]
    proc = subprocess.run(cmd)
    if proc.returncode and (not silent):
        raise SystemExit(proc.returncode)
    if proc.returncode and silent:
        print(f'[WARN] Exportação BibLaTeX PRISMA retornou código {proc.returncode}.')
    return proc.returncode

def prisma_artigo_generico_run_export_with_runtime(runtime, /, argv, silent):
    return _invoke_with_runtime(_ap003e_body__prisma_artigo_generico_run_export_1, runtime, (argv, silent), {})

def _prisma_artigo_generico_run_export_impl_001(runtime, /, argv, silent):
    return prisma_artigo_generico_run_export_with_runtime(runtime, argv, silent)

def _ap003e_body__prisma_artigo_generico_run_freeze_1(argv, silent):
    import subprocess, sys
    out_dir = _prisma_artigo_generico_out_dir(argv)
    prefix = out_dir.name if out_dir.name.startswith('relatorio_prisma_') else 'relatorio_prisma_prisma_fluxo_pmf'
    cmd = [sys.executable, '-m', 'app_bundle.scripts.pipeline.prisma_congelar_artigo', '--out-dir', str(out_dir), '--prefix', prefix]
    cfg = _prisma_artigo_generico_get_arg(argv, '--config')
    if cfg:
        cmd += ['--prisma-config', cfg]
    for src, dst in [('--prisma-artigo-dir', '--artigo-dir'), ('--prisma-congelamento-dir', '--dest-dir'), ('--prisma-artigo-toml-output', '--toml-output'), ('--prisma-csl-path', '--csl-path'), ('--prisma-dados-pesquisa-path', '--dados-pesquisa-path'), ('--prisma-artigo-prefix', '--artigo-prefix'), ('--prisma-autor-artigo', '--autor'), ('--prisma-professor-artigo', '--professor'), ('--prisma-openai-model-artigo', '--openai-model')]:
        val = _prisma_artigo_generico_get_arg(argv, src)
        if val:
            cmd += [dst, val]
    if '--prisma-gerar-toml-artigo' in argv or '--prisma-gerar-artigo-final' in argv:
        cmd.append('--gerar-toml-artigo')
    if '--prisma-gerar-artigo-final' in argv:
        cmd.append('--gerar-artigo-final')
    proc = subprocess.run(cmd)
    if proc.returncode and (not silent):
        raise SystemExit(proc.returncode)
    if proc.returncode and silent:
        print(f'[WARN] Congelamento/geração de artigo retornou código {proc.returncode}.')
    return proc.returncode

def prisma_artigo_generico_run_freeze_with_runtime(runtime, /, argv, silent):
    return _invoke_with_runtime(_ap003e_body__prisma_artigo_generico_run_freeze_1, runtime, (argv, silent), {})

def _prisma_artigo_generico_run_freeze_impl_001(runtime, /, argv, silent):
    return prisma_artigo_generico_run_freeze_with_runtime(runtime, argv, silent)

def run_prisma_stage_001(args, runtime):
    if '_prisma_curadoria_dispatch' in runtime:
        _prisma_curadoria_dispatch = runtime['_prisma_curadoria_dispatch']
    if getattr(args, 'prisma_curadoria_menu', False) or getattr(args, 'prisma_curadoria_ia', False) or getattr(args, 'prisma_curadoria_reexportar_xlsx', False) or getattr(args, 'prisma_curadoria_importar', False) or getattr(args, 'prisma_curadoria_fluxo_completo', False):
        return PrismaStageResult(True, _prisma_curadoria_dispatch(args), {})
    _ap003e_values = {}
    for _ap003e_name in ():
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_002(args, runtime):
    if 'cfg' in runtime:
        cfg = runtime['cfg']
    if 'external_search_enabled' in runtime:
        external_search_enabled = runtime['external_search_enabled']
    if 'is_external_prisma_run' in runtime:
        is_external_prisma_run = runtime['is_external_prisma_run']
    if 'output_paths' in runtime:
        output_paths = runtime['output_paths']
    if 'research_output_paths' in runtime:
        research_output_paths = runtime['research_output_paths']
    is_external_prisma_run = external_search_enabled(cfg) and (not args.somente_renderizar)
    out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
    _ap003e_values = {}
    for _ap003e_name in ('is_external_prisma_run', 'out_dir', 'prefix'):
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_003(args, runtime):
    if 'stage' in runtime:
        stage = runtime['stage']
    stage('Validando configuração preventiva')
    _ap003e_values = {}
    for _ap003e_name in ():
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_004(args, runtime):
    if 'PIPELINE_VERSION' in runtime:
        PIPELINE_VERSION = runtime['PIPELINE_VERSION']
    if 'Path' in runtime:
        Path = runtime['Path']
    if 'artifacts' in runtime:
        artifacts = runtime['artifacts']
    if 'cache_dir' in runtime:
        cache_dir = runtime['cache_dir']
    if 'cfg' in runtime:
        cfg = runtime['cfg']
    if 'client' in runtime:
        client = runtime['client']
    if 'is_external_prisma_run' in runtime:
        is_external_prisma_run = runtime['is_external_prisma_run']
    if 'make_client' in runtime:
        make_client = runtime['make_client']
    if 'make_run_report' in runtime:
        make_run_report = runtime['make_run_report']
    if 'model' in runtime:
        model = runtime['model']
    if 'org_path' in runtime:
        org_path = runtime['org_path']
    if 'out_dir' in runtime:
        out_dir = runtime['out_dir']
    if 'outputs' in runtime:
        outputs = runtime['outputs']
    if 'pdf_path' in runtime:
        pdf_path = runtime['pdf_path']
    if 'precheck' in runtime:
        precheck = runtime['precheck']
    if 'prefix' in runtime:
        prefix = runtime['prefix']
    if 'print_outputs' in runtime:
        print_outputs = runtime['print_outputs']
    if 'prisma_outputs' in runtime:
        prisma_outputs = runtime['prisma_outputs']
    if 'prompt_lock' in runtime:
        prompt_lock = runtime['prompt_lock']
    if 'prompt_lock_md' in runtime:
        prompt_lock_md = runtime['prompt_lock_md']
    if 'prompt_lock_path' in runtime:
        prompt_lock_path = runtime['prompt_lock_path']
    if 'render_external_prisma_outputs' in runtime:
        render_external_prisma_outputs = runtime['render_external_prisma_outputs']
    if 'report' in runtime:
        report = runtime['report']
    if 'report_json_path' in runtime:
        report_json_path = runtime['report_json_path']
    if 'run_external_prisma_search' in runtime:
        run_external_prisma_search = runtime['run_external_prisma_search']
    if 'search_cfg' in runtime:
        search_cfg = runtime['search_cfg']
    if 'stage' in runtime:
        stage = runtime['stage']
    if 'warnings' in runtime:
        warnings = runtime['warnings']
    if 'work_dir' in runtime:
        work_dir = runtime['work_dir']
    if 'write_json' in runtime:
        write_json = runtime['write_json']
    if 'write_outputs_manifest' in runtime:
        write_outputs_manifest = runtime['write_outputs_manifest']
    if 'write_prompt_lock' in runtime:
        write_prompt_lock = runtime['write_prompt_lock']
    if 'write_prompt_lock_markdown' in runtime:
        write_prompt_lock_markdown = runtime['write_prompt_lock_markdown']
    if is_external_prisma_run:
        if args.somente_mapa_mental:
            raise RuntimeError('O perfil de busca PRISMA não produz document.json; use a geração normal ou --prisma-importar-triagem.')
        search_cfg = cfg.get('busca_prisma', {}) if isinstance(cfg.get('busca_prisma'), dict) else {}
        if bool(search_cfg.get('pre_triagem_ia', False)):
            stage('Inicializando cliente OpenAI para pré-triagem assistida')
            client, model = make_client(model)
        stage('Executando busca bibliográfica externa e preparando triagem humana')
        prisma_outputs = run_external_prisma_search(cfg, out_dir, prefix, progress=stage, client=client, model=model)
        org_path, pdf_path = render_external_prisma_outputs(cfg, out_dir, prefix, prisma_outputs, phase='preliminar')
        artifacts = prisma_outputs.setdefault('artefatos', {}) if isinstance(prisma_outputs, dict) else {}
        if org_path:
            artifacts['relatorio_org'] = str(org_path)
        if pdf_path:
            artifacts['relatorio_pdf'] = str(pdf_path)
        report_json_path = artifacts.get('prisma_report_json') if isinstance(artifacts, dict) else ''
        if report_json_path:
            write_json(Path(str(report_json_path)), prisma_outputs)
        prompt_lock_path = out_dir / f'{prefix}.prompt_lock.json'
        prompt_lock_md = out_dir / f'{prefix}.prompt_lock.md'
        stage('Registrando prompt_lock')
        prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
        write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
        outputs = {'output_dir': str(out_dir), 'work_dir': str(work_dir), 'cache_dir': str(cache_dir), 'document_json': None, 'org': str(org_path) if org_path else None, 'bib': None, 'pdf': str(pdf_path) if pdf_path else None, 'docx': None, 'relatorio_pesquisa': prisma_outputs, 'prompt_lock': str(prompt_lock_path)}
        report = make_run_report(cfg=cfg, config_path=Path(str(cfg.get('__config_path__'))), out_dir=out_dir, prefix=prefix, model=None, outputs=outputs, warnings=warnings, extra={'mode': 'prisma_busca_externa', 'precheck': precheck})
        write_json(out_dir / f'{prefix}.run_report.json', report)
        write_outputs_manifest(out_dir / f'{prefix}.outputs.txt', outputs)
        print_outputs(outputs, title=f'academic_pipeline {PIPELINE_VERSION} — busca PRISMA concluída; aguarda triagem humana')
        return PrismaStageResult(True, 0, {})
    _ap003e_values = {}
    for _ap003e_name in ('artifacts', 'client', 'model', 'org_path', 'outputs', 'pdf_path', 'prisma_outputs', 'prompt_lock', 'prompt_lock_md', 'prompt_lock_path', 'report', 'report_json_path', 'search_cfg'):
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_005(args, runtime):
    prisma_outputs = None
    _ap003e_values = {}
    for _ap003e_name in ('prisma_outputs',):
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_006(args, runtime):
    if '_json_or_none' in runtime:
        _json_or_none = runtime['_json_or_none']
    if 'bib_path' in runtime:
        bib_path = runtime['bib_path']
    if 'document' in runtime:
        document = runtime['document']
    if 'document_json_path' in runtime:
        document_json_path = runtime['document_json_path']
    if 'docx_path' in runtime:
        docx_path = runtime['docx_path']
    if 'org_path' in runtime:
        org_path = runtime['org_path']
    if 'out_dir' in runtime:
        out_dir = runtime['out_dir']
    if 'paper_abstract_bundle' in runtime:
        paper_abstract_bundle = runtime['paper_abstract_bundle']
    if 'paper_abstract_path' in runtime:
        paper_abstract_path = runtime['paper_abstract_path']
    if 'pdf_path' in runtime:
        pdf_path = runtime['pdf_path']
    if 'prisma_outputs' in runtime:
        prisma_outputs = runtime['prisma_outputs']
    if 'translated_outputs' in runtime:
        translated_outputs = runtime['translated_outputs']
    outputs = {'output_dir': str(out_dir), 'document_json': str(document_json_path), 'org': str(org_path), 'bib': str(bib_path), 'pdf': str(pdf_path) if pdf_path else None, 'docx': str(docx_path) if docx_path else None, 'resumos_paper': str(paper_abstract_path) if paper_abstract_bundle else None, 'idiomas_adicionais': translated_outputs, 'relatorio_pesquisa': _json_or_none(getattr(document.diagnostics, 'relatorio_pesquisa_json', '')) if getattr(document, 'diagnostics', None) else prisma_outputs}
    _ap003e_values = {}
    for _ap003e_name in ('outputs',):
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_007(args, runtime):
    if 'stage' in runtime:
        stage = runtime['stage']
    stage('Executando conformidade institucional')
    _ap003e_values = {}
    for _ap003e_name in ():
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def run_prisma_stage_008(args, runtime):
    if 'stage' in runtime:
        stage = runtime['stage']
    stage('Gerando relatório de qualidade')
    _ap003e_values = {}
    for _ap003e_name in ():
        if _ap003e_name in locals():
            _ap003e_values[_ap003e_name] = locals()[_ap003e_name]
    return PrismaStageResult(False, None, _ap003e_values)

def _ap003e_body_main_2(*args, **kwargs):
    import sys
    original_argv = list(sys.argv[1:])
    has_import = '--prisma-curadoria-importar' in original_argv or '--prisma-curadoria-fluxo-completo' in original_argv
    wants_export = '--prisma-exportar-bib' in original_argv
    wants_freeze = '--prisma-congelar-artigo' in original_argv
    wants_toml = '--prisma-gerar-toml-artigo' in original_argv
    wants_final = '--prisma-gerar-artigo-final' in original_argv
    if not has_import and wants_export and (not wants_freeze) and (not wants_toml) and (not wants_final):
        return _prisma_artigo_generico_run_export(original_argv, silent=False)
    if not has_import and (wants_freeze or wants_toml or wants_final):
        _prisma_artigo_generico_run_export(original_argv, silent=True)
        return _prisma_artigo_generico_run_freeze(original_argv, silent=False)
    if has_import and (wants_export or wants_freeze or wants_toml or wants_final):
        old_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + _prisma_artigo_generico_strip(original_argv)
        try:
            rc = _ap003f_pipeline_core(*args, **kwargs)
        finally:
            sys.argv = old_argv
    else:
        rc = _ap003f_pipeline_core(*args, **kwargs)
    if has_import:
        _prisma_artigo_generico_run_export(original_argv, silent=True)
    if wants_freeze or wants_toml or wants_final:
        _prisma_artigo_generico_run_freeze(original_argv, silent=False)
    return rc

def run_prisma_generic_with_runtime(runtime, /, *args, **kwargs):
    return _invoke_with_runtime(_ap003e_body_main_2, runtime, (*args,), {**kwargs})

def run_prisma_generic_entrypoint(runtime, /, *args, **kwargs):
    return run_prisma_generic_with_runtime(runtime, *args, **kwargs)
__all__ = ['PrismaStageResult', 'stage_impl_001', '_json_or_none_impl_001', 'make_client_impl_001', '_section_impl_001', 'research_output_paths_impl_001', 'render_external_prisma_outputs_impl_001', '_prisma_curadoria_default_config_impl_001', '_prisma_curadoria_default_out_dir_impl_001', '_prisma_curadoria_default_prompt_impl_001', '_prisma_curadoria_script_path_impl_001', '_prisma_curadoria_arg_impl_001', '_prisma_curadoria_config_from_args_impl_001', '_prisma_curadoria_out_from_args_impl_001', '_prisma_curadoria_prompt_from_args_impl_001', '_prisma_curadoria_input_from_args_impl_001', '_prisma_curadoria_run_command_impl_001', '_prisma_curadoria_build_cmd_impl_001', '_prisma_curadoria_run_ia_impl_001', '_prisma_curadoria_reexportar_xlsx_impl_001', '_prisma_curadoria_pipeline_supports_flag_impl_001', '_prisma_curadoria_importar_no_pipeline_impl_001', '_prisma_curadoria_fluxo_completo_impl_001', '_prisma_curadoria_mostrar_caminhos_impl_001', '_prisma_curadoria_menu_impl_001', '_prisma_curadoria_dispatch_impl_001', '_prisma_artigo_generico_get_arg_impl_001', '_prisma_artigo_generico_strip_impl_001', '_prisma_artigo_generico_out_dir_impl_001', '_prisma_artigo_generico_run_export_impl_001', '_prisma_artigo_generico_run_freeze_impl_001', 'run_prisma_stage_001', 'run_prisma_stage_002', 'run_prisma_stage_003', 'run_prisma_stage_004', 'run_prisma_stage_005', 'run_prisma_stage_006', 'run_prisma_stage_007', 'run_prisma_stage_008', 'run_prisma_generic_entrypoint', 'stage_with_runtime', 'json_or_none_with_runtime', 'make_client_with_runtime', 'section_with_runtime', 'research_output_paths_with_runtime', 'render_external_prisma_outputs_with_runtime', 'prisma_curadoria_default_config_with_runtime', 'prisma_curadoria_default_out_dir_with_runtime', 'prisma_curadoria_default_prompt_with_runtime', 'prisma_curadoria_script_path_with_runtime', 'prisma_curadoria_arg_with_runtime', 'prisma_curadoria_config_from_args_with_runtime', 'prisma_curadoria_out_from_args_with_runtime', 'prisma_curadoria_prompt_from_args_with_runtime', 'prisma_curadoria_input_from_args_with_runtime', 'prisma_curadoria_run_command_with_runtime', 'prisma_curadoria_build_cmd_with_runtime', 'prisma_curadoria_run_ia_with_runtime', 'prisma_curadoria_reexportar_xlsx_with_runtime', 'prisma_curadoria_pipeline_supports_flag_with_runtime', 'prisma_curadoria_importar_no_pipeline_with_runtime', 'prisma_curadoria_fluxo_completo_with_runtime', 'prisma_curadoria_mostrar_caminhos_with_runtime', 'prisma_curadoria_menu_with_runtime', 'prisma_curadoria_dispatch_with_runtime', 'prisma_artigo_generico_get_arg_with_runtime', 'prisma_artigo_generico_strip_with_runtime', 'prisma_artigo_generico_out_dir_with_runtime', 'prisma_artigo_generico_run_export_with_runtime', 'prisma_artigo_generico_run_freeze_with_runtime', 'run_prisma_generic_with_runtime']
