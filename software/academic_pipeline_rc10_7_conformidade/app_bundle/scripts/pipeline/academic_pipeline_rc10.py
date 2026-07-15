#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""academic_pipeline rc10.7.20 — seção [paths] e override de saída por CLI.

A IA gera AcademicDocument/document.json. Renderizadores determinísticos geram
ORG/PDF/DOCX e relatório PRISMA. Esta versão acrescenta:
- --doctor para diagnóstico de ambiente;
- --check-config para validação preventiva do TOML;
- --recompile para recompilar ORG sem chamar IA;
- run_report.json e outputs.txt para rastreabilidade;
- validação básica do DOCX;
- modo --somente-renderizar sem exigir OPENAI_API_KEY;
- --init-project para criar estrutura de projeto;
- --make-doi-manifest para gerar CSV a partir de ZIP/pasta;
- --inspect-bib para diagnóstico bibliográfico;
- quality_report.md/json após a geração;
- perfis institucionais em app_bundle/institutions/<perfil>;
- prompt bank reutilizável em app_bundle/prompts e app_bundle/institutions/<perfil>/prompts;
- conformidade institucional auditável;
- prompt_lock.json/md para rastreabilidade de diretivas;
- schema estrito compatível com OpenAI Structured Outputs;
- impressão de etapas em tempo real com [ETAPA];
- saneamento de menções técnicas no conteúdo visível antes da validação.
- perfil resumo_artigos_local_fgv e perguntas condicionais por tipo documental.
- perfil de resumo de artigos com profundidade analítica configurável, matriz por texto e comparação transversal robusta.
- seção [paths] como fonte única para saídas/work/cache e overrides por linha de comando.
- --tui para central operacional visual de atividades acadêmicas em terminal.
- --gui para interface gráfica opcional de atividades acadêmicas.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .bibliography_manager import build_bibliography
else:
    from bibliography_manager import build_bibliography
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .corpus_manager import (
        collect_orientation_docs,
        copy_documents_to_fulltext_cache,
        discover_local_documents,
    )
else:
    from corpus_manager import (
        collect_orientation_docs,
        copy_documents_to_fulltext_cache,
        discover_local_documents,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .document_builder import build_document_model
else:
    from document_builder import build_document_model
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .document_model import AcademicDocument
else:
    from document_model import AcademicDocument
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .document_validator import (
        raise_if_errors,
        validate_document_model,
        validate_org_text,
        sanitize_document_model_technical_leaks,
        sanitize_document_model_raw_bibkeys,
    )
else:
    from document_validator import (
        raise_if_errors,
        validate_document_model,
        validate_org_text,
        sanitize_document_model_technical_leaks,
        sanitize_document_model_raw_bibkeys,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .diagnostics import (
        PIPELINE_VERSION,
        check_config,
        clean_aux_files,
        make_run_report,
        print_check_config_report,
        print_doctor_report,
        print_outputs,
        run_doctor,
        validate_docx_file,
        write_outputs_manifest,
    )
else:
    from diagnostics import (
        PIPELINE_VERSION,
        check_config,
        clean_aux_files,
        make_run_report,
        print_check_config_report,
        print_doctor_report,
        print_outputs,
        run_doctor,
        validate_docx_file,
        write_outputs_manifest,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .latex_compile import run_compile_sequence
else:
    from latex_compile import run_compile_sequence
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .mindmap_manager import (
        generate_and_attach_mindmap,
        should_generate_mindmap,
        attach_existing_mindmap_if_available,
        delete_existing_mindmap_outputs,
    )
else:
    from mindmap_manager import (
        generate_and_attach_mindmap,
        should_generate_mindmap,
        attach_existing_mindmap_if_available,
        delete_existing_mindmap_outputs,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .render_docx import render_docx
else:
    from render_docx import render_docx
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .render_org_latex import render_org_latex
else:
    from render_org_latex import render_org_latex
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .prisma_pipeline import run_prisma_report_outputs, prisma_enabled
else:
    from prisma_pipeline import run_prisma_report_outputs, prisma_enabled
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .prisma_busca_externa import (
        external_search_enabled,
        import_manual_prisma_triage,
        render_external_prisma_org_report,
        run_external_prisma_search,
    )
else:
    from prisma_busca_externa import (
        external_search_enabled,
        import_manual_prisma_triage,
        render_external_prisma_org_report,
        run_external_prisma_search,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .utils import write_json, resolve_path
else:
    from utils import write_json, resolve_path
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .project_tools import (
        init_project,
        make_doi_manifest,
        inspect_bib,
        render_bib_inspection_markdown,
    )
else:
    from project_tools import (
        init_project,
        make_doi_manifest,
        inspect_bib,
        render_bib_inspection_markdown,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .quality_report import build_quality_report, write_quality_report
else:
    from quality_report import build_quality_report, write_quality_report
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .institution_profiles import apply_institution_profile, describe_institution_profiles
else:
    from institution_profiles import apply_institution_profile, describe_institution_profiles
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .institution_layouts import available_layouts, resolve_layout_spec
else:
    from institution_layouts import available_layouts, resolve_layout_spec
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .prompt_manager import prompt_report_for_cfg, load_prompt_bundle
else:
    from prompt_manager import prompt_report_for_cfg, load_prompt_bundle
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .institution_explainer import explain_profile
else:
    from institution_explainer import explain_profile
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .institution_compliance import (
        run_institution_compliance,
        render_compliance_markdown,
        write_compliance_reports,
    )
else:
    from institution_compliance import (
        run_institution_compliance,
        render_compliance_markdown,
        write_compliance_reports,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .prompt_lock import write_prompt_lock, write_prompt_lock_markdown, build_prompt_lock
else:
    from prompt_lock import write_prompt_lock, write_prompt_lock_markdown, build_prompt_lock
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .document_translation import (
        TranslationError,
        requested_translation_languages,
        translation_batch_size,
        translate_document_model,
    )
else:
    from document_translation import (
        TranslationError,
        requested_translation_languages,
        translation_batch_size,
        translate_document_model,
    )
# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from .paper_abstracts import (
        PaperAbstractError,
        abstract_sidecar_path,
        generate_paper_abstract_bundle,
        inject_paper_abstracts_into_docx,
        inject_paper_abstracts_into_org,
        main_document_abstract_languages,
        paper_abstracts_enabled,
        read_paper_abstract_bundle,
        write_paper_abstract_bundle,
    )
else:
    from paper_abstracts import (
        PaperAbstractError,
        abstract_sidecar_path,
        generate_paper_abstract_bundle,
        inject_paper_abstracts_into_docx,
        inject_paper_abstracts_into_org,
        main_document_abstract_languages,
        paper_abstracts_enabled,
        read_paper_abstract_bundle,
        write_paper_abstract_bundle,
    )

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")


def stage(message: str) -> None:
    """Mostra etapas de execução em tempo real."""
    print(f"[ETAPA] {message}", flush=True)


def _json_or_none(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["__config_path__"] = str(path.resolve())
    cfg["__config_dir__"] = str(path.resolve().parent)
    cfg = apply_institution_profile(cfg)
    return cfg


def make_client(model_override: str | None = None) -> tuple[Any, str]:
    from openai import OpenAI
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY não encontrado no ambiente/.env.")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), (model_override or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name, {})
    return sec if isinstance(sec, dict) else {}


def output_paths(cfg: dict[str, Any]) -> tuple[Path, str]:
    """Resolve a saída final do documento pela seção [paths]."""
    from academic_pipeline.document_orchestration import output_paths_impl as _ap003d_impl_output_paths
    return _ap003d_impl_output_paths({**globals(), **locals()}, cfg)


def research_output_paths(cfg: dict[str, Any]) -> tuple[Path, str]:
    """Resolve a saída canônica da busca e consolidação PRISMA.

    A pesquisa bibliográfica não deve compartilhar ``document_output_dir`` com
    documentos acadêmicos. ``research_output_dir`` e ``research_prefix`` são
    resolvidos em relação ao TOML; na ausência deles, usa-se uma pasta
    ``output_pesquisa`` dentro do próprio projeto.
    """
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths = _section(cfg, "paths")
    projeto = _section(cfg, "projeto")
    project_name = str(projeto.get("nome") or "").strip()
    default_prefix = f"relatorio_prisma_{project_name}" if project_name else "relatorio_prisma"
    prefix = str(paths.get("research_prefix") or default_prefix).strip() or default_prefix
    out_base = resolve_path(paths.get("research_output_dir") or "output_pesquisa", config_dir) or (config_dir / "output_pesquisa")
    out_dir = out_base / prefix if bool(paths.get("create_research_subdir", True)) else out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


def work_cache_paths(cfg: dict[str, Any], prefix: str) -> tuple[Path, Path]:
    """Resolve diretórios operacionais pela seção [paths]."""
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths = _section(cfg, "paths")
    work_base = resolve_path(paths.get("work_dir") or "../../output/work", config_dir) or (config_dir / "output/work")
    cache_base = resolve_path(paths.get("cache_dir") or "../../output/cache", config_dir) or (config_dir / "output/cache")
    work_dir = work_base / prefix if bool(paths.get("create_work_subdir", True)) else work_base
    cache_dir = cache_base / prefix if bool(paths.get("create_cache_subdir", True)) else cache_base
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return work_dir, cache_dir


def apply_cli_path_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Aplica overrides de caminhos informados na linha de comando.

    A prioridade fica: CLI > TOML. Os caminhos continuam sendo resolvidos
    posteriormente em relação ao diretório do TOML, salvo quando absolutos.
    """
    from academic_pipeline.document_orchestration import apply_cli_path_overrides_impl as _ap003d_impl_apply_cli_path_overrides
    return _ap003d_impl_apply_cli_path_overrides({**globals(), **locals()}, cfg, args)


def load_existing_document_json(path: Path) -> AcademicDocument:
    from academic_pipeline.document_orchestration import load_existing_document_json_impl as _ap003d_impl_load_existing_document_json
    return _ap003d_impl_load_existing_document_json({**globals(), **locals()}, path)


def resolve_bib_for_existing_document(document: AcademicDocument, document_json_path: Path, out_dir: Path, prefix: str) -> tuple[Path, list[str]]:
    """Resolve o .bib em modo --somente-renderizar sem exigir que ele já esteja no output_dir."""
    from academic_pipeline.document_orchestration import resolve_bib_for_existing_document_impl as _ap003d_impl_resolve_bib_for_existing_document
    return _ap003d_impl_resolve_bib_for_existing_document({**globals(), **locals()}, document, document_json_path, out_dir, prefix)


def _openai_model_from_cfg(cfg: dict[str, Any]) -> str:
    return str((cfg.get("openai", {}) if isinstance(cfg.get("openai"), dict) else {}).get("model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def _load_optional_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config não encontrada: {p}")
    return load_config(p)


def _resolve_latex_paths_for_recompile(args: argparse.Namespace, cfg: dict[str, Any] | None) -> tuple[Path | None, Path | None, str]:
    from academic_pipeline.document_orchestration import _resolve_latex_paths_for_recompile_impl as _ap003d_impl__resolve_latex_paths_for_recompile
    return _ap003d_impl__resolve_latex_paths_for_recompile({**globals(), **locals()}, args, cfg)


def run_recompile(args: argparse.Namespace, cfg: dict[str, Any] | None) -> int:
    from academic_pipeline.document_orchestration import run_recompile_impl as _ap003d_impl_run_recompile
    return _ap003d_impl_run_recompile({**globals(), **locals()}, args, cfg)



def render_external_prisma_outputs(
    cfg: dict[str, Any],
    out_dir: Path,
    prefix: str,
    prisma_payload: dict[str, Any],
    *,
    phase: str,
) -> tuple[Path | None, Path | None]:
    """Renderiza relatório PRISMA externo em ORG e, quando solicitado, em PDF.

    O perfil de busca não constrói ``document.json``. Por isso, esta rotina
    usa o relatório estruturado da busca/triagem e preserva o layout e a engine
    de LaTeX definidos no TOML, como os demais perfis que exportam PDF.
    """
    report_cfg = cfg.get("relatorio_pesquisa", {}) if isinstance(cfg.get("relatorio_pesquisa"), dict) else {}
    org_requested = bool(report_cfg.get("exportar_org", True))
    pdf_requested = bool(report_cfg.get("exportar_pdf", False))
    if not org_requested and not pdf_requested:
        return None, None

    # PDF é compilado a partir de ORG. Mesmo quando o usuário opta somente
    # pelo PDF, o ORG é mantido para reprodutibilidade e recompilação futura.
    stage(f"Renderizando relatório PRISMA {phase} em ORG")
    org_path = render_external_prisma_org_report(cfg, out_dir, prefix, prisma_payload, phase=phase)
    # >>> PATCH_PRISMA_DIAGRAMA_FLUXO_AUTOMATICO_V1 >>>
    try:
        # Compatibilidade temporária entre pacote e script direto.
        if __package__:
            from .prisma_diagrama_fluxo import ensure_prisma_flow_diagram
        else:
            from prisma_diagrama_fluxo import ensure_prisma_flow_diagram
        ensure_prisma_flow_diagram(
            cfg=cfg,
            out_dir=out_dir,
            prefix=prefix,
            org_path=org_path,
            prisma_payload=prisma_payload,
            phase=phase,
        )
    except Exception as exc:
        print(f"[WARN] Não foi possível gerar/inserir o diagrama PRISMA: {exc}")
    # <<< PATCH_PRISMA_DIAGRAMA_FLUXO_AUTOMATICO_V1 <<<

    pdf_path: Path | None = None
    if pdf_requested:
        latex_cfg = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
        config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
        academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
        latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
        pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")
        stage(f"Compilando PDF PRISMA {phase} via {pdf_engine}")
        pdf_path = run_compile_sequence(
            org_path,
            academic_writing=academic_writing,
            latex_extra_path=latex_extra,
            pdf_engine=pdf_engine,
        )
    return org_path, pdf_path


def render_additional_language_versions(*, client: Any, model: str, cfg: dict[str, Any], document: AcademicDocument, bib_path: Path, bib_keys: list[str], out_dir: Path, prefix: str, doc_cfg: dict[str, Any], latex_cfg: dict[str, Any], config_dir: Path, abstract_bundle: dict[str, Any] | None=None) -> tuple[dict[str, Any], list[str]]:
    """Traduz e renderiza versões adicionais a partir do document.json canônico.

    Cada versão recebe diretório próprio dentro de ``idiomas/<codigo>`` e
    compartilha a bibliografia original, copiada sem tradução. A função nunca
    consulta novamente o corpus nem gera uma segunda análise acadêmica.
    """
    from academic_pipeline.document_orchestration import render_additional_language_versions_impl as _ap003d_impl_render_additional_language_versions
    return _ap003d_impl_render_additional_language_versions({**globals(), **locals()}, client=client, model=model, cfg=cfg, document=document, bib_path=bib_path, bib_keys=bib_keys, out_dir=out_dir, prefix=prefix, doc_cfg=doc_cfg, latex_cfg=latex_cfg, config_dir=config_dir, abstract_bundle=abstract_bundle)



# >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1 >>>
def _prisma_curadoria_default_config() -> str:
    return "app_bundle/projetos/prisma_fluxo_pmf/prisma_fluxo_pmf.toml"


def _prisma_curadoria_default_out_dir() -> str:
    return "app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf"


def _prisma_curadoria_default_prompt() -> str:
    return "/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo/prompt_curadoria_atestmed_ia.yaml"


def _prisma_curadoria_script_path() -> str:
    return "app_bundle/scripts/pipeline/prisma_curadoria_ia_referencias.py"


def _prisma_curadoria_arg(args, name: str, default=None):
    return getattr(args, name, default)


def _prisma_curadoria_config_from_args(args) -> str:
    return (
        _prisma_curadoria_arg(args, "config", None)
        or _prisma_curadoria_arg(args, "cfg", None)
        or _prisma_curadoria_default_config()
    )


def _prisma_curadoria_out_from_args(args) -> str:
    return _prisma_curadoria_arg(args, "prisma_curadoria_out_dir", None) or _prisma_curadoria_default_out_dir()


def _prisma_curadoria_prompt_from_args(args) -> str:
    return _prisma_curadoria_arg(args, "prisma_curadoria_prompt", None) or _prisma_curadoria_default_prompt()


def _prisma_curadoria_input_from_args(args, *, default_xlsx: bool = False) -> str:
    explicit = _prisma_curadoria_arg(args, "prisma_curadoria_input", None)
    if explicit:
        return explicit
    if default_xlsx:
        from pathlib import Path
        return str(Path(_prisma_curadoria_out_from_args(args)) / "relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_referencias.xlsx")
    return ""


def _prisma_curadoria_run_command(cmd: list[str]) -> int:
    import subprocess
    print()
    print("[ETAPA] Executando:")
    print(" ".join(cmd))
    print()
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print("[OK] Etapa concluída.")
    else:
        print(f"[ERRO] Etapa falhou com código {proc.returncode}.")
    return proc.returncode


def _prisma_curadoria_build_cmd(args, *, usar_ia: bool, reexportar_xlsx: bool = False) -> list[str]:
    import sys
    from pathlib import Path

    script = Path(_prisma_curadoria_script_path())
    if not script.exists():
        raise SystemExit(
            "Script de curadoria IA não encontrado: "
            f"{script}. Rode o aplicador da curadoria IA v2 antes."
        )

    cmd = [
        sys.executable,
        str(script),
        "--config",
        _prisma_curadoria_config_from_args(args),
        "--out-dir",
        _prisma_curadoria_out_from_args(args),
    ]

    prompt = _prisma_curadoria_prompt_from_args(args)
    if prompt:
        cmd += ["--prompt-curadoria", prompt]

    if reexportar_xlsx:
        input_path = _prisma_curadoria_input_from_args(args, default_xlsx=True)
        cmd += ["--input", input_path, "--reexportar-xlsx"]
    else:
        input_path = _prisma_curadoria_input_from_args(args)
        if input_path:
            cmd += ["--input", input_path]
        if usar_ia:
            cmd += ["--usar-ia"]

    max_incluir = _prisma_curadoria_arg(args, "prisma_curadoria_max_incluir", None)
    if max_incluir:
        cmd += ["--max-incluir", str(max_incluir)]

    top_n = _prisma_curadoria_arg(args, "prisma_curadoria_top_n_candidatos", None)
    if top_n:
        cmd += ["--top-n-candidatos", str(top_n)]

    limiar = _prisma_curadoria_arg(args, "prisma_curadoria_limiar_minimo", None)
    if limiar:
        cmd += ["--limiar-minimo-inclusao", str(limiar)]

    return cmd


def _prisma_curadoria_run_ia(args, *, usar_ia: bool = True) -> int:
    cmd = _prisma_curadoria_build_cmd(args, usar_ia=usar_ia, reexportar_xlsx=False)
    return _prisma_curadoria_run_command(cmd)


def _prisma_curadoria_reexportar_xlsx(args) -> int:
    cmd = _prisma_curadoria_build_cmd(args, usar_ia=False, reexportar_xlsx=True)
    return _prisma_curadoria_run_command(cmd)


def _prisma_curadoria_pipeline_supports_flag(flag: str) -> bool:
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except Exception:
        return False
    return flag in (proc.stdout or "")


def _prisma_curadoria_importar_no_pipeline(args) -> int:
    import sys
    from pathlib import Path

    cfg = _prisma_curadoria_config_from_args(args)
    out_dir = Path(_prisma_curadoria_out_from_args(args))
    triagem = out_dir / "relatorio_prisma_prisma_fluxo_pmf.triagem_humana.csv"

    if not triagem.exists():
        print(f"[ERRO] CSV de triagem humana não encontrado: {triagem}")
        print("[INFO] Rode primeiro a curadoria IA ou a reexportação do XLSX.")
        return 1

    if _prisma_curadoria_pipeline_supports_flag("--prisma-importar-triagem"):
        cmd = [sys.executable, __file__, "--config", cfg, "--prisma-importar-triagem", str(triagem)]
    else:
        print("[WARN] --prisma-importar-triagem não apareceu no --help.")
        print("[WARN] O CSV ficará no OUT e o pipeline será executado normalmente.")
        cmd = [sys.executable, __file__, "--config", cfg]

    return _prisma_curadoria_run_command(cmd)


def _prisma_curadoria_fluxo_completo(args) -> int:
    rc = _prisma_curadoria_run_ia(
        args,
        usar_ia=not bool(_prisma_curadoria_arg(args, "prisma_curadoria_sem_ia", False)),
    )
    if rc:
        return rc
    return _prisma_curadoria_importar_no_pipeline(args)


def _prisma_curadoria_mostrar_caminhos(args) -> None:
    from pathlib import Path

    out_dir = Path(_prisma_curadoria_out_from_args(args))
    print()
    print("Caminhos da curadoria PRISMA")
    print("=" * 72)
    print(f"Config TOML:        {_prisma_curadoria_config_from_args(args)}")
    print(f"Prompt curadoria:  {_prisma_curadoria_prompt_from_args(args)}")
    print(f"Output PRISMA:     {out_dir}")
    print(f"Script curadoria:  {_prisma_curadoria_script_path()}")
    print()
    print("Arquivos esperados/gerados:")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.xlsx'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_referencias.xlsx'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.triagem_humana.csv'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_seminario.csv'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_resumo.txt'}")
    print(f"- {out_dir / 'relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_log.json'}")
    print()


def _prisma_curadoria_menu(args) -> int:
    while True:
        print()
        print("PRISMA — Curadoria IA de referências")
        print("=" * 72)
        print("1. Rodar curadoria IA v2 com prompt estruturado")
        print("2. Rodar curadoria sem IA, por heurística local")
        print("3. Reexportar XLSX revisado para triagem_humana.csv")
        print("4. Importar triagem_humana.csv e gerar PRISMA final")
        print("5. Fluxo completo: curadoria + importação/geração PRISMA")
        print("6. Mostrar caminhos/arquivos da curadoria")
        print("0. Sair")
        print()

        escolha = input("Escolha uma opção: ").strip()

        if escolha == "1":
            rc = _prisma_curadoria_run_ia(args, usar_ia=True)
        elif escolha == "2":
            rc = _prisma_curadoria_run_ia(args, usar_ia=False)
        elif escolha == "3":
            rc = _prisma_curadoria_reexportar_xlsx(args)
        elif escolha == "4":
            rc = _prisma_curadoria_importar_no_pipeline(args)
        elif escolha == "5":
            rc = _prisma_curadoria_fluxo_completo(args)
        elif escolha == "6":
            _prisma_curadoria_mostrar_caminhos(args)
            rc = 0
        elif escolha in {"0", "q", "Q", "sair", "Sair"}:
            return 0
        else:
            print("[WARN] Opção inválida.")
            continue

        if rc:
            return rc


def _prisma_curadoria_dispatch(args) -> int:
    if _prisma_curadoria_arg(args, "prisma_curadoria_menu", False):
        return _prisma_curadoria_menu(args)
    if _prisma_curadoria_arg(args, "prisma_curadoria_reexportar_xlsx", False):
        return _prisma_curadoria_reexportar_xlsx(args)
    if _prisma_curadoria_arg(args, "prisma_curadoria_fluxo_completo", False):
        return _prisma_curadoria_fluxo_completo(args)
    if _prisma_curadoria_arg(args, "prisma_curadoria_importar", False):
        return _prisma_curadoria_importar_no_pipeline(args)
    if _prisma_curadoria_arg(args, "prisma_curadoria_ia", False):
        usar_ia = not bool(_prisma_curadoria_arg(args, "prisma_curadoria_sem_ia", False))
        return _prisma_curadoria_run_ia(args, usar_ia=usar_ia)
    return 0
# <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1 <<<


def main() -> int:
    from academic_pipeline.cli_parser import parse_args as parse_cli_args

    args = parse_cli_args(pipeline_version=PIPELINE_VERSION)

    # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH >>>
    if (
        getattr(args, "prisma_curadoria_menu", False)
        or getattr(args, "prisma_curadoria_ia", False)
        or getattr(args, "prisma_curadoria_reexportar_xlsx", False)
        or getattr(args, "prisma_curadoria_importar", False)
        or getattr(args, "prisma_curadoria_fluxo_completo", False)
    ):
        return _prisma_curadoria_dispatch(args)
    # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH <<<
    from academic_pipeline.command_dispatch import (
        dispatch_stage_001 as _ap003c_dispatch_001,
    )

    _ap003c_result_001 = _ap003c_dispatch_001(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_001.handled:
        return _ap003c_result_001.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_002 as _ap003c_dispatch_002,
    )

    _ap003c_result_002 = _ap003c_dispatch_002(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_002.handled:
        return _ap003c_result_002.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_003 as _ap003c_dispatch_003,
    )

    _ap003c_result_003 = _ap003c_dispatch_003(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_003.handled:
        return _ap003c_result_003.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_004 as _ap003c_dispatch_004,
    )

    _ap003c_result_004 = _ap003c_dispatch_004(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_004.handled:
        return _ap003c_result_004.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_005 as _ap003c_dispatch_005,
    )

    _ap003c_result_005 = _ap003c_dispatch_005(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_005.handled:
        return _ap003c_result_005.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_006 as _ap003c_dispatch_006,
    )

    _ap003c_result_006 = _ap003c_dispatch_006(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_006.handled:
        return _ap003c_result_006.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_007 as _ap003c_dispatch_007,
    )

    _ap003c_result_007 = _ap003c_dispatch_007(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_007.handled:
        return _ap003c_result_007.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_008 as _ap003c_dispatch_008,
    )

    _ap003c_result_008 = _ap003c_dispatch_008(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_008.handled:
        return _ap003c_result_008.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_009 as _ap003c_dispatch_009,
    )

    _ap003c_result_009 = _ap003c_dispatch_009(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_009.handled:
        return _ap003c_result_009.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_010 as _ap003c_dispatch_010,
    )

    _ap003c_result_010 = _ap003c_dispatch_010(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_010.handled:
        return _ap003c_result_010.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_011 as _ap003c_dispatch_011,
    )

    _ap003c_result_011 = _ap003c_dispatch_011(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_011.handled:
        return _ap003c_result_011.value

    from academic_pipeline.document_orchestration import (
        run_document_stage_001 as _ap003d_stage_001,
    )

    _ap003d_result_001 = _ap003d_stage_001(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_001.terminal:
        return _ap003d_result_001.value
    if 'bib_entry_key' in _ap003d_result_001.values:
        bib_entry_key = _ap003d_result_001.values['bib_entry_key']
    if 'bib_keys' in _ap003d_result_001.values:
        bib_keys = _ap003d_result_001.values['bib_keys']
    if 'bib_path' in _ap003d_result_001.values:
        bib_path = _ap003d_result_001.values['bib_path']
    if 'document' in _ap003d_result_001.values:
        document = _ap003d_result_001.values['document']
    if 'document_json' in _ap003d_result_001.values:
        document_json = _ap003d_result_001.values['document_json']
    if 'e' in _ap003d_result_001.values:
        e = _ap003d_result_001.values['e']
    if 'k' in _ap003d_result_001.values:
        k = _ap003d_result_001.values['k']
    if 'org' in _ap003d_result_001.values:
        org = _ap003d_result_001.values['org']
    if 'out' in _ap003d_result_001.values:
        out = _ap003d_result_001.values['out']
    if 'report' in _ap003d_result_001.values:
        report = _ap003d_result_001.values['report']
    if 'split_bib_entries' in _ap003d_result_001.values:
        split_bib_entries = _ap003d_result_001.values['split_bib_entries']

    cfg = _load_optional_config(args.config) if args.config else None
    if cfg:
        cfg = apply_cli_path_overrides(cfg, args)

    from academic_pipeline.command_dispatch import (
        dispatch_stage_012 as _ap003c_dispatch_012,
    )

    _ap003c_result_012 = _ap003c_dispatch_012(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_012.handled:
        return _ap003c_result_012.value
    from academic_pipeline.command_dispatch import (
        dispatch_stage_013 as _ap003c_dispatch_013,
    )

    _ap003c_result_013 = _ap003c_dispatch_013(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_013.handled:
        return _ap003c_result_013.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_014 as _ap003c_dispatch_014,
    )

    _ap003c_result_014 = _ap003c_dispatch_014(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_014.handled:
        return _ap003c_result_014.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_015 as _ap003c_dispatch_015,
    )

    _ap003c_result_015 = _ap003c_dispatch_015(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_015.handled:
        return _ap003c_result_015.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_016 as _ap003c_dispatch_016,
    )

    _ap003c_result_016 = _ap003c_dispatch_016(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_016.handled:
        return _ap003c_result_016.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_017 as _ap003c_dispatch_017,
    )

    _ap003c_result_017 = _ap003c_dispatch_017(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_017.handled:
        return _ap003c_result_017.value

    from academic_pipeline.command_dispatch import (
        dispatch_stage_018 as _ap003c_dispatch_018,
    )

    _ap003c_result_018 = _ap003c_dispatch_018(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_018.handled:
        return _ap003c_result_018.value

    if not cfg:
        raise RuntimeError("Informe --config, ou use --doctor sem config.")

    from academic_pipeline.command_dispatch import (
        dispatch_stage_019 as _ap003c_dispatch_019,
    )

    _ap003c_result_019 = _ap003c_dispatch_019(
        args,
        {**globals(), **locals()},
    )
    if _ap003c_result_019.handled:
        return _ap003c_result_019.value

    cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
    is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
    out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
    work_dir, cache_dir = work_cache_paths(cfg, prefix)
    from academic_pipeline.document_orchestration import (
        run_document_stage_002 as _ap003d_stage_002,
    )

    _ap003d_result_002 = _ap003d_stage_002(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_002.terminal:
        return _ap003d_result_002.value
    if 'doc_cfg' in _ap003d_result_002.values:
        doc_cfg = _ap003d_result_002.values['doc_cfg']
    latex_cfg = cfg.get("latex", {}) if isinstance(cfg.get("latex"), dict) else {}
    config_dir = Path(str(cfg.get("__config_dir__"))).resolve()
    warnings: list[str] = []
    client: Any | None = None
    model = _openai_model_from_cfg(cfg)

    # Validação preventiva leve; não bloqueia warnings.
    stage("Validando configuração preventiva")
    precheck = check_config(cfg)
    if precheck.get("warnings"):
        warnings.extend(precheck["warnings"])
    if precheck.get("errors"):
        raise RuntimeError("Configuração inválida:\n- " + "\n- ".join(precheck["errors"]))

    document_json_path = Path(args.document_json).expanduser().resolve() if args.document_json else out_dir / f"{prefix}.document.json"

    if is_external_prisma_run:
        if args.somente_mapa_mental:
            raise RuntimeError("O perfil de busca PRISMA não produz document.json; use a geração normal ou --prisma-importar-triagem.")
        search_cfg = cfg.get("busca_prisma", {}) if isinstance(cfg.get("busca_prisma"), dict) else {}
        if bool(search_cfg.get("pre_triagem_ia", False)):
            stage("Inicializando cliente OpenAI para pré-triagem assistida")
            client, model = make_client(model)
        stage("Executando busca bibliográfica externa e preparando triagem humana")
        prisma_outputs = run_external_prisma_search(
            cfg,
            out_dir,
            prefix,
            progress=stage,
            client=client,
            model=model,
        )
        org_path, pdf_path = render_external_prisma_outputs(
            cfg,
            out_dir,
            prefix,
            prisma_outputs,
            phase="preliminar",
        )
        artifacts = prisma_outputs.setdefault("artefatos", {}) if isinstance(prisma_outputs, dict) else {}
        if org_path:
            artifacts["relatorio_org"] = str(org_path)
        if pdf_path:
            artifacts["relatorio_pdf"] = str(pdf_path)
        report_json_path = artifacts.get("prisma_report_json") if isinstance(artifacts, dict) else ""
        if report_json_path:
            write_json(Path(str(report_json_path)), prisma_outputs)
        prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
        prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
        stage("Registrando prompt_lock")
        prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
        write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
        outputs = {
            "output_dir": str(out_dir),
            "work_dir": str(work_dir),
            "cache_dir": str(cache_dir),
            "document_json": None,
            "org": str(org_path) if org_path else None,
            "bib": None,
            "pdf": str(pdf_path) if pdf_path else None,
            "docx": None,
            "relatorio_pesquisa": prisma_outputs,
            "prompt_lock": str(prompt_lock_path),
        }
        report = make_run_report(
            cfg=cfg,
            config_path=Path(str(cfg.get("__config_path__"))),
            out_dir=out_dir,
            prefix=prefix,
            model=None,
            outputs=outputs,
            warnings=warnings,
            extra={"mode": "prisma_busca_externa", "precheck": precheck},
        )
        write_json(out_dir / f"{prefix}.run_report.json", report)
        write_json(out_dir / f"{prefix}.rc10_report.json", outputs)
        write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
        print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — busca PRISMA concluída; aguarda triagem humana")
        return 0

    from academic_pipeline.document_orchestration import (
        run_document_stage_003 as _ap003d_stage_003,
    )

    _ap003d_result_003 = _ap003d_stage_003(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_003.terminal:
        return _ap003d_result_003.value
    if 'client' in _ap003d_result_003.values:
        client = _ap003d_result_003.values['client']
    if 'document' in _ap003d_result_003.values:
        document = _ap003d_result_003.values['document']
    if 'mm_diag' in _ap003d_result_003.values:
        mm_diag = _ap003d_result_003.values['mm_diag']
    if 'model' in _ap003d_result_003.values:
        model = _ap003d_result_003.values['model']
    if 'outputs' in _ap003d_result_003.values:
        outputs = _ap003d_result_003.values['outputs']
    if 'removed_mindmap_files' in _ap003d_result_003.values:
        removed_mindmap_files = _ap003d_result_003.values['removed_mindmap_files']
    if 'report' in _ap003d_result_003.values:
        report = _ap003d_result_003.values['report']
    if 'w' in _ap003d_result_003.values:
        w = _ap003d_result_003.values['w']

    prisma_outputs = None
    source_info: dict[str, Any] | None = None
    paper_abstract_bundle: dict[str, Any] = {}
    paper_abstract_path = abstract_sidecar_path(out_dir, prefix)

    if args.somente_renderizar:
        if not document_json_path.exists():
            raise FileNotFoundError(f"document.json não encontrado para --somente-renderizar: {document_json_path}")
        stage("Carregando document.json existente")
        document = load_existing_document_json(document_json_path)
        stage("Resolvendo bibliografia para renderização")
        bib_path, bib_keys = resolve_bib_for_existing_document(document, document_json_path, out_dir, prefix)
        stage("Saneando document_model existente")
        document, leak_repairs = sanitize_document_model_technical_leaks(document)
        if leak_repairs:
            warnings.append("Menções técnicas removidas/reescritas no document_model existente: " + ", ".join(leak_repairs[:20]))
        document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
        if raw_key_repairs:
            warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model existente: " + ", ".join(raw_key_repairs[:20]))
        if paper_abstracts_enabled(cfg):
            if paper_abstract_path.exists():
                paper_abstract_bundle = read_paper_abstract_bundle(paper_abstract_path)
            else:
                warnings.append(
                    "RESUMO: arquivo de resumos não encontrado no modo --somente-renderizar; "
                    "o ORG/DOCX será recompilado sem inserir resumo. Execute uma geração completa para criá-lo."
                )
        if args.forcar_regeneracao_mapa_mental:
            if not should_generate_mindmap(cfg):
                raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --forcar-regeneracao-mapa-mental.")
            stage("Removendo mapa mental existente")
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
            if removed_mindmap_files:
                warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
            stage("Inicializando cliente OpenAI para regenerar mapa mental")
            client, model = make_client(model)
            stage("Regenerando mapa mental antes da renderização")
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
            document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
            stage("Salvando document.json atualizado com novo mapa mental")
            write_json(document_json_path, document.model_dump())
        elif args.reusar_mapa_mental:
            stage("Tentando reutilizar mapa mental existente")
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if mm_diag:
                document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
                write_json(document_json_path, document.model_dump())
            else:
                warnings.append("--reusar-mapa-mental informado, mas nenhum mapa existente foi encontrado; renderização seguirá com figuras já registradas no document.json.")
    else:
        pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg.get("pipeline"), dict) else {}
        executar_documento = bool(pipeline_cfg.get("executar_documento", True))
        stage("Inicializando cliente OpenAI")
        client, model = make_client(model)
        stage("Descobrindo e extraindo documentos locais")
        docs, source_info = discover_local_documents(cfg, work_dir)
        clean_cache = bool((cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}).get("limpar_cache_anterior", True))
        stage("Copiando documentos para fulltext_cache")
        copy_documents_to_fulltext_cache(docs, cache_dir, clean=clean_cache)
        stage("Carregando orientações do projeto")
        orientations = collect_orientation_docs(cfg, work_dir)
        stage("Gerando e validando bibliografia")
        bib_result = build_bibliography(cfg, docs, out_dir, prefix, client, model)
        bib_path = bib_result.bib_path
        bib_keys = bib_result.keys
        stage("Verificando geração de relatório PRISMA")
        prisma_outputs = run_prisma_report_outputs(cfg, docs, orientations, bib_result, out_dir, prefix) if prisma_enabled(cfg) else None

        if not executar_documento:
            outputs = {
                "output_dir": str(out_dir),
        "work_dir": str(work_dir),
        "cache_dir": str(cache_dir),
                "work_dir": str(work_dir),
                "cache_dir": str(cache_dir),
                "document_json": None,
                "org": None,
                "bib": str(bib_path),
                "pdf": None,
                "docx": None,
                "relatorio_pesquisa": prisma_outputs,
            }
            prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
            prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
            stage("Registrando prompt_lock")
            prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
            write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
            outputs["prompt_lock"] = str(prompt_lock_path)
            report = make_run_report(
                cfg=cfg,
                config_path=Path(str(cfg.get("__config_path__"))),
                out_dir=out_dir,
                prefix=prefix,
                model=model,
                outputs=outputs,
                warnings=warnings,
                extra={"mode": "research_only", "source_info": source_info, "work_dir": str(work_dir), "cache_dir": str(cache_dir)},
            )
            write_json(out_dir / f"{prefix}.run_report.json", report)
            write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade
            write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
            print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} concluído sem documento acadêmico")
            return 0

        resumo_cfg_for_stage = cfg.get("resumo_artigos", {}) if isinstance(cfg.get("resumo_artigos"), dict) else {}
        if resumo_cfg_for_stage.get("ativo") and bool(resumo_cfg_for_stage.get("geracao_em_etapas", True)):
            stage("Gerando document.json canônico com IA em etapas")
        else:
            stage("Gerando document.json canônico com IA")
        document = build_document_model(
            client,
            model,
            cfg,
            docs,
            orientations,
            bib_keys,
            bib_path,
            progress=stage,
            checkpoint_dir=out_dir,
            prefix=prefix,
        )
        if not document.bibliography.entries_used:
            document.bibliography.entries_used = bib_keys
        if not document.bibliography.bib_path:
            document.bibliography.bib_path = bib_path.name
        stage("Saneando linguagem técnica do document_model")
        document, leak_repairs = sanitize_document_model_technical_leaks(document)
        if leak_repairs:
            warnings.append("Menções técnicas removidas/reescritas no document_model: " + ", ".join(leak_repairs[:20]))
        document, raw_key_repairs = sanitize_document_model_raw_bibkeys(document, bib_keys)
        if raw_key_repairs:
            warnings.append("Chaves BibTeX cruas convertidas em citações LaTeX no document_model: " + ", ".join(raw_key_repairs[:20]))
        stage("Validando document_model")
        raise_if_errors(validate_document_model(document, bib_keys), "Validação do document_model falhou")
        if paper_abstracts_enabled(cfg):
            stage("Gerando resumo e palavras-chave do paper")
            try:
                paper_abstract_bundle = generate_paper_abstract_bundle(client, model, document, cfg)
                write_paper_abstract_bundle(paper_abstract_path, paper_abstract_bundle)
            except PaperAbstractError as exc:
                raise RuntimeError("Falha ao gerar o resumo acadêmico do paper: " + str(exc)) from exc
        stage("Gerando/anexando mapa mental, se configurado")
        mm_diag = None
        if args.forcar_regeneracao_mapa_mental:
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
            if removed_mindmap_files:
                warnings.append("Arquivos de mapa mental removidos antes da regeneração: " + ", ".join(removed_mindmap_files[:10]))
        if args.reusar_mapa_mental:
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if not mm_diag:
                warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
        if not mm_diag:
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
        document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
        document.diagnostics.source_info_json = json.dumps(source_info or {}, ensure_ascii=False)
        if prisma_outputs:
            document.diagnostics.relatorio_pesquisa_json = json.dumps(prisma_outputs, ensure_ascii=False)
        stage("Salvando document.json")
        write_json(document_json_path, document.model_dump())

    from academic_pipeline.document_orchestration import (
        run_document_stage_004 as _ap003d_stage_004,
    )

    _ap003d_result_004 = _ap003d_stage_004(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_004.terminal:
        return _ap003d_result_004.value
    org_path = out_dir / f"{prefix}.org"
    from academic_pipeline.document_orchestration import (
        run_document_stage_005 as _ap003d_stage_005,
    )

    _ap003d_result_005 = _ap003d_stage_005(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_005.terminal:
        return _ap003d_result_005.value
    if 'org_text' in _ap003d_result_005.values:
        org_text = _ap003d_result_005.values['org_text']

    pdf_path = None
    from academic_pipeline.document_orchestration import (
        run_document_stage_006 as _ap003d_stage_006,
    )

    _ap003d_result_006 = _ap003d_stage_006(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_006.terminal:
        return _ap003d_result_006.value
    if 'academic_writing' in _ap003d_result_006.values:
        academic_writing = _ap003d_result_006.values['academic_writing']
    if 'latex_extra' in _ap003d_result_006.values:
        latex_extra = _ap003d_result_006.values['latex_extra']
    if 'pdf_engine' in _ap003d_result_006.values:
        pdf_engine = _ap003d_result_006.values['pdf_engine']
    if 'pdf_path' in _ap003d_result_006.values:
        pdf_path = _ap003d_result_006.values['pdf_path']

    docx_path = None
    docx_validation: dict[str, Any] | None = None
    from academic_pipeline.document_orchestration import (
        run_document_stage_007 as _ap003d_stage_007,
    )

    _ap003d_result_007 = _ap003d_stage_007(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_007.terminal:
        return _ap003d_result_007.value
    if 'docx_cfg' in _ap003d_result_007.values:
        docx_cfg = _ap003d_result_007.values['docx_cfg']
    if 'docx_path' in _ap003d_result_007.values:
        docx_path = _ap003d_result_007.values['docx_path']
    if 'docx_validation' in _ap003d_result_007.values:
        docx_validation = _ap003d_result_007.values['docx_validation']
    if 'ref' in _ap003d_result_007.values:
        ref = _ap003d_result_007.values['ref']
    if 'w' in _ap003d_result_007.values:
        w = _ap003d_result_007.values['w']

    translated_outputs: dict[str, Any] = {}
    from academic_pipeline.document_orchestration import (
        run_document_stage_008 as _ap003d_stage_008,
    )

    _ap003d_result_008 = _ap003d_stage_008(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_008.terminal:
        return _ap003d_result_008.value
    if 'exc' in _ap003d_result_008.values:
        exc = _ap003d_result_008.values['exc']
    if 'translated_outputs' in _ap003d_result_008.values:
        translated_outputs = _ap003d_result_008.values['translated_outputs']
    if 'translation_warnings' in _ap003d_result_008.values:
        translation_warnings = _ap003d_result_008.values['translation_warnings']

    outputs = {
        "output_dir": str(out_dir),
        "document_json": str(document_json_path),
        "org": str(org_path),
        "bib": str(bib_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "docx": str(docx_path) if docx_path else None,
        "resumos_paper": str(paper_abstract_path) if paper_abstract_bundle else None,
        "idiomas_adicionais": translated_outputs,
        "relatorio_pesquisa": _json_or_none(getattr(document.diagnostics, "relatorio_pesquisa_json", "")) if getattr(document, "diagnostics", None) else prisma_outputs,
    }

    # Prompt lock: rastreabilidade exata dos prompts/diretivas usados.
    prompt_lock_path = out_dir / f"{prefix}.prompt_lock.json"
    prompt_lock_md = out_dir / f"{prefix}.prompt_lock.md"
    prompt_lock = write_prompt_lock(cfg, prompt_lock_path)
    write_prompt_lock_markdown(prompt_lock, prompt_lock_md)
    outputs["prompt_lock"] = str(prompt_lock_path)

    # Conformidade institucional: valida artefatos contra o perfil escolhido.
    stage("Executando conformidade institucional")
    from academic_pipeline.document_orchestration import (
        run_document_stage_009 as _ap003d_stage_009,
    )

    _ap003d_result_009 = _ap003d_stage_009(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_009.terminal:
        return _ap003d_result_009.value
    if 'compliance_report' in _ap003d_result_009.values:
        compliance_report = _ap003d_result_009.values['compliance_report']
    compliance_md, compliance_json = write_compliance_reports(compliance_report, out_dir / prefix)
    from academic_pipeline.document_orchestration import (
        run_document_stage_010 as _ap003d_stage_010,
    )

    _ap003d_result_010 = _ap003d_stage_010(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_010.terminal:
        return _ap003d_result_010.value
    if compliance_report.get("warnings"):
        warnings.extend([f"CONFORMIDADE: {w.get('message')}" for w in compliance_report.get("warnings", [])])
    if not compliance_report.get("ok"):
        warnings.extend([f"CONFORMIDADE CRÍTICA: {e.get('message')}" for e in compliance_report.get("errors", [])])

    stage("Gerando relatório de qualidade")
    from academic_pipeline.document_orchestration import (
        run_document_stage_011 as _ap003d_stage_011,
    )

    _ap003d_result_011 = _ap003d_stage_011(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_011.terminal:
        return _ap003d_result_011.value
    if 'quality' in _ap003d_result_011.values:
        quality = _ap003d_result_011.values['quality']
    quality_path = out_dir / f"{prefix}.quality_report.md"
    write_quality_report(quality, quality_path)
    if quality.get("warnings"):
        warnings.extend([f"QUALIDADE: {w}" for w in quality.get("warnings", [])])
    from academic_pipeline.document_orchestration import (
        run_document_stage_012 as _ap003d_stage_012,
    )

    _ap003d_result_012 = _ap003d_stage_012(
        args,
        {**globals(), **locals()},
    )
    if _ap003d_result_012.terminal:
        return _ap003d_result_012.value
    if 'report' in _ap003d_result_012.values:
        report = _ap003d_result_012.values['report']
    write_json(out_dir / f"{prefix}.run_report.json", report)
    write_json(out_dir / f"{prefix}.rc10_report.json", outputs)  # compatibilidade com scripts antigos
    write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
    print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} concluído")
    if warnings:
        print("Avisos:")
        for w in warnings:
            print(f"- {w}")
    return 0




# >>> PATCH_REFERENCIAS_FORMAIS_EFETIVAS_V6_RUNTIME >>>
# Política unificada: uma escolha explícita no TOML sempre prevalece. Para
# compatibilidade, TOMLs antigos com todos os flags bibliográficos desligados
# também entram no modo sem referências.
def _refs_v6_disabled(cfg: dict[str, Any] | None) -> bool:
    from academic_pipeline.document_orchestration import _refs_v6_disabled_impl as _ap003d_impl__refs_v6_disabled
    return _ap003d_impl__refs_v6_disabled({**globals(), **locals()}, cfg)


def _refs_v6_apply_runtime_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    from academic_pipeline.document_orchestration import _refs_v6_apply_runtime_policy_impl as _ap003d_impl__refs_v6_apply_runtime_policy
    return _ap003d_impl__refs_v6_apply_runtime_policy({**globals(), **locals()}, cfg)


# Carrega a política antes de qualquer rotina de descoberta, bibliografia ou IA.
_refs_v6_original_load_config = load_config
def load_config(path: Path) -> dict[str, Any]:
    from academic_pipeline.document_orchestration import load_config_impl as _ap003d_impl_load_config
    return _ap003d_impl_load_config({**globals(), **locals()}, path)


# Impede a construção física do .bib. Um Path sentinela mantém compatibilidade
# com funções que recebem bib_path, mas o arquivo não é criado e as chaves ficam vazias.
_refs_v6_original_build_bibliography = build_bibliography
def build_bibliography(cfg: dict[str, Any], docs: Any, out_dir: Path, prefix: str, client: Any, model: str) -> Any:
    from academic_pipeline.document_orchestration import build_bibliography_impl as _ap003d_impl_build_bibliography
    return _ap003d_impl_build_bibliography({**globals(), **locals()}, cfg, docs, out_dir, prefix, client, model)


def _refs_v6_clear_document_bibliography(document: Any) -> Any:
    from academic_pipeline.document_orchestration import _refs_v6_clear_document_bibliography_impl as _ap003d_impl__refs_v6_clear_document_bibliography
    return _ap003d_impl__refs_v6_clear_document_bibliography({**globals(), **locals()}, document)


def _refs_v6_strip_org(text: str) -> str:
    from academic_pipeline.document_orchestration import _refs_v6_strip_org_impl as _ap003d_impl__refs_v6_strip_org
    return _ap003d_impl__refs_v6_strip_org({**globals(), **locals()}, text)


# Garante que PDF/ORG não exibam citações ou referências mesmo se um artefato
# intermediário trouxer marcas bibliográficas inesperadas.
_refs_v6_original_render_org_latex = render_org_latex
def render_org_latex(document: Any, org_path: Path, bib_filename: str, *, cfg: dict[str, Any], bib_keys: list[str] | None=None) -> str:
    from academic_pipeline.document_orchestration import render_org_latex_impl as _ap003d_impl_render_org_latex
    return _ap003d_impl_render_org_latex({**globals(), **locals()}, document, org_path, bib_filename, cfg=cfg, bib_keys=bib_keys)
# <<< PATCH_REFERENCIAS_FORMAIS_EFETIVAS_V6_RUNTIME <<<










# >>> PATCH_PRISMA_ARTIGO_GENERICO_WRAPPER_V1_5 >>>
def _prisma_artigo_generico_get_arg(argv, name):
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None

def _prisma_artigo_generico_strip(argv):
    bool_flags = {"--prisma-exportar-bib", "--prisma-congelar-artigo", "--prisma-gerar-toml-artigo", "--prisma-gerar-artigo-final"}
    value_flags = {"--prisma-bib-input","--prisma-bib-output","--prisma-artigo-dir","--prisma-congelamento-dir","--prisma-artigo-toml-output","--prisma-csl-path","--prisma-dados-pesquisa-path","--prisma-artigo-prefix","--prisma-autor-artigo","--prisma-professor-artigo","--prisma-openai-model-artigo"}
    result=[]; i=0
    while i < len(argv):
        item=argv[i]
        if item in bool_flags:
            i+=1; continue
        if item in value_flags:
            i+=2; continue
        if any(item.startswith(flag+"=") for flag in value_flags):
            i+=1; continue
        result.append(item); i+=1
    return result

def _prisma_artigo_generico_out_dir(argv):
    from pathlib import Path
    out_arg=_prisma_artigo_generico_get_arg(argv,"--prisma-curadoria-out-dir") or _prisma_artigo_generico_get_arg(argv,"--prisma-out-dir")
    if out_arg: return Path(out_arg)
    cfg=_prisma_artigo_generico_get_arg(argv,"--config")
    if cfg:
        cfg_path=Path(cfg)
        if cfg_path.exists(): return cfg_path.resolve().parent/"output_pesquisa"/f"relatorio_prisma_{cfg_path.stem}"
    return Path("app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf")

def _prisma_artigo_generico_run_export(argv, silent=False):
    import subprocess, sys
    from pathlib import Path
    helper=Path(__file__).with_name("prisma_exportar_bib.py")
    if not helper.exists():
        msg=f"Helper de exportação BibLaTeX não encontrado: {helper}"
        if silent:
            print(f"[WARN] {msg}"); return 1
        raise SystemExit(msg)
    out_dir=_prisma_artigo_generico_out_dir(argv)
    prefix=out_dir.name if out_dir.name.startswith("relatorio_prisma_") else "relatorio_prisma_prisma_fluxo_pmf"
    cmd=[sys.executable,str(helper),"--out-dir",str(out_dir),"--prefix",prefix]
    val=_prisma_artigo_generico_get_arg(argv,"--prisma-bib-input")
    if val: cmd+=["--input",val]
    val=_prisma_artigo_generico_get_arg(argv,"--prisma-bib-output")
    if val: cmd+=["--output",val]
    proc=subprocess.run(cmd)
    if proc.returncode and not silent: raise SystemExit(proc.returncode)
    if proc.returncode and silent: print(f"[WARN] Exportação BibLaTeX PRISMA retornou código {proc.returncode}.")
    return proc.returncode

def _prisma_artigo_generico_run_freeze(argv, silent=False):
    import subprocess, sys
    from pathlib import Path
    helper=Path(__file__).with_name("prisma_congelar_artigo.py")
    if not helper.exists():
        msg=f"Helper de congelamento de insumos não encontrado: {helper}"
        if silent:
            print(f"[WARN] {msg}"); return 1
        raise SystemExit(msg)
    out_dir=_prisma_artigo_generico_out_dir(argv)
    prefix=out_dir.name if out_dir.name.startswith("relatorio_prisma_") else "relatorio_prisma_prisma_fluxo_pmf"
    cmd=[sys.executable,str(helper),"--out-dir",str(out_dir),"--prefix",prefix]
    cfg=_prisma_artigo_generico_get_arg(argv,"--config")
    if cfg: cmd+=["--prisma-config",cfg]
    cmd+=["--pipeline-script",str(Path(__file__).resolve())]
    for src,dst in [("--prisma-artigo-dir","--artigo-dir"),("--prisma-congelamento-dir","--dest-dir"),("--prisma-artigo-toml-output","--toml-output"),("--prisma-csl-path","--csl-path"),("--prisma-dados-pesquisa-path","--dados-pesquisa-path"),("--prisma-artigo-prefix","--artigo-prefix"),("--prisma-autor-artigo","--autor"),("--prisma-professor-artigo","--professor"),("--prisma-openai-model-artigo","--openai-model")]:
        val=_prisma_artigo_generico_get_arg(argv,src)
        if val: cmd += [dst,val]
    if "--prisma-gerar-toml-artigo" in argv or "--prisma-gerar-artigo-final" in argv: cmd.append("--gerar-toml-artigo")
    if "--prisma-gerar-artigo-final" in argv: cmd.append("--gerar-artigo-final")
    proc=subprocess.run(cmd)
    if proc.returncode and not silent: raise SystemExit(proc.returncode)
    if proc.returncode and silent: print(f"[WARN] Congelamento/geração de artigo retornou código {proc.returncode}.")
    return proc.returncode

_original_main_before_prisma_artigo_generico_wrapper = main

def main(*args, **kwargs):
    import sys
    original_argv=list(sys.argv[1:])
    has_import="--prisma-curadoria-importar" in original_argv or "--prisma-curadoria-fluxo-completo" in original_argv
    wants_export="--prisma-exportar-bib" in original_argv
    wants_freeze="--prisma-congelar-artigo" in original_argv
    wants_toml="--prisma-gerar-toml-artigo" in original_argv
    wants_final="--prisma-gerar-artigo-final" in original_argv
    if not has_import and wants_export and not wants_freeze and not wants_toml and not wants_final:
        return _prisma_artigo_generico_run_export(original_argv, silent=False)
    if not has_import and (wants_freeze or wants_toml or wants_final):
        _prisma_artigo_generico_run_export(original_argv, silent=True)
        return _prisma_artigo_generico_run_freeze(original_argv, silent=False)
    if has_import and (wants_export or wants_freeze or wants_toml or wants_final):
        old_argv=sys.argv[:]
        sys.argv=[sys.argv[0]]+_prisma_artigo_generico_strip(original_argv)
        try: rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
        finally: sys.argv=old_argv
    else:
        rc=_original_main_before_prisma_artigo_generico_wrapper(*args, **kwargs)
    if has_import: _prisma_artigo_generico_run_export(original_argv, silent=True)
    if wants_freeze or wants_toml or wants_final: _prisma_artigo_generico_run_freeze(original_argv, silent=False)
    return rc
# <<< PATCH_PRISMA_ARTIGO_GENERICO_WRAPPER_V1_5 <<<


if __name__ == "__main__":
    raise SystemExit(main())
