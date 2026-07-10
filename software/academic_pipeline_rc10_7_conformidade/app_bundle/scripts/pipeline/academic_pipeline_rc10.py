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

from bibliography_manager import build_bibliography
from corpus_manager import collect_orientation_docs, copy_documents_to_fulltext_cache, discover_local_documents
from document_builder import build_document_model
from document_model import AcademicDocument
from document_validator import raise_if_errors, validate_document_model, validate_org_text, sanitize_document_model_technical_leaks, sanitize_document_model_raw_bibkeys
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
from latex_compile import run_compile_sequence
from mindmap_manager import generate_and_attach_mindmap, should_generate_mindmap, attach_existing_mindmap_if_available, delete_existing_mindmap_outputs
from render_docx import render_docx
from render_org_latex import render_org_latex
from prisma_pipeline import run_prisma_report_outputs, prisma_enabled
from prisma_busca_externa import external_search_enabled, import_manual_prisma_triage, render_external_prisma_org_report, run_external_prisma_search
from utils import write_json, resolve_path
from project_tools import init_project, make_doi_manifest, inspect_bib, render_bib_inspection_markdown
from quality_report import build_quality_report, write_quality_report
from institution_profiles import apply_institution_profile, describe_institution_profiles
from institution_layouts import available_layouts, resolve_layout_spec
from prompt_manager import prompt_report_for_cfg, load_prompt_bundle
from institution_explainer import explain_profile
from institution_compliance import run_institution_compliance, render_compliance_markdown, write_compliance_reports
from prompt_lock import write_prompt_lock, write_prompt_lock_markdown, build_prompt_lock
from document_translation import TranslationError, requested_translation_languages, translation_batch_size, translate_document_model
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
    config_dir = Path(str(cfg.get("__config_dir__") or Path.cwd())).resolve()
    paths = _section(cfg, "paths")
    projeto = _section(cfg, "projeto")
    prefix = str(paths.get("document_prefix") or projeto.get("nome") or "documento").strip() or "documento"
    out_base = resolve_path(paths.get("document_output_dir") or "../../output/documento", config_dir) or (config_dir / "output/documento")
    out_dir = out_base / prefix if bool(paths.get("create_document_subdir", True)) else out_base
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, prefix


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
    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        cfg["paths"] = paths
    if getattr(args, "output_dir", ""):
        paths["document_output_dir"] = args.output_dir
    if getattr(args, "work_dir", ""):
        paths["work_dir"] = args.work_dir
    if getattr(args, "cache_dir", ""):
        paths["cache_dir"] = args.cache_dir
    if getattr(args, "research_output_dir", ""):
        paths["research_output_dir"] = args.research_output_dir
    if getattr(args, "output_prefix", ""):
        paths["document_prefix"] = args.output_prefix
    if getattr(args, "no_output_subdir", False):
        paths["create_document_subdir"] = False

    doc = cfg.setdefault("documento", {})
    if not isinstance(doc, dict):
        doc = {}
        cfg["documento"] = doc
    if getattr(args, "layout", ""):
        doc["layout"] = args.layout
    if getattr(args, "tipo_conteudo", ""):
        doc["tipo_conteudo"] = args.tipo_conteudo
    if getattr(args, "genero_academico", ""):
        doc["genero_academico"] = args.genero_academico
    return cfg


def load_existing_document_json(path: Path) -> AcademicDocument:
    return AcademicDocument.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_bib_for_existing_document(document: AcademicDocument, document_json_path: Path, out_dir: Path, prefix: str) -> tuple[Path, list[str]]:
    """Resolve o .bib em modo --somente-renderizar sem exigir que ele já esteja no output_dir."""
    raw = str(document.bibliography.bib_path or f"{prefix}.bib").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([document_json_path.parent / p, out_dir / p])
    candidates.extend([
        document_json_path.with_suffix(".bib"),
        document_json_path.with_name(prefix + ".bib"),
        out_dir / f"{prefix}.bib",
    ])
    found = next((c.resolve() for c in candidates if c.exists()), out_dir / f"{prefix}.bib")
    target = out_dir / found.name
    if found.exists() and found.resolve() != target.resolve():
        import shutil
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, target)
        found = target
    keys = list(document.bibliography.entries_used or [])
    if found.exists() and not keys:
        import re
        text = found.read_text(encoding="utf-8", errors="ignore")
        keys = [m.group(1).strip() for m in re.finditer(r"@[^{}]+\{\s*([^,]+)\s*,", text)]
    return found, keys


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
    base = Path(str((cfg or {}).get("__config_dir__") or Path.cwd())).resolve()
    latex_cfg = (cfg or {}).get("latex", {}) if isinstance((cfg or {}).get("latex", {}), dict) else {}
    academic_writing = resolve_path(args.academic_writing or latex_cfg.get("org_latex_class_init"), base)
    latex_extra = resolve_path(args.latex_extra_path or latex_cfg.get("latex_extra_path"), base)
    pdf_engine = str(args.pdf_engine or latex_cfg.get("pdf_engine") or "lualatex")
    return academic_writing, latex_extra, pdf_engine


def run_recompile(args: argparse.Namespace, cfg: dict[str, Any] | None) -> int:
    if not args.org:
        raise RuntimeError("Use --recompile --org caminho/arquivo.org")
    org_path = Path(args.org).expanduser().resolve()
    if not org_path.exists():
        raise FileNotFoundError(f"ORG não encontrado: {org_path}")
    academic_writing, latex_extra, pdf_engine = _resolve_latex_paths_for_recompile(args, cfg)
    removed = [] if args.no_clean else clean_aux_files(org_path)
    pdf = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)
    out_dir = org_path.parent
    prefix = org_path.stem
    outputs = {"org": str(org_path), "pdf": str(pdf), "removed_aux": removed}
    write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
    stage("Gerando run_report e manifestos")
    report = make_run_report(
        cfg=cfg or {"__config_dir__": str(Path.cwd())},
        config_path=Path(str((cfg or {}).get("__config_path__"))) if cfg and cfg.get("__config_path__") else None,
        out_dir=out_dir,
        prefix=prefix,
        model=None,
        outputs=outputs,
        warnings=[],
        extra={"mode": "recompile"},
    )
    write_json(out_dir / f"{prefix}.run_report.json", report)
    print_outputs(outputs, title="Recompilação concluída")
    return 0



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


def render_additional_language_versions(
    *,
    client: Any,
    model: str,
    cfg: dict[str, Any],
    document: AcademicDocument,
    bib_path: Path,
    bib_keys: list[str],
    out_dir: Path,
    prefix: str,
    doc_cfg: dict[str, Any],
    latex_cfg: dict[str, Any],
    config_dir: Path,
    abstract_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Traduz e renderiza versões adicionais a partir do document.json canônico.

    Cada versão recebe diretório próprio dentro de ``idiomas/<codigo>`` e
    compartilha a bibliografia original, copiada sem tradução. A função nunca
    consulta novamente o corpus nem gera uma segunda análise acadêmica.
    """
    result: dict[str, Any] = {}
    warnings: list[str] = []
    languages = requested_translation_languages(cfg)
    if not languages:
        return result, warnings
    base_dir = out_dir / "idiomas"
    max_chars = translation_batch_size(cfg)
    docx_cfg = cfg.get("docx", {}) if isinstance(cfg.get("docx"), dict) else {}
    reference_docx = resolve_path(docx_cfg.get("reference_docx") or doc_cfg.get("docx_reference"), config_dir)
    academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
    latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
    pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")

    for language_code, language_label in languages:
        stage(f"Traduzindo paper para {language_label}")
        translated_document, audit = translate_document_model(
            client,
            model,
            document,
            language_code,
            max_chars=max_chars,
        )
        language_dir = base_dir / language_code
        language_dir.mkdir(parents=True, exist_ok=True)
        language_prefix = f"{prefix}_{language_code}"
        language_bib = language_dir / bib_path.name
        if bib_path.exists() and bib_path.resolve() != language_bib.resolve():
            shutil.copy2(bib_path, language_bib)
        document_json = language_dir / f"{language_prefix}.document.json"
        write_json(document_json, translated_document.model_dump())
        audit_path = language_dir / f"{language_prefix}.translation_audit.json"
        write_json(audit_path, audit)

        stage(f"Renderizando ORG traduzido ({language_label})")
        org_path = language_dir / f"{language_prefix}.org"
        org_text = render_org_latex(
            translated_document,
            org_path,
            language_bib.name,
            cfg=cfg,
            bib_keys=bib_keys,
        )
        if abstract_bundle:
            org_text = inject_paper_abstracts_into_org(org_path, abstract_bundle, [language_code])
        raise_if_errors(validate_org_text(org_text, bib_keys), f"Validação do ORG traduzido falhou ({language_label})")

        pdf_path: Path | None = None
        if bool(doc_cfg.get("exportar_pdf", True)):
            stage(f"Compilando PDF traduzido ({language_label})")
            pdf_path = run_compile_sequence(
                org_path,
                academic_writing=academic_writing,
                latex_extra_path=latex_extra,
                pdf_engine=pdf_engine,
            )

        docx_path: Path | None = None
        if bool(doc_cfg.get("exportar_docx", True)):
            stage(f"Renderizando DOCX traduzido ({language_label})")
            docx_path = render_docx(
                translated_document,
                language_dir / f"{language_prefix}.docx",
                bib_path=language_bib,
                reference_docx=reference_docx,
                cfg=cfg,
            )
            if abstract_bundle:
                inject_paper_abstracts_into_docx(docx_path, abstract_bundle, [language_code])
            validation = validate_docx_file(
                docx_path,
                expected_title=translated_document.metadata.titulo,
                require_references=bool(translated_document.bibliography.entries_used),
            )
            if validation and validation.get("warnings"):
                warnings.extend([f"DOCX {language_code}: {item}" for item in validation.get("warnings", [])])

        quality = build_quality_report(translated_document, org_path=org_path, bib_keys=bib_keys)
        quality_path = language_dir / f"{language_prefix}.quality_report.md"
        write_quality_report(quality, quality_path)
        if quality.get("warnings"):
            warnings.extend([f"QUALIDADE {language_code}: {item}" for item in quality.get("warnings", [])])
        result[language_code] = {
            "idioma": language_label,
            "output_dir": str(language_dir),
            "document_json": str(document_json),
            "translation_audit": str(audit_path),
            "org": str(org_path),
            "pdf": str(pdf_path) if pdf_path else None,
            "docx": str(docx_path) if docx_path else None,
            "bib": str(language_bib),
            "quality_report": str(quality_path),
        }
    return result, warnings



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
    parser = argparse.ArgumentParser(description=f"academic_pipeline {PIPELINE_VERSION} — document_model canônico")
    parser.add_argument("--config", default="", help="Arquivo TOML")
    parser.add_argument("--tui", action="store_true", help="Abre a Central Operacional FGV em terminal (prompt_toolkit)")
    parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica FGV de atividades acadêmicas")
    parser.add_argument("--init-toml", action="store_true", help="Abre o gerador interativo completo de TOML")
    parser.add_argument("--toml-profile", default="", help="Preset inicial para --init-toml, ex.: atividade_local_fgv")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre etapas do --init-toml")
    parser.add_argument("--list-toml-profiles", action="store_true", help="Lista presets do gerador de TOML")
    parser.add_argument("--list-institutions", action="store_true", help="Lista perfis institucionais disponíveis")
    parser.add_argument("--list-layouts", action="store_true", help="Lista layouts disponíveis do perfil institucional informado no TOML")
    parser.add_argument("--explain-profile", default="", nargs="?", const="fgv", help="Explica um perfil institucional, ex.: --explain-profile fgv")
    parser.add_argument("--show-prompts", action="store_true", help="Mostra os prompts/diretivas ativos para o TOML informado")
    parser.add_argument("--write-prompt-lock", action="store_true", help="Gera prompt_lock.json/md para o TOML e encerra")
    parser.add_argument("--check-institution-compliance", action="store_true", help="Valida conformidade institucional de artefatos já gerados")
    parser.add_argument("--doctor", action="store_true", help="Diagnostica ambiente, ferramentas, arquivos FGV e estilo bibliográfico")
    parser.add_argument("--check-config", action="store_true", help="Valida preventivamente o TOML e encerra")
    parser.add_argument("--recompile", action="store_true", help="Recompila um .org existente sem chamar IA")
    parser.add_argument("--org", default="", help="Arquivo .org para --recompile")
    parser.add_argument("--academic-writing", default="", help="Override do academic-writing.el para --recompile")
    parser.add_argument("--latex-extra-path", default="", help="Override do latex_extra_path para --recompile")
    parser.add_argument("--pdf-engine", default="", help="Override do pdf_engine para --recompile")
    parser.add_argument("--no-clean", action="store_true", help="Não remove auxiliares no --recompile")
    parser.add_argument("--somente-renderizar", action="store_true", help="Usa document.json existente e só renderiza saídas")
    parser.add_argument("--somente-mapa-mental", action="store_true", help="Usa document.json existente e gera/renderiza apenas o mapa mental")
    parser.add_argument("--reusar-mapa-mental", action="store_true", help="Reaproveita imagem de mapa mental existente quando disponível, sem chamar IA/PlantUML")
    parser.add_argument("--forcar-regeneracao-mapa-mental", action="store_true", help="Remove mapa mental existente e recria PlantUML/imagem quando a etapa de mapa mental for executada")
    parser.add_argument("--document-json", default="", help="Caminho de document.json existente")
    parser.add_argument("--prisma-importar-triagem", default="", help="Importa CSV de triagem humana do perfil relatorio_prisma_busca_orientada_fgv e consolida matriz/relatório PRISMA")
    parser.add_argument("--init-project", default="", help="Cria app_bundle/projetos/<nome> com TOML, ZIPs placeholder e doi_manifest.csv")
    parser.add_argument("--project-type", default="paper", choices=["paper", "atividade", "prisma", "atividade_prisma", "paper_prisma"], help="Tipo de projeto para --init-project")
    parser.add_argument("--institution", default="fgv", help="Perfil institucional usado por --init-project, ex.: fgv")
    parser.add_argument("--base-dir", default="", help="Raiz do academic_pipeline ou app_bundle para --init-project")
    parser.add_argument("--overwrite-project", action="store_true", help="Permite sobrescrever arquivos seguros criados por --init-project")
    parser.add_argument("--make-doi-manifest", action="store_true", help="Gera doi_manifest.csv a partir de --input-zip ou --input-dir")
    parser.add_argument("--input-zip", default="", help="ZIP de documentos para --make-doi-manifest")
    parser.add_argument("--input-dir", default="", help="Pasta de documentos para --make-doi-manifest")
    parser.add_argument("--output", default="", help="Arquivo de saída para --make-doi-manifest")
    parser.add_argument("--output-dir", default="", help="Override de [paths].document_output_dir para a geração/renderização do documento")
    parser.add_argument("--work-dir", default="", help="Override de [paths].work_dir para extrações temporárias")
    parser.add_argument("--cache-dir", default="", help="Override de [paths].cache_dir para fulltext_cache")
    parser.add_argument("--research-output-dir", default="", help="Override de [paths].research_output_dir para relatório PRISMA")
    parser.add_argument("--output-prefix", default="", help="Override de [paths].document_prefix")
    parser.add_argument("--layout", default="", help="Override de [documento].layout, ex.: atividade_fgv")
    parser.add_argument("--tipo-conteudo", default="", help="Override de [documento].tipo_conteudo, ex.: resumo_artigos")
    parser.add_argument("--genero-academico", default="", help="Override de [documento].genero_academico, ex.: atividade")
    parser.add_argument("--no-output-subdir", action="store_true", help="Não cria subdiretório com document_prefix dentro de --output-dir/[paths].document_output_dir")
    parser.add_argument("--inspect-bib", default="", help="Inspeciona arquivo .bib e gera relatório .md/.json")
    parser.add_argument("--quality-report", action="store_true", help="Gera quality_report.md a partir de --document-json e opcionalmente --org")
    parser.add_argument("--bib", default="", help="Arquivo .bib opcional para --quality-report ou --check-institution-compliance")
    parser.add_argument("--docx", default="", help="Arquivo .docx opcional para --check-institution-compliance")
    parser.add_argument("--pdf", default="", help="Arquivo .pdf opcional para --check-institution-compliance")
    # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS >>>
    parser.add_argument(
        "--prisma-curadoria-menu",
        action="store_true",
        help="Abre o sub-menu PRISMA de curadoria IA de referências.",
    )
    parser.add_argument(
        "--prisma-curadoria-ia",
        action="store_true",
        help="Executa a curadoria IA v2 de referências e gera XLSX/CSV para o PRISMA.",
    )
    parser.add_argument(
        "--prisma-curadoria-sem-ia",
        action="store_true",
        help="Usa a etapa de curadoria sem chamada à IA, apenas com heurística local.",
    )
    parser.add_argument(
        "--prisma-curadoria-reexportar-xlsx",
        action="store_true",
        help="Reexporta o XLSX de curadoria revisado para triagem_humana.csv.",
    )
    parser.add_argument(
        "--prisma-curadoria-importar",
        action="store_true",
        help="Importa triagem_humana.csv e executa a geração final do PRISMA.",
    )
    parser.add_argument(
        "--prisma-curadoria-fluxo-completo",
        action="store_true",
        help="Executa curadoria IA e depois importa a triagem para gerar o PRISMA final.",
    )
    parser.add_argument(
        "--prisma-curadoria-prompt",
        default="",
        help="Caminho do YAML de prompt estruturado da curadoria.",
    )
    parser.add_argument(
        "--prisma-curadoria-input",
        default="",
        help="Entrada específica para a curadoria: XLSX/CSV de triagem ou XLSX revisado.",
    )
    parser.add_argument(
        "--prisma-curadoria-out-dir",
        default="",
        help="Diretório de saída do relatório PRISMA.",
    )
    parser.add_argument(
        "--prisma-curadoria-max-incluir",
        type=int,
        default=0,
        help="Número máximo de referências incluídas pela curadoria.",
    )
    parser.add_argument(
        "--prisma-curadoria-top-n-candidatos",
        type=int,
        default=0,
        help="Número de candidatos enviados/avaliados pela curadoria IA.",
    )
    parser.add_argument(
        "--prisma-curadoria-limiar-minimo",
        type=int,
        default=0,
        help="Limiar mínimo de inclusão da curadoria v2.",
    )
    # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS <<<
    args = parser.parse_args()

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
    if args.gui:
        from academic_pipeline_gui import run_gui
        return run_gui()

    if args.tui:
        from academic_pipeline_tui import run_tui
        return run_tui(no_clear=bool(args.no_clear))

    if args.list_toml_profiles:
        from academic_pipeline_toml_generator_interativo import print_profiles
        print_profiles()
        return 0

    if args.init_toml:
        from academic_pipeline_toml_generator_interativo import generate_interactive
        generate_interactive(non_interactive_profile=args.toml_profile or None, no_clear=bool(args.no_clear))
        return 0

    if args.list_institutions:
        print(describe_institution_profiles())
        return 0

    if args.list_layouts:
        if not args.config:
            raise RuntimeError("--list-layouts exige --config caminho.toml")
        cfg_layouts = load_config(Path(args.config).expanduser().resolve())
        layouts = available_layouts(cfg_layouts)
        if not layouts:
            print("Nenhum layout declarado no perfil institucional.")
        else:
            print("Layouts disponíveis:")
            for layout_id, spec in layouts.items():
                desc = str(spec.get("description") or spec.get("descricao") or "").strip()
                genero = str(spec.get("genero_academico") or "").strip()
                print(f"- {layout_id}" + (f" ({genero})" if genero else "") + (f": {desc}" if desc else ""))
            resolved = resolve_layout_spec(cfg_layouts)
            print(f"Layout resolvido para este TOML: {resolved.id}")
        return 0

    if args.explain_profile:
        print(explain_profile(args.explain_profile))
        return 0

    if args.show_prompts:
        if not args.config:
            raise RuntimeError("--show-prompts exige --config caminho.toml")
        cfg_preview = load_config(Path(args.config).expanduser().resolve())
        print(json.dumps(prompt_report_for_cfg(cfg_preview), ensure_ascii=False, indent=2))
        return 0

    if args.init_project:
        base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None
        result = init_project(args.init_project, project_type=args.project_type, base_dir=base_dir, overwrite=bool(args.overwrite_project), institution=args.institution)
        print("Projeto criado:")
        print(f"- Diretório: {result.project_dir}")
        print(f"- TOML: {result.config_path}")
        print(f"- DOI manifest: {result.doi_manifest_path}")
        print(f"- Documentos ZIP: {result.documentos_zip_path}")
        print(f"- Orientações ZIP: {result.orientacoes_zip_path}")
        print(f"- README: {result.readme_path}")
        return 0

    if args.make_doi_manifest:
        input_zip = Path(args.input_zip).expanduser().resolve() if args.input_zip else None
        input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
        if args.output:
            output = Path(args.output).expanduser().resolve()
        else:
            if input_zip:
                output = input_zip.parent / "doi_manifest.csv"
            elif input_dir:
                output = input_dir / "doi_manifest.csv"
            else:
                raise RuntimeError("Use --make-doi-manifest com --input-zip ou --input-dir.")
        result = make_doi_manifest(input_zip, input_dir, output, overwrite=True)
        print("DOI manifest gerado:")
        print(f"- Fonte: {result['source']}")
        print(f"- Saída: {result['output']}")
        print(f"- Arquivos listados: {result['total_files']}")
        return 0

    if args.inspect_bib:
        bib = Path(args.inspect_bib).expanduser().resolve()
        prefix = bib.with_name(bib.name + "_inspection")
        report = inspect_bib(bib, output_prefix=prefix)
        print(render_bib_inspection_markdown(report))
        print(f"Relatórios: {str(prefix)}.md e {str(prefix)}.json")
        return 0 if report.get("ok") else 1

    if args.quality_report:
        if not args.document_json:
            raise RuntimeError("--quality-report exige --document-json caminho/document.json")
        document_json = Path(args.document_json).expanduser().resolve()
        document = load_existing_document_json(document_json)
        org = Path(args.org).expanduser().resolve() if args.org else None
        bib_keys: list[str] = []
        if args.bib:
            from bibliography_manager import split_bib_entries, bib_entry_key
            bib_path = Path(args.bib).expanduser().resolve()
            if bib_path.exists():
                bib_keys = [k for e in split_bib_entries(bib_path.read_text(encoding='utf-8', errors='ignore')) if (k := bib_entry_key(e))]
        report = build_quality_report(document, org_path=org, bib_keys=bib_keys or list(document.bibliography.entries_used or []))
        out = document_json.with_suffix(".quality_report.md")
        write_quality_report(report, out)
        print(f"Relatório de qualidade: {out}")
        return 0 if report.get("ok") else 1

    cfg = _load_optional_config(args.config) if args.config else None
    if cfg:
        cfg = apply_cli_path_overrides(cfg, args)

    if args.somente_renderizar and args.somente_mapa_mental:
        raise RuntimeError("Use apenas um entre --somente-renderizar e --somente-mapa-mental.")
    if args.reusar_mapa_mental and args.forcar_regeneracao_mapa_mental:
        raise RuntimeError("Use apenas um entre --reusar-mapa-mental e --forcar-regeneracao-mapa-mental.")

    if args.write_prompt_lock:
        if not cfg:
            raise RuntimeError("--write-prompt-lock exige --config caminho.toml")
        out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
        lock_path = out_dir / f"{prefix}.prompt_lock.json"
        lock_md = out_dir / f"{prefix}.prompt_lock.md"
        lock = write_prompt_lock(cfg, lock_path)
        write_prompt_lock_markdown(lock, lock_md)
        print(f"Prompt lock gerado: {lock_path}")
        print(f"Prompt lock markdown: {lock_md}")
        return 0

    if args.check_institution_compliance:
        if not cfg:
            raise RuntimeError("--check-institution-compliance exige --config caminho.toml")
        out_dir, prefix = output_paths(cfg)
        org = Path(args.org).expanduser().resolve() if args.org else out_dir / f"{prefix}.org"
        bib = Path(args.bib).expanduser().resolve() if args.bib else out_dir / f"{prefix}.bib"
        docx = Path(args.docx).expanduser().resolve() if args.docx else out_dir / f"{prefix}.docx"
        pdf = Path(args.pdf).expanduser().resolve() if args.pdf else out_dir / f"{prefix}.pdf"
        report = run_institution_compliance(cfg, org_path=org, bib_path=bib, docx_path=docx, pdf_path=pdf)
        md_path, json_path = write_compliance_reports(report, out_dir / prefix)
        print(render_compliance_markdown(report))
        print(f"Relatórios: {md_path} e {json_path}")
        return 0 if report.get("ok") else 2

    if args.doctor:
        report = run_doctor(cfg)
        print_doctor_report(report)
        if cfg:
            out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
            write_json(out_dir / f"{prefix}.doctor_report.json", report)
        return 0 if report.get("ok") else 2

    if args.check_config:
        if not cfg:
            raise RuntimeError("--check-config exige --config caminho.toml")
        report = check_config(cfg)
        print_check_config_report(report)
        out_dir, prefix = research_output_paths(cfg) if external_search_enabled(cfg) else output_paths(cfg)
        write_json(out_dir / f"{prefix}.check_config_report.json", report)
        return 0 if report.get("ok") else 2

    if args.recompile:
        return run_recompile(args, cfg)

    if not cfg:
        raise RuntimeError("Informe --config, ou use --doctor sem config.")

    if args.prisma_importar_triagem:
        if not external_search_enabled(cfg):
            raise RuntimeError("--prisma-importar-triagem exige um TOML do perfil relatorio_prisma_busca_orientada_fgv.")
        out_dir, prefix = research_output_paths(cfg)
        stage("Importando planilha de triagem PRISMA preenchida")
        prisma_outputs = import_manual_prisma_triage(cfg, out_dir, prefix, Path(args.prisma_importar_triagem))
        org_path, pdf_path = render_external_prisma_outputs(
            cfg,
            out_dir,
            prefix,
            prisma_outputs,
            phase="final",
        )
        artifacts = prisma_outputs.setdefault("artefatos", {}) if isinstance(prisma_outputs, dict) else {}
        if org_path:
            artifacts["relatorio_org"] = str(org_path)
        if pdf_path:
            artifacts["relatorio_pdf"] = str(pdf_path)
        report_json_path = artifacts.get("prisma_report_json") if isinstance(artifacts, dict) else ""
        if report_json_path:
            write_json(Path(str(report_json_path)), prisma_outputs)
        outputs = {
            "output_dir": str(out_dir),
            "org": str(org_path) if org_path else None,
            "pdf": str(pdf_path) if pdf_path else None,
            "relatorio_pesquisa": prisma_outputs,
        }
        report = make_run_report(
            cfg=cfg,
            config_path=Path(str(cfg.get("__config_path__"))),
            out_dir=out_dir,
            prefix=prefix,
            model=None,
            outputs=outputs,
            warnings=[],
            extra={"mode": "prisma_importar_triagem"},
        )
        write_json(out_dir / f"{prefix}.run_report.json", report)
        write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
        print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — triagem PRISMA consolidada")
        return 0

    cfg["__somente_renderizar__"] = bool(args.somente_renderizar)
    is_external_prisma_run = external_search_enabled(cfg) and not args.somente_renderizar
    out_dir, prefix = research_output_paths(cfg) if is_external_prisma_run else output_paths(cfg)
    work_dir, cache_dir = work_cache_paths(cfg, prefix)
    doc_cfg = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
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

    if args.somente_mapa_mental:
        if not document_json_path.exists():
            raise FileNotFoundError(f"document.json não encontrado para --somente-mapa-mental: {document_json_path}")
        if not should_generate_mindmap(cfg):
            raise RuntimeError("[mapa_mental] não está ativo no TOML. Ative gerar=true/ativo=true para usar --somente-mapa-mental.")
        stage("Carregando document.json existente")
        document = load_existing_document_json(document_json_path)
        removed_mindmap_files: list[str] = []
        if args.forcar_regeneracao_mapa_mental:
            stage("Removendo mapa mental existente")
            removed_mindmap_files = delete_existing_mindmap_outputs(cfg, out_dir)
        mm_diag = None
        if args.reusar_mapa_mental:
            stage("Tentando reutilizar mapa mental existente")
            mm_diag = attach_existing_mindmap_if_available(document, cfg, out_dir)
            if not mm_diag:
                warnings.append("Mapa mental existente não encontrado; gerando novo mapa mental.")
        if not mm_diag:
            stage("Inicializando cliente OpenAI")
            client, model = make_client(model)
            stage("Gerando/renderizando apenas o mapa mental")
            mm_diag = generate_and_attach_mindmap(client, model, cfg, document, out_dir)
        if removed_mindmap_files:
            mm_diag = dict(mm_diag or {})
            mm_diag["removed_before_regeneration"] = removed_mindmap_files
        document.diagnostics.mindmap_json = json.dumps(mm_diag, ensure_ascii=False)
        stage("Salvando document.json atualizado")
        write_json(document_json_path, document.model_dump())
        outputs = {
            "output_dir": str(out_dir),
            "document_json": str(document_json_path),
            "mindmap_puml": (mm_diag or {}).get("puml_path") if mm_diag else None,
            "mindmap_image": (mm_diag or {}).get("image_path") if mm_diag else None,
            "mindmap_reused": bool((mm_diag or {}).get("reused")),
            "mindmap_removed": removed_mindmap_files,
        }
        report = make_run_report(
            cfg=cfg,
            config_path=Path(str(cfg.get("__config_path__"))),
            out_dir=out_dir,
            prefix=prefix,
            model=model,
            outputs=outputs,
            warnings=warnings,
            extra={"mode": "somente_mapa_mental"},
        )
        write_json(out_dir / f"{prefix}.run_report.json", report)
        write_outputs_manifest(out_dir / f"{prefix}.outputs.txt", outputs)
        print_outputs(outputs, title=f"academic_pipeline {PIPELINE_VERSION} — mapa mental renderizado")
        if warnings:
            print("Avisos:")
            for w in warnings:
                print(f"- {w}")
        return 0

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

    stage("Renderizando ORG/LaTeX")
    org_path = out_dir / f"{prefix}.org"
    org_text = render_org_latex(document, org_path, bib_path.name if 'bib_path' in locals() else f"{prefix}.bib", cfg=cfg, bib_keys=bib_keys if 'bib_keys' in locals() else None)
    if paper_abstract_bundle:
        stage("Inserindo resumo e palavras-chave no ORG")
        org_text = inject_paper_abstracts_into_org(org_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
    stage("Validando ORG renderizado")
    raise_if_errors(validate_org_text(org_text, bib_keys), "Validação do ORG renderizado falhou")

    pdf_path = None
    if bool(doc_cfg.get("exportar_pdf", True)):
        academic_writing = resolve_path(latex_cfg.get("org_latex_class_init"), config_dir)
        latex_extra = resolve_path(latex_cfg.get("latex_extra_path"), config_dir)
        pdf_engine = str(latex_cfg.get("pdf_engine") or "lualatex")
        stage("Compilando PDF via Emacs/LaTeX")
        pdf_path = run_compile_sequence(org_path, academic_writing=academic_writing, latex_extra_path=latex_extra, pdf_engine=pdf_engine)

    docx_path = None
    docx_validation: dict[str, Any] | None = None
    if bool(doc_cfg.get("exportar_docx", True)):
        docx_cfg = cfg.get("docx", {}) if isinstance(cfg.get("docx"), dict) else {}
        ref = resolve_path(docx_cfg.get("reference_docx") or doc_cfg.get("docx_reference"), config_dir)
        stage("Renderizando DOCX")
        docx_path = render_docx(document, out_dir / f"{prefix}.docx", bib_path=bib_path, reference_docx=ref, cfg=cfg)
        if paper_abstract_bundle:
            stage("Inserindo resumo e palavras-chave no DOCX")
            inject_paper_abstracts_into_docx(docx_path, paper_abstract_bundle, main_document_abstract_languages(cfg))
        docx_validation = validate_docx_file(docx_path, expected_title=document.metadata.titulo, require_references=bool(document.bibliography.entries_used))
        if docx_validation and docx_validation.get("warnings"):
            warnings.extend([f"DOCX: {w}" for w in docx_validation.get("warnings", [])])

    translated_outputs: dict[str, Any] = {}
    if args.somente_renderizar:
        if requested_translation_languages(cfg):
            warnings.append(
                "Versões adicionais por IA não foram atualizadas no modo --somente-renderizar. "
                "Execute a geração completa para traduzir o document.json canônico."
            )
    elif requested_translation_languages(cfg):
        try:
            translated_outputs, translation_warnings = render_additional_language_versions(
                client=client,
                model=model,
                cfg=cfg,
                document=document,
                bib_path=bib_path,
                bib_keys=bib_keys,
                out_dir=out_dir,
                prefix=prefix,
                doc_cfg=doc_cfg,
                latex_cfg=latex_cfg,
                config_dir=config_dir,
                abstract_bundle=paper_abstract_bundle or None,
            )
            warnings.extend(translation_warnings)
        except TranslationError as exc:
            # Traduções são saídas opcionais: uma falha nelas não invalida o
            # paper principal que já foi gerado e validado.
            warnings.append(f"TRADUÇÃO: {exc}")

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
    compliance_report = run_institution_compliance(
        cfg,
        org_path=org_path,
        bib_path=bib_path,
        docx_path=docx_path,
        pdf_path=pdf_path,
    )
    compliance_md, compliance_json = write_compliance_reports(compliance_report, out_dir / prefix)
    outputs["compliance_report"] = str(compliance_md)
    if compliance_report.get("warnings"):
        warnings.extend([f"CONFORMIDADE: {w.get('message')}" for w in compliance_report.get("warnings", [])])
    if not compliance_report.get("ok"):
        warnings.extend([f"CONFORMIDADE CRÍTICA: {e.get('message')}" for e in compliance_report.get("errors", [])])

    stage("Gerando relatório de qualidade")
    quality = build_quality_report(document, org_path=org_path, bib_keys=bib_keys)
    quality_path = out_dir / f"{prefix}.quality_report.md"
    write_quality_report(quality, quality_path)
    if quality.get("warnings"):
        warnings.extend([f"QUALIDADE: {w}" for w in quality.get("warnings", [])])
    outputs["quality_report"] = str(quality_path)

    report = make_run_report(
        cfg=cfg,
        config_path=Path(str(cfg.get("__config_path__"))),
        out_dir=out_dir,
        prefix=prefix,
        model=model,
        outputs=outputs,
        warnings=warnings,
        extra={
            "mode": "somente_renderizar" if args.somente_renderizar else "full",
            "work_dir": str(work_dir),
            "cache_dir": str(cache_dir),
            "precheck": precheck,
            "docx_validation": docx_validation,
        },
    )
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
    if not isinstance(cfg, dict):
        return False
    bibliography = cfg.get("bibliografia", {}) if isinstance(cfg.get("bibliografia"), dict) else {}
    document = cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}
    local = cfg.get("documentos_locais", {}) if isinstance(cfg.get("documentos_locais"), dict) else {}

    if "ativo" in bibliography:
        return not bool(bibliography.get("ativo"))
    if "referencias_formais" in document:
        return not bool(document.get("referencias_formais"))
    return (
        local.get("auto_detect_bib") is False
        and local.get("gerar_bib_revisado_ia") is False
        and document.get("usar_citacoes_latex_diretas") is False
    )


def _refs_v6_apply_runtime_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    if not _refs_v6_disabled(cfg):
        return cfg
    bibliography = cfg.setdefault("bibliografia", {})
    if not isinstance(bibliography, dict):
        bibliography = {}
        cfg["bibliografia"] = bibliography
    bibliography["ativo"] = False
    bibliography["gerar_arquivo_bib"] = False
    bibliography["buscar_metadados_por_doi"] = False
    bibliography["enriquecer_metadados_buscadores"] = False

    document = cfg.setdefault("documento", {})
    if not isinstance(document, dict):
        document = {}
        cfg["documento"] = document
    document["referencias_formais"] = False
    document["usar_citacoes_latex_diretas"] = False

    local = cfg.setdefault("documentos_locais", {})
    if isinstance(local, dict):
        local["auto_detect_bib"] = False
        local["gerar_bib_revisado_ia"] = False
        local["enriquecer_metadados_buscadores"] = False
        local["extrair_doi_dos_pdfs"] = False
        local["buscar_metadados_por_doi"] = False

    orientations = cfg.setdefault("orientacoes", {})
    if not isinstance(orientations, dict):
        orientations = {}
        cfg["orientacoes"] = orientations
    instruction = (
        "Não inclua citações no corpo do texto, notas bibliográficas, seção Referências, "
        "lista bibliográfica ou arquivo .bib. Use exclusivamente o corpus local e não invente fontes."
    )
    current = str(orientations.get("inline") or "").strip()
    if instruction not in current:
        orientations["inline"] = (current + "\n\n" + instruction).strip()
    return cfg


# Carrega a política antes de qualquer rotina de descoberta, bibliografia ou IA.
_refs_v6_original_load_config = load_config
def load_config(path: Path) -> dict[str, Any]:
    return _refs_v6_apply_runtime_policy(_refs_v6_original_load_config(path))


# Impede a construção física do .bib. Um Path sentinela mantém compatibilidade
# com funções que recebem bib_path, mas o arquivo não é criado e as chaves ficam vazias.
_refs_v6_original_build_bibliography = build_bibliography
def build_bibliography(
    cfg: dict[str, Any],
    docs: Any,
    out_dir: Path,
    prefix: str,
    client: Any,
    model: str,
) -> Any:
    if _refs_v6_disabled(cfg):
        from types import SimpleNamespace
        return SimpleNamespace(bib_path=Path(out_dir) / f"{prefix}.bib", keys=[])
    return _refs_v6_original_build_bibliography(cfg, docs, out_dir, prefix, client, model)


def _refs_v6_clear_document_bibliography(document: Any) -> Any:
    bibliography = getattr(document, "bibliography", None)
    if bibliography is not None:
        try:
            bibliography.entries_used = []
        except Exception:
            pass
        try:
            bibliography.bib_path = ""
        except Exception:
            pass
    return document


def _refs_v6_strip_org(text: str) -> str:
    import re as _re
    # Diretivas e comandos de bibliografia.
    text = _re.sub(
        r"(?im)^.*(?:#\+(?:print_)?bibliography|\\addbibresource|\\printbibliography).*(?:\n|$)",
        "",
        text,
    )
    # Citações Org e LaTeX que possam ter sido produzidas antes da renderização.
    text = _re.sub(r"(?is)\[cite(?:/[\w-]+)?\s*:[^\]]*\]", "", text)
    text = _re.sub(r"(?is)\[@[A-Za-z0-9_:.+/\-]+(?:;\s*@[A-Za-z0-9_:.+/\-]+)*\]", "", text)
    text = _re.sub(
        r"(?is)\\(?:auto|text|para|smart|foot|super)?cite(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^}]*\}",
        "",
        text,
    )
    # A seção final é removida apenas quando usa um título inequívoco.
    text = _re.sub(
        r"(?ims)^\*+\s*(?:refer[eê]ncias|bibliografia)\s*$.*?(?=^\*+\s+|\Z)",
        "",
        text,
    )
    text = _re.sub(
        r"(?is)\\(?:section|section\*|chapter|chapter\*)\{\s*(?:refer[eê]ncias|bibliografia)\s*\}.*?(?=\\(?:section|chapter)\{|\\end\{document\}|\Z)",
        "",
        text,
    )
    # Normalização visual após a remoção.
    return _re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# Garante que PDF/ORG não exibam citações ou referências mesmo se um artefato
# intermediário trouxer marcas bibliográficas inesperadas.
_refs_v6_original_render_org_latex = render_org_latex
def render_org_latex(
    document: Any,
    org_path: Path,
    bib_filename: str,
    *,
    cfg: dict[str, Any],
    bib_keys: list[str] | None = None,
) -> str:
    if not _refs_v6_disabled(cfg):
        return _refs_v6_original_render_org_latex(
            document,
            org_path,
            bib_filename,
            cfg=cfg,
            bib_keys=bib_keys,
        )
    document = _refs_v6_clear_document_bibliography(document)
    rendered = _refs_v6_original_render_org_latex(
        document,
        org_path,
        bib_filename,
        cfg=cfg,
        bib_keys=[],
    )
    clean = _refs_v6_strip_org(rendered)
    Path(org_path).write_text(clean, encoding="utf-8")
    return clean
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
