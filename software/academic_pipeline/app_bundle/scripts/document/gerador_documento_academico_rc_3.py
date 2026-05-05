#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador standalone de documento acadêmico em Org-mode, alinhado ao ecossistema atual do projeto.

Principais ajustes em relação ao script legado:
- nomenclatura de templates por tipo documental: template_paper.org / template_dissertacao.org / template_research.org;
- suporte a TOML via --config;
- suporte a bundle/pesquisa existente;
- nomenclatura unificada de orientações: orientacoes_paths + orientacao_inline;
- possibilidade de artigos extras;
- dry_run e pacote final de entrega.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import textwrap
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None

import gerar_documento_org_ai_interativo_rc_1 as core

DEFAULT_STANDALONE_TEMPLATE = "template_paper.org"
DEFAULT_RESEARCH_TEMPLATE_FALLBACK = "template_research.org"
DEFAULT_BASENAME = "documento"
STATE_FILE = ".gerador_documento_academico_rc_3_state.json"

DEFAULT_TEMPLATE_BY_DOC_TYPE = {
    "paper": "template_paper.org",
    "dissertacao": "template_dissertacao.org",
}
DEFAULT_BASENAME_BY_DOC_TYPE = {
    "paper": "paper",
    "dissertacao": "dissertacao",
}


def normalize_document_type(raw: str | None) -> str:
    value = (raw or "paper").strip().lower()
    aliases = {
        "papel": "paper",
        "artigo": "paper",
        "paper": "paper",
        "dissertacao": "dissertacao",
        "dissertação": "dissertacao",
        "thesis": "dissertacao",
    }
    return aliases.get(value, value)



def load_toml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if tomllib is None:
        raise RuntimeError("Python sem suporte a tomllib. Use Python 3.11+.")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def load_state() -> dict[str, Any]:
    path = script_dir() / STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(data: dict[str, Any]) -> None:
    path = script_dir() / STATE_FILE
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bool_cfg(cfg: dict[str, Any], *keys: str, default: bool = False) -> bool:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return bool(cur)


def get_cfg(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raw = str(value).strip()
    return [raw] if raw else []


def resolve_orientation_docs(paths: list[str], inline_text: str, workdir: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(os.path.expanduser(raw)).resolve()
        if p.exists() and p.is_file():
            key = str(p)
            if key not in seen:
                out.append(p)
                seen.add(key)
    if inline_text.strip():
        temp = workdir / "orientacao_inline.txt"
        temp.write_text(inline_text.strip() + "\n", encoding="utf-8")
        key = str(temp)
        if key not in seen:
            out.append(temp)
            seen.add(key)
    return out


def default_template_path(doc_type: str = "paper") -> Path:
    doc_type = normalize_document_type(doc_type)
    template_name = DEFAULT_TEMPLATE_BY_DOC_TYPE.get(doc_type, "template_paper.org")
    candidates = [
        Path.cwd() / "templates" / template_name,
        Path.cwd() / template_name,
        Path.cwd() / "templates" / DEFAULT_RESEARCH_TEMPLATE_FALLBACK,
        script_dir().parent.parent / "templates" / template_name,
        script_dir().parent.parent / "templates" / DEFAULT_RESEARCH_TEMPLATE_FALLBACK,
        script_dir() / template_name,
        script_dir() / DEFAULT_RESEARCH_TEMPLATE_FALLBACK,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def collect_bundle_artifacts(bundle_dir: Path) -> dict[str, Path | list[Path] | None]:
    artifacts: dict[str, Path | list[Path] | None] = {
        "research_org": None,
        "research_context": None,
        "research_bib": None,
        "research_manifest": None,
        "fulltexts": [],
        "documento_org": None,
    }
    manifest = bundle_dir / "manifest.json"
    if manifest.exists():
        artifacts["research_manifest"] = manifest
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            art = data.get("artifacts", {}) if isinstance(data, dict) else {}
            for key_src, key_dst in [("research_org", "research_org"), ("bibliography", "research_bib"), ("documento_org", "documento_org")]:
                raw = art.get(key_src)
                if raw:
                    p = Path(raw)
                    if not p.is_absolute():
                        p = (bundle_dir / p).resolve()
                    if p.exists():
                        artifacts[key_dst] = p
            raw_ctx = art.get("documento_context_json") or art.get("provenance")
            if raw_ctx:
                p = Path(raw_ctx)
                if not p.is_absolute():
                    p = (bundle_dir / p).resolve()
                if p.exists():
                    artifacts["research_context"] = p
            raw_full = art.get("fulltexts_dir")
            if raw_full:
                p = Path(raw_full)
                if not p.is_absolute():
                    p = (bundle_dir / p).resolve()
                if p.exists() and p.is_dir():
                    artifacts["fulltexts"] = sorted([x.resolve() for x in p.rglob("*.pdf") if x.is_file()])
        except Exception:
            pass
    if artifacts["research_org"] is None:
        for name in ["pesquisa_manual.org", "pesquisa.org", "research.org"]:
            p = bundle_dir / name
            if p.exists():
                artifacts["research_org"] = p.resolve()
                break
    if artifacts["research_bib"] is None:
        for name in ["referencias.bib", "research.bib", "documento.bib"]:
            p = bundle_dir / name
            if p.exists():
                artifacts["research_bib"] = p.resolve()
                break
    full_dir = bundle_dir / "fulltexts"
    if not artifacts["fulltexts"] and full_dir.exists():
        artifacts["fulltexts"] = sorted([x.resolve() for x in full_dir.rglob("*.pdf") if x.is_file()])
    return artifacts


def build_docs_from_paths(paths: list[Path], kind: str, max_chars: int = 30000) -> list[core.SourceDoc]:
    items = [core.InputItem(path=p, label=p.name, metadata={"source": str(p)}) for p in paths if p.exists() and p.is_file()]
    return core.build_source_docs(items, kind, max_chars=max_chars)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gerador standalone de documento acadêmico em Org-mode, alinhado ao pipeline unificado.")
    p.add_argument("--config")
    p.add_argument("--tipo-documento")
    p.add_argument("--bundle-dir")
    p.add_argument("--template")
    p.add_argument("--output-dir")
    p.add_argument("--basename")
    p.add_argument("--model")
    p.add_argument("--citation-style", choices=["apa", "abnt"])
    p.add_argument("--exportar-pdf", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--nao-salvar-prompts", dest="salvar_prompts", action="store_false")
    p.set_defaults(salvar_prompts=True)
    return p.parse_args()


def main() -> int:
    core.load_env()
    args = parse_args()
    core.DEBUG = bool(args.debug)
    state = load_state()
    cfg = load_toml(Path(args.config).expanduser().resolve()) if args.config else {}
    prompt_log: list[tuple[str, str]] = []

    nao_interativo = bool_cfg(cfg, "controle", "nao_interativo", default=bool(args.config))
    dry_run = bool_cfg(cfg, "controle", "dry_run", default=False)

    doc_type = normalize_document_type(args.tipo_documento or get_cfg(cfg, "documento", "tipo_documento") or state.get("tipo_documento") or "documento")
    bundle_dir_raw = args.bundle_dir or get_cfg(cfg, "documento", "bundle_dir") or get_cfg(cfg, "pipeline", "bundle_dir") or get_cfg(cfg, "pipeline", "pesquisa_dir_existente")
    bundle_dir = Path(os.path.expanduser(str(bundle_dir_raw))).resolve() if bundle_dir_raw else None
    bundle_artifacts = collect_bundle_artifacts(bundle_dir) if bundle_dir else {}

    template_default = args.template or get_cfg(cfg, "documento", "template_org") or state.get("last_template") or str(default_template_path(doc_type))
    output_default = args.output_dir or get_cfg(cfg, "documento", "output_dir") or state.get("last_output_dir") or "."
    basename_default = args.basename or get_cfg(cfg, "documento", "prefixo") or state.get("last_basename") or DEFAULT_BASENAME_BY_DOC_TYPE.get(doc_type, DEFAULT_BASENAME)
    style_default = core.normalize_style(args.citation_style or get_cfg(cfg, "documento", "estilo_citacao") or get_cfg(cfg, "bibliografia", "estilo_citacao") or state.get("citation_style") or core.DEFAULT_STYLE)
    model = args.model or get_cfg(cfg, "openai", "model") or core.DEFAULT_MODEL

    emacs_init_default = get_cfg(cfg, "latex", "org_latex_class_init") or state.get("last_emacs_init")
    academic_writing_default = get_cfg(cfg, "latex", "org_latex_class_init") or state.get("last_academic_writing")
    latex_extra_default = get_cfg(cfg, "latex", "latex_extra_path") or state.get("last_latex_extra_path")

    emacs_init = Path(os.path.expanduser(emacs_init_default)).resolve() if emacs_init_default else core.default_emacs_init()
    academic_writing = Path(os.path.expanduser(academic_writing_default)).resolve() if academic_writing_default else core.default_academic_writing()
    latex_extra_path = Path(os.path.expanduser(latex_extra_default)).resolve() if latex_extra_default else (Path(core.DEFAULT_LATEX_EXTRA_PATH).expanduser().resolve() if Path(core.DEFAULT_LATEX_EXTRA_PATH).expanduser().exists() else None)

    exportar_pdf = bool(args.exportar_pdf or bool_cfg(cfg, "documento", "exportar_pdf", default=False))

    core.preflight_checks(exportar_pdf=bool(exportar_pdf), emacs_init=emacs_init, academic_writing=academic_writing, latex_extra_path=latex_extra_path)
    if args.preflight_only:
        print("Pré-check concluído com sucesso.")
        return 0

    client = core.make_client()

    if nao_interativo:
        template_path = Path(os.path.expanduser(str(template_default))).resolve()
        if not template_path.exists():
            raise RuntimeError(f"Template do documento não encontrado: {template_path}")
        output_root_dir = Path(os.path.expanduser(str(output_default))).resolve()
        basename = str(basename_default).strip() or DEFAULT_BASENAME
    else:
        template_path = core.prompt_path("Template .org do documento acadêmico", template_default, must_exist=True)
        output_root_dir = core.prompt_path("Diretório de saída", output_default, only_directories=True)
        basename = core.prompt_text("Nome-base dos arquivos de saída do documento", basename_default)

    output_dir = output_root_dir / basename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_workspace_dir = output_dir / ".documento_inputs_tmp"
    input_workspace_dir.mkdir(parents=True, exist_ok=True)

    style = style_default if nao_interativo else core.normalize_style(core.prompt_text("Estilo bibliográfico/citacional (apa ou abnt)", style_default, completer=core._word_completer(["apa", "abnt"])))

    # Base docs and guidance docs
    base_paths: list[Path] = []
    guidance_paths: list[Path] = []
    imported_bib_paths: list[Path] = []

    # Bundle/proveniência da pesquisa
    if bundle_artifacts:
        if bool_cfg(cfg, "documento", "usar_artigos_selecionados_pesquisa", default=True):
            base_paths.extend(bundle_artifacts.get("fulltexts", []) or [])
        research_org = bundle_artifacts.get("research_org")
        research_ctx = bundle_artifacts.get("research_context")
        if isinstance(research_org, Path):
            guidance_paths.append(research_org)
        if isinstance(research_ctx, Path):
            guidance_paths.append(research_ctx)
        research_bib = bundle_artifacts.get("research_bib")
        if isinstance(research_bib, Path):
            imported_bib_paths.append(research_bib)
        documento_org_prev = bundle_artifacts.get("documento_org")
        if bool_cfg(cfg, "documento", "modo_escrita", default="novo") in {"reescrever", "expandir"}:
            if isinstance(documento_org_prev, Path):
                guidance_paths.append(documento_org_prev)

    # Config documento base docs / extras
    documento_base_paths = [Path(os.path.expanduser(x)).resolve() for x in coerce_list(get_cfg(cfg, "documento", "base_docs_paths", default=[]))]
    extra_paths_raw = coerce_list(get_cfg(cfg, "documento", "artigos_extras_paths", default=[]))
    extra_paths: list[Path] = []
    for raw in extra_paths_raw:
        p = Path(os.path.expanduser(raw)).resolve()
        if p.is_dir():
            extra_paths.extend(sorted([x.resolve() for x in p.rglob("*") if x.is_file() and x.suffix.lower() in core.READABLE_SUFFIXES]))
        elif p.is_file():
            extra_paths.append(p)
    base_paths.extend([p for p in documento_base_paths if p.exists()])
    base_paths.extend([p for p in extra_paths if p.exists()])

    # Unified orientations
    saida_orient = resolve_orientation_docs(
        coerce_list(get_cfg(cfg, "saida", "orientacoes_paths", default=[])),
        str(get_cfg(cfg, "saida", "orientacao_inline", default="") or ""),
        input_workspace_dir,
    )
    documento_orient = resolve_orientation_docs(
        coerce_list(get_cfg(cfg, "documento", "orientacoes_paths", default=[])),
        str(get_cfg(cfg, "documento", "orientacao_inline", default="") or ""),
        input_workspace_dir,
    )
    guidance_paths.extend(saida_orient)
    guidance_paths.extend(documento_orient)

    # Documento alvo como orientação explícita para a IA
    doc_type_inline = input_workspace_dir / "tipo_documento_alvo.txt"
    doc_type_inline.write_text(
        f"Tipo de documento alvo: {doc_type}.\n"
        "A redação deve respeitar o template selecionado e as convenções estruturais do tipo documental indicado.\n",
        encoding="utf-8",
    )
    guidance_paths.append(doc_type_inline)

    # Reescrita/expansão a partir de org anterior
    documento_org_existente = get_cfg(cfg, "documento", "documento_org_existente")
    if documento_org_existente:
        p = Path(os.path.expanduser(str(documento_org_existente))).resolve()
        if p.exists() and p.is_file():
            guidance_paths.append(p)

    # Imported bib from config
    imported_bib_paths.extend([Path(os.path.expanduser(x)).resolve() for x in coerce_list(get_cfg(cfg, "documento", "bib_paths", default=[])) if Path(os.path.expanduser(x)).exists()])
    # dedupe paths
    def dedupe_paths(items: list[Path]) -> list[Path]:
        seen: set[str] = set()
        out: list[Path] = []
        for p in items:
            k = str(p)
            if k not in seen and p.exists() and p.is_file():
                out.append(p)
                seen.add(k)
        return out
    base_paths = dedupe_paths(base_paths)
    guidance_paths = dedupe_paths(guidance_paths)
    imported_bib_paths = dedupe_paths(imported_bib_paths)

    if not nao_interativo:
        if not base_paths:
            base_paths = core.prompt_multi_paths("Selecione os documentos-base do documento", required=True)
        else:
            print(f"Documentos-base resolvidos automaticamente: {len(base_paths)}")
            if core.prompt_yes_no("Deseja adicionar mais documentos-base manualmente?", default=False):
                base_paths.extend(core.prompt_multi_paths("Selecione documentos-base adicionais do documento", required=False))
        if guidance_paths:
            print(f"Arquivos de orientação resolvidos automaticamente: {len(guidance_paths)}")
            if core.prompt_yes_no("Deseja adicionar mais arquivos de orientação manualmente?", default=False):
                guidance_paths.extend(core.prompt_multi_paths("Selecione arquivos de orientação adicionais", required=False))
        else:
            guidance_paths = core.prompt_multi_paths("Selecione arquivos de orientação", required=False)
        if imported_bib_paths:
            print(f"Arquivos .bib resolvidos automaticamente: {len(imported_bib_paths)}")
            if core.prompt_yes_no("Deseja adicionar mais .bib manualmente?", default=False):
                imported_bib_paths.extend([p for p in core.prompt_multi_paths("Selecione arquivos .bib adicionais", required=False) if p.suffix.lower()=='.bib'])
        elif core.prompt_yes_no("Deseja importar um .bib existente para complementar/mesclar as referências?", default=False):
            imported_bib_paths = [p for p in core.prompt_multi_paths("Selecione um ou mais arquivos .bib existentes", required=True) if p.suffix.lower()=='.bib']

    if dry_run:
        snapshot = {
            "template_path": str(template_path),
            "output_dir": str(output_dir),
            "basename": basename,
            "style": style,
            "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "tipo_documento": doc_type,
            "base_paths": [str(p) for p in base_paths],
            "guidance_paths": [str(p) for p in guidance_paths],
            "imported_bib_paths": [str(p) for p in imported_bib_paths],
        }
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return 0

    if not base_paths:
        raise RuntimeError("Nenhum documento-base foi resolvido. Informe base_docs_paths, bundle_dir/pesquisa_dir_existente ou adicione documentos manualmente.")

    bib_filename = f"{basename}.bib"
    template_raw = core.read_template_raw(template_path)
    template_fields = core.parse_template_fields(template_raw)
    # respostas acadêmicas guiadas por config, sem depender de None como interface principal
    author_default = str(get_cfg(cfg, "atividade", "aluno", default="") or core.DEFAULT_AUTHOR)
    institution_default = str(get_cfg(cfg, "documento", "institution_name", default="") or core.DEFAULT_INSTITUTION)
    final_answers = {f.key: f.default for f in template_fields}
    final_answers.update({
        "title": core.AUTO_HEADER_TITLE,
        "author": author_default or core.DEFAULT_AUTHOR,
        "institution_name": institution_default or core.DEFAULT_INSTITUTION,
        "course_name": str(get_cfg(cfg, "atividade", "curso", default="") or ""),
        "discipline_name": str(get_cfg(cfg, "atividade", "disciplina", default="") or ""),
        "professor_name": str(get_cfg(cfg, "atividade", "professor", default="") or ""),
        "city_name": str(get_cfg(cfg, "atividade", "polo", default="") or ""),
        "documento_type": core.AUTO_HEADER_PAPER_TYPE,
        "cover_note": core.AUTO_HEADER_COVER_NOTE,
        "bibliography_file": bib_filename,
    })
    template_text = core.materialize_template(template_raw, template_fields, final_answers)
    template_text = core.apply_citation_style(template_text, bib_filename, style)

    base_items = core.collect_input_items(base_paths, input_workspace_dir)
    guidance_items = core.collect_input_items(guidance_paths, input_workspace_dir) if guidance_paths else []
    base_docs = core.build_source_docs(base_items, "base", max_chars=40000)
    guidance_docs = core.build_source_docs(guidance_items, "orientacao", max_chars=25000) if guidance_items else []

    # Se preservar estrutura do org anterior foi pedido, injeta instrução textual
    if bool_cfg(cfg, "documento", "preservar_estrutura_do_org_anterior", default=False):
        hint_path = input_workspace_dir / "preservar_estrutura_hint.txt"
        hint_path.write_text(
            "Preserve, tanto quanto possível, a arquitetura global do org anterior usado como orientação, reescrevendo e aprofundando o conteúdo sem desmontar a estrutura principal.\n",
            encoding="utf-8",
        )
        guidance_docs.extend(core.build_source_docs([core.InputItem(path=hint_path, label=hint_path.name, metadata={})], "orientacao", max_chars=5000))

    base_context, narrowed_context, tema_adicional_1, tema_adicional_2, context, strategy_answers = core.infer_context_with_ai(
        client, model, template_text, base_docs, guidance_docs, prompt_log
    )

    # Optionally allow reformulation overrides from config
    if not bool_cfg(cfg, "documento", "usar_contexto_consolidado_da_pesquisa", default=True) or bool_cfg(cfg, "documento", "reformular_tema_recorte_objetivo", default=False):
        context.tema = str(get_cfg(cfg, "documento", "tema", default=context.tema) or context.tema)
        context.recorte = str(get_cfg(cfg, "documento", "recorte", default=context.recorte) or context.recorte)
        context.objetivo = str(get_cfg(cfg, "documento", "objetivo", default=context.objetivo) or context.objetivo)

    base_docs = core.build_base_doc_bibliography(client, model, base_docs, prompt_log)
    for doc in base_docs:
        try:
            doc.summary = core.summarize_document(client, model, doc, prompt_log)
        except Exception:
            doc.summary = core.shorten_text(doc.extracted_text, 1200)

    correlated_docs: list[core.SourceDoc] = []
    related_info: dict[str, Any] = {"used": False}
    if bool_cfg(cfg, "documento", "permitir_busca_correlata_extra", default=False):
        correlated_docs, related_info = core.run_related_search_flow(client, model, context, output_dir, basename, prompt_log)

    imported_bib_entries, used_keys, imported_bib_files = core.load_existing_bib_entries(imported_bib_paths)
    bib_entries: list[str] = list(imported_bib_entries)
    for doc in [*base_docs, *correlated_docs]:
        if doc.bib_key is None:
            doc.bib_key = core.unique_key(core.slugify(Path(doc.path).stem), used_keys)
        else:
            doc.bib_key = core.unique_key(doc.bib_key, used_keys)
        if doc.bib_entry is None:
            meta = core.BibMetadataOutput(entry_type="misc", title=Path(doc.path).stem.replace("_", " "), note="Metadados incompletos; revisar manualmente.")
            doc.bib_entry = core.render_biblatex_entry(doc.bib_key, meta)
        else:
            doc.bib_entry = re.sub(r"^@([^{]+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{doc.bib_key},", doc.bib_entry, count=1)
        bib_entries.append(doc.bib_entry.strip())

    final_answers.update({k: v for k, v in strategy_answers.items() if v not in (None, "")})
    final_answers["title"] = core.AUTO_HEADER_TITLE
    final_answers["author"] = final_answers.get("author", core.DEFAULT_AUTHOR) or core.DEFAULT_AUTHOR
    final_answers["institution_name"] = final_answers.get("institution_name", core.DEFAULT_INSTITUTION) or core.DEFAULT_INSTITUTION
    final_answers["documento_type"] = core.AUTO_HEADER_PAPER_TYPE
    final_answers["cover_note"] = core.AUTO_HEADER_COVER_NOTE
    final_answers.setdefault("tema_principal", context.tema)
    final_answers.setdefault("recorte_empirico", context.recorte)
    final_answers.setdefault("objetivo_geral", context.objetivo)
    if context.pergunta_pesquisa:
        final_answers.setdefault("pergunta_de_pesquisa", context.pergunta_pesquisa)
    if context.hipotese:
        final_answers.setdefault("hipotese", context.hipotese)
    template_text_final = core.materialize_template(template_raw, template_fields, final_answers)
    template_text_final = core.apply_citation_style(template_text_final, bib_filename, style)

    org_text = core.generate_documento_org(client, model, template_text_final, context, base_docs, guidance_docs, correlated_docs, bib_filename, style, prompt_log)
    front_matter = core.infer_final_front_matter(client, model, context, org_text, prompt_log)
    org_text = core.apply_final_front_matter(
        org_text,
        title=front_matter.title.strip(),
        author=final_answers.get("author", core.DEFAULT_AUTHOR) or core.DEFAULT_AUTHOR,
        documento_type=front_matter.documento_type.strip(),
        cover_note=front_matter.cover_note.strip(),
        course_name=final_answers.get("course_name", ""),
        institution_name=final_answers.get("institution_name", core.DEFAULT_INSTITUTION),
    )

    bib_path = output_dir / bib_filename
    org_path = output_dir / f"{basename}.org"
    json_path = output_dir / f"{basename}_contexto.json"
    prompt_audit_path = output_dir / f"{basename}_prompts_auditoria.txt"
    provenance_path = output_dir / f"{basename}_proveniencia.json"
    entrega_dir = output_dir / "entrega_final"
    entrega_dir.mkdir(parents=True, exist_ok=True)

    core.write_text(bib_path, "\n\n".join(bib_entries).strip() + "\n")
    core.write_text(org_path, org_text)
    core.write_text(json_path, json.dumps({
        "generated_at": datetime.now().isoformat(),
        "template": str(template_path),
        "citation_style": style,
        "template_field_answers": final_answers,
        "context": asdict(context),
        "base_context": base_context.model_dump(),
        "narrowed_context": narrowed_context.model_dump(),
        "base_docs": [asdict(d) for d in base_docs],
        "guidance_docs": [asdict(d) for d in guidance_docs],
        "correlated_docs": [asdict(d) for d in correlated_docs],
        "related_info": related_info,
        "imported_bib_files": imported_bib_files,
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "tipo_documento": doc_type,
    }, ensure_ascii=False, indent=2))
    if args.salvar_prompts:
        core.write_prompt_audit(prompt_audit_path, prompt_log)

    pdf_path = None
    if exportar_pdf:
        pdf_path = core.run_compile_sequence(org_path, emacs_init=emacs_init, academic_writing=academic_writing, latex_extra_path=latex_extra_path)

    # provenance + final package
    provenance = {
        "generated_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "bundle_dir": str(bundle_dir) if bundle_dir else "",
        "tipo_documento": doc_type,
        "base_docs_used": [d.label for d in base_docs],
        "guidance_docs_used": [d.label for d in guidance_docs],
        "correlated_docs_used": [d.label for d in correlated_docs],
        "imported_bib_files": imported_bib_files,
        "template": str(template_path),
    }
    core.write_text(provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2))
    for p in [org_path, bib_path, json_path, provenance_path]:
        shutil.copy2(p, entrega_dir / p.name)
    if pdf_path:
        shutil.copy2(pdf_path, entrega_dir / pdf_path.name)
    if args.salvar_prompts:
        shutil.copy2(prompt_audit_path, entrega_dir / prompt_audit_path.name)

    print("\nArquivos gerados:")
    print(f"- Org: {org_path}")
    print(f"- Bib: {bib_path}")
    print(f"- Contexto JSON: {json_path}")
    print(f"- Proveniência: {provenance_path}")
    if args.salvar_prompts:
        print(f"- Auditoria de prompts: {prompt_audit_path}")
    if pdf_path:
        print(f"- PDF: {pdf_path}")
    print(f"- Entrega final: {entrega_dir}")

    save_state({
        "last_template": str(template_path),
        "last_output_dir": str(output_root_dir),
        "last_project_dir": str(output_dir),
        "last_basename": basename,
        "citation_style": style,
        "last_export_pdf": bool(exportar_pdf),
        "last_emacs_init": str(emacs_init) if emacs_init else "",
        "last_academic_writing": str(academic_writing) if academic_writing else "",
        "last_latex_extra_path": str(latex_extra_path) if latex_extra_path else "",
        "last_bundle_dir": str(bundle_dir) if bundle_dir else "",
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        if core.DEBUG:
            traceback.print_exc()
        raise SystemExit(1)
