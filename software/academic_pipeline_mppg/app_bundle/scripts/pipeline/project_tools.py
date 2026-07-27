#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ferramentas de usabilidade para academic_pipeline rc10.7.

Inclui:
- criação de pasta de projeto com TOML e templates de entrada;
- geração de doi_manifest.csv a partir de ZIP ou diretório;
- inspeção bibliográfica de arquivos .bib.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .bibliography_manager import (
        bib_entry_key,
        deduplicate_entries,
        entry_identity,
        extract_field,
        normalize_doi,
        split_bib_entries,
    )
else:
    from bibliography_manager import (
        bib_entry_key,
        deduplicate_entries,
        entry_identity,
        extract_field,
        normalize_doi,
        split_bib_entries,
    )
# Compatibilidade temporária entre pacote oficial e execução direta.
if __package__:
    from .utils import normalize_title_loose, slugify, write_json, write_text
else:
    from utils import normalize_title_loose, slugify, write_json, write_text

SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".org", ".rst", ".tex"}


@dataclass
class InitProjectResult:
    project_dir: Path
    config_path: Path
    doi_manifest_path: Path
    documentos_zip_path: Path
    orientacoes_zip_path: Path
    readme_path: Path


def _packaged_app_bundle() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if candidate.name == "app_bundle":
            return candidate
    raise RuntimeError("Não consegui localizar o app_bundle instalado.")


def _find_app_bundle(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if candidate.name == "app_bundle":
            return candidate
        if (candidate / "app_bundle").exists():
            return (candidate / "app_bundle").resolve()
    return _packaged_app_bundle()


def _destination_app_bundle(base_dir: Path | None, resource_app_bundle: Path) -> Path:
    if base_dir is None:
        return resource_app_bundle
    base = Path(base_dir).expanduser().resolve()
    if base.name == "app_bundle":
        return base
    if (base / "app_bundle").is_dir():
        return (base / "app_bundle").resolve()
    return base / "app_bundle"


def _write_placeholder_zip(path: Path, readme_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme_text)


def _template_config(app_bundle: Path, project_slug: str, project_type: str, institution: str = "fgv") -> str:
    examples_dir = app_bundle / "config" / "examples"
    if project_type == "atividade":
        source = examples_dir / "atividade_exemplo.toml"
    elif project_type in {"prisma", "atividade_prisma", "paper_prisma"}:
        source = examples_dir / "relatorio_prisma_exemplo.toml"
    else:
        source = examples_dir / "paper_exemplo.toml"
    if not source.exists():
        raise FileNotFoundError(f"Template TOML não encontrado: {source}")
    text = source.read_text(encoding="utf-8")
    # Ajusta exemplos antigos para o novo projeto. Como o TOML ficará dentro de
    # app_bundle/projetos/<slug>, os caminhos relativos corretos para templates/output
    # continuam sendo ../../templates, ../../output etc.; já os insumos ficam no diretório atual.
    replacements = {
        "paper_nome_do_tema": project_slug,
        "atividade_aula_2": project_slug,
        "../../projetos/" + project_slug + "/documentos-base.zip": "documentos-base.zip",
        "../../projetos/" + project_slug + "/orientacoes.zip": "orientacoes.zip",
        "../../projetos/" + project_slug + "/doi_manifest.csv": "doi_manifest.csv",
    }
    # Substituições genéricas para quando o template fonte ainda contém os nomes originais.
    text = text.replace("../../projetos/paper_nome_do_tema/documentos-base.zip", "documentos-base.zip")
    text = text.replace("../../projetos/paper_nome_do_tema/orientacoes.zip", "orientacoes.zip")
    text = text.replace("../../projetos/paper_nome_do_tema/doi_manifest.csv", "doi_manifest.csv")
    text = text.replace("../../projetos/atividade_aula_2/documentos-base.zip", "documentos-base.zip")
    text = text.replace("../../projetos/atividade_aula_2/orientacoes.zip", "orientacoes.zip")
    text = text.replace("../../projetos/atividade_aula_2/doi_manifest.csv", "doi_manifest.csv")
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "[instituicao]" not in text:
        text = f"[instituicao]\nperfil = \"{institution}\"\n\n" + text
    else:
        text = re.sub(r"(?ms)^\[instituicao\].*?(?=^\[|\Z)", f"[instituicao]\nperfil = \"{institution}\"\n\n", text)

    # Ajusta prefixos mais comuns.
    text = re.sub(r'prefixo\s*=\s*"(?:paper_nome_do_tema|atividade_aula_2|relatorio_prisma_atividade_aula_2)"', f'prefixo = "{project_slug}"', text)
    return text


def init_project(name: str, project_type: str = "paper", base_dir: Path | None = None, overwrite: bool = False, institution: str = "fgv") -> InitProjectResult:
    if not name or not name.strip():
        raise ValueError("Informe um nome de projeto.")
    project_slug = slugify(name)
    resource_app_bundle = _find_app_bundle()
    app_bundle = _destination_app_bundle(base_dir, resource_app_bundle)
    project_dir = app_bundle / "projetos" / project_slug
    if project_dir.exists() and any(project_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Projeto já existe: {project_dir}. Use --overwrite-project para sobrescrever arquivos seguros.")
    project_dir.mkdir(parents=True, exist_ok=True)

    documentos_zip = project_dir / "documentos-base.zip"
    orientacoes_zip = project_dir / "orientacoes.zip"
    doi_manifest = project_dir / "doi_manifest.csv"
    config_path = project_dir / "paper_config.toml"
    readme_path = project_dir / "README_PROJETO.md"

    _write_placeholder_zip(documentos_zip, "Substitua este ZIP pelos PDFs/DOCX/TXT/ORG que compõem o corpus do projeto.\n")
    _write_placeholder_zip(orientacoes_zip, "Substitua este ZIP pelo roteiro, rubrica, enunciado e orientações do professor.\n")
    if not doi_manifest.exists() or overwrite:
        write_text(doi_manifest, "arquivo,doi\n")
    if not config_path.exists() or overwrite:
        write_text(config_path, _template_config(resource_app_bundle, project_slug, project_type, institution=institution))
    if not readme_path.exists() or overwrite:
        write_text(readme_path, f"""# Projeto {project_slug}

Estrutura criada pelo `academic_pipeline rc10.7` com perfil institucional.

## Perfil institucional

Este projeto foi criado com o perfil institucional `{institution}`.

## Arquivos de entrada

- `documentos-base.zip`: substitua pelo ZIP com PDFs/DOCX/TXT/ORG do corpus.
- `orientacoes.zip`: substitua pelo ZIP com roteiro, enunciado, rubrica ou orientações.
- `doi_manifest.csv`: preencha os DOIs conhecidos no formato `arquivo,doi`.
- `paper_config.toml`: ajuste tema, disciplina, professor, saída e estilo bibliográfico.

## Fluxo recomendado

```bash
cd /caminho/para/app_bundle/projetos/{project_slug}

academic-pipeline \\
  --config paper_config.toml \\
  --check-config

academic-pipeline \\
  --config paper_config.toml
```

## Gerar DOI manifest a partir do ZIP real

```bash
academic-pipeline \\
  --make-doi-manifest \\
  --input-zip documentos-base.zip \\
  --output doi_manifest.csv
```
""")
    return InitProjectResult(project_dir, config_path, doi_manifest, documentos_zip, orientacoes_zip, readme_path)


def _iter_zip_sources(input_zip: Path) -> list[str]:
    if not input_zip.exists():
        raise FileNotFoundError(f"ZIP não encontrado: {input_zip}")
    names: list[str] = []
    with zipfile.ZipFile(input_zip, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix in SUPPORTED_SOURCE_SUFFIXES:
                names.append(name)
    return sorted(dict.fromkeys(names))


def _iter_dir_sources(input_dir: Path, recursive: bool = True) -> list[str]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    out = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES:
            try:
                out.append(str(path.relative_to(input_dir)))
            except Exception:
                out.append(path.name)
    return sorted(dict.fromkeys(out))


def make_doi_manifest(input_zip: Path | None, input_dir: Path | None, output: Path, overwrite: bool = True) -> dict[str, Any]:
    if not input_zip and not input_dir:
        raise ValueError("Informe --input-zip ou --input-dir.")
    if input_zip:
        files = _iter_zip_sources(input_zip)
        source = str(input_zip)
    else:
        files = _iter_dir_sources(input_dir)  # type: ignore[arg-type]
        source = str(input_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Arquivo já existe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", "doi"])
        for name in files:
            writer.writerow([name, ""])
    return {"source": source, "output": str(output), "total_files": len(files), "files": files}


DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", re.I)
HTML_RE = re.compile(r"<[^>]+>")


def inspect_bib(path: Path, output_prefix: Path | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f".bib não encontrado: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries = split_bib_entries(text)
    by_identity: dict[str, list[str]] = {}
    issues: list[dict[str, Any]] = []
    keys: list[str] = []
    for entry in entries:
        key = bib_entry_key(entry) or ""
        if key:
            keys.append(key)
        identity = entry_identity(entry)
        by_identity.setdefault(identity, []).append(key or "<sem-chave>")
        entry_head = re.match(r"\s*@([^\{]+)\{", entry)
        entry_type = (entry_head.group(1).lower() if entry_head else "")
        title = extract_field(entry, "title")
        doi = normalize_doi(extract_field(entry, "doi"))
        author = extract_field(entry, "author") or extract_field(entry, "editor")
        year = extract_field(entry, "year")
        source_field = extract_field(entry, "journaltitle") or extract_field(entry, "journal") or extract_field(entry, "booktitle") or extract_field(entry, "publisher")
        pages = extract_field(entry, "pages")
        note = extract_field(entry, "note")
        entry_issues: list[str] = []
        if not key:
            entry_issues.append("entrada_sem_chave")
        if not title:
            entry_issues.append("titulo_ausente")
        if title and HTML_RE.search(title):
            entry_issues.append("titulo_com_html_xml")
        if not author:
            entry_issues.append("autor_ou_editor_ausente")
        if not year:
            entry_issues.append("ano_ausente")
        if doi and not DOI_RE.match(doi):
            entry_issues.append("doi_malformado")
        if entry_type == "article" and not source_field:
            entry_issues.append("artigo_sem_periodico")
        if entry_type in {"article", "incollection", "inbook"} and not pages:
            entry_issues.append("paginas_ausentes")
        if entry_type in {"", "misc"}:
            entry_issues.append("tipo_misc_ou_ausente")
        low = (note + " " + author).lower()
        if "metadados" in low or "material fornecido" in low or "fornecido pelo professor" in low:
            entry_issues.append("nota_ou_autor_generico")
        if entry_issues:
            issues.append({"key": key or None, "type": entry_type or None, "title": title or None, "issues": entry_issues})
    duplicates = {ident: ks for ident, ks in by_identity.items() if len([k for k in ks if k]) > 1}
    report = {
        "bib_path": str(path),
        "entries_total": len(entries),
        "keys_total": len(keys),
        "duplicate_groups_total": len(duplicates),
        "duplicate_groups": duplicates,
        "issues_total": len(issues),
        "issues": issues,
        "ok": not duplicates and not issues,
    }
    if output_prefix:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        write_json(Path(str(output_prefix) + ".json"), report)
        write_text(Path(str(output_prefix) + ".md"), render_bib_inspection_markdown(report))
    return report


def render_bib_inspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Inspeção bibliográfica",
        "",
        f"- Arquivo: `{report.get('bib_path')}`",
        f"- Entradas: {report.get('entries_total', 0)}",
        f"- Chaves: {report.get('keys_total', 0)}",
        f"- Grupos duplicados prováveis: {report.get('duplicate_groups_total', 0)}",
        f"- Entradas com alertas: {report.get('issues_total', 0)}",
        f"- Status: {'OK' if report.get('ok') else 'REVISAR'}",
        "",
    ]
    duplicates = report.get("duplicate_groups") or {}
    if duplicates:
        lines += ["## Duplicatas prováveis", ""]
        for ident, keys in duplicates.items():
            lines.append(f"- `{ident}`: {', '.join('`' + str(k) + '`' for k in keys)}")
        lines.append("")
    issues = report.get("issues") or []
    if issues:
        lines += ["## Alertas por entrada", ""]
        for item in issues:
            key = item.get("key") or "<sem-chave>"
            title = item.get("title") or "<sem título>"
            lines.append(f"- `{key}` — {title}: {', '.join(item.get('issues') or [])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
