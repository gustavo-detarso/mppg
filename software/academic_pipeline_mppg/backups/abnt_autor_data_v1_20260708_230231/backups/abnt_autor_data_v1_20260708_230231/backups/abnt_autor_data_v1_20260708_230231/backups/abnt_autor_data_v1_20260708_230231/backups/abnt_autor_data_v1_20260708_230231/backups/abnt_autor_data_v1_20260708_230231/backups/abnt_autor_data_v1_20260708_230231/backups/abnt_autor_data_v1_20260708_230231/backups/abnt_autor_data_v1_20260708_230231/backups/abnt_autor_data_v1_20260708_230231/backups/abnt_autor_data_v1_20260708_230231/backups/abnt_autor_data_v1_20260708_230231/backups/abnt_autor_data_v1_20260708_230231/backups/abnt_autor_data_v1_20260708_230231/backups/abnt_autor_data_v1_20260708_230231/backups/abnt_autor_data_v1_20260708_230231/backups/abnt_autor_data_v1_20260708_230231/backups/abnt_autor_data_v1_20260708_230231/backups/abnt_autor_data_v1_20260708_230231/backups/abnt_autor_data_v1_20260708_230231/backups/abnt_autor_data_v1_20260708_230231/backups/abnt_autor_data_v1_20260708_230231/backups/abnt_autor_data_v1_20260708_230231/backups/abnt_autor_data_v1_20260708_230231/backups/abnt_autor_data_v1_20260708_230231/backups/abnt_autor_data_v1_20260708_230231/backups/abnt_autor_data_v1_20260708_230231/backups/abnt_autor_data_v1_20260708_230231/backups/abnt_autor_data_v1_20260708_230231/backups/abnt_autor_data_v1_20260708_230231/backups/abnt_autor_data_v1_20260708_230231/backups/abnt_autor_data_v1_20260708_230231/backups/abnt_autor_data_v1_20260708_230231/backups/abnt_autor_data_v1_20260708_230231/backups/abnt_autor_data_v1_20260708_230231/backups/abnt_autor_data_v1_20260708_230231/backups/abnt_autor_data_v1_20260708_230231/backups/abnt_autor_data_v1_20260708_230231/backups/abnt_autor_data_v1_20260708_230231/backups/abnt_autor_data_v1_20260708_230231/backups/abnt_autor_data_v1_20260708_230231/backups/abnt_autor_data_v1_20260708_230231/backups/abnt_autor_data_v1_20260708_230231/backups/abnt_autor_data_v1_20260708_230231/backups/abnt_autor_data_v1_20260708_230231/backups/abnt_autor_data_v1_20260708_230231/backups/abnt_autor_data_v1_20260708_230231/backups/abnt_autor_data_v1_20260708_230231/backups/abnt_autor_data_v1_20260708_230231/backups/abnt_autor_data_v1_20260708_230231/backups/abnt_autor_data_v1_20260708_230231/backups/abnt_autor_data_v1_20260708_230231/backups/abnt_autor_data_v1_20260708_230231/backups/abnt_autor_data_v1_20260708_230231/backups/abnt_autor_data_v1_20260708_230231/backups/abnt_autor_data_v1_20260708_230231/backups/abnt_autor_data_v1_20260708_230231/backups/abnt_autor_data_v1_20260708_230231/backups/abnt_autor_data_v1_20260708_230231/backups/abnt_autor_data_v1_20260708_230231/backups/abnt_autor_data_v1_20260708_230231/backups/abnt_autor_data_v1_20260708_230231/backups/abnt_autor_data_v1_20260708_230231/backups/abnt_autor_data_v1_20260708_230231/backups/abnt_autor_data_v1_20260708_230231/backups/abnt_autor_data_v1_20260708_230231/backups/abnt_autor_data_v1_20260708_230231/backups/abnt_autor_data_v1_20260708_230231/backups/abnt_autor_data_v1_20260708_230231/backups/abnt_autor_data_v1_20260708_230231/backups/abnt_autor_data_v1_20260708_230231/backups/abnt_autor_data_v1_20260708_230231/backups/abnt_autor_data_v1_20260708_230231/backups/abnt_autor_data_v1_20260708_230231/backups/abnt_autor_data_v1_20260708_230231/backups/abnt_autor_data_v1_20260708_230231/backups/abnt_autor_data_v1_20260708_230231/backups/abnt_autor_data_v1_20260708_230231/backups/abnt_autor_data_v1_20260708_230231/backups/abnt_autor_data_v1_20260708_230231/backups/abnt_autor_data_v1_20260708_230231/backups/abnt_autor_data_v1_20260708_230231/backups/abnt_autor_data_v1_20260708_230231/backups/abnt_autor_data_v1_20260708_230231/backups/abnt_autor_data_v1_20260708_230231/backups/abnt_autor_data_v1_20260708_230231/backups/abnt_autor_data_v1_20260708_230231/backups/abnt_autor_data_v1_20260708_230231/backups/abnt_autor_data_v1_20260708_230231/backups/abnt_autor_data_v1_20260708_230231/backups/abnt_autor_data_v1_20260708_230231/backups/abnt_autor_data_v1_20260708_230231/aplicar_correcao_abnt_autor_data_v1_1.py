#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r'''
aplicar_correcao_abnt_autor_data_v1_1.py

Correção consolidada para o academic_pipeline_rc10_7_conformidade:

1. Corrige render_org_latex.py para não gerar BibLaTeX sem style.
2. Garante que latex_style="abnt" seja tratado explicitamente como ABNT autor-data.
3. Injeta style=abnt nas opções do \usepackage{biblatex}.
4. Corrige casos literais de \usepackage{biblatex} sem opções em fontes Python/TEX/ORG geradas.
5. Ajusta TOMLs principais para explicitar ABNT autor-data.
6. Limpa artefatos antigos de PDF/TEX/Biber/BibLaTeX no output PRISMA.
7. Opcionalmente roda check-config e regeneração PRISMA com --run-prisma.

Uso típico:

cd ~/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade
python3 ~/Downloads/aplicar_correcao_abnt_autor_data_v1_1.py

Para também rerodar o pipeline PRISMA:

python3 ~/Downloads/aplicar_correcao_abnt_autor_data_v1_1.py --run-prisma
'''

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PATCH_ID = "ABNT_AUTOR_DATA_V1"


def agora() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def erro(msg: str) -> None:
    print(f"[ERRO] {msg}")


def raiz_projeto() -> Path:
    root = Path.cwd().resolve()
    esperado = root / "app_bundle" / "scripts" / "pipeline" / "render_org_latex.py"
    if not esperado.exists():
        erro("Execute este script a partir da raiz do projeto academic_pipeline_rc10_7_conformidade.")
        erro(f"Arquivo esperado não encontrado: {esperado}")
        sys.exit(2)
    return root


def backup_file(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    rel = path.relative_to(Path.cwd())
    destino = backup_root / rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destino)


def write_if_changed(path: Path, new_text: str, backup_root: Path) -> bool:
    old = path.read_text(encoding="utf-8")
    if old == new_text:
        return False
    backup_file(path, backup_root)
    path.write_text(new_text, encoding="utf-8")
    return True


def replace_function(text: str, func_name: str, replacement: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"^def {re.escape(func_name)}\([^\n]*\).*?\n(?=^def |\Z|^class )",
        re.M | re.S,
    )
    new_text, n = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    return new_text, n > 0


def normalized_function_source() -> str:
    return "\n".join([
        "def normalize_biblatex_style(style: str | None) -> str:",
        "    # Normaliza estilos BibLaTeX usados pelo pipeline.",
        "    #",
        "    # Correção ABNT_AUTOR_DATA_V1:",
        "    # - \"abnt\" deve permanecer \"abnt\", pois no biblatex-abnt esse é o estilo",
        "    #   autor-data ABNT esperado.",
        "    # - entradas ambíguas como \"autor-data\" e \"abnt_autor_data\" passam a cair",
        "    #   em \"abnt\", evitando saída numérica [1].",
        "    # - \"numeric\"/\"num\" continuam possíveis, mas só quando explicitamente usados.",
        "    raw = (style or \"\").strip().lower()",
        "    raw = raw.replace(\"_\", \"-\").replace(\" \", \"-\")",
        "",
        "    aliases = {",
        "        \"\": \"apa\",",
        "        \"abnt\": \"abnt\",",
        "        \"abnt-autor-data\": \"abnt\",",
        "        \"abntauthoryear\": \"abnt\",",
        "        \"abnt-author-year\": \"abnt\",",
        "        \"autor-data\": \"abnt\",",
        "        \"author-date\": \"abnt\",",
        "        \"authordate\": \"abnt\",",
        "        \"authoryear\": \"authoryear\",",
        "        \"autor-data-generico\": \"authoryear\",",
        "        \"apa\": \"apa\",",
        "        \"apa7\": \"apa\",",
        "        \"ieee\": \"ieee\",",
        "        \"numeric\": \"numeric\",",
        "        \"num\": \"numeric\",",
        "        \"numerico\": \"numeric\",",
        "        \"numérico\": \"numeric\",",
        "        \"vancouver\": \"numeric\",",
        "    }",
        "    return aliases.get(raw, raw or \"apa\")",
        "",
    ])


def patch_render_org_latex(path: Path, backup_root: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    text2, replaced = replace_function(text, "normalize_biblatex_style", normalized_function_source())
    if replaced:
        text = text2
    else:
        warn(f"Não consegui substituir normalize_biblatex_style em {path}. Vou aplicar apenas injeções pontuais.")

    marker = f"# {PATCH_ID}: saneamento defensivo das opções BibLaTeX"
    header_pattern = r'(\n\s*)(f?"#\+LATEX_HEADER:\s*\\+usepackage\[\{options\}\]\{\{biblatex\}\}",)'
    if marker not in text:
        m = re.search(header_pattern, text)
        if m:
            base_indent = m.group(1)
            code = f'''{base_indent}# {PATCH_ID}: saneamento defensivo das opções BibLaTeX
{base_indent}try:
{base_indent}    _style_normalizado = normalize_biblatex_style(style)
{base_indent}except Exception:
{base_indent}    _style_normalizado = str(style or "").strip().lower()
{base_indent}_opts_parts = [p.strip() for p in str(options).split(",") if p.strip()]
{base_indent}_opts_parts = [
{base_indent}    p for p in _opts_parts
{base_indent}    if not p.startswith("style=") and not p.startswith("citestyle=")
{base_indent}]
{base_indent}if not any(p.startswith("backend=") for p in _opts_parts):
{base_indent}    _opts_parts.insert(0, "backend=biber")
{base_indent}if _style_normalizado == "abnt":
{base_indent}    _opts_parts.insert(0, "style=abnt")
{base_indent}elif _style_normalizado == "authoryear":
{base_indent}    _opts_parts.insert(0, "citestyle=authoryear")
{base_indent}    _opts_parts.insert(0, "style=authoryear")
{base_indent}elif _style_normalizado:
{base_indent}    _opts_parts.insert(0, f"style={{_style_normalizado}}")
{base_indent}options = ",".join(_opts_parts)
'''
            text = text[:m.start()] + code + text[m.start():]
        else:
            warn("Não encontrei a linha padrão do LATEX_HEADER biblatex para injetar saneamento.")

    text = text.replace(
        r"\usepackage{biblatex}",
        r"\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
    )
    text = text.replace(
        r"\usepackage[backend=biber,sorting=nty,giveninits=true]{biblatex}",
        r"\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
    )

    changed = write_if_changed(path, text, backup_root)
    if changed:
        ok(f"Corrigido: {path}")
    else:
        ok(f"Sem alterações necessárias: {path}")
    return changed or (text != original)


def patch_biblatex_literals(root: Path, backup_root: Path) -> int:
    ex_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".patch_backups",
        "execucoes_anteriores",
    }
    ex_suffixes = {
        ".py",
        ".tex",
        ".org",
        ".sty",
        ".toml",
        ".md",
    }

    changed_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ex_dirs for part in path.parts):
            continue
        if path.suffix.lower() not in ex_suffixes:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        new = text

        new = new.replace(
            r"\usepackage{biblatex}",
            r"\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
        )
        new = new.replace(
            r"\usepackage[backend=biber,sorting=nty,giveninits=true]{biblatex}",
            r"\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
        )

        new = new.replace(
            r"\\usepackage{biblatex}",
            r"\\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
        )
        new = new.replace(
            r"\\usepackage[backend=biber,sorting=nty,giveninits=true]{biblatex}",
            r"\\usepackage[backend=biber,style=abnt,sorting=nty,giveninits=true]{biblatex}",
        )

        if new != text:
            write_if_changed(path, new, backup_root)
            changed_count += 1
            ok(f"BibLaTeX literal corrigido: {path.relative_to(root)}")

    return changed_count


def ensure_key_in_section(text: str, section: str, key: str, value: str) -> str:
    section_re = re.compile(rf"(?ms)^(\[{re.escape(section)}\]\s*)(.*?)(?=^\[|\Z)")
    m = section_re.search(text)
    line = f'{key} = "{value}"'

    if not m:
        return text.rstrip() + f"\n\n[{section}]\n{line}\n"

    header, body = m.group(1), m.group(2)
    key_re = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=\s*.*$")
    if key_re.search(body):
        body2 = key_re.sub(line, body)
    else:
        body2 = body.rstrip() + "\n" + line + "\n"

    return text[:m.start()] + header + body2 + text[m.end():]


def patch_toml(path: Path, backup_root: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    new = text

    new = ensure_key_in_section(new, "bibliografia", "estilo_citacao", "abnt")
    new = ensure_key_in_section(new, "bibliografia", "latex_style", "abnt")
    new = ensure_key_in_section(new, "bibliografia", "sistema_citacao", "autor-data")
    new = ensure_key_in_section(new, "bibliografia", "backend", "biber")

    if re.search(r"(?m)^\[documento\]\s*$", new):
        new = ensure_key_in_section(new, "documento", "estilo_citacao", "abnt")
        new = ensure_key_in_section(new, "documento", "sistema_citacao", "autor-data")

    changed = write_if_changed(path, new, backup_root)
    if changed:
        ok(f"TOML ajustado: {path}")
    else:
        ok(f"TOML já compatível: {path}")
    return changed


def patch_tomls(root: Path, backup_root: Path) -> int:
    alvos = [
        root / "app_bundle" / "institutions" / "fgv" / "institution_profile.toml",
        root / "app_bundle" / "projetos" / "prisma_fluxo_pmf" / "prisma_fluxo_pmf.toml",
    ]

    count = 0
    for path in alvos:
        if patch_toml(path, backup_root):
            count += 1
    return count


def clean_prisma_outputs(root: Path) -> int:
    out = root / "app_bundle" / "projetos" / "prisma_fluxo_pmf" / "output_pesquisa" / "relatorio_prisma_prisma_fluxo_pmf"
    if not out.exists():
        warn(f"Output PRISMA não encontrado para limpeza: {out}")
        return 0

    suffixes = {
        ".pdf",
        ".tex",
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".run.xml",
        ".log",
        ".out",
        ".toc",
        ".lof",
        ".lot",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
    }

    removed = 0
    for path in out.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(name.endswith(s) for s in suffixes):
            try:
                path.unlink()
                removed += 1
                info(f"Removido artefato antigo: {path.relative_to(root)}")
            except OSError as exc:
                warn(f"Não consegui remover {path}: {exc}")

    ok(f"Artefatos antigos removidos do PRISMA: {removed}")
    return removed


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print()
    info("Executando: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode == 0:
        ok("Comando concluído.")
    else:
        erro(f"Comando retornou código {proc.returncode}.")
    return proc.returncode


def validate(root: Path) -> bool:
    render = root / "app_bundle" / "scripts" / "pipeline" / "render_org_latex.py"
    text = render.read_text(encoding="utf-8")

    checks = {
        "normalize_abnt": '"abnt": "abnt"' in text,
        "saneamento_marker": PATCH_ID in text,
        "style_abnt_literal": "style=abnt" in text,
    }

    print()
    info("Validação da correção:")
    for k, v in checks.items():
        print(f"  - {k}: {'OK' if v else 'FALHOU'}")

    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-prisma",
        action="store_true",
        help="Depois de aplicar o patch, roda check-config e regenera o perfil prisma_fluxo_pmf.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Não remove artefatos antigos do output PRISMA.",
    )
    args = parser.parse_args()

    root = raiz_projeto()
    os.chdir(root)

    backup_root = root / "backups" / f"abnt_autor_data_v1_{agora()}"
    backup_root.mkdir(parents=True, exist_ok=True)

    info(f"Raiz do projeto: {root}")
    info(f"Backup em: {backup_root}")

    render = root / "app_bundle" / "scripts" / "pipeline" / "render_org_latex.py"

    patch_render_org_latex(render, backup_root)
    n_lit = patch_biblatex_literals(root, backup_root)
    n_toml = patch_tomls(root, backup_root)

    if not args.no_clean:
        clean_prisma_outputs(root)

    valid = validate(root)

    print()
    info("Resumo:")
    print(f"  - BibLaTeX literal corrigido em arquivos: {n_lit}")
    print(f"  - TOMLs ajustados: {n_toml}")
    print(f"  - Validação interna: {'OK' if valid else 'FALHOU'}")
    print(f"  - Backup: {backup_root}")

    if not valid:
        erro("A validação interna falhou. Revise render_org_latex.py antes de regenerar.")
        return 1

    if args.run_prisma:
        cfg = "app_bundle/projetos/prisma_fluxo_pmf/prisma_fluxo_pmf.toml"
        pipe = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"

        rc = run_cmd(["pipenv", "run", "python", pipe, "--config", cfg, "--check-config"], root)
        if rc != 0:
            return rc

        rc = run_cmd(["pipenv", "run", "python", pipe, "--config", cfg], root)
        if rc != 0:
            return rc

        out = root / "app_bundle" / "projetos" / "prisma_fluxo_pmf" / "output_pesquisa" / "relatorio_prisma_prisma_fluxo_pmf"
        print()
        info("Conferência rápida dos arquivos gerados:")
        for path in sorted(out.glob("*")):
            if path.suffix.lower() in {".tex", ".org", ".pdf", ".bib"}:
                print(f"  - {path.relative_to(root)}")

        texs = list(out.glob("*.tex"))
        if texs:
            print()
            info("Linhas BibLaTeX encontradas no TEX:")
            for tex in texs:
                for i, line in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    if "usepackage" in line and "biblatex" in line:
                        print(f"{tex.name}:{i}: {line}")
        else:
            warn("Nenhum .tex encontrado após a execução.")

    else:
        print()
        info("Patch aplicado. Para regenerar o PRISMA agora, rode:")
        print()
        print('CFG="app_bundle/projetos/prisma_fluxo_pmf/prisma_fluxo_pmf.toml"')
        print('PIPE="app_bundle/scripts/pipeline/academic_pipeline_rc10.py"')
        print('pipenv run python "$PIPE" --config "$CFG" --check-config')
        print('pipenv run python "$PIPE" --config "$CFG"')
        print()
        info("Depois confira:")
        print('OUT="app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf"')
        print('grep -RIn "usepackage.*biblatex\\|style=abnt\\|style=authoryear" "$OUT"/*.tex "$OUT"/*.org 2>/dev/null')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
