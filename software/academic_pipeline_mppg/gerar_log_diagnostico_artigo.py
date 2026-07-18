#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import os
import re
import shutil
import subprocess
import tarfile
import textwrap


def run(cmd, cwd=None, env=None, timeout=300):
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout
    except Exception as e:
        return 999, f"[ERRO AO EXECUTAR] {cmd}\n{type(e).__name__}: {e}\n"


def read(path: Path, max_chars=None):
    if not path.exists():
        return f"[AUSENTE] {path}\n"
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(txt) > max_chars:
        return txt[:max_chars] + f"\n\n[TRUNCADO em {max_chars} caracteres]\n"
    return txt


def section(log, title):
    log.append("\n" + "=" * 100)
    log.append(title)
    log.append("=" * 100 + "\n")


def grep_lines(txt, pattern, flags=0, context=0):
    lines = txt.splitlines()
    out = []
    rx = re.compile(pattern, flags)
    for i, line in enumerate(lines, start=1):
        if rx.search(line):
            start = max(1, i - context)
            end = min(len(lines), i + context)
            for j in range(start, end + 1):
                out.append(f"{j:05d}: {lines[j-1]}")
            out.append("-" * 80)
    return "\n".join(out) if out else "[sem ocorrências]\n"


def tex_window(tex_txt, center, radius=20):
    lines = tex_txt.splitlines()
    start = max(1, center - radius)
    end = min(len(lines), center + radius)
    return "\n".join(f"{i:05d}: {lines[i-1]}" for i in range(start, end + 1))


def parse_bib_keys(bib_txt):
    return re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)", bib_txt)


def parse_tex_cite_keys(tex_txt):
    keys = []
    for m in re.finditer(r"\\(?:parencite|textcite|autocite|cite|citep|citet)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", tex_txt):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.append(k)
    return keys


def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--prefix", default="artigo_final_atestmed_abnt")
    args = ap.parse_args()

    art = Path(args.art_dir).resolve()
    out = art / "output"
    prefix = args.prefix

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag = out / f"diagnostico_latex_biber_{stamp}"
    build = diag / "build"

    diag.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)

    org = out / f"{prefix}.org"
    tex = out / f"{prefix}.tex"
    bib = out / f"{prefix}.bib"
    pdf = out / f"{prefix}.pdf"
    bbl = out / f"{prefix}.bbl"
    bcf = out / f"{prefix}.bcf"
    blg = out / f"{prefix}.blg"
    latex_log = out / f"{prefix}.log"

    for p in [org, tex, bib, pdf, bbl, bcf, blg, latex_log, out / f"{prefix}_export_pdf.el"]:
        copy_if_exists(p, diag)

    log = []
    section(log, "DIAGNÓSTICO ARTIGO — LATEX / BIBER / ABNT")
    log.append(f"Data/hora: {datetime.now().isoformat(timespec='seconds')}")
    log.append(f"ART: {art}")
    log.append(f"OUT: {out}")
    log.append(f"PREFIX: {prefix}")

    section(log, "VERSÕES E AMBIENTE")
    for cmd in [
        ["uname", "-a"],
        ["which", "pdflatex"],
        ["pdflatex", "--version"],
        ["which", "biber"],
        ["biber", "--version"],
        ["which", "emacs"],
        ["emacs", "--version"],
        ["kpsewhich", "abnt.bbx"],
        ["kpsewhich", "abnt.cbx"],
        ["kpsewhich", "biblatex.sty"],
        ["kpsewhich", "fgv-paper.sty"],
    ]:
        code, output = run(cmd, timeout=60)
        log.append(f"\n$ {' '.join(cmd)}\n[exit={code}]\n{output}")

    section(log, "ARQUIVOS EXISTENTES")
    for p in [org, tex, bib, pdf, bbl, bcf, blg, latex_log]:
        if p.exists():
            log.append(f"{p} | {p.stat().st_size} bytes")
        else:
            log.append(f"[AUSENTE] {p}")

    org_txt = read(org)
    tex_txt = read(tex)
    bib_txt = read(bib)
    bbl_txt = read(bbl)
    log_txt = read(latex_log)

    section(log, "BIBTEX — CHAVES DETECTADAS")
    bib_keys = parse_bib_keys(bib_txt)
    log.append(f"Total de entradas BibTeX: {len(bib_keys)}")
    for i, k in enumerate(bib_keys, start=1):
        log.append(f"{i:02d}. {k}")

    section(log, "CITAÇÕES NO TEX — CHAVES CITADAS VS CHAVES NO BIB")
    cited = parse_tex_cite_keys(tex_txt)
    log.append(f"Total de comandos de citação no TEX: {len(cited)}")
    missing = sorted(set(cited) - set(bib_keys))
    unused = sorted(set(bib_keys) - set(cited))
    log.append(f"Chaves citadas ausentes no BIB: {len(missing)}")
    for k in missing:
        log.append(f"  MISSING: {k}")
    log.append(f"Chaves do BIB não citadas diretamente: {len(unused)}")
    for k in unused:
        log.append(f"  UNUSED: {k}")

    section(log, "ORG — HEADERS BIBLATEX / REFERÊNCIAS / CITAÇÕES LITERAIS")
    log.append("Headers biblatex/addbibresource:")
    log.append(grep_lines(org_txt, r"biblatex|addbibresource|printbibliography|bibliography|abntex2cite", re.I, context=1))
    log.append("\nCitações literais em colchetes no ORG:")
    log.append(grep_lines(org_txt, r"\[[0-9]+(?:,\s*[0-9]+)*\]|\[[A-Za-z][A-Za-z0-9_:\-]{15,}\]", context=1))

    section(log, "TEX — PREÂMBULO BIBLATEX / REFERÊNCIAS")
    log.append(grep_lines(tex_txt, r"biblatex|addbibresource|printbibliography|bibliographystyle|bibliography|abntex2cite", re.I, context=2))

    section(log, "TEX — CITAÇÕES LITERAIS REMANESCENTES")
    log.append(grep_lines(tex_txt, r"\[[0-9]+(?:,\s*[0-9]+)*\]|\[[A-Za-z][A-Za-z0-9_:\-]{15,}\]", context=1))

    section(log, "TEX — OCORRÊNCIAS DA CHAVE PROBLEMÁTICA hospital2016")
    log.append(grep_lines(tex_txt, r"hospital2016mentaldisabilityaretrospectivestudyofsocioclinica", context=6))

    section(log, "TEX — JANELA EM TORNO DA LINHA 195")
    if tex.exists():
        log.append(tex_window(tex_txt, 195, radius=35))
    else:
        log.append("[TEX ausente]")

    section(log, "TEX — TABELAS / LONGTABLE PRÓXIMAS")
    log.append(grep_lines(tex_txt, r"\\begin\{(?:longtable|tabular|tabularx)\}|\\end\{(?:longtable|tabular|tabularx)\}|\\hline|\\\\", context=2))

    section(log, "LOG LATEX ATUAL — ERROS E AVISOS RELEVANTES")
    if latex_log.exists():
        log.append(grep_lines(
            log_txt,
            r"! |Fatal error|LaTeX Error|Package .* Warning|Citation .* undefined|Missing \\cr|Misplaced \\cr|alignment tab|Runaway argument|xkeyval Error|Undefined control sequence",
            re.I,
            context=2,
        ))
    else:
        log.append("[LOG ausente]")

    section(log, "BBL ATUAL — PRIMEIROS 30000 CARACTERES")
    log.append(bbl_txt[:30000] if bbl.exists() else "[BBL ausente]")

    section(log, "COMPILAÇÃO ISOLADA EM DIRETÓRIO DE DIAGNÓSTICO")
    for p in [tex, bib]:
        copy_if_exists(p, build)

    env = os.environ.copy()
    env["TEXINPUTS"] = f".:{out}:{diag}:{build}:" + env.get("TEXINPUTS", "")
    env["BIBINPUTS"] = f".:{out}:{diag}:{build}:" + env.get("BIBINPUTS", "")

    compile_cmds = [
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
        ["biber", prefix],
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
        ["pdflatex", "-file-line-error", "-interaction=nonstopmode", f"{prefix}.tex"],
    ]

    for cmd in compile_cmds:
        code, output = run(cmd, cwd=build, env=env, timeout=300)
        log.append(f"\n$ {' '.join(cmd)}\n[exit={code}]\n")
        log.append(output[-60000:])

    section(log, "ARQUIVOS GERADOS NA COMPILAÇÃO ISOLADA")
    for p in sorted(build.glob("*")):
        log.append(f"{p.name} | {p.stat().st_size} bytes")

    isolated_log = build / f"{prefix}.log"
    isolated_blg = build / f"{prefix}.blg"
    isolated_bbl = build / f"{prefix}.bbl"

    section(log, "LOG ISOLADO — ERROS E AVISOS RELEVANTES")
    if isolated_log.exists():
        iso_txt = read(isolated_log)
        log.append(grep_lines(
            iso_txt,
            r"! |Fatal error|LaTeX Error|Package .* Warning|Citation .* undefined|Missing \\cr|Misplaced \\cr|alignment tab|Runaway argument|xkeyval Error|Undefined control sequence",
            re.I,
            context=2,
        ))
    else:
        log.append("[LOG isolado ausente]")

    section(log, "BLG ISOLADO — BIBER")
    log.append(read(isolated_blg, max_chars=60000))

    section(log, "BBL ISOLADO — PRIMEIROS 30000 CARACTERES")
    log.append(read(isolated_bbl, max_chars=30000))

    final_log = diag / "diagnostico_artigo_latex_biber.txt"
    final_log.write_text("\n".join(log), encoding="utf-8")

    tar_path = out / f"diagnostico_latex_biber_{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(diag, arcname=diag.name)

    print("[OK] Diagnóstico gerado:")
    print(final_log)
    print()
    print("[OK] Pacote compactado:")
    print(tar_path)
    print()
    print("Envie preferencialmente este arquivo:")
    print(final_log)


if __name__ == "__main__":
    main()
