#!/usr/bin/env python3
from pathlib import Path
import argparse
import importlib.util
import sys


def ask_path(label: str, default: Path) -> Path:
    value = input(f"{label} [{default}]: ").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def clean_aux_files(org_path: Path) -> None:
    patterns = [
        "*.aux", "*.bcf", "*.bbl", "*.blg", "*.log", "*.out",
        "*.run.xml", "*.toc", "*.tex", "*.lof", "*.lot", "*.fls",
        "*.fdb_latexmk"
    ]

    for pattern in patterns:
        for file in org_path.parent.glob(pattern):
            try:
                file.unlink()
                print(f"Removido: {file.name}")
            except Exception as exc:
                print(f"Aviso: não consegui remover {file}: {exc}")


def import_pipeline(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script do pipeline não encontrado: {script_path}")

    sys.path.insert(0, str(script_path.parent))
    sys.path.insert(0, str(script_path.parent.parent.parent))

    spec = importlib.util.spec_from_file_location("academic_pipeline_rc7", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar o módulo: {script_path}")

    mod = importlib.util.module_from_spec(spec)

    # Necessário para @dataclass funcionar corretamente com importlib.
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)
    return mod


def main():
    base_default = Path("/home/gustavodetarso/Documentos/mppg/software/academic_pipeline")
    app_default = base_default / "app_bundle"

    parser = argparse.ArgumentParser(
        description="Recompila um arquivo .org já gerado pelo fluxo do academic_pipeline, preservando o .org e o .bib."
    )

    parser.add_argument("--base", default=str(base_default), help="Diretório base do academic_pipeline.")
    parser.add_argument("--org", default="", help="Caminho do arquivo .org a recompilar.")
    parser.add_argument("--script", default="", help="Caminho do script academic_pipeline.py.")
    parser.add_argument("--academic-writing", default="", help="Caminho do academic-writing.el.")
    parser.add_argument("--fgv-path", default="", help="Caminho da pasta misc/fgv.")
    parser.add_argument("--no-clean", action="store_true", help="Não remove arquivos auxiliares antes de recompilar.")
    parser.add_argument("--interactive", action="store_true", help="Pergunta os caminhos antes de compilar.")

    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    app = base / "app_bundle"

    org_default = app / "output/documento/paper_politica_brasileira_contemporanea/paper_politica_brasileira_contemporanea.org"
    script_default = app / "scripts/pipeline/academic_pipeline.py"
    academic_default = app / "misc/academic-writing.el"
    fgv_default = app / "misc/fgv"

    if args.interactive:
        org_path = ask_path("Arquivo ORG", Path(args.org) if args.org else org_default)
        script_path = ask_path("Script do pipeline", Path(args.script) if args.script else script_default)
        academic_writing = ask_path("academic-writing.el", Path(args.academic_writing) if args.academic_writing else academic_default)
        fgv_path = ask_path("Pasta FGV/LaTeX", Path(args.fgv_path) if args.fgv_path else fgv_default)
    else:
        org_path = Path(args.org).expanduser().resolve() if args.org else org_default.resolve()
        script_path = Path(args.script).expanduser().resolve() if args.script else script_default.resolve()
        academic_writing = Path(args.academic_writing).expanduser().resolve() if args.academic_writing else academic_default.resolve()
        fgv_path = Path(args.fgv_path).expanduser().resolve() if args.fgv_path else fgv_default.resolve()

    for label, path in [
        ("ORG", org_path),
        ("Script", script_path),
        ("academic-writing.el", academic_writing),
        ("Pasta FGV", fgv_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} não encontrado: {path}")

    bib_path = org_path.with_suffix(".bib")
    if not bib_path.exists():
        print(f"Aviso: não encontrei .bib ao lado do .org: {bib_path}")

    if not args.no_clean:
        print("Limpando auxiliares de compilação...")
        clean_aux_files(org_path)

    print("Carregando pipeline...")
    mod = import_pipeline(script_path)

    print("Recompilando pelo fluxo do pipeline...")
    pdf = mod.run_compile_sequence(
        org_path.resolve(),
        academic_writing=academic_writing.resolve(),
        latex_extra_path=fgv_path.resolve(),
    )

    print(f"\nPDF gerado em: {pdf}")


if __name__ == "__main__":
    main()
