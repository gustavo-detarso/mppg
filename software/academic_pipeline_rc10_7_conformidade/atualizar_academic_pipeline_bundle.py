#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atualizador seguro para bundles academic_pipeline/MPPG.

O script:
1. descompacta o ZIP corrigido em diretório temporário;
2. localiza automaticamente a raiz que contém app_bundle/;
3. faz backup dos arquivos locais que serão sobrescritos;
4. copia os arquivos seguros do bundle novo para o bundle atual;
5. executa testes básicos:
   - py_compile dos scripts Python do pipeline;
   - --list-toml-profiles;
   - --doctor, --check-config e opcionalmente --show-prompts, se --config for informado;
   - --somente-renderizar, se --render e --document-json forem informados.
6. opcionalmente faz rollback automático se algum teste falhar.

Uso típico:
python atualizar_academic_pipeline_bundle.py \
  --zip /home/gustavodetarso/Downloads/academic_pipeline_rc10_7_14.zip \
  --dst /home/gustavodetarso/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade \
  --config app_bundle/projetos/resumo_artigos_encontro_4/resumo_artigos_config.toml \
  --document-json app_bundle/output/documento/resumo_artigos_encontro_4/resumo_artigos_encontro_4.document.json \
  --render \
  --rollback-on-fail
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_COPY_PREFIXES = [
    "app_bundle/scripts/",
    "app_bundle/prompts/",
    "app_bundle/institutions/",
    "app_bundle/misc/",
    "app_bundle/templates/",
]

DEFAULT_ROOT_FILES = [
    "README.md",
    "MANUAL.md",
    "CHANGELOG.md",
    "Pipfile",
    "Pipfile.lock",
    "requirements.txt",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

PROTECTED_PREFIXES = [
    "app_bundle/output/",
    "app_bundle/projetos/",
    "app_bundle/fulltext_cache/",
    ".env",
]


def norm_rel(path: Path) -> str:
    return path.as_posix().lstrip("./")


def is_protected(rel: str) -> bool:
    rel = norm_rel(Path(rel))
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in PROTECTED_PREFIXES)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"\n[ERRO] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def unzip_to_temp(zip_path: Path, tmp_root: Path) -> Path:
    log(f"[ETAPA] Descompactando ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_root)
    return tmp_root


def find_src_root(extract_dir: Path) -> Path:
    log("[ETAPA] Localizando app_bundle no ZIP extraído")
    candidates = []
    for p in extract_dir.rglob("app_bundle"):
        if p.is_dir() and (p / "scripts" / "pipeline" / "academic_pipeline_rc10.py").exists():
            candidates.append(p)

    if not candidates:
        fail("Não encontrei app_bundle/scripts/pipeline/academic_pipeline_rc10.py dentro do ZIP.")

    candidates.sort(key=lambda p: len(p.parts))
    app_bundle = candidates[0]
    src_root = app_bundle.parent
    log(f"[OK] Raiz do bundle novo: {src_root}")
    return src_root


def iter_files_under(base: Path) -> Iterable[Path]:
    if not base.exists():
        return
    if base.is_file():
        yield base
        return

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
        root_path = Path(root)
        for name in files:
            yield root_path / name


def collect_files_to_copy(src_root: Path, copy_prefixes: list[str], include_root_files: bool) -> list[Path]:
    files: list[Path] = []

    for prefix in copy_prefixes:
        prefix = prefix.strip()
        if not prefix:
            continue
        base = src_root / prefix
        for f in iter_files_under(base):
            rel = norm_rel(f.relative_to(src_root))
            if not is_protected(rel):
                files.append(f)

    if include_root_files:
        for name in DEFAULT_ROOT_FILES:
            f = src_root / name
            if f.exists() and f.is_file():
                rel = norm_rel(f.relative_to(src_root))
                if not is_protected(rel):
                    files.append(f)

    # remove duplicatas preservando ordem
    seen = set()
    unique = []
    for f in files:
        rel = norm_rel(f.relative_to(src_root))
        if rel not in seen:
            seen.add(rel)
            unique.append(f)

    return unique


def make_backup_and_copy(
    src_root: Path,
    dst_root: Path,
    files: list[Path],
    backup_root: Path,
    dry_run: bool = False,
) -> dict:
    log("[ETAPA] Fazendo backup e copiando arquivos")
    manifest = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "backup_root": str(backup_root),
        "copied": [],
        "new_files": [],
        "backed_up": [],
        "dry_run": dry_run,
    }

    for src_file in files:
        rel = norm_rel(src_file.relative_to(src_root))
        dst_file = dst_root / rel
        backup_file = backup_root / rel

        if is_protected(rel):
            log(f"[SKIP protegido] {rel}")
            continue

        if dst_file.exists():
            manifest["backed_up"].append(rel)
            log(f"[BACKUP] {rel}")
            if not dry_run:
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst_file, backup_file)
        else:
            manifest["new_files"].append(rel)
            log(f"[NOVO] {rel}")

        manifest["copied"].append(rel)
        log(f"[COPY] {rel}")
        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    if not dry_run:
        backup_root.mkdir(parents=True, exist_ok=True)
        (backup_root / "manifest_update.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    log(f"[OK] Total de arquivos preparados/copied: {len(manifest['copied'])}")
    return manifest


def rollback(manifest: dict) -> None:
    log("[ETAPA] Rollback automático iniciado")
    dst_root = Path(manifest["dst_root"])
    backup_root = Path(manifest["backup_root"])

    backed_up = set(manifest.get("backed_up", []))
    new_files = set(manifest.get("new_files", []))
    copied = list(manifest.get("copied", []))

    for rel in reversed(copied):
        dst_file = dst_root / rel
        backup_file = backup_root / rel

        if rel in backed_up and backup_file.exists():
            log(f"[ROLLBACK restore] {rel}")
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, dst_file)
        elif rel in new_files and dst_file.exists():
            log(f"[ROLLBACK remove novo] {rel}")
            dst_file.unlink()

    log("[OK] Rollback finalizado")


def run_cmd(cmd: list[str], cwd: Path, log_file: Path, check: bool = True) -> int:
    rendered = " ".join(cmd)
    log(f"\n[TESTE] {rendered}")

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write("\n" + "=" * 100 + "\n")
        lf.write(f"$ {rendered}\n")
        lf.write("=" * 100 + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)

        rc = proc.wait()
        lf.write(f"\n[returncode] {rc}\n")

    if check and rc != 0:
        raise RuntimeError(f"Comando falhou com código {rc}: {rendered}")

    return rc


def run_tests(
    dst_root: Path,
    config: str | None,
    document_json: str | None,
    do_render: bool,
    do_show_prompts: bool,
    do_pipenv_sync: bool,
    log_file: Path,
) -> None:
    log("[ETAPA] Rodando testes")

    if do_pipenv_sync:
        run_cmd(["pipenv", "sync"], cwd=dst_root, log_file=log_file)

    pipeline_dir = dst_root / "app_bundle" / "scripts" / "pipeline"
    py_files = sorted(str(p.relative_to(dst_root)) for p in pipeline_dir.glob("*.py"))

    if py_files:
        run_cmd(["pipenv", "run", "python", "-m", "py_compile", *py_files], cwd=dst_root, log_file=log_file)

    main_script = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"

    run_cmd(
        ["pipenv", "run", "python", main_script, "--list-toml-profiles"],
        cwd=dst_root,
        log_file=log_file,
    )

    if config:
        run_cmd(
            ["pipenv", "run", "python", main_script, "--config", config, "--doctor"],
            cwd=dst_root,
            log_file=log_file,
        )
        run_cmd(
            ["pipenv", "run", "python", main_script, "--config", config, "--check-config"],
            cwd=dst_root,
            log_file=log_file,
        )
        if do_show_prompts:
            run_cmd(
                ["pipenv", "run", "python", main_script, "--config", config, "--show-prompts"],
                cwd=dst_root,
                log_file=log_file,
            )

    if do_render:
        if not config:
            raise RuntimeError("--render exige --config.")
        if not document_json:
            raise RuntimeError("--render exige --document-json.")
        run_cmd(
            [
                "pipenv", "run", "python", main_script,
                "--config", config,
                "--somente-renderizar",
                "--document-json", document_json,
            ],
            cwd=dst_root,
            log_file=log_file,
        )

    log("[OK] Todos os testes solicitados passaram")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atualiza com segurança um bundle academic_pipeline a partir de um ZIP corrigido."
    )
    parser.add_argument("--zip", required=True, help="Caminho do ZIP corrigido, ex.: academic_pipeline_rc10_7_14.zip")
    parser.add_argument("--dst", required=True, help="Diretório do bundle local já existente.")
    parser.add_argument(
        "--config",
        default=None,
        help="Caminho do TOML relativo ao destino, ex.: app_bundle/projetos/.../config.toml",
    )
    parser.add_argument(
        "--document-json",
        default=None,
        help="Caminho do document.json relativo ao destino, usado com --render.",
    )
    parser.add_argument("--render", action="store_true", help="Testa também --somente-renderizar.")
    parser.add_argument("--show-prompts", action="store_true", help="Testa também --show-prompts.")
    parser.add_argument("--rollback-on-fail", action="store_true", help="Restaura backup se algum teste falhar.")
    parser.add_argument("--dry-run", action="store_true", help="Mostra o que faria, sem copiar arquivos.")
    parser.add_argument(
        "--copy-root-files",
        action="store_true",
        help="Copia também arquivos de raiz como README, Pipfile e requirements.txt.",
    )
    parser.add_argument(
        "--pipenv-sync",
        action="store_true",
        help="Roda pipenv sync antes dos testes. Use apenas se quiser atualizar dependências conforme Pipfile.lock.",
    )
    parser.add_argument(
        "--copy-prefix",
        action="append",
        default=None,
        help=(
            "Prefixo adicional/alternativo a copiar, relativo à raiz do ZIP. "
            "Pode repetir. Se usado, substitui os prefixos padrão."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not zip_path.exists():
        fail(f"ZIP não encontrado: {zip_path}")
    if not dst_root.exists():
        fail(f"Diretório destino não encontrado: {dst_root}")
    if not (dst_root / "app_bundle" / "scripts" / "pipeline" / "academic_pipeline_rc10.py").exists():
        fail(f"O destino não parece ser um bundle academic_pipeline válido: {dst_root}")

    version_label = zip_path.stem.replace(".", "_").replace("-", "_")
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dst_root / f"_backup_pre_{version_label}_{timestamp}"
    log_file = backup_root / "update_test_log.txt"

    copy_prefixes = args.copy_prefix if args.copy_prefix else DEFAULT_COPY_PREFIXES

    manifest = None
    with tempfile.TemporaryDirectory(prefix="academic_pipeline_update_") as tmp:
        tmp_root = Path(tmp)
        unzip_to_temp(zip_path, tmp_root)
        src_root = find_src_root(tmp_root)

        files = collect_files_to_copy(
            src_root=src_root,
            copy_prefixes=copy_prefixes,
            include_root_files=args.copy_root_files,
        )

        if not files:
            fail("Nenhum arquivo selecionado para cópia. Verifique o ZIP ou os prefixos.")

        log("\n[INFO] Arquivos que serão copiados:")
        for f in files:
            log(f"  - {norm_rel(f.relative_to(src_root))}")

        if args.dry_run:
            log("\n[DRY-RUN] Nenhum arquivo será copiado e nenhum teste será executado.")
            return 0

        backup_root.mkdir(parents=True, exist_ok=True)
        manifest = make_backup_and_copy(
            src_root=src_root,
            dst_root=dst_root,
            files=files,
            backup_root=backup_root,
            dry_run=False,
        )

    try:
        run_tests(
            dst_root=dst_root,
            config=args.config,
            document_json=args.document_json,
            do_render=args.render,
            do_show_prompts=args.show_prompts,
            do_pipenv_sync=args.pipenv_sync,
            log_file=log_file,
        )
    except Exception as exc:
        log(f"\n[ERRO] Teste falhou: {exc}")
        log(f"[INFO] Log completo: {log_file}")

        if args.rollback_on_fail and manifest is not None:
            rollback(manifest)
        else:
            log("[INFO] Rollback automático não foi solicitado. Use o backup manualmente se necessário.")
            log(f"[INFO] Backup: {backup_root}")

        return 1

    log("\n[OK] Atualização concluída com sucesso.")
    log(f"[OK] Backup: {backup_root}")
    log(f"[OK] Log:    {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
