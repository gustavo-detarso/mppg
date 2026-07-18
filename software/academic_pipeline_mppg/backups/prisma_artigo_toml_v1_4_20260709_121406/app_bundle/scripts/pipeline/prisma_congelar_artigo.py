#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_ARTIGO_DIR = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_info(path: Path, copied_from: str | None = None) -> dict:
    st = path.stat()
    return {
        "arquivo": path.name,
        "caminho": str(path),
        "origem": copied_from,
        "tamanho_bytes": st.st_size,
        "mtime_iso": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }

def copy_one(src: Path, dest_dir: Path, required: bool, copied: list[dict], missing: list[str]) -> None:
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {src}")
        missing.append(str(src))
        return
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    copied.append(file_info(dest, copied_from=str(src)))

def freeze_inputs(out_dir: Path, artigo_dir: Path | None = None, dest_dir: Path | None = None, prefix: str | None = None) -> Path:
    out_dir = out_dir.resolve()
    prefix = prefix or out_dir.name
    artigo_dir = (artigo_dir or DEFAULT_ARTIGO_DIR).resolve()
    dest_dir = (dest_dir.resolve() if dest_dir else artigo_dir / "dados_prisma")
    dest_dir.mkdir(parents=True, exist_ok=True)

    required_names = [
        f"{prefix}.referencias_incluidas.bib",
        f"{prefix}.referencias_incluidas_seminario.csv",
        f"{prefix}.triagem_humana.csv",
        f"{prefix}.relatorio_prisma_final.pdf",
    ]
    optional_names = [
        f"{prefix}.curadoria_ia_referencias.xlsx",
        f"{prefix}.relatorio_prisma_preliminar.pdf",
        f"{prefix}.diagrama_prisma.png",
        f"{prefix}.diagrama_prisma_contagens.json",
        f"{prefix}.busca_prisma_log.json",
        f"{prefix}.triagem_titulo_resumo.csv",
        f"{prefix}.triagem_titulo_resumo.xlsx",
        f"{prefix}.curadoria_ia_resumo.txt",
        f"{prefix}.curadoria_ia_log.json",
    ]

    copied: list[dict] = []
    missing_optional: list[str] = []

    for name in required_names:
        copy_one(out_dir / name, dest_dir, required=True, copied=copied, missing=missing_optional)
    for name in optional_names:
        copy_one(out_dir / name, dest_dir, required=False, copied=copied, missing=missing_optional)

    manifest_lines = [f"{item['sha256']}  {item['arquivo']}" for item in sorted(copied, key=lambda x: x["arquivo"])]
    (dest_dir / "MANIFESTO_SHA256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    arquivo_lines = [
        "Arquivo\tTamanho_bytes\tModificado_em\tSHA256",
        *[
            f"{item['arquivo']}\t{item['tamanho_bytes']}\t{item['mtime_iso']}\t{item['sha256']}"
            for item in sorted(copied, key=lambda x: x["arquivo"])
        ],
    ]
    (dest_dir / "ARQUIVOS_CONGELADOS.txt").write_text("\n".join(arquivo_lines) + "\n", encoding="utf-8")

    manifest_json = {
        "gerado_em": dt.datetime.now().isoformat(timespec="seconds"),
        "out_dir_origem": str(out_dir),
        "artigo_dir": str(artigo_dir),
        "destino": str(dest_dir),
        "prefixo": prefix,
        "arquivos": sorted(copied, key=lambda x: x["arquivo"]),
        "opcionais_ausentes": missing_optional,
    }
    (dest_dir / "MANIFESTO_ARTIGO.json").write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Insumos congelados em: {dest_dir}")
    print(f"[OK] Arquivos copiados: {len(copied)}")
    print(f"[OK] Manifesto SHA256: {dest_dir / 'MANIFESTO_SHA256.txt'}")
    print(f"[OK] Manifesto JSON: {dest_dir / 'MANIFESTO_ARTIGO.json'}")
    if missing_optional:
        print(f"[INFO] Arquivos opcionais ausentes: {len(missing_optional)}")
    return dest_dir

def main(argv=None):
    p = argparse.ArgumentParser(description="Congela insumos finais do artigo e cria manifestos.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--artigo-dir", default=str(DEFAULT_ARTIGO_DIR))
    p.add_argument("--dest-dir", default=None)
    p.add_argument("--prefix", default=None)
    a = p.parse_args(argv)
    out_dir = Path(a.out_dir)
    artigo_dir = Path(a.artigo_dir) if a.artigo_dir else DEFAULT_ARTIGO_DIR
    dest_dir = Path(a.dest_dir) if a.dest_dir else None
    prefix = a.prefix or out_dir.name
    freeze_inputs(out_dir, artigo_dir=artigo_dir, dest_dir=dest_dir, prefix=prefix)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
