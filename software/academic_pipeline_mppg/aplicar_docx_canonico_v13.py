#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import py_compile
import re
import shutil

ROOT = Path.cwd()
MODULE_PATH = ROOT / "app_bundle" / "scripts" / "pipeline" / "render_docx_canonico.py"
WRAPPER_PATH = ROOT / "gerar_docx_canonico.py"

TAG = "docx_canonico_v13_cover_bottom_city_year"
CITY_SPACE_BEFORE_PT = 240


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_name(path.name + f".bak_{TAG}_{stamp()}")
    shutil.copy2(path, bak)
    print(f"[OK] Backup: {bak}")
    return bak


def patch_renderer(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"ERRO: não encontrei {path}. Aplique antes o patch v12, ou confira se está na raiz do projeto."
        )

    txt = path.read_text(encoding="utf-8", errors="ignore")
    original = txt

    # Reposiciona local/ano da capa: a versão anterior deixava Brasília/ano
    # visualmente acima do rodapé da capa. A ABNT usualmente coloca esses dados
    # centralizados no bloco inferior da página, imediatamente acima da margem.
    city_pattern = re.compile(
        r'add_centered_paragraph\(\s*doc,\s*metadata\.get\("city",\s*"Brasília"\),\s*size=12,\s*before=\d+,\s*after=0\s*\)'
    )
    replacement = (
        'add_centered_paragraph(doc, metadata.get("city", "Brasília"), '
        f'size=12, before={CITY_SPACE_BEFORE_PT}, after=0)'
    )
    if not city_pattern.search(txt):
        raise SystemExit(
            "ERRO: não consegui localizar a linha de cidade/ano em add_cover(). "
            "Confirme se o renderizador DOCX canônico está na versão v11/v12."
        )
    txt = city_pattern.sub(replacement, txt, count=1)

    # Atualiza marcadores de versão quando existirem.
    txt = txt.replace('"schema_version": "docx-canonico-v12"', '"schema_version": "docx-canonico-v13"')
    txt = txt.replace('"schema_version": "docx-canonico-v11"', '"schema_version": "docx-canonico-v13"')
    txt = txt.replace(
        'Versão v12 com texto integralmente preto.',
        'Versão v13 com local e ano posicionados no bloco inferior da capa.'
    )
    txt = txt.replace(
        'Versão v11 sem resíduos LaTeX/export.',
        'Versão v13 com local e ano posicionados no bloco inferior da capa.'
    )

    # Acrescenta uma pequena evidência no relatório, sem alterar a lógica principal.
    if 'report["cover_city_year_space_before_pt"]' not in txt:
        marker = '    report["font_color_policy"] = "all_word_color_tags_forced_to_000000"\n'
        if marker in txt:
            txt = txt.replace(
                marker,
                marker + f'    report["cover_city_year_space_before_pt"] = {CITY_SPACE_BEFORE_PT}\n',
                1,
            )

    if txt == original:
        print(f"[OK] {path} já parecia estar no estado esperado.")
    else:
        backup(path)
        path.write_text(txt, encoding="utf-8")

    py_compile.compile(str(path), doraise=True)
    print(f"[OK] Renderizador DOCX canônico v13 instalado: {path}")


def patch_wrapper(path: Path) -> None:
    if not path.exists():
        path.write_text(
            '#!/usr/bin/env python3\n'
            'from __future__ import annotations\n\n'
            'from app_bundle.scripts.pipeline.render_docx_canonico import main\n\n'
            'if __name__ == "__main__":\n'
            '    raise SystemExit(main())\n',
            encoding="utf-8",
        )
        print(f"[OK] Wrapper criado: {path}")
    py_compile.compile(str(path), doraise=True)


def main() -> int:
    if not (ROOT / "app_bundle").exists():
        raise SystemExit("ERRO: execute este aplicador na raiz do projeto academic_pipeline_mppg.")
    patch_renderer(MODULE_PATH)
    patch_wrapper(WRAPPER_PATH)
    print("[OK] Patch v13 aplicado: local e ano da capa foram deslocados para o bloco inferior da página.")
    print("[OK] Gere novamente com: pipenv run python gerar_docx_canonico.py --art-dir $ART --cfg-art $CFG_ART --quiet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
