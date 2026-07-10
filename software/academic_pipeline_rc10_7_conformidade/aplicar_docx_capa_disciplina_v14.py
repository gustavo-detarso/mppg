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

TAG = "docx_canonico_v14_capa_disciplina"


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
            f"ERRO: não encontrei {path}. Execute este aplicador na raiz do projeto "
            "academic_pipeline_rc10_7_conformidade e confirme que o DOCX canônico já está instalado."
        )

    txt = path.read_text(encoding="utf-8", errors="ignore")
    original = txt

    # 1) Inclui a disciplina no bloco institucional superior da capa:
    #    FUNDAÇÃO GETÚLIO VARGAS / CURSO / DISCIPLINA.
    old_inst = 'inst_lines_raw = [metadata.get("institution", ""), metadata.get("program", ""), metadata.get("course", "")]'
    new_inst = (
        'inst_lines_raw = [\n'
        '        metadata.get("institution", ""),\n'
        '        metadata.get("program", ""),\n'
        '        metadata.get("course", ""),\n'
        '        metadata.get("discipline", ""),\n'
        '    ]'
    )
    if old_inst in txt:
        txt = txt.replace(old_inst, new_inst, 1)
    elif 'metadata.get("discipline", ""),' in txt and 'inst_lines_raw = [' in txt:
        print("[OK] A disciplina já parece estar no bloco institucional superior.")
    else:
        raise SystemExit(
            "ERRO: não consegui localizar a lista inst_lines_raw em add_cover(). "
            "Confirme se o renderizador está nas versões v11-v13 do DOCX canônico."
        )

    # 2) Ajusta a nota inferior para não repetir a disciplina como linha solta "Disciplina — Professor".
    #    A disciplina já fica no bloco superior e permanece na frase da covernote:
    #    "Trabalho acadêmico elaborado para a disciplina ...". Na linha seguinte fica apenas Professor.
    old_note = '''        parts: list[str] = []\n        if metadata.get("discipline"):\n            parts.append(metadata["discipline"])\n        if metadata.get("professor"):\n            parts.append(metadata["professor"])\n        if parts:\n            note = (note + "\\n" if note else "") + " — ".join(parts)'''
    new_note = '''        parts: list[str] = []\n        if metadata.get("professor"):\n            professor = metadata["professor"].strip()\n            if professor and not professor.lower().startswith("professor"):\n                professor = "Professor: " + professor\n            if professor:\n                parts.append(professor)\n        elif metadata.get("discipline") and not note:\n            # Fallback raro: se não houver nota nem professor, preserva a disciplina.\n            parts.append(metadata["discipline"])\n        if parts:\n            note = (note + "\\n" if note else "") + " — ".join(parts)'''
    if old_note in txt:
        txt = txt.replace(old_note, new_note, 1)
    elif 'Professor: " + professor' in txt:
        print("[OK] A nota da capa já parece estar ajustada para Professor.")
    else:
        print("[AVISO] Não localizei o bloco antigo da nota da capa; mantive a lógica atual da nota.")

    # 3) Atualiza marcadores de versão/relatório, se existirem.
    txt = txt.replace('"schema_version": "docx-canonico-v13"', '"schema_version": "docx-canonico-v14"')
    txt = txt.replace('"schema_version": "docx-canonico-v12"', '"schema_version": "docx-canonico-v14"')
    txt = txt.replace('"schema_version": "docx-canonico-v11"', '"schema_version": "docx-canonico-v14"')
    txt = txt.replace(
        'Versão v13 com local e ano posicionados no bloco inferior da capa.',
        'Versão v14 com disciplina no bloco institucional superior da capa.'
    )
    txt = txt.replace(
        'Versão v12 com texto integralmente preto.',
        'Versão v14 com disciplina no bloco institucional superior da capa.'
    )
    txt = txt.replace(
        'Versão v11 sem resíduos LaTeX/export.',
        'Versão v14 com disciplina no bloco institucional superior da capa.'
    )

    if 'report["cover_discipline_in_top_block"]' not in txt:
        markers = [
            '    report["cover_city_year_space_before_pt"] = 240\n',
            '    report["font_color_policy"] = "all_word_color_tags_forced_to_000000"\n',
        ]
        for marker in markers:
            if marker in txt:
                txt = txt.replace(marker, marker + '    report["cover_discipline_in_top_block"] = True\n', 1)
                break

    if txt == original:
        print(f"[OK] {path} já parecia estar no estado esperado.")
    else:
        backup(path)
        path.write_text(txt, encoding="utf-8")

    py_compile.compile(str(path), doraise=True)
    print(f"[OK] Renderizador DOCX canônico v14 instalado: {path}")


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
        raise SystemExit("ERRO: execute este aplicador na raiz do projeto academic_pipeline_rc10_7_conformidade.")
    patch_renderer(MODULE_PATH)
    patch_wrapper(WRAPPER_PATH)
    print("[OK] Patch v14 aplicado: a disciplina passa a aparecer no bloco institucional superior da capa do DOCX.")
    print("[OK] Gere novamente com: pipenv run python gerar_docx_canonico.py --art-dir $ART --cfg-art $CFG_ART --quiet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
