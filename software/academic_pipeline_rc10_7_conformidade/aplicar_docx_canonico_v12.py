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

TAG = "docx_canonico_v12_black_text"

NEW_FONT_FUNCS = '''def set_run_font(run: Any, name: str = "Times New Roman", size_pt: int = 12, bold: bool | None = None, italic: bool | None = None) -> None:
    run.font.name = name
    run.font.size = Pt(size_pt)
    # ABNT/FGV: todo texto visível deve permanecer em preto. O estilo padrão
    # de headings do Word costuma herdar azul temático; por isso a cor é
    # fixada explicitamente no nível do run.
    try:
        run.font.color.rgb = RGBColor(0, 0, 0)
    except Exception:
        pass
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
        rfonts.set(qn(attr), name)


def set_style_font(style: Any, name: str = "Times New Roman", size_pt: int = 12, bold: bool | None = None) -> None:
    style.font.name = name
    style.font.size = Pt(size_pt)
    # Corrige a cor azul herdada dos estilos Heading 1/2/3 do template padrão.
    try:
        style.font.color.rgb = RGBColor(0, 0, 0)
    except Exception:
        pass
    if bold is not None:
        style.font.bold = bold
    rpr = style._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
        rfonts.set(qn(attr), name)


'''

FORCE_BLACK_FUNC = '''
def force_ooxml_black_docx(path: Path) -> None:
    """Força todas as definições OOXML de cor de fonte para preto.

    python-docx corrige a maioria dos runs, mas o Word/LibreOffice pode manter
    cores temáticas nos estilos Heading*. Esta etapa final atua diretamente em
    word/styles.xml, word/document.xml, headers, footers e notas, removendo
    themeColor/themeShade/themeTint e substituindo qualquer w:color por preto.
    """
    if not path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp_black")
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                try:
                    xml = data.decode("utf-8", errors="ignore")
                except Exception:
                    zout.writestr(item, data)
                    continue
                xml = re.sub(r"<w:color\\b[^>]*/>", '<w:color w:val="000000"/>', xml)
                xml = re.sub(r"<w:color\\b[^>]*>", '<w:color w:val="000000"/>', xml)
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)

'''


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
            f"ERRO: não encontrei {path}. Aplique antes o patch v11, ou confira se está na raiz do projeto."
        )
    txt = path.read_text(encoding="utf-8", errors="ignore")
    original = txt

    # Importar RGBColor.
    if "from docx.shared import Cm, Pt, RGBColor" not in txt:
        txt = txt.replace("from docx.shared import Cm, Pt", "from docx.shared import Cm, Pt, RGBColor")

    # Substituir funções de fonte por versões que fixam preto.
    pattern_fonts = re.compile(r"def set_run_font\(.*?\n(?=def configure_doc_styles\()", re.S)
    if not pattern_fonts.search(txt):
        raise SystemExit("ERRO: não consegui localizar set_run_font/set_style_font no renderizador DOCX.")
    txt = pattern_fonts.sub(NEW_FONT_FUNCS, txt, count=1)

    # Inserir função OOXML final antes de validate_docx.
    if "def force_ooxml_black_docx" not in txt:
        marker = "def validate_docx(path: Path) -> dict[str, Any]:"
        if marker not in txt:
            raise SystemExit("ERRO: não consegui localizar validate_docx no renderizador DOCX.")
        txt = txt.replace(marker, FORCE_BLACK_FUNC + marker, 1)

    # Garantir chamada após salvar.
    if "force_ooxml_black_docx(paths.docx)" not in txt:
        txt = txt.replace("    doc.save(paths.docx)\n", "    doc.save(paths.docx)\n    force_ooxml_black_docx(paths.docx)\n", 1)

    # Validador deve ler styles.xml para detectar azul temático residual.
    if "style_xml = zf.read(\"word/styles.xml\")" not in txt:
        txt = txt.replace(
            '            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")\n',
            '            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")\n'
            '            try:\n'
            '                style_xml = zf.read("word/styles.xml").decode("utf-8", errors="ignore")\n'
            '            except KeyError:\n'
            '                style_xml = ""\n',
            1,
        )
    if "non_black_colors =" not in txt:
        txt = txt.replace(
            '    report["paragraphs_xml_count"] = xml.count("<w:p")\n',
            '    color_xml = xml + "\\n" + style_xml\n'
            '    color_tags = re.findall(r\'<w:color\\b[^>]*/>\', color_xml)\n'
            '    non_black_colors = [tag for tag in color_tags if \'w:val="000000"\' not in tag and \'w:val="auto"\' not in tag]\n'
            '    if non_black_colors:\n'
            '        report["warnings"].append(f"Cores de fonte não pretas detectadas no DOCX: {len(non_black_colors)} ocorrência(s).")\n'
            '    report["font_color_policy"] = "all_word_color_tags_forced_to_000000"\n'
            '    report["paragraphs_xml_count"] = xml.count("<w:p")\n',
            1,
        )

    txt = txt.replace('"schema_version": "docx-canonico-v11"', '"schema_version": "docx-canonico-v12"')
    txt = txt.replace(
        'description="Gera DOCX ABNT/FGV canônico a partir do ORG/document.json/BIB finais. Versão v11 sem resíduos LaTeX/export."',
        'description="Gera DOCX ABNT/FGV canônico a partir do ORG/document.json/BIB finais. Versão v12 com texto integralmente preto."',
    )

    if txt == original:
        print(f"[OK] {path} já parecia estar no estado esperado.")
    else:
        backup(path)
        path.write_text(txt, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"[OK] Renderizador DOCX canônico v12 instalado: {path}")


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
    print("[OK] Patch v12 aplicado: títulos, subtítulos e corpo passam a ser renderizados em fonte preta.")
    print("[OK] Gere novamente com: pipenv run python gerar_docx_canonico.py --art-dir $ART --cfg-art $CFG_ART --quiet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
