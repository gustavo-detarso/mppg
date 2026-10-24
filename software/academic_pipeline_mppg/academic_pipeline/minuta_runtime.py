"""Fluxo nativo de minutas FGV para Academic Pipeline MPPG.

Integração aditiva: não altera os renderizadores existentes.
Comandos: python -m academic_pipeline minuta init|check|build.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _safe_name(s: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", s):
        raise ValueError("[projeto].nome deve usar somente letras ASCII, números, '_' e '-'.")
    return s


def _path_in_project(base: Path, value: str, *, field: str) -> Path:
    p = Path(value).expanduser()
    resolved = (p if p.is_absolute() else base / p).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{field} deve apontar para dentro da pasta do projeto: {value}")
    return resolved


def _value(cfg: dict, section: str, key: str, default: str = "") -> str:
    sect = cfg.get(section, {})
    if not isinstance(sect, dict):
        raise ValueError(f"[{section}] deve ser uma tabela TOML")
    val = sect.get(key, default)
    if not isinstance(val, str):
        raise ValueError(f"[{section}].{key} deve ser texto")
    return val.strip()


def _load(path: Path) -> tuple[dict, Path, Path, Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TOML ausente: {path}")
    with path.open("rb") as fd:
        cfg = tomllib.load(fd)
    base = path.parent
    name = _safe_name(_value(cfg, "projeto", "nome"))
    if _value(cfg, "instituicao", "perfil").lower() != "fgv":
        raise ValueError("Este modelo suporta o perfil institucional 'fgv'.")
    source = _path_in_project(base, _value(cfg, "paths", "org_conteudo", "conteudo.org"), field="org_conteudo")
    output = _path_in_project(base, _value(cfg, "paths", "document_output_dir", "saida"), field="document_output_dir")
    if not source.is_file():
        raise FileNotFoundError(f"Conteúdo Org ausente: {source}")
    templ = _root() / "app_bundle/institutions/fgv/templates/template_minuta.org"
    logo = _root() / "app_bundle/institutions/fgv/assets/fgv.png"
    if not templ.is_file() or not logo.is_file():
        raise FileNotFoundError(f"Recursos FGV ausentes: modelo={templ.is_file()}, logo={logo.is_file()}")
    if not _value(cfg, "documento", "titulo_trabalho"):
        raise ValueError("[documento].titulo_trabalho é obrigatório")
    if not _value(cfg, "documento", "autor"):
        raise ValueError("[documento].autor é obrigatório")
    sections = source.read_text(encoding="utf-8")
    for label, pattern in {"Tema": r"(?im)^\*\s+(?:1[.\s]+)?tema\s*$", "Problema": r"(?im)^\*\s+(?:2[.\s]+)?problema\s*$", "Protocolo da revisão": r"(?im)^\*\s+(?:3[.\s]+)?protocolo\s+da\s+revis[aã]o\s*$"}.items():
        if not re.search(pattern, sections):
            raise ValueError(f"Seção obrigatória não encontrada no conteúdo Org: {label}")
    for chave in ("exportar_pdf", "exportar_docx"):
        if not isinstance(cfg.get("saidas", {}).get(chave, True), bool):
            raise ValueError(f"[saidas].{chave} deve ser true ou false (booleano TOML)")
    cfg.setdefault("_internal", {})["nome"] = name
    return cfg, base, source, output


def _body(org: str) -> str:
    match = re.search(r"(?im)^\*\s+(?:1[.\s]+)?tema\s*$", org)
    if not match:
        raise ValueError("Seção 'Tema' não encontrada")
    return org[match.start():].strip() + "\n"


def _latex_escape(s: str) -> str:
    """Escapa conteúdo de metadados TOML antes de inserção em LaTeX."""
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in s.replace("\n", " ").replace("\r", " "))


def _identity_rows(cfg: dict) -> list[tuple[str, str]]:
    curso = _value(cfg, "documento", "curso", "Mestrado Profissional em Políticas Públicas e Governo")
    professor = _value(cfg, "documento", "professor") or _value(cfg, "documento", "orientador")
    pairs = [
        ("Curso / programa", curso),
        ("Mestrando", _value(cfg, "documento", "autor")),
        ("Professor orientador", professor),
        ("Documento", _value(cfg, "documento", "titulo_trabalho")),
        ("Data", _value(cfg, "documento", "data", "Outubro de 2026")),
    ]
    for title, field in (("Turma", "turma"), ("Polo", "polo"), ("Disciplina", "disciplina")):
        value = _value(cfg, "documento", field)
        if value:
            pairs.append((title, value))
    return [(k, v) for k, v in pairs if v]


def _identity_latex(cfg: dict) -> str:
    # MPPG_MINUTA_QUADRO_IDENTIFICACAO_V29: baixa ligeiramente o quadro, aproxima Curso do título.
    # MPPG_MINUTA_QUADRO_IDENTIFICACAO_V210: quadro +2pt; primeira linha -2pt em relação ao quadro.
    # A classe institucional fornece tcolorbox e define FGVHeaderBlueDark.
    # Usar fonte 10 pt apenas na ficha, texto acadêmico mantém 12 pt.
    rows = [_latex_escape(label) + " & " + _latex_escape(val) + r" \\"
            for label, val in _identity_rows(cfg)]
    return "\n".join([
        "#+begin_export latex",
        # MPPG_MINUTA_CABECALHO_COMPACTO_V27: referência histórica do ajuste anterior.
        # MPPG_MINUTA_CABECALHO_COMPACTO_V28: compactação interna do quadro e aproximação da régua.
        r"\vspace*{-21pt}",
        r"\begingroup",
        r"\setlength{\parindent}{0pt}",
        r"\begin{tcolorbox}[colback=FGVHeaderBlueLight!9,colframe=FGVHeaderBlueDark,"
        r"colbacktitle=FGVHeaderBlueDark,coltitle=white,title={IDENTIFICAÇÃO ACADÊMICA --- FGV},"
        r"fonttitle=\bfseries,toptitle=3pt,bottomtitle=2pt,boxrule=0.55pt,sharp corners,arc=0pt,"
        r"left=7pt,right=7pt,top=0pt,bottom=3pt,boxsep=0pt,before skip=0pt,after skip=4pt]",
        r"\begingroup\fontsize{10}{12}\selectfont",
        r"\renewcommand{\arraystretch}{0.84}",
        r"\vspace*{-6pt}",
        r"\begin{tabularx}{\linewidth}{@{}>{\bfseries}p{4.0cm}X@{}}",
        *rows,
        r"\end{tabularx}",
        r"\endgroup",
        r"\end{tcolorbox}",
        r"\endgroup",
        "#+end_export",
    ])


def _render(cfg: dict, source: Path, dest: Path) -> tuple[Path, Path]:
    templ = _root() / "app_bundle/institutions/fgv/templates/template_minuta.org"
    logo = _root() / "app_bundle/institutions/fgv/assets/fgv.png"
    texto = templ.read_text(encoding="utf-8")
    autor = _value(cfg, "documento", "autor")
    data = _value(cfg, "documento", "data", "Outubro de 2026")
    titulo = _value(cfg, "documento", "titulo_trabalho")
    # Metadados em diretivas Org nunca devem permitir inclusão de novas diretivas.
    def org_escape(v):
        return v.replace("\n", " ").replace("\r", " ").replace("{{", "(").replace("}}", ")")
    substitutions = {"TITULO_TRABALHO": org_escape(titulo),
                     "ALUNO": org_escape(autor), "DATA": org_escape(data),
                     "QUADRO_LATEX": _identity_latex(cfg),
                     "CONTEUDO": _body(source.read_text(encoding="utf-8"))}
    for key, val in substitutions.items():
        placeholder = "{{" + key + "}}"
        if placeholder not in texto:
            raise ValueError(f"Placeholder obrigatório não encontrado: {placeholder}")
        texto = texto.replace(placeholder, val)
    if re.search(r"{{[A-Z_]+}}", texto):
        raise ValueError("Modelo contém placeholders não resolvidos")
    dest.mkdir(parents=True, exist_ok=True)
    result_org = dest / f"{cfg['_internal']['nome']}.org"
    result_org.write_text(texto.rstrip() + "\n", encoding="utf-8")
    result_logo = dest / "fgv.png"
    shutil.copy2(logo, result_logo)
    return result_org, result_logo


def _find_aw(root: Path) -> Path:
    hits = sorted((root / "app_bundle").rglob("academic-writing.el"))
    if not hits:
        raise FileNotFoundError("academic-writing.el não localizado na instalação MPPG")
    return hits[0]


def _run_pdf(org: Path, root: Path, log: Path) -> Path:
    aw = _find_aw(root)
    latex_path = root / "app_bundle/institutions/fgv/latex"
    if not latex_path.is_dir():
        raise FileNotFoundError(f"Estilos LaTeX FGV não localizados: {latex_path}")
    args = [sys.executable, "-m", "academic_pipeline", "--recompile", "--org", str(org),
            "--pdf-engine", "lualatex", "--academic-writing", str(aw),
            "--latex-extra-path", str(latex_path)]
    process = subprocess.run(args, cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + " ".join(args) + "\n\n" + process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"Compilação nativa do MPPG falhou (código {process.returncode}). Consulte {log}")
    pdf = org.with_suffix(".pdf")
    if not pdf.is_file():
        raise FileNotFoundError(f"MPPG retornou sucesso mas não criou PDF: {pdf}")
    return pdf


def _pdf_pages(pdf: Path) -> tuple[int, str]:
    p = subprocess.run(["pdfinfo", str(pdf)], text=True, capture_output=True, check=True)
    n = re.search(r"^Pages:\s+(\d+)", p.stdout, re.M)
    size = re.search(r"^Page size:\s+(.+)", p.stdout, re.M)
    if not n:
        raise RuntimeError("Não foi possível obter o número de páginas do PDF")
    if not size or ("A4" not in size.group(1) and "595" not in size.group(1)):
        raise RuntimeError("O PDF não parece estar em papel A4")
    return int(n.group(1)), size.group(1)


def _org_for_pandoc(s: str) -> str:
    # Pandoc não implementa todos os modificadores de exportação HTML do Emacs.
    s = re.sub(r"(?im)^#\+(?:TITLE|AUTHOR|DATE):[^\r\n]*\r?\n", "", s)
    def patch(m):
        opt = [v for v in m.group(2).split() if v.lower() != "html-postamble:nil"]
        return m.group(1) + " " + " ".join(opt)
    return re.sub(r"(?im)^(#\+OPTIONS:)([^\r\n]*)$", patch, s)


def _cell_shading(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def _insert_identity_docx(doc, cfg: dict) -> None:
    """Mesmo quadro de identificação do LaTeX, implementado em DOCX nativo."""
    from docx.shared import Cm, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    rows = _identity_rows(cfg)
    table = doc.add_table(rows=1 + len(rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(4.25)
    table.columns[1].width = Cm(11.75)
    top = table.cell(0, 0).merge(table.cell(0, 1))
    top.text = "IDENTIFICAÇÃO ACADÊMICA — FGV"
    _cell_shading(top, "003B71")
    for paragraph in top.paragraphs:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(255, 255, 255)
    for idx, (key, value) in enumerate(rows, 1):
        for ci, txt in enumerate((key, value)):
            cell = table.cell(idx, ci)
            cell.text = txt
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.width = Cm(4.25 if ci == 0 else 11.75)
            if idx % 2 == 1:
                _cell_shading(cell, "F2F7FC")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
                    if ci == 0:
                        run.bold = True
    # Moldura azul única e margens internas discretas
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH"):
        b = OxmlElement(f"w:{edge}")
        for key, val in (("val", "single"), ("sz", "5" if edge != "insideH" else "2"),
                         ("color", "003B71" if edge != "insideH" else "D2DFEA")):
            b.set(qn("w:" + key), val)
        borders.append(b)
    tblPr.append(borders)
    mar = OxmlElement("w:tblCellMar")
    for edge, twips in (("top", "45"), ("bottom", "45"), ("left", "80"), ("right", "80")):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:w"), twips)
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tblPr.append(mar)
    # doc.add_table insere ao fim: reposicionar antes do primeiro parágrafo de conteúdo.
    body = doc._element.body
    first = next((child for child in body if child.tag == qn("w:p")), None)
    if first is not None:
        first.addprevious(table._tbl)
        # Linha em branco de altura controlada (6 pt), antes da ficha azul.
        # Preserva margens FGV e não desloca o cabeçalho ou altera a ficha.
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_before = Pt(0)
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing = Pt(2)
        table._tbl.addprevious(spacer._p)


def _docx(org: Path, logo: Path, dest: Path, root: Path, log: Path, cfg: dict) -> None:
    from docx import Document
    from docx.shared import Cm, Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    reference = root / "app_bundle/institutions/fgv/docx/reference_fgv.docx"
    if not reference.is_file():
        hits = sorted((root / "app_bundle").rglob("reference_fgv.docx"))
        if not hits:
            raise FileNotFoundError("reference_fgv.docx não encontrado no MPPG")
        reference = hits[0]
    with tempfile.TemporaryDirectory(prefix="mppg_minuta_docx_") as tmp:
        clean_org = Path(tmp) / "fonte_pandoc.org"
        clean_org.write_text(_org_for_pandoc(org.read_text(encoding="utf-8")), encoding="utf-8")
        args = ["pandoc", str(clean_org), "--from=org", "--to=docx", "--reference-doc", str(reference), "--output", str(dest)]
        proc = subprocess.run(args, text=True, capture_output=True)
        log.write_text("$ " + " ".join(args) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"Falha na exportação DOCX pelo Pandoc. Consulte {log}")
    doc = Document(dest)
    sty = doc.styles["Normal"]
    sty.font.name = "Times New Roman"
    sty.font.size = Pt(12)
    sty.paragraph_format.line_spacing = 1.5
    sty.paragraph_format.space_after = Pt(0)
    for name in ("Heading 1", "Heading 2"):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(12)
        st.font.bold = True
        st.paragraph_format.space_before = Pt(3)
        st.paragraph_format.space_after = Pt(0)
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin, sec.right_margin = Cm(3), Cm(2)
        sec.top_margin, sec.bottom_margin = Cm(3), Cm(2)
        sec.header_distance = Cm(.65)
        header = sec.header
        # MPPG_MINUTA_LOGO_PADRAO_V24
        # O reference_fgv.docx determina as medidas das imagens existentes.
        # Não redimensionar desenhos preexistentes da identidade visual FGV.
        # Se a referência não trouxer logo, aplicar 1,05 cm (padrão Org).
        if not header._element.xpath('.//w:drawing') and not header._element.xpath('.//w:pict'):
            para = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
            para.add_run().add_picture(str(logo), height=Cm(1.05))
            border = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            for k, v in (('val', 'single'), ('sz', '8'), ('color', '006BB6'), ('space', '4')):
                bottom.set(qn('w:' + k), v)
            border.append(bottom)
            para._p.get_or_add_pPr().append(border)
    _insert_identity_docx(doc, cfg)
    # Espaço vertical entre a ficha azul e a primeira seção do texto.
    first_content = next((p for p in doc.paragraphs if p.text.strip()), None)
    if first_content is not None:
        first_content.paragraph_format.space_before = Pt(8)
    doc.save(dest)


def _check(config: Path) -> dict:
    cfg, base, source, output = _load(config)
    return {"status": "OK", "config": str(config.expanduser().resolve()),
            "modelo": str(_root() / "app_bundle/institutions/fgv/templates/template_minuta.org"),
            "fonte_org": str(source), "destino": str(output / cfg['_internal']['nome']),
            "arquivo_fonte_sha256": _sha256(source)}


def _init(args) -> int:
    project = Path(args.project).expanduser().resolve()
    if project.exists() and (project / "minuta.toml").exists() and not args.force:
        raise FileExistsError("minuta.toml já existe. Utilize outra pasta ou --force (substitui SOMENTE o TOML).")
    project.mkdir(parents=True, exist_ok=True)
    if not (project / "conteudo.org").exists():
        sample = _root() / "app_bundle/institutions/fgv/templates/exemplo_conteudo_minuta.org"
        if not sample.is_file():
            raise FileNotFoundError(f"Exemplo não localizado: {sample}")
        shutil.copy2(sample, project / "conteudo.org")
    name = _safe_name(args.name)
    # tomlkit não é necessário para dados literais; json.dumps emite strings válidas em TOML.
    def q(x): return json.dumps(x, ensure_ascii=False)
    document = (f"[projeto]\nnome = {q(name)}\n\n[instituicao]\nperfil = \"fgv\"\n\n"
                f"[documento]\ntipo_documento = \"minuta\"\n"
                f"titulo_trabalho = {q(args.title)}\n"
                f"autor = {q(args.author)}\norientador = {q(args.adviser)}\n"

                f"data = {q(args.date)}\n"
                f"curso = \"Mestrado Profissional em Políticas Públicas e Governo\"\n\n"
                "[paths]\norg_conteudo = \"conteudo.org\"\n"
                "document_output_dir = \"saida\"\n\n"
                "[minuta]\nmax_paginas = 0\n\n"
                "[saidas]\nexportar_pdf = true\nexportar_docx = true\n")
    target = project / "minuta.toml"
    target.write_text(document, encoding="utf-8")
    print(json.dumps({"criado_toml": str(target), "conteudo_org": str(project / 'conteudo.org')}, ensure_ascii=False, indent=2))
    return 0


def _build(config: Path, overwrite: bool) -> int:
    cfg, _, source, output = _load(config)
    name = cfg['_internal']['nome']
    path = output / name
    if path.exists() and not overwrite:
        raise FileExistsError(f"Saída já existe: {path}. Use --overwrite para uma nova compilação.")
    # A nova compilação começa em diretório limpo temporário, sem tocar na anterior.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".minuta_{name}_", dir=output) as td:
        candidate = Path(td)
        org, logo = _render(cfg, source, candidate)
        meta = {"perfil": "fgv", "tipo": "minuta", "gerador": "academic_pipeline.minuta_runtime",
                "versao_formato": 2, "origem_toml": str(config.resolve()),
                "origem_conteudo_org": str(source), "origem_hash": _sha256(source),
                "modelo_hash": _sha256(_root() / "app_bundle/institutions/fgv/templates/template_minuta.org")}
        want_pdf = bool(cfg.get("saidas", {}).get("exportar_pdf", True))
        want_docx = bool(cfg.get("saidas", {}).get("exportar_docx", True))
        if not want_pdf and not want_docx:
            raise ValueError("Habilite pelo menos uma saída em [saidas].")
        try:
            if want_pdf:
                pdf = _run_pdf(org, _root(), candidate / "compilacao_mppg.log")
                pages, paper = _pdf_pages(pdf)
                meta.update({"pdf_paginas": pages, "pdf_papel": paper})
            if want_docx:
                _docx(org, logo, candidate / f"{name}.docx", _root(), candidate / "compilacao_docx.log", cfg)
                meta["docx_paginacao"] = "conferir no LibreOffice/Word; PDF é validado automaticamente"
        except Exception:
            # Preservar evidências diagnósticas quando o compilador nativo falhar.
            failures = output / (name + ".falha_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            shutil.copytree(candidate, failures)
            print("Arquivos e logs da falha preservados em:", failures, file=sys.stderr)
            raise
        meta["artefatos"] = {x.name: _sha256(x) for x in candidate.iterdir() if x.is_file()}
        (candidate / "manifesto.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        limit = cfg.get("minuta", {}).get("max_paginas", 0)
        # MPPG_MINUTA_SEM_LIMITE_V26: zero desativa o limite de páginas.
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("[minuta].max_paginas deve ser inteiro >= 0 (0 = sem limite)")
        if want_pdf and limit > 0 and pages > limit:
            failures = output / (name + ".revisar_paginacao_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            shutil.copytree(candidate, failures)
            raise RuntimeError(f"Minuta gerou {pages} páginas; limite={limit}. PDF para revisão preservado em {failures}. Não foi promovido à saída final.")
        if path.exists():
            backup = output / (name + ".backup_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            path.rename(backup)
            try:
                shutil.copytree(candidate, path)
            except Exception:
                if path.exists(): shutil.rmtree(path)
                backup.rename(path)
                raise
            meta["backup_anterior"] = str(backup)
        else:
            shutil.copytree(candidate, path)
    print(json.dumps({"status": "GERADO", "saida": str(path), **{k:v for k,v in meta.items() if k.startswith('pdf_') or k == 'docx_paginacao'}}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m academic_pipeline minuta", description="Minutas acadêmicas FGV em Org/PDF/DOCX, com TOML local")
    sub = parser.add_subparsers(dest="operation", required=True)
    ini = sub.add_parser("init", help="Criar minuta.toml e conteudo.org editáveis")
    ini.add_argument("--project", required=True, help="Pasta da minuta")
    ini.add_argument("--name", default="minuta_pmf_buta")
    ini.add_argument("--title", default="Minuta inicial de dissertação")
    ini.add_argument("--author", default="Gustavo M. Mendes de Tarso")
    ini.add_argument("--adviser", default="Bernardo Buta")
    ini.add_argument("--date", default="Outubro de 2026")
    ini.add_argument("--force", action="store_true", help="Sobrescrever apenas TOML existente")
    ck = sub.add_parser("check", help="Validar TOML, conteúdo e recursos institucionais")
    ck.add_argument("--config", required=True)
    bu = sub.add_parser("build", help="Compilar usando o MPPG e o modelo FGV")
    bu.add_argument("--config", required=True)
    bu.add_argument("--overwrite", action="store_true", help="Guardar versão anterior e recompilar")
    args = parser.parse_args(argv)
    try:
        if args.operation == "init": return _init(args)
        if args.operation == "check":
            print(json.dumps(_check(Path(args.config)), ensure_ascii=False, indent=2)); return 0
        if args.operation == "build": return _build(Path(args.config).expanduser().resolve(), args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERRO MINUTA MPPG: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
