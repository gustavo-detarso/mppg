#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

def load_env(root: Path) -> None:
    for candidate in [root / ".env", Path.cwd() / ".env"]:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)

def word_count(text: str) -> int:
    text = re.sub(r"(?s)#\+begin_src.*?#\+end_src", " ", text)
    text = re.sub(r"(?m)^#\+.*$", " ", text)
    text = re.sub(r"\\cite\{[^}]+\}", " ", text)
    return len(re.findall(r"\b[\wÀ-ÿ-]+\b", text))

def read_text(path: Path, default: str = "") -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return default

def backup(path: Path) -> None:
    if path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dst = path.with_name(path.name + f".bak_secional_{stamp}")
        shutil.copy2(path, dst)
        print(f"[OK] Backup: {dst}")

def parse_bib_entries(bib_text: str):
    starts = [m.start() for m in re.finditer(r"(?m)^@\w+\s*\{", bib_text)]
    entries = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(bib_text)
        raw = bib_text[start:end]
        km = re.search(r"^@\w+\s*\{\s*([^,\s]+)", raw)
        if not km:
            continue
        key = km.group(1).strip()

        def field(name):
            m = re.search(rf"(?is)\b{name}\s*=\s*[\{{\"](.+?)[\}}\"]\s*,", raw)
            if not m:
                return ""
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            val = val.replace("{", "").replace("}", "")
            return val

        title = field("title") or key
        year = field("year")
        author = field("author")
        entries.append({
            "key": key,
            "title": title,
            "year": year,
            "author": author,
        })
    return entries

def select_context(corpus: str, keywords, max_chars=55000) -> str:
    parts = re.split(r"\n\s*\n", corpus)
    scored = []
    for p in parts:
        low = p.lower()
        score = sum(low.count(k.lower()) for k in keywords)
        if score:
            scored.append((score, len(p), p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    total = 0
    for _, _, p in scored:
        p = p.strip()
        if not p:
            continue
        if total + len(p) > max_chars:
            continue
        out.append(p)
        total += len(p)
        if total >= max_chars:
            break
    if not out:
        return corpus[:max_chars]
    return "\n\n".join(out)

def openai_chat(prompt: str, max_tokens=3600) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente no .env.")

    model = (
        os.environ.get("OPENAI_MODEL_ARTIGO_LONGO")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )

    base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions").strip()
    if not base.endswith("/chat/completions"):
        base = base.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "temperature": 0.2,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um pesquisador acadêmico brasileiro. "
                    "Escreva em português formal, em padrão de artigo científico ABNT/FGV. "
                    "Use linguagem substantiva, analítica e metodológica. "
                    "Não produza texto curto, genérico ou ensaístico. "
                    "Use citações LaTeX no formato \\cite{chave} quando forem fornecidas chaves BibTeX."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Erro HTTP OpenAI {e.code}: {detail}") from e

    return data["choices"][0]["message"]["content"].strip()

def sanitize_section(text: str) -> str:
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def make_evidence_matrix(entries):
    rows = []
    rows.append("* Matriz de evidências dos 20 estudos incluídos")
    rows.append("")
    rows.append("A matriz a seguir consolida o corpus final de 20 estudos com full text recuperado, vinculando cada referência ao seu papel na síntese analítica do artigo.")
    rows.append("")
    rows.append("| Nº | Estudo | Ano | Papel na síntese | Citação |")
    rows.append("|----+--------+-----+-----------------+---------|")
    for i, e in enumerate(entries[:20], 1):
        title = e["title"][:95].replace("|", "/")
        year = e["year"] or "s.d."
        if i == 1:
            role = "Eixo nacional sobre ATESTMED, concessão documental e efeitos institucionais."
        elif "tele" in e["title"].lower() or "video" in e["title"].lower():
            role = "Evidência sobre telemedicina, consulta remota ou mediação digital da avaliação."
        elif "disability" in e["title"].lower() or "incap" in e["title"].lower():
            role = "Evidência sobre avaliação de incapacidade, funcionalidade e certificação."
        elif "whodas" in e["title"].lower() or "classification" in e["title"].lower() or "functioning" in e["title"].lower():
            role = "Evidência sobre instrumentos funcionais, ICF/WHODAS e padronização avaliativa."
        else:
            role = "Evidência complementar para desenho decisório, triagem, aceitabilidade ou implementação."
        rows.append(f"| {i} | {title} | {year} | {role} | \\cite{{{e['key']}}} |")
    rows.append("")
    return "\n".join(rows)

def make_prompt(title, target_words, keywords, entries, refs_md, stats_md, corpus, extra_instruction=""):
    keys = ", ".join(e["key"] for e in entries[:20])
    context = select_context(corpus, keywords)
    return f"""
Escreva a seção "{title}" de um artigo científico longo sobre ATESTMED, saúde digital e decisão baseada em evidências na Perícia Médica Federal.

Tamanho mínimo da seção: {target_words} palavras.
Não escreva resumo. Não escreva em tópicos, salvo quando indispensável.
Use parágrafos densos, com encadeamento argumentativo.
Use o termo "full text" ao tratar da curadoria de textos completos quando pertinente.
Cite estudos usando apenas chaves BibTeX existentes, no formato \\cite{{chave}}.
Chaves BibTeX disponíveis: {keys}

Instrução adicional da seção:
{extra_instruction}

Estatísticas/metodologia PRISMA disponíveis:
{stats_md[:12000]}

Referências incluídas e metadados:
{refs_md[:18000]}

Excertos selecionados do corpus full text:
{context}
""".strip()

def extract_org_header(old_org: str) -> str:
    m = re.search(r"(?m)^\* ", old_org)
    if m:
        header = old_org[:m.start()].rstrip()
    else:
        header = old_org.strip()

    if not header:
        header = "\n".join([
            "#+title: ATESTMED, saúde digital e decisão baseada em evidências",
            "#+language: pt_BR",
            "#+options: toc:nil num:t",
        ])

    # Remove bibliografias antigas no cabeçalho apenas se forem manifestamente duplicadas.
    return header.rstrip() + "\n\n"

def compile_pdf(out_dir: Path, org_path: Path) -> None:
    export_el = out_dir / "artigo_final_atestmed_abnt_export_pdf.el"

    commands = []
    if export_el.exists():
        commands.append(["emacs", "--batch", "-l", str(export_el)])

    commands.append([
        "emacs", "--batch",
        "--eval",
        f'(progn (require (quote org)) (find-file "{org_path}") (org-latex-export-to-pdf))'
    ])

    for cmd in commands:
        print("\n$", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception as e:
            print(f"[AVISO] Falha na tentativa de exportação: {e}")

    raise RuntimeError("Não foi possível compilar o PDF por Emacs/Org.")

def try_docx(org_path: Path, docx_path: Path, bib_path: Path) -> None:
    cmd = ["pandoc", str(org_path), "-o", str(docx_path), "--bibliography", str(bib_path)]
    print("\n$", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[AVISO] DOCX via pandoc não gerado: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", required=True)
    ap.add_argument("--min-palavras", type=int, default=8500)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    load_env(root)

    art = Path(args.art_dir)
    out = art / "output"
    out.mkdir(parents=True, exist_ok=True)

    org_path = out / "artigo_final_atestmed_abnt.org"
    bib_path = out / "artigo_final_atestmed_abnt.bib"
    pdf_path = out / "artigo_final_atestmed_abnt.pdf"
    docx_path = out / "artigo_final_atestmed_abnt.docx"
    json_path = out / "artigo_final_atestmed_abnt.document.json"

    refs_md_path = art / "dados_prisma/artigo_longo_fulltext/referencias_incluidas_20.md"
    stats_md_path = art / "dados_prisma/artigo_longo_fulltext/estatisticas_prisma_fulltext.md"
    corpus_path = art / "dados_prisma/artigo_longo_fulltext/corpus_fulltext_compilado.md"

    refs_md = read_text(refs_md_path)
    stats_md = read_text(stats_md_path)
    corpus = read_text(corpus_path)

    if not bib_path.exists():
        src_bib = art / "dados_prisma/relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas.bib"
        if src_bib.exists():
            shutil.copy2(src_bib, bib_path)

    bib_text = read_text(bib_path)
    entries = parse_bib_entries(bib_text)

    if len(entries) < 20:
        raise SystemExit(f"ERRO: BibTeX com {len(entries)} entradas; esperado mínimo de 20.")

    old_org = read_text(org_path)
    backup(org_path)
    backup(pdf_path)
    backup(docx_path)
    backup(json_path)

    sections_plan = [
        (
            "Introdução",
            1100,
            ["ATESTMED", "saúde digital", "Perícia Médica Federal", "benefícios por incapacidade"],
            "Apresente o problema público, a relevância previdenciária, a tensão entre escala administrativa e qualidade decisória, e a pergunta do artigo.",
        ),
        (
            "Referencial analítico: decisão baseada em evidências e análise de políticas públicas",
            1000,
            ["evidence", "policy", "decision", "public policy", "assessment"],
            "Articule decisão baseada em evidências, análise de políticas públicas, legitimidade, eficiência, efetividade e desenho institucional.",
        ),
        (
            "Método: revisão estruturada PRISMA com curadoria de full text",
            1300,
            ["PRISMA", "full text", "included", "screening", "eligibility", "exclusion"],
            "Descreva o procedimento metodológico, bases, triagem, elegibilidade, inclusão, full text garantido, curadoria assistida por IA e decisão humana final.",
        ),
        (
            "Resultados da revisão estruturada",
            1400,
            ["disability", "telemedicine", "work disability", "sickness benefit", "ICF", "WHODAS"],
            "Sintetize os achados por eixos: ATESTMED, avaliação documental, telemedicina, avaliação funcional, IA, certificação médica e retorno ao trabalho.",
        ),
        (
            "Síntese temática das evidências",
            1300,
            ["telehealth", "video consultation", "capacity", "functioning", "certification", "acceptability"],
            "Integre os estudos em eixos substantivos, apontando convergências, divergências e implicações para a Perícia Médica Federal.",
        ),
        (
            "Discussão",
            1800,
            ["ATESTMED", "telemedicine", "artificial intelligence", "disability assessment", "public administration"],
            "Discuta criticamente validade, risco de erro decisório, assimetria informacional, padronização, auditoria, governança algorítmica e capacidade estatal.",
        ),
        (
            "Proposta de redesenho do fluxo decisório do ATESTMED",
            1500,
            ["workflow", "triage", "risk", "decision", "monitoring", "audit"],
            "Proponha um fluxo decisório redesenhado com triagem por risco, critérios documentais, encaminhamento para perícia presencial/remota, auditoria e retroalimentação por evidências.",
        ),
        (
            "Indicadores, monitoramento e avaliação",
            900,
            ["indicator", "monitoring", "evaluation", "outcome", "quality"],
            "Defina indicadores de processo, resultado, qualidade decisória, equidade, tempo de resposta, reversão, judicialização, auditoria e aprendizado institucional.",
        ),
        (
            "Limitações",
            650,
            ["limitation", "bias", "evidence", "full text"],
            "Declare limitações da revisão, heterogeneidade dos estudos, uso de proxies, transferibilidade internacional e lacunas empíricas brasileiras.",
        ),
        (
            "Conclusão",
            750,
            ["ATESTMED", "evidence", "digital health", "decision"],
            "Conclua retomando a pergunta, os achados, a contribuição do artigo e a recomendação central para a Perícia Médica Federal.",
        ),
    ]

    generated_sections = []
    document_json_sections = []

    for title, target, keywords, instruction in sections_plan:
        print(f"\n[ETAPA] Gerando seção: {title} ({target} palavras-alvo)")
        prompt = make_prompt(title, target, keywords, entries, refs_md, stats_md, corpus, instruction)
        text = sanitize_section(openai_chat(prompt, max_tokens=4300))
        wc = word_count(text)
        print(f"[OK] {title}: {wc} palavras aproximadas")
        generated_sections.append(f"* {title}\n\n{text}\n")
        document_json_sections.append({"title": title, "text": text, "words": wc})

    matrix = make_evidence_matrix(entries)

    body = "\n".join(generated_sections[:4])
    body += "\n" + matrix + "\n\n"
    body += "\n".join(generated_sections[4:])

    total = word_count(body)
    attempts = 0
    while total < args.min_palavras and attempts < 4:
        attempts += 1
        remaining = args.min_palavras - total + 600
        print(f"\n[ETAPA] Complemento analítico {attempts}: faltam cerca de {args.min_palavras - total} palavras")
        prompt = make_prompt(
            "Complemento analítico de integração das evidências",
            min(1800, max(900, remaining)),
            ["ATESTMED", "full text", "PRISMA", "telemedicine", "disability assessment", "decision"],
            entries,
            refs_md,
            stats_md,
            corpus,
            "Aprofunde lacunas argumentativas do artigo. Não repita literalmente seções anteriores. Integre os 20 estudos, a matriz de evidências, PRISMA/full text e a proposta decisória.",
        )
        extra = sanitize_section(openai_chat(prompt, max_tokens=4300))
        body += f"\n* Complemento analítico de integração das evidências {attempts}\n\n{extra}\n"
        total = word_count(body)
        print(f"[OK] Total parcial: {total} palavras")

    body = re.sub(r"\b14\s+estudos\b", "20 estudos", body, flags=re.I)
    body = re.sub(r"\bquatorze\s+estudos\b", "20 estudos", body, flags=re.I)

    refs_block = """
* Referências

\\bibliographystyle{abntex2-alf}
\\bibliography{artigo_final_atestmed_abnt}
""".strip()

    header = extract_org_header(old_org)
    final_org = header + body.strip() + "\n\n" + refs_block + "\n"

    org_path.write_text(final_org, encoding="utf-8")

    doc = {
        "title": "ATESTMED, saúde digital e decisão baseada em evidências",
        "generated_by": "gerar_artigo_longo_fulltext_secional.py v1.14",
        "target_min_words": args.min_palavras,
        "actual_words_org_body": word_count(final_org),
        "bib_entries": len(entries),
        "sections": document_json_sections,
    }
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] ORG longo regravado:", org_path)
    print("[OK] document.json longo regravado:", json_path)
    print("[OK] Palavras aproximadas no ORG:", word_count(final_org))

    if args.compile:
        compile_pdf(out, org_path)
        try_docx(org_path, docx_path, bib_path)

if __name__ == "__main__":
    main()
