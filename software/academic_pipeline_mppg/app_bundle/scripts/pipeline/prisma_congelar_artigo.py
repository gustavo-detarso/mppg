#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, re, shutil, subprocess, sys
from pathlib import Path
try:
    import tomllib
except Exception:
    tomllib = None

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIGO_DIR = Path.cwd() / "artigo"
DEFAULT_AUTOR = "Gustavo M. Mendes de Tarso"
DEFAULT_PROFESSOR = "Marcos Aurélio Pereira Valadão"

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def file_info(path, origem=None):
    st=path.stat()
    return {"arquivo":path.name,"caminho":str(path),"origem":origem,"tamanho_bytes":st.st_size,"mtime_iso":dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),"sha256":sha256_file(path)}

def copy_one(src,dest_dir,required,copied,missing):
    if not src.exists():
        if required: raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {src}")
        missing.append(str(src)); return
    dest=dest_dir/src.name; shutil.copy2(src,dest); copied.append(file_info(dest,str(src)))

def esc(v): return str(v or "").replace("\\","\\\\").replace('"','\\"')
def arr(items): return "["+", ".join(f'"{esc(x)}"' for x in items)+"]"
def slug(v,max_len=70):
    import unicodedata
    t=unicodedata.normalize("NFKD",v or "")
    t="".join(ch for ch in t if not unicodedata.combining(ch))
    t=re.sub(r"[^a-zA-Z0-9]+","_",t.lower()).strip("_")
    return (re.sub(r"_+","_",t)[:max_len].strip("_") or "artigo_final_abnt")

def read_toml(path):
    if not path or not path.exists() or tomllib is None: return {}
    with path.open("rb") as f: return tomllib.load(f)

def resolve_relative(base,path):
    p=Path(str(path)).expanduser()
    return p.resolve() if p.is_absolute() else (base.parent/p).resolve()

def dados_path_from_config(config):
    if not config or not config.exists(): return None
    d=read_toml(config)
    for item in [d.get("pesquisa",{}).get("dados_pesquisa_path"), d.get("busca_prisma",{}).get("estrategia_busca_path"), d.get("busca_prisma",{}).get("criterios_path")]:
        if item: return resolve_relative(config,str(item))
    return None

def deep_find(data,names):
    wanted={n.lower() for n in names}; found=None
    def walk(obj):
        nonlocal found
        if found is not None: return
        if isinstance(obj,dict):
            for k,v in obj.items():
                if str(k).lower() in wanted and v not in (None,"",[],{}): found=v; return
            for v in obj.values(): walk(v)
        elif isinstance(obj,list):
            for x in obj: walk(x)
    walk(data); return found

def as_text(v):
    if v is None: return ""
    if isinstance(v,list): return "; ".join(str(x).strip() for x in v if str(x).strip())
    if isinstance(v,dict): return json.dumps(v,ensure_ascii=False,indent=2)
    return str(v).strip()

def as_list(v):
    if v is None: return []
    if isinstance(v,list): return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[;\n,]+",str(v)) if x.strip()]

def values(meta,config,fallback):
    c={"dados_pesquisa":meta,"config":config}
    titulo=as_text(deep_find(c,["titulo_trabalho","titulo","título","title","nome_artigo","paper_title"]))
    tema=as_text(deep_find(c,["tema","assunto","topic"]))
    recorte=as_text(deep_find(c,["recorte","escopo","delimitacao","delimitação"]))
    objetivo=as_text(deep_find(c,["objetivo","objective","objetivo_geral"]))
    pergunta=as_text(deep_find(c,["pergunta_pesquisa","pergunta_de_pesquisa","research_question","questao_pesquisa","questão_pesquisa"]))
    hipotese=as_text(deep_find(c,["hipotese","hipótese","hypothesis"]))
    tese=as_text(deep_find(c,["tese_central","tese","argumento_central","central_claim"]))
    palavras=as_list(deep_find(c,["palavras_chave","keywords","descritores"]))
    estrutura=as_list(deep_find(c,["estrutura_desejada","estrutura","secoes","seções"]))
    argumentos=as_list(deep_find(c,["argumentos_obrigatorios","argumentos","pontos_obrigatorios"]))
    orientacoes=as_text(deep_find(c,["orientacoes","orientações","instrucoes","instruções","orientacoes_especificas","orientações_específicas"]))
    if not titulo: titulo=tema or fallback.replace("_"," ").title()
    if not tema: tema=titulo
    if not tese: tese=objetivo or pergunta or f"Síntese analítica baseada nas evidências selecionadas sobre {tema}."
    if not estrutura: estrutura=["Introdução","Método","Resultados","Discussão","Limitações","Conclusão"]
    return locals()

def render_template(template_path,repl):
    s=template_path.read_text(encoding="utf-8")
    for k,v in repl.items(): s=s.replace("{{"+k+"}}",str(v))
    return s

def generate_toml(args,dados_dir):
    root=Path(args.root_dir).resolve() if args.root_dir else PACKAGE_ROOT
    artigo_dir=Path(args.artigo_dir).resolve()
    out_dir=artigo_dir/"output"
    csl=Path(args.csl_path).resolve() if args.csl_path else root/"app_bundle/templates/csl/associacao-brasileira-de-normas-tecnicas.csl"
    prisma_config=Path(args.prisma_config).resolve() if args.prisma_config else None
    meta_path=Path(args.dados_pesquisa_path).resolve() if args.dados_pesquisa_path else dados_path_from_config(prisma_config)
    meta=read_toml(meta_path) if meta_path and meta_path.exists() else {}
    config=read_toml(prisma_config) if prisma_config and prisma_config.exists() else {}
    vals=values(meta,config,args.artigo_prefix or "artigo_final_abnt")
    prefix=args.artigo_prefix or slug("artigo_final_"+vals["titulo"])
    if not prefix.startswith("artigo"): prefix="artigo_final_"+prefix
    prefix=slug(prefix,80)
    bib=dados_dir/"relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas.bib"
    csv=dados_dir/"relatorio_prisma_prisma_fluxo_pmf.referencias_incluidas_seminario.csv"
    pdf=dados_dir/"relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_final.pdf"
    missing=[p for p in [bib,csv,pdf] if not p.exists()]
    if missing: raise FileNotFoundError("Arquivos obrigatórios ausentes para gerar TOML:\n"+"\n".join(map(str,missing)))
    inline="\n".join([
        "Produza um paper acadêmico em português, no layout paper_fgv, com citações autor-data e referências finais em ABNT.",
        "",
        f"Arquivo estruturado de pesquisa: {meta_path or ''}",
        f"Tema: {vals['tema']}",
        f"Recorte: {vals['recorte']}",
        f"Objetivo: {vals['objetivo']}",
        f"Pergunta de pesquisa: {vals['pergunta']}",
        f"Tese central: {vals['tese']}",
        "",
        "Orientações específicas extraídas do arquivo estruturado:",
        vals["orientacoes"],
        "",
        "Use exclusivamente os estudos selecionados no PRISMA final e congelados em dados_prisma.",
        "Não invente referências, autores, DOI, periódicos, dados empíricos ou conclusões.",
        "O método deve explicitar revisão estruturada com fluxo PRISMA, curadoria assistida por IA e revisão humana final.",
        "Não crie seção manual de Referências. O renderizador do programa deve inserir a bibliografia automaticamente.",
    ]).replace("'''","’")
    template=root/"app_bundle/templates/prisma_artigo_generico.toml.template"
    if not template.exists(): raise FileNotFoundError(f"Template não encontrado: {template}")
    repl={
        "PREFIX":esc(prefix), "OPENAI_MODEL":esc(args.openai_model), "OUT_DIR":esc(out_dir),
        "WORK_DIR":esc(artigo_dir/".academic_pipeline/work"), "CACHE_DIR":esc(artigo_dir/".academic_pipeline/cache"),
        "ORIENTACOES_PATHS":arr([str(meta_path)] if meta_path else []), "ORIENTACOES_INLINE":inline,
        "DADOS_ATIVO":str(bool(meta_path)).lower(), "DADOS_PESQUISA_PATH":esc(meta_path or ""),
        "PRISMA_CONFIG_PATH":esc(prisma_config or ""), "DADOS_DIR":esc(dados_dir), "TEMA":esc(vals["tema"]),
        "RECORTE":esc(vals["recorte"]), "OBJETIVO":esc(vals["objetivo"]), "PERGUNTA":esc(vals["pergunta"]),
        "HIPOTESE":esc(vals["hipotese"]), "PALAVRAS":arr(vals["palavras"]),
        "GERAR_PALAVRAS_IA":str(not bool(vals["palavras"])).lower(), "TESE":esc(vals["tese"]),
        "ESTRUTURA":arr(vals["estrutura"]), "ARGUMENTOS":arr(vals["argumentos"]), "PROFESSOR":esc(args.professor),
        "AUTOR":esc(args.autor), "TITULO":esc(vals["titulo"]), "BIB_PATH":esc(bib), "CSL_PATH":esc(csl),
        "ORG_LATEX_CLASS_INIT":esc(root/"app_bundle/misc/academic-writing.el"), "LATEX_EXTRA_PATH":esc(root/"app_bundle/misc/fgv"),
        "FGV_LOGO_PATH":esc(root/"app_bundle/misc/fgv.png"), "PROMPT_GLOBAL":esc(root/"app_bundle/prompts/global/orientacao_geral_execucao.txt"),
        "PROMPT_PAPER":esc(root/"app_bundle/prompts/document/paper.txt"),
    }
    toml_out=Path(args.toml_output).resolve() if args.toml_output else artigo_dir/"artigo_final_abnt.toml"
    toml_out.parent.mkdir(parents=True,exist_ok=True)
    toml_out.write_text(render_template(template,repl),encoding="utf-8")
    print(f"[OK] TOML do artigo gerado: {toml_out}")
    if meta_path: print(f"[OK] Dados estruturados usados: {meta_path}")
    if not csl.exists(): print(f"[WARN] CSL ABNT não encontrado: {csl}")
    return toml_out

def freeze(args):
    out=Path(args.out_dir).resolve(); prefix=args.prefix or out.name
    artigo_dir=Path(args.artigo_dir).resolve()
    dest=Path(args.dest_dir).resolve() if args.dest_dir else artigo_dir/"dados_prisma"
    dest.mkdir(parents=True,exist_ok=True)
    required=[f"{prefix}.referencias_incluidas.bib",f"{prefix}.referencias_incluidas_seminario.csv",f"{prefix}.triagem_humana.csv",f"{prefix}.relatorio_prisma_final.pdf"]
    optional=[f"{prefix}.curadoria_ia_referencias.xlsx",f"{prefix}.relatorio_prisma_preliminar.pdf",f"{prefix}.diagrama_prisma.png",f"{prefix}.diagrama_prisma_contagens.json",f"{prefix}.busca_prisma_log.json",f"{prefix}.triagem_titulo_resumo.csv",f"{prefix}.triagem_titulo_resumo.xlsx",f"{prefix}.curadoria_ia_resumo.txt",f"{prefix}.curadoria_ia_log.json"]
    copied=[]; missing=[]
    for n in required: copy_one(out/n,dest,True,copied,missing)
    for n in optional: copy_one(out/n,dest,False,copied,missing)
    (dest/"MANIFESTO_SHA256.txt").write_text("\n".join(f"{x['sha256']}  {x['arquivo']}" for x in sorted(copied,key=lambda i:i["arquivo"]))+"\n",encoding="utf-8")
    (dest/"ARQUIVOS_CONGELADOS.txt").write_text("\n".join(["Arquivo\tTamanho_bytes\tModificado_em\tSHA256"]+[f"{x['arquivo']}\t{x['tamanho_bytes']}\t{x['mtime_iso']}\t{x['sha256']}" for x in sorted(copied,key=lambda i:i["arquivo"])])+"\n",encoding="utf-8")
    toml_out=None
    if args.gerar_toml_artigo or args.gerar_artigo_final:
        toml_out=generate_toml(args,dest)
    manifest={"gerado_em":dt.datetime.now().isoformat(timespec="seconds"),"out_dir_origem":str(out),"artigo_dir":str(artigo_dir),"destino":str(dest),"prefixo_prisma":prefix,"arquivos":sorted(copied,key=lambda i:i["arquivo"]),"opcionais_ausentes":missing,"toml_artigo_gerado":str(toml_out) if toml_out else None,"prisma_config":args.prisma_config,"dados_pesquisa_path_override":args.dados_pesquisa_path}
    (dest/"MANIFESTO_ARTIGO.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[OK] Insumos congelados em: {dest}")
    print(f"[OK] Arquivos copiados: {len(copied)}")
    print(f"[OK] Manifesto SHA256: {dest/'MANIFESTO_SHA256.txt'}")
    print(f"[OK] Manifesto JSON: {dest/'MANIFESTO_ARTIGO.json'}")
    if args.gerar_artigo_final:
        if not toml_out: raise RuntimeError("TOML não gerado.")
        print(f"[ETAPA] Gerando artigo final com: {toml_out}")
        if args.pipeline_script:
            pipeline = Path(args.pipeline_script).resolve()
            command = [sys.executable, str(pipeline), "--config", str(toml_out)]
        else:
            command = [sys.executable, "-m", "academic_pipeline", "--config", str(toml_out)]
        proc=subprocess.run(command)
        if proc.returncode: raise SystemExit(proc.returncode)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True); p.add_argument("--artigo-dir",default=str(DEFAULT_ARTIGO_DIR)); p.add_argument("--dest-dir")
    p.add_argument("--prefix"); p.add_argument("--gerar-toml-artigo",action="store_true"); p.add_argument("--gerar-artigo-final",action="store_true")
    p.add_argument("--toml-output"); p.add_argument("--root-dir"); p.add_argument("--csl-path"); p.add_argument("--prisma-config"); p.add_argument("--dados-pesquisa-path")
    p.add_argument("--artigo-prefix"); p.add_argument("--autor",default=DEFAULT_AUTOR); p.add_argument("--professor",default=DEFAULT_PROFESSOR)
    p.add_argument("--openai-model",default="gpt-4.1-mini"); p.add_argument("--pipeline-script")
    args=p.parse_args(argv); freeze(args); return 0
if __name__=="__main__": raise SystemExit(main())
