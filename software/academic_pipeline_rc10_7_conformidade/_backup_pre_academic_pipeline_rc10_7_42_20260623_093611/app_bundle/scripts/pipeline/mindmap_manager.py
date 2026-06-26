#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from document_model import AcademicDocument, FigureSpec
from utils import write_text, slugify, shorten_text
from prompt_manager import load_prompt_bundle


class MindmapOutput(BaseModel):
    plantuml: str


# Paleta padrão extraída do modelo visual FGV já aprovado pelo usuário:
# raiz azul clara, ramos principais verdes, subcategorias amarelas e folhas rosa/salmão.
# Níveis mais profundos repetem o rosa para manter a identidade visual do mapa antigo.
DEFAULT_MINDMAP_LEVEL_COLORS = [
    "#D9EAF7",  # raiz: azul claro
    "#DFF2E1",  # nível 1: verde claro
    "#FFF2CC",  # nível 2: amarelo claro
    "#F8D7DA",  # nível 3: rosa/salmão claro
    "#F8D7DA",  # nível 4+: mantém rosa/salmão
    "#F8D7DA",
    "#F8D7DA",
]


def _normalize_level_colors(raw: Any) -> list[str]:
    """Resolve a paleta de cores por nível do mapa mental.

    Aceita tanto lista TOML:
        cores_niveis = ["#D9EAF7", "#DFF2E1", ...]

    quanto tabela TOML:
        [mapa_mental.cores_por_nivel]
        nivel_0 = "#D9EAF7"
        nivel_1 = "#DFF2E1"
    """
    if isinstance(raw, list):
        vals = [str(x).strip() for x in raw if str(x).strip()]
        return vals or DEFAULT_MINDMAP_LEVEL_COLORS[:]
    if isinstance(raw, dict):
        items: list[tuple[int, str]] = []
        for k, v in raw.items():
            m = re.search(r"(\d+)", str(k))
            if not m:
                continue
            color = str(v).strip()
            if color:
                items.append((int(m.group(1)), color))
        if items:
            ordered = [c for _, c in sorted(items, key=lambda x: x[0])]
            return ordered or DEFAULT_MINDMAP_LEVEL_COLORS[:]
    return DEFAULT_MINDMAP_LEVEL_COLORS[:]


def colorize_mindmap_plantuml(code: str, cfg: dict[str, Any]) -> str:
    """Aplica cores determinísticas por nível aos nós do PlantUML mindmap.

    O PlantUML aceita cores de nó no formato:
        *[#D9EAF7] Raiz
        **[#DFF2E1] Nível 1

    Essa etapa evita depender da IA para estilizar o mapa. Por padrão,
    também substitui cores preexistentes para manter a paleta FGV aprovada
    de forma consistente. Para preservar cores vindas da IA/usuário, defina:
        [mapa_mental]
        sobrescrever_cores_existentes = false
    """
    mm = mindmap_config(cfg)
    if not bool(mm.get("colorido", mm.get("colorir", True))):
        return code

    raw_colors = mm.get("cores_niveis") or mm.get("level_colors") or mm.get("cores_por_nivel")
    colors = _normalize_level_colors(raw_colors)
    overwrite_existing = bool(mm.get("sobrescrever_cores_existentes", mm.get("forcar_cores", True)))

    out: list[str] = []
    for line in (code or "").splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if not stripped or stripped.startswith("@") or stripped.startswith("'"):
            out.append(line)
            continue
        m = re.match(r"^([*+-]+)(\s*)(.*)$", stripped)
        if not m:
            out.append(line)
            continue
        markers, spacing, rest = m.groups()
        level = max(0, len(markers) - 1)
        color = colors[min(level, len(colors) - 1)]
        rest_clean = rest
        # Substitui/remover cor já informada pela IA/usuário: *[#ABCDEF] Texto
        if rest.lstrip().startswith("[#"):
            if not overwrite_existing:
                out.append(line)
                continue
            rest_clean = re.sub(r"^\s*\[#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})\]\s*", "", rest, count=1)
        sep = spacing if spacing else " "
        out.append(f"{indent}{markers}[{color}]{sep}{rest_clean}".rstrip())
    return "\n".join(out).strip() + "\n"


def mindmap_config(cfg: dict[str, Any]) -> dict[str, Any]:
    mm = cfg.get("mapa_mental", {}) if isinstance(cfg.get("mapa_mental"), dict) else {}
    return mm


def should_generate_mindmap(cfg: dict[str, Any]) -> bool:
    mm = mindmap_config(cfg)
    return bool(mm.get("gerar") or mm.get("enabled") or mm.get("ativo"))


def sanitize_plantuml(text: str) -> str:
    raw = re.sub(r"(?is)^```(?:plantuml)?\s*|```$", "", text or "").strip()
    m = re.search(r"(?is)@startmindmap.*?@endmindmap", raw)
    if m:
        raw = m.group(0)
    if not raw.lower().startswith("@startmindmap"):
        raw = "@startmindmap\n" + raw
    if not raw.lower().endswith("@endmindmap"):
        raw += "\n@endmindmap"
    return raw.strip() + "\n"


def build_mindmap_prompt(doc: AcademicDocument, cfg: dict[str, Any] | None = None) -> str:
    prompt_bundle = load_prompt_bundle(cfg or {}, "mindmap") if cfg else None
    prompt_extras = (prompt_bundle.text if prompt_bundle else "") or "Nenhuma diretiva complementar carregada."
    return f"""
Gere um mapa mental em PlantUML mindmap para o documento canônico abaixo.
Regras: retorne apenas código PlantUML @startmindmap ... @endmindmap; use rótulos curtos; não use citações; no máximo 45 nós. Não use blocos de estilo complexos; as cores serão aplicadas automaticamente pelo pipeline.

Diretivas complementares carregadas pelo prompt bank:
{prompt_extras}

Documento:
{shorten_text(doc.model_dump_json(), 45000)}
""".strip()




def mindmap_artifact_paths(cfg: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Resolve caminhos padrão dos artefatos do mapa mental.

    Mantém compatibilidade com a configuração histórica:
      [mapa_mental]
      diretorio_imagens = "images"
      arquivo = "mapa_mental"
      formato = "png"
    """
    mm = mindmap_config(cfg)
    images_dir = output_dir / str(mm.get("diretorio_imagens") or "images")
    stem = slugify(str(mm.get("arquivo") or "mapa_mental"))
    formato = str(mm.get("formato") or "png").lower().lstrip(".")
    if formato not in {"png", "svg"}:
        formato = "png"
    return {
        "images_dir": images_dir,
        "puml": images_dir / f"{stem}.puml",
        "image": images_dir / f"{stem}.{formato}",
    }


def attach_mindmap_figure(doc: AcademicDocument, cfg: dict[str, Any], output_dir: Path, image_path: Path) -> None:
    """Anexa/atualiza a figura do mapa mental no document_model."""
    mm = mindmap_config(cfg)
    title = str(mm.get("titulo") or "Mapa mental dos textos analisados")
    rel = str(image_path.relative_to(output_dir)).replace("\\", "/")
    doc.figures = [f for f in doc.figures if f.id != "mapa_mental"]
    doc.figures.append(
        FigureSpec(
            id="mapa_mental",
            title=title,
            path=rel,
            placement="after_references",
            page_break_before=True,
            page_break_after=False,
        )
    )


def render_existing_mindmap(cfg: dict[str, Any], doc: AcademicDocument, output_dir: Path) -> dict[str, Any] | None:
    """Recolore e renderiza um .puml existente, sem chamar IA.

    Retorna diagnóstico quando o .puml existe. Retorna None se não há .puml.
    """
    paths = mindmap_artifact_paths(cfg, output_dir)
    puml = paths["puml"]
    if not puml.exists():
        return None
    code = sanitize_plantuml(puml.read_text(encoding="utf-8", errors="ignore"))
    code = colorize_mindmap_plantuml(code, cfg)
    write_text(puml, code)
    img, err = render_plantuml(puml, cfg)
    if err and bool(mindmap_config(cfg).get("falhar_se_nao_renderizar", True)):
        raise RuntimeError(err)
    if img:
        attach_mindmap_figure(doc, cfg, output_dir, img)
    return {
        "mode": "existing_puml",
        "puml_path": str(puml),
        "image_path": str(img) if img else None,
        "error": err,
    }


def render_or_generate_mindmap(
    client: Any | None,
    model: str,
    cfg: dict[str, Any],
    doc: AcademicDocument,
    output_dir: Path,
    *,
    force_regenerate: bool = False,
    prefer_existing: bool = True,
) -> dict[str, Any] | None:
    """Renderiza ou gera o mapa mental como etapa autônoma.

    - Se houver .puml existente e ``force_regenerate`` for falso, apenas recolore e renderiza.
    - Se não houver .puml, ou se ``force_regenerate`` for verdadeiro, chama IA para gerar novo PlantUML.
    - Atualiza ``doc.figures`` para apontar para a imagem final.
    """
    if prefer_existing and not force_regenerate:
        diag = render_existing_mindmap(cfg, doc, output_dir)
        if diag is not None:
            return diag
    if client is None:
        raise RuntimeError(
            "Mapa mental ainda não possui .puml existente. "
            "Para gerar novo mapa, rode com OPENAI_API_KEY disponível ou use --forcar-regeneracao-mapa-mental."
        )
    diag = generate_and_attach_mindmap(client, model, cfg, doc, output_dir)
    if diag is not None:
        diag["mode"] = "generated_with_ai"
    return diag

def render_plantuml(puml_path: Path, cfg: dict[str, Any]) -> tuple[Path | None, str | None]:
    mm = mindmap_config(cfg)
    formato = str(mm.get("formato") or "png").lower().lstrip(".")
    if formato not in {"png", "svg"}:
        formato = "png"
    output = puml_path.with_suffix("." + formato)
    env = os.environ.copy()
    env["PLANTUML_LIMIT_SIZE"] = str(int(mm.get("plantuml_limit_size") or 8192))
    cmds: list[list[str]] = []
    if shutil.which("plantuml"):
        cmds.append(["plantuml", f"-t{formato}", puml_path.name])
    jar = mm.get("plantuml_jar_path") or (cfg.get("documento", {}) if isinstance(cfg.get("documento"), dict) else {}).get("plantuml_jar_path") or os.getenv("PLANTUML_JAR")
    if jar and shutil.which("java") and Path(str(jar)).expanduser().exists():
        cmds.append(["java", "-DPLANTUML_LIMIT_SIZE=" + env["PLANTUML_LIMIT_SIZE"], "-jar", str(Path(str(jar)).expanduser()), f"-t{formato}", puml_path.name])
    if not cmds:
        return None, "PlantUML não encontrado. Informe [mapa_mental].plantuml_jar_path ou instale plantuml."
    errors = []
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=str(puml_path.parent), text=True, capture_output=True, env=env)
        if proc.returncode == 0 and output.exists():
            return output, None
        errors.append(" ".join(cmd) + "\n" + proc.stderr)
    return (output if output.exists() else None), "\n---\n".join(errors)



def mindmap_output_paths(cfg: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Retorna caminhos previstos do mapa mental sem chamar a IA."""
    mm = mindmap_config(cfg)
    images_dir = output_dir / str(mm.get("diretorio_imagens") or "images")
    stem = slugify(str(mm.get("arquivo") or "mapa_mental"))
    formato = str(mm.get("formato") or "png").lower().lstrip(".")
    if formato not in {"png", "svg"}:
        formato = "png"
    return {
        "images_dir": images_dir,
        "puml": images_dir / f"{stem}.puml",
        "image": images_dir / f"{stem}.{formato}",
        "png": images_dir / f"{stem}.png",
        "svg": images_dir / f"{stem}.svg",
    }


def delete_existing_mindmap_outputs(cfg: dict[str, Any], output_dir: Path) -> list[str]:
    """Remove arquivos conhecidos do mapa mental para forçar regeneração."""
    paths = mindmap_output_paths(cfg, output_dir)
    removed: list[str] = []
    for key in ("puml", "image", "png", "svg"):
        p = paths.get(key)
        if p and p.exists() and p.is_file():
            p.unlink()
            removed.append(str(p))
    return sorted(set(removed))


def attach_existing_mindmap_if_available(doc: AcademicDocument, cfg: dict[str, Any], output_dir: Path) -> dict[str, Any] | None:
    """Anexa ao document_model uma imagem de mapa mental já existente.

    Retorna diagnóstico quando conseguiu reaproveitar; retorna None quando
    não há imagem disponível.
    """
    if not should_generate_mindmap(cfg):
        return None
    mm = mindmap_config(cfg)
    title = str(mm.get("titulo") or "Mapa mental dos textos analisados")
    paths = mindmap_output_paths(cfg, output_dir)
    candidates = [paths.get("image"), paths.get("png"), paths.get("svg")]
    img = next((p for p in candidates if p and p.exists() and p.is_file()), None)
    if not img:
        return None
    rel = str(img.relative_to(output_dir)).replace("\\", "/")
    doc.figures = [f for f in doc.figures if f.id != "mapa_mental"]
    doc.figures.append(FigureSpec(id="mapa_mental", title=title, path=rel, placement="after_references", page_break_before=True, page_break_after=False))
    diag = {"reused": True, "puml_path": str(paths.get("puml")), "image_path": str(img), "error": None}
    try:
        diag["prompts"] = load_prompt_bundle(cfg, "mindmap").report()
    except Exception:
        pass
    return diag

def generate_and_attach_mindmap(client: Any, model: str, cfg: dict[str, Any], doc: AcademicDocument, output_dir: Path) -> dict[str, Any] | None:
    if not should_generate_mindmap(cfg):
        return None
    mm = mindmap_config(cfg)
    paths = mindmap_artifact_paths(cfg, output_dir)
    images_dir = paths["images_dir"]
    images_dir.mkdir(parents=True, exist_ok=True)
    puml = paths["puml"]
    resp = client.responses.parse(model=model, input=[{"role": "user", "content": build_mindmap_prompt(doc, cfg)}], text_format=MindmapOutput)
    if resp.output_parsed is None:
        raise RuntimeError("IA não retornou PlantUML do mapa mental.")
    code = sanitize_plantuml(resp.output_parsed.plantuml)
    code = colorize_mindmap_plantuml(code, cfg)
    write_text(puml, code)
    img, err = render_plantuml(puml, cfg)
    if err and bool(mm.get("falhar_se_nao_renderizar", True)):
        raise RuntimeError(err)
    if img:
        attach_mindmap_figure(doc, cfg, output_dir, img)
    diag = {"puml_path": str(puml), "image_path": str(img) if img else None, "error": err}
    diag["prompts"] = load_prompt_bundle(cfg, "mindmap").report()
    return diag
