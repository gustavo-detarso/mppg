#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validações estruturais para o fluxo de artigo PRISMA."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import STATUS_BLOCKED, STATUS_ERROR, STATUS_OK, STATUS_PENDING, STAGES, WorkflowState

BAD_PDF_PATTERNS = [
    r"LATEX",
    r"LaTeX",
    r"\[[0-9]+(?:,\s*[0-9]+)*\]",
    r"Soares, Ninin e Lima \(Soares",
    r"Gabbay et al\. \(Gabbay",
    r"Dijkstra \(Dijkstra",
]


@dataclass
class StageValidation:
    key: str
    ok: bool
    status: str
    message: str
    evidence: list[str] = field(default_factory=list)


class ArticleWorkflow:
    """Controla estado, inferência por arquivos e contratos de saída.

    Parâmetros principais:
    - ``art_dir``: pasta do artigo, onde fica ``artigo_state.json``;
    - ``cfg_art``: TOML do artigo final, usado para prefixo do ORG/BIB/PDF;
    - ``prisma_cfg``: TOML PRISMA preliminar/final, usado para localizar saídas PRISMA.
    """

    def __init__(self, art_dir: Path, *, cfg_art: Path | None = None, prisma_cfg: Path | None = None) -> None:
        self.art_dir = Path(art_dir).expanduser().resolve()
        self.cfg_art = Path(cfg_art).expanduser().resolve() if cfg_art else self._guess_article_cfg()
        self.prisma_cfg = Path(prisma_cfg).expanduser().resolve() if prisma_cfg else None
        self.state = WorkflowState(self.art_dir / "artigo_state.json")
        self._store_metadata()

    def _store_metadata(self) -> None:
        meta = self.state.data.setdefault("metadata", {})
        meta["art_dir"] = str(self.art_dir)
        if self.cfg_art:
            meta["cfg_art"] = str(self.cfg_art)
        if self.prisma_cfg:
            meta["prisma_cfg"] = str(self.prisma_cfg)
        self.state.save()

    def _guess_article_cfg(self) -> Path | None:
        tomls = sorted(self.art_dir.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True) if self.art_dir.exists() else []
        return tomls[0].resolve() if tomls else None

    @property
    def output_dir(self) -> Path:
        return self.art_dir / "output"

    @property
    def dados_prisma_dir(self) -> Path:
        return self.art_dir / "dados_prisma"

    @property
    def prefix(self) -> str:
        if self.cfg_art:
            return self.cfg_art.stem
        return "artigo_final"

    @property
    def org_path(self) -> Path:
        return self.output_dir / f"{self.prefix}.org"

    @property
    def bib_path(self) -> Path:
        return self.output_dir / f"{self.prefix}.bib"

    @property
    def pdf_path(self) -> Path:
        return self.output_dir / f"{self.prefix}.pdf"

    def _read_toml(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _resolve_config_path(config_path: Path, raw: Any) -> Path | None:
        value = str(raw or "").strip()
        if not value or value.startswith("profile://"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return path.resolve()

    def research_output_dir(self) -> Path | None:
        if not self.prisma_cfg or not self.prisma_cfg.exists():
            return None
        try:
            cfg = self._read_toml(self.prisma_cfg)
        except Exception:
            return None
        paths = self._section(cfg, "paths")
        project = self._section(cfg, "projeto")
        raw_base = paths.get("research_output_dir") or "output_pesquisa"
        base = self._resolve_config_path(self.prisma_cfg, raw_base) or (self.prisma_cfg.parent / "output_pesquisa")
        prefix = str(paths.get("research_prefix") or f"relatorio_prisma_{project.get('nome') or self.prisma_cfg.stem}").strip()
        return base / prefix if bool(paths.get("create_research_subdir", True)) else base

    def _glob_any(self, bases: list[Path], patterns: list[str]) -> list[Path]:
        found: list[Path] = []
        for base in bases:
            if not base.exists():
                continue
            for pattern in patterns:
                found.extend(p for p in base.glob(pattern) if p.is_file())
        return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)

    def find_prisma_artifacts(self, patterns: list[str]) -> list[Path]:
        bases = [self.dados_prisma_dir]
        research = self.research_output_dir()
        if research:
            bases.insert(0, research)
        return self._glob_any(bases, patterns)

    def validate_briefing(self) -> StageValidation:
        candidates = self._glob_any([self.art_dir], ["briefing_artigo.txt", "briefing*.txt", "briefing*.md", "tema*.txt"])
        for p in candidates:
            try:
                if len(p.read_text(encoding="utf-8", errors="ignore").strip()) >= 80:
                    return StageValidation("briefing", True, STATUS_OK, "Briefing localizado e preenchido.", [str(p)])
            except Exception:
                pass
        return StageValidation("briefing", False, STATUS_PENDING, "Crie ou preencha o briefing do artigo.", [str(p) for p in candidates[:3]])

    def validate_toml_prisma(self) -> StageValidation:
        if not self.prisma_cfg or not self.prisma_cfg.exists():
            return StageValidation("toml_prisma", False, STATUS_PENDING, "Selecione ou gere o TOML PRISMA preliminar.", [])
        try:
            cfg = self._read_toml(self.prisma_cfg)
        except Exception as exc:
            return StageValidation("toml_prisma", False, STATUS_ERROR, f"TOML PRISMA ilegível: {exc}", [str(self.prisma_cfg)])
        if not self._section(cfg, "projeto") or not self._section(cfg, "documento"):
            return StageValidation("toml_prisma", False, STATUS_ERROR, "TOML não parece ser configuração válida da pipeline.", [str(self.prisma_cfg)])
        return StageValidation("toml_prisma", True, STATUS_OK, "TOML PRISMA válido.", [str(self.prisma_cfg)])

    def validate_prisma_preliminar(self) -> StageValidation:
        files = self.find_prisma_artifacts(["*.busca_prisma_log.json", "*.triagem_titulo_resumo.csv", "*.relatorio_prisma_preliminar.pdf"])
        if files:
            return StageValidation("prisma_preliminar", True, STATUS_OK, "Saídas da busca PRISMA preliminar localizadas.", [str(p) for p in files[:5]])
        return StageValidation("prisma_preliminar", False, STATUS_PENDING, "Rode a busca PRISMA preliminar.", [])

    def validate_xlsx_cut(self) -> StageValidation:
        files = self.find_prisma_artifacts(["*.curadoria_ia_referencias.xlsx"])
        if files:
            return StageValidation("xlsx_cut", True, STATUS_OK, "XLSX cut de curadoria IA localizado.", [str(files[0])])
        return StageValidation("xlsx_cut", False, STATUS_PENDING, "Gere o XLSX cut de referências pela IA.", [])

    def validate_revisao_humana(self) -> StageValidation:
        hr = self.state.data.get("human_review", {}) if isinstance(self.state.data.get("human_review"), dict) else {}
        confirmed = bool(hr.get("confirmed"))
        xlsx_path = Path(str(hr.get("xlsx_path") or "")).expanduser() if hr.get("xlsx_path") else None
        if confirmed and xlsx_path and xlsx_path.exists():
            return StageValidation("revisao_humana_xlsx", True, STATUS_OK, "Revisão humana do XLSX confirmada.", [str(xlsx_path)])
        files = self.find_prisma_artifacts(["*.curadoria_ia_referencias.xlsx"])
        if files:
            return StageValidation("revisao_humana_xlsx", False, STATUS_PENDING, "Abra, revise, salve e confirme a revisão humana do XLSX.", [str(files[0])])
        return StageValidation("revisao_humana_xlsx", False, STATUS_BLOCKED, "Aguardando geração do XLSX cut.", [])

    def validate_prisma_final(self) -> StageValidation:
        files = self.find_prisma_artifacts(["*.referencias_incluidas.bib", "*.referencias_incluidas.csv", "*.triagem_humana.csv", "*.relatorio_prisma_final.pdf"])
        if files:
            return StageValidation("prisma_final", True, STATUS_OK, "PRISMA final e referências incluídas localizados.", [str(p) for p in files[:5]])
        return StageValidation("prisma_final", False, STATUS_PENDING, "Gere/importa o PRISMA final após a revisão humana.", [])

    def validate_fulltext(self) -> StageValidation:
        files = self._glob_any([self.dados_prisma_dir], [
            "artigo_longo_fulltext/corpus_fulltext_compilado.md",
            "artigo_longo_fulltext/referencias_incluidas_*.md",
            "fulltext_garantido/*.referencias_incluidas_fulltext_garantido.bib",
            "fulltext_garantido/*.matriz_prisma_fulltext_garantido.csv",
        ])
        if files:
            return StageValidation("fulltext", True, STATUS_OK, "Corpus/full text preparado localizado.", [str(p) for p in files[:5]])
        return StageValidation("fulltext", False, STATUS_PENDING, "Prepare full text/corpus pelo gerador unificado.", [])

    def validate_artigo_org(self) -> StageValidation:
        evidence: list[str] = []
        if self.org_path.exists():
            evidence.append(str(self.org_path))
        if self.bib_path.exists():
            evidence.append(str(self.bib_path))
        if self.org_path.exists() and self.bib_path.exists():
            text = self.org_path.read_text(encoding="utf-8", errors="ignore")
            problems = []
            if "style=abnt" not in text:
                problems.append("ORG sem biblatex style=abnt")
            if "\\printbibliography" not in text:
                problems.append("ORG sem \\printbibliography")
            if "evitando tabela larga no LATEX" in text or "evitando tabela larga no LaTeX" in text:
                problems.append("ORG ainda contém artefato LATEX")
            if problems:
                return StageValidation("artigo_org", False, STATUS_ERROR, "; ".join(problems), evidence)
            return StageValidation("artigo_org", True, STATUS_OK, "ORG/BIB finalizados em padrão ABNT/Biber.", evidence)
        return StageValidation("artigo_org", False, STATUS_PENDING, "Gere o artigo ORG/BIB final.", evidence)

    def validate_pdf_final(self) -> StageValidation:
        if not self.pdf_path.exists():
            return StageValidation("pdf_final", False, STATUS_PENDING, "PDF final ainda não localizado.", [])
        evidence = [str(self.pdf_path)]
        if shutil.which("pdftotext"):
            proc = subprocess.run(["pdftotext", str(self.pdf_path), "-"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            txt = (proc.stdout or b"").decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return StageValidation("pdf_final", False, STATUS_ERROR, "pdftotext falhou ao validar PDF.", evidence)
            hits: list[str] = []
            for i, line in enumerate(txt.splitlines(), start=1):
                for pat in BAD_PDF_PATTERNS:
                    if re.search(pat, line):
                        hits.append(f"{i}:{line}")
                        break
            if hits:
                return StageValidation("pdf_final", False, STATUS_ERROR, "PDF contém padrões inválidos: " + " | ".join(hits[:3]), evidence)
        return StageValidation("pdf_final", True, STATUS_OK, "PDF final validado.", evidence)

    def validations(self) -> list[StageValidation]:
        validators = [
            self.validate_briefing,
            self.validate_toml_prisma,
            self.validate_prisma_preliminar,
            self.validate_xlsx_cut,
            self.validate_revisao_humana,
            self.validate_prisma_final,
            self.validate_fulltext,
            self.validate_artigo_org,
            self.validate_pdf_final,
        ]
        return [fn() for fn in validators]

    def refresh_from_files(self) -> list[StageValidation]:
        results = self.validations()
        previous_ok = True
        for result in results:
            status = result.status
            if not previous_ok and not result.ok:
                status = STATUS_BLOCKED
            self.state.mark(result.key, status, evidence=result.evidence, message=result.message)
            if not result.ok:
                previous_ok = False
        return results

    def mark_human_review(self, xlsx_path: Path) -> None:
        xlsx = Path(xlsx_path).expanduser().resolve()
        self.state.data.setdefault("human_review", {})
        self.state.data["human_review"] = {
            "confirmed": True,
            "xlsx_path": str(xlsx),
            "confirmed_at": self.state._now(),
            "xlsx_mtime": xlsx.stat().st_mtime if xlsx.exists() else None,
        }
        self.state.mark("revisao_humana_xlsx", STATUS_OK, evidence=[str(xlsx)], message="Revisão humana confirmada pelo usuário.")

    def mark_stage_ok(self, stage: str, *, evidence: list[str] | None = None, message: str = "") -> None:
        self.state.mark(stage, STATUS_OK, evidence=evidence or [], message=message or "Etapa concluída.")

    def can_run(self, stage: str) -> tuple[bool, str]:
        self.refresh_from_files()
        return self.state.can_run(stage)

    def format_status(self) -> str:
        self.refresh_from_files()
        lines = ["Status estrutural do artigo PRISMA:", ""]
        for key, label, _deps in STAGES:
            rec = self.state.record(key)
            marker = {"ok": "[OK]", "pendente": "[PENDENTE]", "bloqueado": "[BLOQUEADO]", "erro": "[ERRO]", "stale": "[STALE]"}.get(rec.status, f"[{rec.status.upper()}]")
            lines.append(f"{marker:<12} {label}")
            if rec.message:
                lines.append(f"             {rec.message}")
            if rec.evidence:
                lines.append(f"             Evidência: {rec.evidence[0]}")
        next_stage = self.state.next_pending_stage()
        if next_stage:
            lines.append("")
            lines.append("Próxima ação recomendada: " + next(label for key, label, _deps in STAGES if key == next_stage))
        else:
            lines.append("")
            lines.append("Fluxo concluído: PDF final validado.")
        lines.append("")
        lines.append(f"Estado: {self.state.path}")
        return "\n".join(lines)
