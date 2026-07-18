#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estado persistente do fluxo de artigo PRISMA.

O arquivo fica no diretório do artigo, em ``artigo_state.json``. Ele não é
um log operacional: é um painel de progresso com bloqueios, evidências e a
próxima ação recomendada.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

STATUS_BLOCKED = "bloqueado"
STATUS_PENDING = "pendente"
STATUS_OK = "ok"
STATUS_ERROR = "erro"
STATUS_STALE = "stale"

STAGES: list[tuple[str, str, list[str]]] = [
    ("briefing", "Briefing do artigo", []),
    ("toml_prisma", "TOML PRISMA preliminar", ["briefing"]),
    ("prisma_preliminar", "Pesquisa PRISMA preliminar", ["toml_prisma"]),
    ("xlsx_cut", "XLSX cut de referências pela IA", ["prisma_preliminar"]),
    ("revisao_humana_xlsx", "Revisão humana do XLSX", ["xlsx_cut"]),
    ("prisma_final", "PRISMA final", ["revisao_humana_xlsx"]),
    ("fulltext", "Full text e corpus", ["prisma_final"]),
    ("artigo_org", "Artigo ORG/BIB", ["fulltext"]),
    ("pdf_final", "PDF final validado", ["artigo_org"]),
]

STAGE_LABELS = {key: label for key, label, _deps in STAGES}
STAGE_DEPS = {key: deps for key, _label, deps in STAGES}


@dataclass
class StageRecord:
    status: str = STATUS_BLOCKED
    evidence: list[str] = field(default_factory=list)
    message: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidence": list(self.evidence),
            "message": self.message,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "StageRecord":
        if not isinstance(raw, dict):
            return cls()
        evidence = raw.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        return cls(
            status=str(raw.get("status") or STATUS_BLOCKED),
            evidence=[str(x) for x in evidence],
            message=str(raw.get("message") or ""),
            updated_at=str(raw.get("updated_at") or ""),
        )


class WorkflowState:
    """Leitura/gravação do ``artigo_state.json``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = self._default()
        self.load()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "current_stage": "briefing",
            "metadata": {},
            "stages": {key: StageRecord().to_dict() for key, _label, _deps in STAGES},
            "human_review": {},
            "history": [],
        }

    def load(self) -> None:
        if not self.path.exists():
            self._normalize()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.data = raw
        except Exception:
            # Estado corrompido não derruba o fluxo. Mantém default e deixa o
            # arquivo antigo para diagnóstico humano.
            self.data = self._default()
        self._normalize()

    def _normalize(self) -> None:
        self.data.setdefault("schema_version", 1)
        self.data.setdefault("created_at", self._now())
        self.data.setdefault("updated_at", self._now())
        self.data.setdefault("metadata", {})
        self.data.setdefault("stages", {})
        self.data.setdefault("human_review", {})
        self.data.setdefault("history", [])
        stages = self.data["stages"]
        for key, _label, deps in STAGES:
            rec = StageRecord.from_dict(stages.get(key)).to_dict()
            if deps and rec["status"] == STATUS_PENDING:
                pass
            stages[key] = rec
        self._apply_dependency_blocks()
        self.data["current_stage"] = self.next_pending_stage() or "concluido"

    def _apply_dependency_blocks(self) -> None:
        stages = self.data["stages"]
        for key, _label, deps in STAGES:
            if not deps:
                if stages[key].get("status") == STATUS_BLOCKED:
                    stages[key]["status"] = STATUS_PENDING
                continue
            if any(stages.get(dep, {}).get("status") != STATUS_OK for dep in deps):
                if stages[key].get("status") not in {STATUS_OK, STATUS_ERROR, STATUS_STALE}:
                    stages[key]["status"] = STATUS_BLOCKED
            elif stages[key].get("status") == STATUS_BLOCKED:
                stages[key]["status"] = STATUS_PENDING

    def save(self) -> None:
        self._normalize()
        self.data["updated_at"] = self._now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record(self, stage: str) -> StageRecord:
        self._normalize()
        return StageRecord.from_dict(self.data["stages"].get(stage))

    def status(self, stage: str) -> str:
        return self.record(stage).status

    def is_ok(self, stage: str) -> bool:
        return self.status(stage) == STATUS_OK

    def can_run(self, stage: str) -> tuple[bool, str]:
        self._normalize()
        deps = STAGE_DEPS.get(stage, [])
        missing = [STAGE_LABELS.get(dep, dep) for dep in deps if not self.is_ok(dep)]
        if missing:
            return False, "Etapa bloqueada. Antes conclua: " + ", ".join(missing)
        return True, ""

    def mark(self, stage: str, status: str, *, evidence: list[str] | None = None, message: str = "") -> None:
        if stage not in STAGE_LABELS:
            raise KeyError(f"Etapa desconhecida: {stage}")
        rec = StageRecord(status=status, evidence=evidence or [], message=message, updated_at=self._now())
        self.data["stages"][stage] = rec.to_dict()
        self.data["history"].append({
            "at": self._now(),
            "stage": stage,
            "status": status,
            "message": message,
            "evidence": evidence or [],
        })
        self.save()

    def next_pending_stage(self) -> str | None:
        stages = self.data.get("stages", {})
        for key, _label, _deps in STAGES:
            if stages.get(key, {}).get("status") != STATUS_OK:
                return key
        return None
