#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Limpeza/normalização da árvore do academic_pipeline.

rc10.7.32 — clean institutional tree

Este módulo remove sobras globais duplicadas após a migração para perfis
institucionais, cria a estrutura canônica em app_bundle/institutions/<perfil>/
e reorganiza exemplos em app_bundle/examples/.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class CleanAction:
    action: str
    path: str
    target: str | None = None
    status: str = "planned"
    note: str = ""


def find_app_bundle(start: Path | None = None) -> Path:
    """Localiza app_bundle a partir de start/cwd/script."""
    candidates: list[Path] = []
    if start:
        start = start.expanduser().resolve()
        candidates.extend([start, *start.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.name == "app_bundle" and (c / "scripts" / "pipeline").exists():
            return c
        if (c / "app_bundle" / "scripts" / "pipeline").exists():
            return (c / "app_bundle").resolve()
    raise RuntimeError("Não foi possível localizar app_bundle.")


def _rel(app: Path, p: Path) -> str:
    try:
        return str(p.relative_to(app)).replace("\\", "/")
    except Exception:
        return str(p)


def _add_create(actions: list[CleanAction], app: Path, path: Path, note: str = "") -> None:
    actions.append(CleanAction("mkdir", _rel(app, path), note=note))


def _add_delete(actions: list[CleanAction], app: Path, path: Path, note: str = "") -> None:
    actions.append(CleanAction("delete", _rel(app, path), note=note))


def _add_move(actions: list[CleanAction], app: Path, src: Path, dst: Path, note: str = "") -> None:
    actions.append(CleanAction("move", _rel(app, src), target=_rel(app, dst), note=note))


def build_clean_plan(
    app_bundle: Path,
    *,
    remove_outputs: bool = False,
    remove_projects: bool = False,
    remove_backups: bool = True,
    remove_legacy_examples: bool = True,
) -> list[CleanAction]:
    """Gera plano de limpeza/normalização.

    Por padrão, remove apenas duplicatas e resíduos seguros. Outputs e projetos
    só entram no plano mediante flags explícitas.
    """
    app = app_bundle.resolve()
    root = app.parent
    actions: list[CleanAction] = []

    # Diretórios canônicos.
    for d in [
        app / "examples" / "doi",
        app / "examples" / "toml",
        app / "institutions" / "fgv" / "assets",
        app / "institutions" / "fgv" / "latex",
        app / "institutions" / "fgv" / "templates",
        app / "institutions" / "fgv" / "docx",
        app / "institutions" / "fgv" / "validators",
        app / "institutions" / "fgv" / "prompts",
    ]:
        _add_create(actions, app, d, "estrutura canônica institucional")

    # Reorganização de exemplos.
    for csv in [app / "examples" / "doi_manifest_template.csv", app / "examples" / "doi_manifest_template_com_exemplos.csv"]:
        if csv.exists():
            _add_move(actions, app, csv, app / "examples" / "doi" / csv.name, "mover exemplos DOI para examples/doi")

    legacy_examples = app / "config" / "examples"
    if legacy_examples.exists():
        for toml in sorted(legacy_examples.glob("*.toml")):
            _add_move(actions, app, toml, app / "examples" / "toml" / toml.name, "mover exemplos TOML para examples/toml")
        if remove_legacy_examples:
            _add_delete(actions, app, legacy_examples, "remover pasta antiga config/examples")
            if (app / "config").exists():
                _add_delete(actions, app, app / "config", "remover app_bundle/config se ficar vazio")

    # Duplicatas globais FGV substituídas por institutions/fgv/*.
    for p in [
        app / "templates" / "template_atividade.org",
        app / "templates" / "template_paper.org",
        app / "templates" / "template_dissertacao.org",
        app / "templates" / "reference_fgv.docx",
        app / "templates" / "make_reference_fgv_docx.py",
        app / "misc" / "fgv.png",
        app / "misc" / "fgv",
    ]:
        if p.exists():
            _add_delete(actions, app, p, "duplicata global; usar institutions/fgv")

    # Resíduos de execução.
    for pycache in app.rglob("__pycache__"):
        _add_delete(actions, app, pycache, "cache Python")

    if remove_outputs and (app / "output").exists():
        _add_delete(actions, app, app / "output", "outputs/work/cache gerados")

    if remove_projects and (app / "projetos").exists():
        _add_delete(actions, app, app / "projetos", "projetos locais/pessoais não devem integrar distribuição limpa")

    if remove_backups:
        for b in sorted(root.glob("_backup_pre_*")):
            if b.exists():
                actions.append(CleanAction("delete", str(b), note="backup local de atualização"))

    return actions


def _rm_empty_dir(path: Path) -> bool:
    try:
        path.rmdir()
        return True
    except OSError:
        return False


def apply_clean_plan(app_bundle: Path, actions: list[CleanAction], *, dry_run: bool = True) -> list[CleanAction]:
    app = app_bundle.resolve()
    root = app.parent
    executed: list[CleanAction] = []

    def resolve_action_path(raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return app / raw

    for a in actions:
        act = CleanAction(**asdict(a))
        try:
            if act.action == "mkdir":
                p = resolve_action_path(act.path)
                if dry_run:
                    act.status = "would_create" if not p.exists() else "exists"
                else:
                    p.mkdir(parents=True, exist_ok=True)
                    act.status = "created" if p.exists() else "failed"

            elif act.action == "move":
                src = resolve_action_path(act.path)
                dst = resolve_action_path(act.target or "")
                if not src.exists():
                    act.status = "missing"
                elif dst.exists():
                    act.status = "target_exists"
                elif dry_run:
                    act.status = "would_move"
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    act.status = "moved"

            elif act.action == "delete":
                p = resolve_action_path(act.path)
                # Se path absoluto fora do app_bundle, resolve_action_path já preservou.
                if Path(act.path).is_absolute():
                    p = Path(act.path)
                if not p.exists():
                    act.status = "missing"
                elif dry_run:
                    act.status = "would_delete"
                else:
                    if p.is_dir():
                        # app_bundle/config pode ser removido apenas se vazio.
                        if p == app / "config":
                            act.status = "deleted_empty" if _rm_empty_dir(p) else "kept_not_empty"
                        else:
                            shutil.rmtree(p)
                            act.status = "deleted"
                    else:
                        p.unlink()
                        act.status = "deleted"
            else:
                act.status = "unknown_action"
        except Exception as exc:
            act.status = "error"
            act.note = (act.note + " | " if act.note else "") + str(exc)
        executed.append(act)
    return executed


def render_clean_report(actions: list[CleanAction], *, dry_run: bool) -> str:
    title = "Plano de limpeza" if dry_run else "Limpeza aplicada"
    lines = [f"{title}:"]
    for a in actions:
        target = f" -> {a.target}" if a.target else ""
        note = f" ({a.note})" if a.note else ""
        lines.append(f"- [{a.status}] {a.action}: {a.path}{target}{note}")
    return "\n".join(lines)


def clean_institutional_tree(
    *,
    base_dir: Path | None = None,
    apply: bool = False,
    remove_outputs: bool = False,
    remove_projects: bool = False,
    remove_backups: bool = True,
    remove_legacy_examples: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    app = find_app_bundle(base_dir)
    plan = build_clean_plan(
        app,
        remove_outputs=remove_outputs,
        remove_projects=remove_projects,
        remove_backups=remove_backups,
        remove_legacy_examples=remove_legacy_examples,
    )
    result_actions = apply_clean_plan(app, plan, dry_run=not apply)
    report = {
        "app_bundle": str(app),
        "applied": bool(apply),
        "remove_outputs": bool(remove_outputs),
        "remove_projects": bool(remove_projects),
        "actions": [asdict(a) for a in result_actions],
    }
    if write_report:
        report_path = app / "clean_institutional_tree_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Limpa e normaliza a árvore institucional do academic_pipeline.")
    parser.add_argument("--base-dir", default="", help="Raiz do bundle ou app_bundle. Vazio = autodetectar.")
    parser.add_argument("--apply", action="store_true", help="Aplica a limpeza. Sem isso, roda em dry-run.")
    parser.add_argument("--remove-output", action="store_true", help="Remove app_bundle/output.")
    parser.add_argument("--remove-projects", action="store_true", help="Remove app_bundle/projetos.")
    parser.add_argument("--keep-backups", action="store_true", help="Não remove _backup_pre_* na raiz do bundle.")
    args = parser.parse_args()
    res = clean_institutional_tree(
        base_dir=Path(args.base_dir).expanduser().resolve() if args.base_dir else None,
        apply=args.apply,
        remove_outputs=args.remove_output,
        remove_projects=args.remove_projects,
        remove_backups=not args.keep_backups,
    )
    actions = [CleanAction(**a) for a in res["actions"]]
    print(render_clean_report(actions, dry_run=not args.apply))
    if res.get("report_path"):
        print(f"Relatório: {res['report_path']}")
