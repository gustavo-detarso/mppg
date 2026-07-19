#!/usr/bin/env python3
"""Validate AP-006D.1 runtime-consumer migration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OLD_NAME = "academic_pipeline_rc10_7_conformidade"
NEW_NAME = "academic_pipeline_mppg"
OLD_REL = Path("software") / OLD_NAME
NEW_REL = Path("software") / NEW_NAME

MIGRATED = (
    NEW_REL / "aplicar_docx_canonico_v11.py",
    NEW_REL / "aplicar_docx_canonico_v12.py",
    NEW_REL / "aplicar_docx_canonico_v13.py",
    NEW_REL / "aplicar_docx_capa_disciplina_v14.py",
    NEW_REL / "app_bundle" / ".academic_pipeline_tui_state.json",
    NEW_REL / "atualizar_academic_pipeline_bundle.py",
)

REPORT = NEW_REL / "app_bundle" / "clean_institutional_tree_report.json"
MANIFEST = (
    Path("docs")
    / "refactor"
    / "academic-pipeline"
    / "AP-006"
    / "ap006d1_runtime_consumer_migration.json"
)


def validate(repo_root: Path) -> list[str]:
    errors: list[str] = []
    repo_root = repo_root.resolve()

    bridge = repo_root / OLD_REL
    canonical = repo_root / NEW_REL

    if not bridge.is_symlink():
        errors.append(f"compatibility bridge is not a symlink: {bridge}")
    else:
        target = bridge.readlink()
        if target.as_posix() != NEW_NAME:
            errors.append(
                f"compatibility bridge target is {target!s}, expected {NEW_NAME}"
            )

    if not canonical.is_dir() or canonical.is_symlink():
        errors.append(f"canonical root is invalid: {canonical}")

    for relative in MIGRATED:
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"migrated file missing: {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        if OLD_NAME in text:
            errors.append(f"legacy root remains in migrated file: {relative}")

    state_path = repo_root / MIGRATED[4]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        serialized = json.dumps(state, ensure_ascii=False)
        if OLD_NAME in serialized:
            errors.append("legacy root remains in TUI state")
        if serialized.count(NEW_NAME) != 4:
            errors.append(
                "TUI state should contain four canonical-root references"
            )

    report_path = repo_root / REPORT
    if not report_path.is_file():
        errors.append(f"historical report missing: {REPORT}")
    else:
        report_text = report_path.read_text(
            encoding="utf-8",
            errors="surrogateescape",
        )
        if report_text.count(OLD_NAME) != 2:
            errors.append(
                "historical report must preserve exactly two legacy references"
            )

    manifest_path = repo_root / MANIFEST
    if not manifest_path.is_file():
        errors.append(f"manifest missing: {MANIFEST}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = manifest.get("summary", {})
        if summary.get("migrated_reference_count") != 10:
            errors.append("manifest migrated-reference count is not 10")
        if summary.get("migrated_file_count") != 6:
            errors.append("manifest migrated-file count is not 6")
        if summary.get("preserved_historical_reference_count") != 2:
            errors.append("manifest preserved-reference count is not 2")
        bridge_contract = manifest.get("compatibility_bridge", {})
        if bridge_contract.get("must_remain") is not True:
            errors.append("manifest does not preserve compatibility bridge")
        if bridge_contract.get("removal_phase") != "AP-006F":
            errors.append("manifest bridge-removal phase is not AP-006F")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    errors = validate(args.repo_root)
    if errors:
        for error in errors:
            print(f"[ERRO] {error}", file=sys.stderr)
        return 1

    print("[OK] AP-006D.1 runtime-consumer migration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
