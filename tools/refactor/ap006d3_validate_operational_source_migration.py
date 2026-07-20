#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD_NAME = "academic_pipeline_rc10_7_conformidade"
NEW_NAME = "academic_pipeline_mppg"
OLD_REL = Path("software") / OLD_NAME
NEW_REL = Path("software") / NEW_NAME

README = NEW_REL / "app_bundle/docs/README_rc10.md"
SETUP = NEW_REL / "app_bundle/docs/SETUP_PIPENV.md"
UPDATER = NEW_REL / "atualizar_academic_pipeline_bundle.py"
TOML = Path(
    "disciplinas/04_decisoes_baseadas_em_evidencia/"
    "atividades/artigo/artigo_final_atestmed_abnt.toml"
)
DOC = Path(
    "docs/refactor/academic-pipeline/AP-006/"
    "AP-006D3_OPERATIONAL_SOURCE_MIGRATION.md"
)
CONTRACT = Path(
    "docs/refactor/academic-pipeline/AP-006/"
    "ap006d3_operational_source_migration.json"
)
TEST = (
    NEW_REL
    / "tests/characterization/"
    "test_ap006d3_operational_source_migration.py"
)
VALIDATOR = Path(
    "tools/refactor/"
    "ap006d3_validate_operational_source_migration.py"
)

EXPECTED = {
    README: (0, 3),
    SETUP: (1, 1),
    UPDATER: (0, 1),
    TOML: (0, 7),
}


def validate(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    errors: list[str] = []
    observed: dict[str, dict[str, int]] = {}

    for relative, (expected_old, expected_new) in EXPECTED.items():
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"arquivo ausente: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        if relative.suffix == ".py":
            try:
                compile(text, str(relative), "exec")
            except SyntaxError as exc:
                errors.append(f"syntax error em {relative}: {exc}")
        old_count = text.count(OLD_NAME)
        new_count = text.count(NEW_NAME)
        observed[str(relative)] = {
            "old": old_count,
            "new": new_count,
        }
        if old_count != expected_old:
            errors.append(
                f"{relative}: old={old_count}, esperado={expected_old}"
            )
        if new_count != expected_new:
            errors.append(
                f"{relative}: new={new_count}, esperado={expected_new}"
            )

    bridge = repo_root / OLD_REL
    canonical = repo_root / NEW_REL
    if not bridge.is_symlink():
        errors.append("ponte ausente")
    elif bridge.readlink() != Path(NEW_NAME):
        errors.append(f"target da ponte divergente: {bridge.readlink()}")
    if not canonical.is_dir() or canonical.is_symlink():
        errors.append("raiz canônica inválida")

    for relative in (DOC, CONTRACT, TEST, VALIDATOR):
        if not (repo_root / relative).is_file():
            errors.append(f"artefato ausente: {relative}")

    contract_path = repo_root / CONTRACT
    if contract_path.is_file():
        contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        summary = contract.get("summary", {})
        constraints = contract.get("constraints", {})
        if contract.get("phase") != "AP-006D.3":
            errors.append("fase incorreta")
        if summary.get("migrated_line_count") != 11:
            errors.append("contagem migrada incorreta")
        if summary.get("modified_source_file_count") != 3:
            errors.append("contagem de fontes incorreta")
        if len(
            contract.get("external_regeneration", {}).get("entries", [])
        ) != 3:
            errors.append("regeneração externa deve ter três entradas")
        if not constraints.get(
            "nonoperational_setup_reference_preserved"
        ):
            errors.append("preservação SETUP não registrada")
        if not constraints.get("do_not_mix_worktrees"):
            errors.append("separação de worktrees não registrada")

    if errors:
        raise AssertionError("\n".join(errors))

    return {
        "status": "valid",
        "observed": observed,
        "bridge": str(bridge),
        "canonical": str(canonical),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print("[OK] AP-006D.3 operational-source migration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
