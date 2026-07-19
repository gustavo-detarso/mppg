#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

OLD_NAME = "academic_pipeline_rc10_7_conformidade"
NEW_NAME = "academic_pipeline_mppg"
OLD_REL = Path("software") / OLD_NAME
NEW_REL = Path("software") / NEW_NAME

EXPECTED_OLD_COUNTS = {
    "tools/refactor/ap004e_inventory_compatibility.py": 2,
    "tools/refactor/ap004f_generate_closure.py": 0,
    "tools/refactor/ap005d_inventory_facades.py": 0,
    "tools/refactor/ap005e1_inventory_installation_entrypoints.py": 4,
    "tools/refactor/ap005e2_characterize_isolated_build_installation.py": 0,
}
EXPECTED_NEW_COUNTS = {
    "tools/refactor/ap004e_inventory_compatibility.py": 3,
    "tools/refactor/ap004f_generate_closure.py": 2,
    "tools/refactor/ap005d_inventory_facades.py": 2,
    "tools/refactor/ap005e1_inventory_installation_entrypoints.py": 13,
    "tools/refactor/ap005e2_characterize_isolated_build_installation.py": 3,
}
REQUIRED_ARTIFACTS = (
    "docs/refactor/academic-pipeline/AP-006/AP-006D2_CONTRACT_VALIDATOR_MIGRATION.md",
    "docs/refactor/academic-pipeline/AP-006/ap006d2_contract_validator_migration.json",
    "software/academic_pipeline_mppg/tests/characterization/test_ap006d2_contract_validator_migration.py",
    "tools/refactor/ap006d2_validate_contract_validator_migration.py",
)


def _load_dual_root_function(source: str, filename: str):
    module = ast.parse(source, filename=filename)
    selected = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "module_name_from_path"
    ]
    namespace: dict[str, Any] = {"Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace["module_name_from_path"]


def validate(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    errors: list[str] = []
    observed: dict[str, dict[str, int]] = {}

    for relative, expected_old in EXPECTED_OLD_COUNTS.items():
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"arquivo ausente: {relative}")
            continue

        text = path.read_text(encoding="utf-8")
        try:
            compile(text, relative, "exec")
        except SyntaxError as exc:
            errors.append(f"syntax error em {relative}: {exc}")

        old_count = text.count(OLD_NAME)
        new_count = text.count(NEW_NAME)
        observed[relative] = {"old": old_count, "new": new_count}

        if old_count != expected_old:
            errors.append(f"{relative}: old={old_count}, esperado={expected_old}")

        expected_new = EXPECTED_NEW_COUNTS[relative]
        if new_count != expected_new:
            errors.append(f"{relative}: new={new_count}, esperado={expected_new}")

    bridge = repo_root / OLD_REL
    canonical = repo_root / NEW_REL

    if not bridge.is_symlink():
        errors.append("ponte de compatibilidade ausente")
    elif bridge.readlink() != Path(NEW_NAME):
        errors.append(f"target da ponte divergente: {bridge.readlink()}")

    if not canonical.is_dir() or canonical.is_symlink():
        errors.append("raiz canônica física inválida")

    compat = repo_root / "tools/refactor/ap004e_inventory_compatibility.py"
    if compat.is_file():
        text = compat.read_text(encoding="utf-8")

        for marker in ("project_root_names", "markers = ("):
            if marker not in text:
                errors.append(f"marcador dual-root ausente: {marker}")

        function = _load_dual_root_function(text, str(compat))
        fake_root = Path("/tmp/repo")

        for root_name in (NEW_NAME, OLD_NAME):
            result = function(
                fake_root,
                fake_root / "software" / root_name / "academic_pipeline" / "cli.py",
            )
            if result != "academic_pipeline.cli":
                errors.append(
                    f"module_name_from_path falhou para {root_name}: {result}"
                )

    for relative in REQUIRED_ARTIFACTS:
        if not (repo_root / relative).is_file():
            errors.append(f"artefato ausente: {relative}")

    contract_path = repo_root / REQUIRED_ARTIFACTS[1]
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        if contract.get("phase") != "AP-006D.2":
            errors.append("phase incorreta no contrato JSON")
        if contract.get("summary", {}).get("direct_migration_count") != 19:
            errors.append("contagem direta incorreta no contrato JSON")
        if contract.get("summary", {}).get("historical_record_count") != 4:
            errors.append("contagem histórica incorreta no contrato JSON")

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
    print("[OK] AP-006D.2 contract-validator migration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
