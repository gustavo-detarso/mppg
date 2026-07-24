from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JSON_PATH = REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e3_isolated_installation_matrix.json'
EXPECTED_SCHEMA = "ap007e3_isolated_installation_matrix.v1"
EXPECTED_PATHS = ['docs/refactor/academic-pipeline/AP-007/AP-007E3_ISOLATED_INSTALLATION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-007/ap007e3_isolated_installation_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap007e3_isolated_installation_matrix_contract.py', 'tools/refactor/ap007e3_validate_isolated_installation_matrix.py']


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks = [
        (data.get("schema") == EXPECTED_SCHEMA, "schema"),
        (data.get("phase") == "AP-007E.3", "phase"),
        (data.get("status") in {"installation_approved", "installation_approved_with_classified_findings"}, "status"),
        (data.get("scope", {}).get("materialized_paths") == EXPECTED_PATHS, "materialized paths"),
        (data.get("scope", {}).get("artifact_installation_executed") is True, "artifact installation"),
        (data.get("scope", {}).get("dependency_installation_executed") is False, "no dependency installation"),
        (data.get("scope", {}).get("network_allowed") is False, "no network"),
        (data.get("scope", {}).get("pythonpath_removed") is True, "pythonpath removed"),
        (data.get("scope", {}).get("canonical_environment_modified") is False, "canonical env unchanged"),
        (data.get("scope", {}).get("productive_modules_modified") == [], "no productive changes"),
        (data.get("scope", {}).get("git_write_executed") is False, "no git write"),
        (len(data.get("installations", [])) == 2, "two installations"),
        ({item.get("origin") for item in data.get("installations", [])} == {"wheel", "sdist"}, "wheel and sdist origins"),
        (data.get("canonical_environment", {}).get("preserved") is True, "canonical fingerprint"),
        (data.get("resources", {}).get("critical_resources_present_in_both_installations") is True, "critical resources"),
        (data.get("summary", {}).get("source_leak_count") == 0, "no source leaks"),
        (data.get("summary", {}).get("module_locations_inside_disposable_venvs") is True, "module locations"),
        (data.get("summary", {}).get("distribution_locations_inside_disposable_venvs") is True, "distribution locations"),
        (data.get("summary", {}).get("console_scripts_valid") is True, "console scripts"),
        (data.get("summary", {}).get("pythonpath_removed") is True, "summary pythonpath"),
        (data.get("summary", {}).get("normalized_artifact_provenance_validated") is True, "artifact provenance"),
        (data.get("summary", {}).get("blocking_finding_count") == 0, "no blocking findings"),
    ]
    for passed, label in checks:
        if not passed:
            failures.append(label)
    for installation in data.get("installations", []):
        runtime = installation.get("runtime", {})
        summary = runtime.get("summary", {})
        if summary.get("critical_resources_present") is not True:
            failures.append(f"critical resources {installation.get('origin')}")
        if summary.get("module_locations_inside_venv") is not True:
            failures.append(f"module location {installation.get('origin')}")
        if summary.get("source_leaks") != 0:
            failures.append(f"source leak {installation.get('origin')}")
        if runtime.get("console_script", {}).get("exists") is not True:
            failures.append(f"console missing {installation.get('origin')}")
        if runtime.get("console_script", {}).get("executable") is not True:
            failures.append(f"console not executable {installation.get('origin')}")
    for rel in EXPECTED_PATHS:
        if not (REPO / rel).is_file():
            failures.append(f"missing {rel}")
    print(f"AP007E3_VALIDATION_FAILURES={len(failures)}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        return 1
    print("[GATE] AP-007E.3: WHEEL E SDIST INSTALADOS EM AMBIENTES DESCARTÁVEIS, IMPORTS, RECURSOS, CONSOLE SCRIPT E PYTHON -M AUDITADOS SEM PYTHONPATH OU REUSO DA FONTE; AMBIENTE CANÔNICO PRESERVADO, SEM DEPENDÊNCIAS ADICIONAIS, STAGING, COMMIT, TAG OU PUSH.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
