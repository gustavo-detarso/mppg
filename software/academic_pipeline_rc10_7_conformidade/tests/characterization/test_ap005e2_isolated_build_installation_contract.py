from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC_DIR = ROOT / "docs/refactor/academic-pipeline/AP-005"
JSON_FILE = DOC_DIR / "ap005e2_isolated_build_installation_characterization.json"
REPORT = DOC_DIR / "AP-005E2_ISOLATED_BUILD_INSTALLATION_CHARACTERIZATION.md"
SCOPE = DOC_DIR / "AP-005E2_CORRECTION_SCOPE.md"
TOOL = ROOT / "tools/refactor/ap005e2_characterize_isolated_build_installation.py"

EXPECTED_BASELINE = "0d553c975ad7948762f74aa4fcff3903578712de"


def _payload() -> dict:
    return json.loads(JSON_FILE.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_ap005e2_schema_baseline_and_fingerprint() -> None:
    payload = _payload()
    assert payload["schema_version"] == (
        "ap005e2.isolated-build-installation-characterization.v1"
    )
    assert payload["phase"] == "AP-005E.2"
    assert payload["baseline"] == {
        "branch": "ap-refactor/04-consumer-canonicalization",
        "commit": EXPECTED_BASELINE,
        "upstream": "origin/ap-refactor/04-consumer-canonicalization",
    }

    fingerprint = payload.pop("fingerprint")
    assert fingerprint == hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def test_ap005e2_build_artifacts_and_entrypoint_metadata() -> None:
    payload = _payload()
    build = payload["dynamic_evidence"]["build"]
    source = payload["source_snapshot"]

    assert build["wheel"]["python_file_count"] == 66
    assert source["packaged_python_source_count"] == 66
    assert len(source["tracked_python_outside_wheel"]) == 1
    assert source["tracked_python_outside_wheel"][0].endswith(".py")
    assert build["wheel"]["package_non_python_file_count"] == 0
    assert build["wheel"]["strong_residues"] == []
    assert build["sdist"]["strong_residues"] == []
    assert build["metadata"] == {
        "console_script": "academic-pipeline = academic_pipeline.cli:main",
        "name": "academic-pipeline-mppg",
        "requires_dist": [],
        "requires_python": ">=3.11",
        "version": "0.1.0",
    }
    assert build["reproducible_wheel_hash_observations"] == 3


def test_ap005e2_dependency_metadata_defect_is_confirmed() -> None:
    payload = _payload()
    source = payload["source_snapshot"]
    wheel_only = payload["dynamic_evidence"]["wheel_only_environment"]

    assert source["pyproject"]["dependencies"] == []
    assert wheel_only["pip_check_returncode"] == 0
    assert wheel_only["legacy_runtime_imported"] is False
    assert "dotenv" in wheel_only["legacy_runtime_error"]

    finding = next(
        item for item in payload["findings"]
        if item["id"] == "distribution-dependencies-empty"
    )
    assert finding["severity"] == "blocking"


def test_ap005e2_package_data_defect_is_confirmed() -> None:
    payload = _payload()
    counts = payload["source_snapshot"]["tracked_data_counts"]
    commands = payload["dynamic_evidence"]["installed_operational_commands"]

    assert counts == {
        "institutions": 18,
        "misc": 6,
        "projetos": 111,
        "prompts": 9,
        "templates": 9,
        "tracked_non_python_total": 184,
    }
    assert payload["dynamic_evidence"]["build"]["wheel"][
        "package_non_python_file_count"
    ] == 0
    assert commands["list_institutions"]["stdout"] == (
        "Nenhum perfil institucional encontrado."
    )
    assert commands["explain_profile_fgv"]["returncode"] == 1
    assert commands["init_project_fgv"]["returncode"] == 1


def test_ap005e2_installed_import_defect_is_confirmed() -> None:
    imports = _payload()["dynamic_evidence"]["passive_module_imports"]
    assert imports["requirements_only"] == {
        "failed": 1,
        "passed": 64,
        "total": 65,
    }
    assert imports["pipfile_direct_dependencies"] == {
        "failed": 1,
        "passed": 64,
        "total": 65,
    }
    assert imports["failure"]["module"] == (
        "app_bundle.scripts.pipeline.artigo_prisma_workflow"
    )
    assert "article_workflow" in imports["failure"]["error"]


def test_ap005e2_prisma_portability_defects_are_recorded() -> None:
    defects = _payload()["source_snapshot"]["source_defects"]
    assert defects["hardcoded_user_prompt_path"] is True
    assert defects["self_invocation_via_dunder_file"] is True
    assert defects["helper_export_sibling_assumption"] is True
    assert defects["helper_freeze_sibling_assumption"] is True
    assert len(defects["article_workflow_absolute_imports"]) == 1


def test_ap005e2_excludes_invalid_or_environmental_probes() -> None:
    commands = _payload()["dynamic_evidence"]["installed_operational_commands"]
    assert commands["list_layouts_attempt"]["classification"] == (
        "invalid_probe_not_used_as_defect_evidence"
    )
    assert commands["doctor"]["classification"] == (
        "environmental_diagnostic_not_packaging_gate"
    )


def test_ap005e2_decision_requires_scoped_ap005e3() -> None:
    payload = _payload()
    decisions = payload["decisions"]
    assert decisions["productive_change_required_in_ap005e2"] is False
    assert decisions["ap005e2_is_characterization_only"] is True
    assert decisions["ap005e3_correction_required"] is True
    assert decisions["ap005e3_may_be_noop"] is False
    assert decisions["preserve_console_script"] is True
    assert decisions["preserve_module_entrypoint"] is True
    assert decisions["package_all_tracked_non_python_allowed"] is False
    assert decisions["package_projects_outputs_or_historical_docs_allowed"] is False

    assert "package app_bundle/projetos" in payload["ap005e3_scope"]["forbidden"]
    assert "package app_bundle/output" in payload["ap005e3_scope"]["forbidden"]


def test_ap005e2_documents_and_tool_are_reproducible() -> None:
    report = REPORT.read_text(encoding="utf-8")
    scope = SCOPE.read_text(encoding="utf-8")

    for path, text in ((REPORT, report), (SCOPE, scope)):
        assert text.endswith("\n"), path
        assert not text.endswith("\n\n"), path

    assert "AP-005E.3 obrigatória" in report
    assert "Não é permitido incluir os 184 arquivos" in scope
    assert "65 módulos passivos" in scope

    tool_source = TOOL.read_text(encoding="utf-8")
    assert "if branch:" in tool_source
    assert "upstream_proc = run_git(" in tool_source
    assert "upstream = None" in tool_source
    assert "Descendentes temporários são validados em HEAD destacado." in tool_source

    proc = subprocess.run(
        [str(TOOL), "--check"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fingerprint=" in proc.stdout
