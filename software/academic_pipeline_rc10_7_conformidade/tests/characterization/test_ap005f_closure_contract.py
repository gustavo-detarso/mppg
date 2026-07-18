from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005"
    / "ap005f_closure_manifest.json"
)
REPORT = (
    ROOT
    / "docs/refactor/academic-pipeline/AP-005"
    / "AP-005F_CLOSURE_REPORT.md"
)
VALIDATOR = ROOT / "tools/refactor/ap005f_validate_closure.py"

EXPECTED_HEAD = "e5e0d85178d8498c303ad2e8ccc9102f2c8222c8"
EXPECTED_BRANCH = "ap-refactor/04-consumer-canonicalization"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_ap005f_manifest_schema_and_fingerprint() -> None:
    data = load_manifest()
    fingerprint = data.pop("fingerprint")
    raw = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert data["schema"] == "ap005f.closure-manifest.v1"
    assert fingerprint == hashlib.sha256(raw).hexdigest()


def test_ap005f_baseline_identity_is_frozen() -> None:
    data = load_manifest()
    assert data["phase"] == "AP-005F"
    assert data["baseline"] == {
        "branch": EXPECTED_BRANCH,
        "head": EXPECTED_HEAD,
        "upstream": "origin/ap-refactor/04-consumer-canonicalization",
        "origin": "git@github.com:gustavo-detarso/mppg.git",
    }


def test_ap005f_phase_commit_chain_is_complete() -> None:
    data = load_manifest()
    commits = data["phase_commits"]

    assert len(commits) == 10
    assert commits[0]["hash"] == (
        "6ef568b250390e12dc2e86b86a8c530188604a28"
    )
    assert commits[-1]["hash"] == EXPECTED_HEAD
    assert [item["hash"] for item in commits] == list(
        dict.fromkeys(item["hash"] for item in commits)
    )


def test_ap005f_preclosure_inventory_is_preserved() -> None:
    data = load_manifest()
    documents = data["tracked_documentation_before_closure"]
    contracts = data["characterization_contracts_before_closure"]

    assert len(documents) == 21
    assert len(contracts) == 14
    assert all((ROOT / path).is_file() for path in documents)
    assert all((ROOT / path).is_file() for path in contracts)


def test_ap005f_final_validation_evidence_is_frozen() -> None:
    data = load_manifest()
    validation = data["validation"]

    assert validation["canonical_suite"] == {
        "passed": 573,
        "xfailed": 3,
        "returncode": 0,
    }
    assert validation["distribution"]["package_data_resources"] == 38
    assert validation["distribution"]["passive_modules"] == 65
    assert validation["distribution"]["build_warnings_classified"] == 0
    assert validation["isolated_installation"]["passive_import_failures"] == 0
    assert len(data["known_xfails"]) == 3


def test_ap005f_closure_artifact_manifest_is_exact() -> None:
    data = load_manifest()
    expected = {
        "docs/refactor/academic-pipeline/AP-005/AP-005F_CLOSURE_REPORT.md",
        "docs/refactor/academic-pipeline/AP-005/ap005f_closure_manifest.json",
        (
            "software/academic_pipeline_rc10_7_conformidade/"
            "tests/characterization/test_ap005f_closure_contract.py"
        ),
        "tools/refactor/ap005f_validate_closure.py",
    }

    assert set(data["closure_artifacts"]) == expected
    assert all((ROOT / path).is_file() for path in expected)
    assert data["scope"]["production_code_changes"] == 0
    assert data["closure_decision"] == (
        "ready_for_explicit_commit_and_publication_approval"
    )


def test_ap005f_report_mirrors_closure_decision() -> None:
    data = load_manifest()
    report = REPORT.read_text(encoding="utf-8")

    assert "# AP-005F — Relatório de encerramento da AP-005" in report
    assert data["fingerprint"] in report
    assert "573 passed e 3 xfailed" in report
    assert "38" in report
    assert "65" in report
    assert "AP-006" in report


def test_ap005f_validator_check_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--root",
            str(ROOT),
            "--check",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, (
        result.stdout,
        result.stderr,
    )
    assert "schema=ap005f.closure-manifest.v1" in result.stdout
    assert "artefatos de encerramento=4" in result.stdout
