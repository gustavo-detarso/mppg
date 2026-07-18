from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TEST_FILE = Path(__file__).resolve()
REPO_ROOT = TEST_FILE.parents[4]

INVENTORY = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005d_facade_inventory.json"
)
STRATEGY = (
    REPO_ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "AP-005D_FACADE_STRATEGY.md"
)
TOOL = REPO_ROOT / "tools/refactor/ap005d_inventory_facades.py"

CANONICAL_MODULES = (
    "academic_pipeline.cli_parser",
    "academic_pipeline.command_dispatch",
    "academic_pipeline.document_orchestration",
    "academic_pipeline.prisma_generic_orchestration",
)


def _payload() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _by_module() -> dict[str, dict]:
    return {
        item["module"]: item
        for item in _payload()["candidates"]
    }


def test_inventory_schema_fingerprint_and_baseline() -> None:
    payload = _payload()

    assert payload["schema"] == "ap005d.facade-inventory.v1"
    assert payload["baseline_commit"] == (
        "78f3be0fce0dd8f79e55729a7111a9359c9edb8d"
    )
    assert payload["scope"]["auditable_python_files"] == 145
    assert payload["source_manifest"] == (
        "git ls-tree -r -z --name-only 78f3be0fce0dd8f79e55729a7111a9359c9edb8d"
    )

    canonical = dict(payload)
    expected = canonical.pop("fingerprint")

    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == expected


def test_candidate_manifest_is_exact() -> None:
    assert set(_by_module()) == {
        "app_bundle.scripts.pipeline.article_workflow",
        "academic_pipeline",
        *CANONICAL_MODULES,
    }


def test_article_workflow_is_only_true_facade() -> None:
    records = _by_module()
    workflow = records[
        "app_bundle.scripts.pipeline.article_workflow"
    ]

    assert workflow["all"] == [
        "STAGES",
        "StageRecord",
        "WorkflowState",
        "ArticleWorkflow",
        "StageValidation",
    ]
    assert len(workflow["reexports"]) == 5
    assert "facade_reexport_publico_verdadeiro" in (
        workflow["classification"]
    )

    true_facades = [
        record["module"]
        for record in records.values()
        if "facade_reexport_publico_verdadeiro"
        in record["classification"]
    ]

    assert true_facades == [
        "app_bundle.scripts.pipeline.article_workflow"
    ]


def test_public_package_surface_is_preserved() -> None:
    root = _by_module()["academic_pipeline"]

    assert root["all"] == ["main"]
    assert root["local_exports"] == ["main"]
    assert root["reexports"] == []
    assert root["false_positive_names"] == [
        "Sequence",
        "annotations",
    ]
    assert root["decision"] == "preserve_unchanged"


@pytest.mark.parametrize("module", CANONICAL_MODULES)
def test_all_declarative_modules_are_not_facades(
    module: str,
) -> None:
    record = _by_module()[module]

    assert record["all_present"] is True
    assert record["dynamic_all"] is False
    assert record["reexports"] == []
    assert record["local_exports"]
    assert "nao_e_facade" in record["classification"]
    assert record["decision"] == "preserve_unchanged"


def test_no_productive_change_is_required() -> None:
    payload = _payload()

    assert payload["decision"]["productive_changes_required"] is False
    assert payload["summary"]["productive_changes_required"] is False
    assert payload["summary"]["true_facade_count"] == 1
    assert payload["summary"]["canonical_non_facade_count"] == 4


def test_inventory_tool_check_is_reproducible() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert '"ls-tree",' in source
    assert '"--name-only",' in source
    assert "BASELINE_COMMIT," in source
    assert 'run_git(root, "ls-files", "-z")' not in source

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--root",
            str(REPO_ROOT),
            "--check",
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert "candidates=6" in proc.stdout
    assert "productive_changes_required=False" in proc.stdout


def test_strategy_records_preservation_scope() -> None:
    text = STRATEGY.read_text(encoding="utf-8")

    assert "A AP-005D não requer alteração produtiva." in text
    assert "app_bundle.scripts.pipeline.article_workflow" in text
    assert "`Sequence` e `annotations`" in text
    assert "não são facades" in text
    assert "árvore Git do commit baseline" in text
    assert "fases posteriores" in text
