from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DATA = json.loads((REPO / 'docs/refactor/academic-pipeline/AP-007/ap007e2_controlled_build_inventory.json').read_text(encoding="utf-8"))


def test_ap007e2_scope_and_isolation_contract() -> None:
    assert DATA["schema"] == "ap007e2_controlled_build_inventory.v1"
    assert DATA["phase"] == "AP-007E.2"
    assert DATA["scope"]["build_executed"] is True
    assert DATA["scope"]["installation_executed"] is False
    assert DATA["scope"]["dependency_installation_executed"] is False
    assert DATA["scope"]["productive_modules_modified"] == []
    assert DATA["scope"]["git_write_executed"] is False
    assert len(DATA["snapshots"]) == 2
    for snapshot in DATA["snapshots"]:
        assert snapshot["residual_filter"]["applied_before_extraction"] is True
        assert snapshot["residual_filter"]["destination_path_constructed_before_classification"] is False
        assert snapshot["archive_member_count_total"] == snapshot["member_count_extracted"] + snapshot["residual_member_count_excluded"]
        assert snapshot["residual_member_count_excluded"] > 0


def test_ap007e2_artifacts_and_metadata_contract() -> None:
    assert len(DATA["builds"]) == 2
    for build in DATA["builds"]:
        assert build["wheel"]["member_count"] > 0
        assert build["sdist"]["member_count"] > 0
        assert build["wheel"]["entrypoints"]["console_scripts"]["academic-pipeline"] == "academic_pipeline.cli:main"
        assert build["wheel"]["required_paths_missing"] == []
        assert build["wheel"]["residual_paths"] == []
        assert build["sdist"]["residual_paths"] == []


def test_ap007e2_reproducibility_contract() -> None:
    assert DATA["reproducibility"]["normalized_reproducible"] is True
    assert DATA["reproducibility"]["wheel"]["equivalent"] is True
    assert DATA["reproducibility"]["sdist"]["equivalent"] is True
    assert DATA["summary"]["entrypoint_metadata_validated"] is True
    assert DATA["summary"]["residual_paths_packaged"] == 0
    assert DATA["summary"]["absolute_worktree_path_leaks"] == 0
