#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

EXPECTED_HEAD = "1b4c71c204a0314aa5bab6db5b49cc1ada86b234"
EXPECTED_TREE = "a95de261751e6274a8fa7fbf8010dc4d039507b3"
EXPECTED_FINGERPRINT = "7d570f3786f18a2a07b150763cc5e6e1fdf4cc646b394951f85fef9ac3bffe45"
OLD_REL = Path("software/academic_pipeline_rc10_7_conformidade")
NEW_REL = Path("software/academic_pipeline_mppg")
DECISION_REL = Path(
    "docs/refactor/academic-pipeline/AP-006/ap006c_physical_materialization.json"
)
RESOLVER_REL = NEW_REL / "academic_pipeline/repository_paths.py"


def _load_resolver(repo: Path):
    path = repo / RESOLVER_REL
    spec = importlib.util.spec_from_file_location("ap006c_repository_paths", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(repo: Path) -> None:
    old_path = repo / OLD_REL
    new_path = repo / NEW_REL

    assert old_path.is_symlink()
    assert os.readlink(old_path) == NEW_REL.name
    assert new_path.is_dir() and not new_path.is_symlink()
    assert old_path.resolve() == new_path.resolve()

    decision = json.loads((repo / DECISION_REL).read_text(encoding="utf-8"))
    assert decision["phase"] == "AP-006C"
    assert decision["baseline_commit"] == EXPECTED_HEAD
    assert decision["source_tree_oid"] == EXPECTED_TREE
    assert decision["move_fingerprint_sha256"] == EXPECTED_FINGERPRINT

    topology = decision["physical_topology"]
    assert topology["canonical_project_path"] == NEW_REL.as_posix()
    assert topology["compatibility_path"] == OLD_REL.as_posix()
    assert topology["relative_symlink_target"] == NEW_REL.name
    assert topology["single_source_of_truth"] == NEW_REL.as_posix()
    assert topology["tracked_source_entry_count"] == 520
    assert topology["ignored_residue_count_moved"] == 112

    resolver = _load_resolver(repo)
    assert resolver.repository_project_root(new_path) == new_path.resolve()
    assert resolver.repository_project_root(old_path) == new_path.resolve()
    assert resolver.repository_resource(
        "pyproject.toml", start=new_path
    ) == (new_path / "pyproject.toml").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    validate(args.repo_root.resolve())
    print("AP-006C physical materialization contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
