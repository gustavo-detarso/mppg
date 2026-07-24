#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JSON_REL = 'docs/refactor/academic-pipeline/AP-007/ap007e0_distribution_isolation_inventory.json'
DOC_REL = 'docs/refactor/academic-pipeline/AP-007/AP-007E0_DISTRIBUTION_ISOLATION_INVENTORY.md'
TEST_REL = 'software/academic_pipeline_mppg/tests/characterization/test_ap007e0_distribution_isolation_inventory_contract.py'
VALIDATOR_REL = 'tools/refactor/ap007e0_validate_distribution_isolation_inventory.py'
EXPECTED_PATHS = [DOC_REL, JSON_REL, TEST_REL, VALIDATOR_REL]
EXPECTED_COMMIT = '766956710435f1c338d2e0332d24e55106b981b7'
EXPECTED_TREE = '1d673e7c324b74f1fef033578aa995e836da1014'
EXPECTED_RUNTIME_SHA = 'b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c'
EXPECTED_SCHEMA = "ap007e0_distribution_isolation_inventory.v1"
EXPECTED_COMMANDS = ['--help', '--list-institutions', '--list-profiles', '--check-config', '--check-institution-compliance', '--make-doi-manifest']
EXPECTED_GATE = '[GATE] AP-007E.0: INVENTÁRIO DE DISTRIBUIÇÃO, ISOLAMENTO E COMPATIBILIDADE MATERIALIZADO E VALIDADO; NENHUM BUILD, INSTALAÇÃO OU MODIFICAÇÃO PRODUTIVA, SEM STAGING, COMMIT, TAG OU PUSH.'


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []
    data = json.loads((REPO / JSON_REL).read_text(encoding="utf-8"))
    checks = [
        (data.get("schema") == EXPECTED_SCHEMA, "schema"),
        (data.get("phase") == "AP-007E.0", "phase"),
        (data.get("baseline", {}).get("commit") == EXPECTED_COMMIT, "baseline.commit"),
        (data.get("baseline", {}).get("tree") == EXPECTED_TREE, "baseline.tree"),
        (data.get("baseline", {}).get("runtime_sha256") == EXPECTED_RUNTIME_SHA, "runtime sha"),
        (data.get("baseline", {}).get("ap007d_commit_paths_count") == 48, "AP-007D path count"),
        (data.get("scope", {}).get("materialized_paths") == EXPECTED_PATHS, "exact scope"),
        (data.get("scope", {}).get("productive_modules_modified") == [], "no productive modules"),
        (data.get("scope", {}).get("build_executed") is False, "no build"),
        (data.get("scope", {}).get("installation_executed") is False, "no install"),
        (data.get("scope", {}).get("git_write_executed") is False, "no git write"),
        (data.get("distribution", {}).get("tracked_paths_total", 0) >= data.get("distribution", {}).get("tracked_paths_audited", 0), "tracked path counts"),
        (isinstance(data.get("distribution", {}).get("tracked_residual_paths_excluded"), list), "residual exclusions"),
        (data.get("runtime_surfaces", {}).get("selected_commands") == EXPECTED_COMMANDS, "selected commands"),
        (data.get("runtime_surfaces", {}).get("official_target") == "academic_pipeline.cli:main", "official target"),
        (len(data.get("runtime_surfaces", {}).get("source_matrix_ap007e1", [])) >= 4, "source matrix"),
        (data.get("gate") == EXPECTED_GATE, "gate"),
    ]
    for passed, label in checks:
        if not passed:
            failures.append(label)

    for rel in EXPECTED_PATHS:
        if not (REPO / rel).is_file():
            failures.append(f"missing path: {rel}")

    if git("rev-parse", "HEAD") != EXPECTED_COMMIT:
        failures.append("HEAD changed")
    if git("rev-parse", "HEAD^{tree}") != EXPECTED_TREE:
        failures.append("tree changed")
    if git("diff", "--cached", "--name-only"):
        failures.append("staging not empty")

    status_lines = [line for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    observed = sorted(line[3:] for line in status_lines if line.startswith("?? "))
    if observed != sorted(EXPECTED_PATHS):
        failures.append(f"unexpected worktree scope: {status_lines}")

    doc = (REPO / DOC_REL).read_text(encoding="utf-8")
    if EXPECTED_GATE not in doc:
        failures.append("gate missing from markdown")
    if not doc.endswith("\n") or doc.endswith("\n\n"):
        failures.append("markdown EOF normalization")

    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        print(f"AP007E0_VALIDATION_FAILURES={len(failures)}")
        return 1

    print(f"JSON_SHA256={sha256(REPO / JSON_REL)}")
    print(f"DOC_SHA256={sha256(REPO / DOC_REL)}")
    print("AP007E0_VALIDATION_FAILURES=0")
    print(EXPECTED_GATE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
