#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

BASELINE_HEAD = '4db60736cfb4d2be53af32babdcdbfed84c3e6b4'
BRIDGE = Path('software/academic_pipeline_rc10_7_conformidade')
CANONICAL = Path('software/academic_pipeline_mppg')
DISPATCH = Path('software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py')
LEGACY = Path('software/academic_pipeline_mppg/academic_pipeline/legacy.py')
EXPECTED_DISPATCH_SHA256 = '9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe'
EXPECTED_LEGACY_SHA256 = 'f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9'
EXPECTED_V5_SHA256 = '1a16d9a5528d70adc8f879a1d65471cc380aa99bf03eec65c5fbe0d5650aa335'
EXPECTED_V6_SHA256 = '091e67b58d9fe5d88c55066723afc9cb34ead840d94df4aafe56fd851bee7661'
F1 = ['docs/refactor/academic-pipeline/AP-006/AP-006F1_DEPENDENCY_DECISION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f1_dependency_decision_matrix_contract.py', 'tools/refactor/ap006f1_validate_dependency_decision_matrix.py']
F3 = ['docs/refactor/academic-pipeline/AP-006/AP-006F3_MINIMAL_MATERIALIZATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f3_minimal_materialization_contract.py', 'tools/refactor/ap006f3_validate_minimal_materialization.py']
F4 = ['docs/refactor/academic-pipeline/AP-006/AP-006F4_COMPARATIVE_SOURCE_DISTRIBUTION_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py', 'tools/refactor/ap006f4_validate_comparative_source_distribution.py']
F5 = ['docs/refactor/academic-pipeline/AP-006/AP-006F5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py', 'tools/refactor/ap006f5_validate_closure.py']


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(repo: Path, requested_mode: str = "auto") -> dict[str, object]:
    repo = repo.resolve()
    head = git(repo, "rev-parse", "HEAD")
    canonical = repo / CANONICAL
    dispatch = repo / DISPATCH
    legacy = repo / LEGACY
    bridge = repo / BRIDGE
    if bridge.exists() or bridge.is_symlink():
        raise AssertionError("ponte de compatibilidade ainda existe")
    if not canonical.is_dir() or not dispatch.is_file() or not legacy.is_file():
        raise AssertionError("raiz canônica, dispatch ou fallback ausente")
    if sha256(dispatch) != EXPECTED_DISPATCH_SHA256:
        raise AssertionError("command_dispatch.py divergente")
    if sha256(legacy) != EXPECTED_LEGACY_SHA256:
        raise AssertionError("legacy.py divergente")
    legacy_names = {
        node.name for node in ast.parse(legacy.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "run_legacy" not in legacy_names:
        raise AssertionError("run_legacy ausente")

    f1 = [Path(x) for x in F1]
    f3 = [Path(x) for x in F3]
    f4 = [Path(x) for x in F4]
    f5 = [Path(x) for x in F5]
    for rel in f1 + f3 + f4:
        if not (repo / rel).is_file():
            raise AssertionError(f"artefato obrigatório ausente: {rel}")

    payload = json.loads((repo / f4[1]).read_text(encoding="utf-8"))
    if payload.get("phase") != "AP-006F.4":
        raise AssertionError("phase F4 divergente")
    if payload.get("status") != "comparative_source_distribution_validation_complete":
        raise AssertionError("status F4 divergente")
    if payload.get("gate_ap006f5") != "PASS":
        raise AssertionError("gate F4 divergente")
    if payload.get("evidence", {}).get("v5", {}).get("sha256") != EXPECTED_V5_SHA256:
        raise AssertionError("hash V5 divergente")
    if payload.get("evidence", {}).get("v6", {}).get("sha256") != EXPECTED_V6_SHA256:
        raise AssertionError("hash V6 divergente")
    tests = payload.get("test_results", {})
    if tests.get("logical", {}) != {"passed": 636, "xfailed": 3}:
        raise AssertionError("síntese lógica divergente")
    if payload.get("wheel_comparison", {}).get("member_count") != 110:
        raise AssertionError("contagem de membros do wheel divergente")
    if payload.get("functional_comparison", {}).get("status") != "source_equals_installed_wheel":
        raise AssertionError("comparação funcional divergente")

    untracked = set(lines(git(repo, "ls-files", "--others", "--exclude-standard")))
    tracked = set(lines(git(repo, "ls-files")))
    diff = lines(git(repo, "diff", "--name-only"))
    staged = lines(git(repo, "diff", "--cached", "--name-only"))
    if staged:
        raise AssertionError("staging não está vazio")

    f5_present = [rel for rel in f5 if (repo / rel).is_file()]
    if f5_present and len(f5_present) != len(f5):
        raise AssertionError(f"artefatos F5 parciais: {f5_present}")
    f5_complete = len(f5_present) == len(f5)

    mode = requested_mode
    if mode == "auto":
        mode = "precommit" if any(str(rel) in untracked for rel in f4) else "postcommit"
    if mode == "precommit":
        expected = {str(rel) for rel in f1 + f3 + f4 + (f5 if f5_complete else [])}
        if untracked != expected:
            raise AssertionError(f"não rastreados divergentes: {sorted(untracked)}")
        if set(diff) != {str(BRIDGE), str(DISPATCH)} or len(diff) != 2:
            raise AssertionError(f"diff F4 divergente: {diff}")
        state = "ap006f5_precommit" if f5_complete else "ap006f4_precommit"
    elif mode == "postcommit":
        if subprocess.run([
            "git", "-C", str(repo), "merge-base", "--is-ancestor",
            BASELINE_HEAD, head,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            raise AssertionError("HEAD não descende do baseline")
        if not all(str(rel) in tracked for rel in f4):
            raise AssertionError("artefatos F4 não rastreados por HEAD")
        if git(repo, "ls-tree", "HEAD", str(BRIDGE)):
            raise AssertionError("ponte ainda registrada em HEAD")
        state = "ap006f5_or_later_postcommit" if all(str(rel) in tracked for rel in f5) else "ap006f4_postcommit"
    else:
        raise AssertionError(f"modo inválido: {mode}")

    return {
        "phase": "AP-006F.4",
        "mode": mode,
        "head": head,
        "status": "ok",
        "gate_ap006f5": "PASS",
        "artifact_count": len(f4),
        "bridge": "absent",
        "fallback": "preserved_active_run_legacy",
        "dispatch_repair": "revalidated",
        "logical_suite": {"passed": 636, "xfailed": 3},
        "descendant_state": state,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("auto", "precommit", "postcommit"), default="auto")
    args = parser.parse_args()
    print(json.dumps(validate(args.repo, args.mode), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
