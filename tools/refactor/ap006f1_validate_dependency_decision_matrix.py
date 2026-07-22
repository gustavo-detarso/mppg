#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
BASELINE_HEAD = '4db60736cfb4d2be53af32babdcdbfed84c3e6b4'
BRIDGE = Path('software/academic_pipeline_rc10_7_conformidade')
DISPATCH = Path('software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py')
F1 = ['docs/refactor/academic-pipeline/AP-006/AP-006F1_DEPENDENCY_DECISION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f1_dependency_decision_matrix_contract.py', 'tools/refactor/ap006f1_validate_dependency_decision_matrix.py']
F3 = ['docs/refactor/academic-pipeline/AP-006/AP-006F3_MINIMAL_MATERIALIZATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f3_minimal_materialization_contract.py', 'tools/refactor/ap006f3_validate_minimal_materialization.py']
F4 = ['docs/refactor/academic-pipeline/AP-006/AP-006F4_COMPARATIVE_SOURCE_DISTRIBUTION_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py', 'tools/refactor/ap006f4_validate_comparative_source_distribution.py']
F5 = ['docs/refactor/academic-pipeline/AP-006/AP-006F5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py', 'tools/refactor/ap006f5_validate_closure.py']

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

def lines(value: str) -> list[str]:
    return [x for x in value.splitlines() if x]

def ancestor(repo: Path, old: str, new: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", old, new], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def validate(repo: Path, requested: str = "auto") -> dict[str, object]:
    repo = repo.resolve()
    head = git(repo, "rev-parse", "HEAD")
    f1, f3, f4, f5 = ([Path(x) for x in values] for values in (F1, F3, F4, F5))
    for rel in f1:
        if not (repo / rel).is_file(): raise AssertionError(f"artefato F1 ausente: {rel}")
    payload = json.loads((repo / f1[1]).read_text(encoding="utf-8"))
    if payload.get("schema") != "ap006f1-dependency-decision-matrix-v1": raise AssertionError("schema F1 divergente")
    if payload.get("gates", {}).get("ap006f2") != "PASS": raise AssertionError("gate F1 divergente")
    decisions = {item.get("surface"): item.get("decision") for item in payload.get("decisions", [])}
    if decisions.get("bridge_symlink") != "preserve_pending_ap006f2_no_bridge_trial": raise AssertionError("decisão histórica da ponte divergente")
    if decisions.get("academic_pipeline.legacy:run_legacy") != "preserve_as_active_runtime_adapter_pending_replacement_trial": raise AssertionError("decisão histórica do fallback divergente")
    untracked = set(lines(git(repo, "ls-files", "--others", "--exclude-standard")))
    tracked = set(lines(git(repo, "ls-files")))
    diff = lines(git(repo, "diff", "--name-only"))
    staged = lines(git(repo, "diff", "--cached", "--name-only"))
    if staged: raise AssertionError(f"staging não vazio: {staged}")
    complete = {"f3": all((repo/r).is_file() for r in f3), "f4": all((repo/r).is_file() for r in f4), "f5": all((repo/r).is_file() for r in f5)}
    for key, values in (("f4", f4), ("f5", f5)):
        present = [r for r in values if (repo/r).is_file()]
        if present and len(present) != len(values): raise AssertionError(f"artefatos {key} parciais: {present}")
    mode = requested
    if mode == "auto": mode = "precommit" if untracked or diff else "postcommit"
    if mode == "precommit":
        if complete["f5"]:
            expected = {str(r) for r in f1+f3+f4+f5}
            if untracked != expected: raise AssertionError(f"não rastreados F5 divergentes: {sorted(untracked)}")
            if set(diff) != {str(BRIDGE), str(DISPATCH)} or len(diff) != 2: raise AssertionError(f"diff F5 divergente: {diff}")
            p5 = json.loads((repo/f5[1]).read_text(encoding="utf-8"))
            if p5.get("phase") != "AP-006F.5" or p5.get("gate_ap006f_commit") != "PASS": raise AssertionError("contrato F5 divergente")
            state = "ap006f5_precommit"
        elif complete["f4"]:
            expected = {str(r) for r in f1+f3+f4}
            if untracked != expected: raise AssertionError(f"não rastreados F4 divergentes: {sorted(untracked)}")
            if set(diff) != {str(BRIDGE), str(DISPATCH)} or len(diff) != 2: raise AssertionError(f"diff F4 divergente: {diff}")
            state = "ap006f4_precommit"
        elif complete["f3"]:
            expected = {str(r) for r in f1+f3}
            if untracked != expected: raise AssertionError(f"não rastreados F3 divergentes: {sorted(untracked)}")
            diff_set = set(diff)
            if diff_set == {str(BRIDGE)} and len(diff) == 1: state = "ap006f3_precommit"
            elif diff_set == {str(BRIDGE), str(DISPATCH)} and len(diff) == 2: state = "ap006f4_dispatch_repair_precommit"
            else: raise AssertionError(f"diff descendente divergente: {diff}")
        else:
            expected = {str(r) for r in f1}
            if untracked != expected or not (repo/BRIDGE).is_symlink() or diff: raise AssertionError("estado histórico F1 divergente")
            state = "ap006f1_precommit"
    elif mode == "postcommit":
        if not ancestor(repo, BASELINE_HEAD, head): raise AssertionError("HEAD não descende do baseline")
        if not all(str(r) in tracked for r in f1): raise AssertionError("artefatos F1 não rastreados por HEAD")
        if all(str(r) in tracked for r in f5):
            if git(repo, "ls-tree", "HEAD", str(BRIDGE)): raise AssertionError("ponte ainda registrada no HEAD F5")
            state = "ap006f5_postcommit"
        elif all(str(r) in tracked for r in f4): state = "ap006f4_or_later_postcommit"
        elif all(str(r) in tracked for r in f3): state = "ap006f3_or_later_postcommit"
        else: state = "ap006f1_postcommit"
    else: raise AssertionError(f"modo inválido: {mode}")
    return {"phase":"AP-006F.1","mode":mode,"head":head,"status":"ok","gate_ap006f2":"PASS","artifact_count":len(f1),"descendant_state":state}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--mode",choices=("auto","precommit","postcommit"),default="auto"); a=p.parse_args(); print(json.dumps(validate(a.repo,a.mode),indent=2,ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
