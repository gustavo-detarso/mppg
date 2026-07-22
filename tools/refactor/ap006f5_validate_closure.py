#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
BASELINE_HEAD='4db60736cfb4d2be53af32babdcdbfed84c3e6b4'
BRIDGE=Path('software/academic_pipeline_rc10_7_conformidade')
CANONICAL=Path('software/academic_pipeline_mppg')
DISPATCH=Path('software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py')
LEGACY=Path('software/academic_pipeline_mppg/academic_pipeline/legacy.py')
EXPECTED_DISPATCH_SHA256='9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe'
EXPECTED_LEGACY_SHA256='f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9'
EXPECTED_F4_FORMAL_EVIDENCE_SHA256='828ddc8e1d4fe7f7a776a6228b5a3da4fd4697a06a1e85470785632ff18a42e1'
F1=['docs/refactor/academic-pipeline/AP-006/AP-006F1_DEPENDENCY_DECISION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f1_dependency_decision_matrix_contract.py', 'tools/refactor/ap006f1_validate_dependency_decision_matrix.py']
F3=['docs/refactor/academic-pipeline/AP-006/AP-006F3_MINIMAL_MATERIALIZATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f3_minimal_materialization_contract.py', 'tools/refactor/ap006f3_validate_minimal_materialization.py']
F4=['docs/refactor/academic-pipeline/AP-006/AP-006F4_COMPARATIVE_SOURCE_DISTRIBUTION_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py', 'tools/refactor/ap006f4_validate_comparative_source_distribution.py']
F5=['docs/refactor/academic-pipeline/AP-006/AP-006F5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py', 'tools/refactor/ap006f5_validate_closure.py']
COMMIT_PATHS=['docs/refactor/academic-pipeline/AP-006/AP-006F1_DEPENDENCY_DECISION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-006/AP-006F3_MINIMAL_MATERIALIZATION.md', 'docs/refactor/academic-pipeline/AP-006/AP-006F4_COMPARATIVE_SOURCE_DISTRIBUTION_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-006/AP-006F5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json', 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json', 'docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json', 'docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json', 'software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f1_dependency_decision_matrix_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f3_minimal_materialization_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py', 'software/academic_pipeline_rc10_7_conformidade', 'tools/refactor/ap006f1_validate_dependency_decision_matrix.py', 'tools/refactor/ap006f3_validate_minimal_materialization.py', 'tools/refactor/ap006f4_validate_comparative_source_distribution.py', 'tools/refactor/ap006f5_validate_closure.py']

def git(repo:Path,*args:str)->str: return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def lines(value:str)->list[str]: return [x for x in value.splitlines() if x]
def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(repo:Path,requested_mode:str="auto")->dict[str,object]:
    repo=repo.resolve(); head=git(repo,"rev-parse","HEAD"); bridge=repo/BRIDGE; canonical=repo/CANONICAL; dispatch=repo/DISPATCH; legacy=repo/LEGACY
    if bridge.exists() or bridge.is_symlink(): raise AssertionError("ponte de compatibilidade ainda existe")
    if not canonical.is_dir() or not dispatch.is_file() or not legacy.is_file(): raise AssertionError("topologia final inválida")
    if sha256(dispatch)!=EXPECTED_DISPATCH_SHA256: raise AssertionError("command_dispatch.py divergente")
    if sha256(legacy)!=EXPECTED_LEGACY_SHA256: raise AssertionError("legacy.py divergente")
    names={n.name for n in ast.parse(legacy.read_text(encoding="utf-8")).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if "run_legacy" not in names: raise AssertionError("run_legacy ausente")
    f1,f3,f4,f5=([Path(x) for x in values] for values in (F1,F3,F4,F5))
    for rel in f1+f3+f4+f5:
        if not (repo/rel).is_file(): raise AssertionError(f"artefato de encerramento ausente: {rel}")
    p4=json.loads((repo/f4[1]).read_text(encoding="utf-8")); p5=json.loads((repo/f5[1]).read_text(encoding="utf-8"))
    if p4.get("phase")!="AP-006F.4" or p4.get("gate_ap006f5")!="PASS": raise AssertionError("contrato F4 divergente")
    if p5.get("phase")!="AP-006F.5" or p5.get("status")!="closure_materialized_precommit": raise AssertionError("manifesto F5 divergente")
    if p5.get("gate_ap006f_commit")!="PASS": raise AssertionError("gate de commit divergente")
    if p5.get("closure_evidence",{}).get("formal_f4_materialization",{}).get("sha256")!=EXPECTED_F4_FORMAL_EVIDENCE_SHA256: raise AssertionError("evidência formal F4 divergente")
    if p5.get("test_results",{}).get("logical")!={"passed":636,"xfailed":3}: raise AssertionError("síntese lógica divergente")
    if p5.get("commit_readiness",{}).get("candidate_paths")!=COMMIT_PATHS: raise AssertionError("caminhos candidatos divergentes")
    untracked=set(lines(git(repo,"ls-files","--others","--exclude-standard"))); tracked=set(lines(git(repo,"ls-files"))); diff=lines(git(repo,"diff","--name-only")); staged=lines(git(repo,"diff","--cached","--name-only"))
    if staged: raise AssertionError("staging não está vazio")
    mode=requested_mode
    if mode=="auto": mode="precommit" if untracked or diff else "postcommit"
    if mode=="precommit":
        expected={str(r) for r in f1+f3+f4+f5}
        if untracked!=expected: raise AssertionError(f"não rastreados F5 divergentes: {sorted(untracked)}")
        if set(diff)!={str(BRIDGE),str(DISPATCH)} or len(diff)!=2: raise AssertionError(f"diff F5 divergente: {diff}")
        state="ready_for_explicit_commit_authorization"
    elif mode=="postcommit":
        if subprocess.run(["git","-C",str(repo),"merge-base","--is-ancestor",BASELINE_HEAD,head],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0: raise AssertionError("HEAD não descende do baseline")
        if not all(str(r) in tracked for r in f1+f3+f4+f5): raise AssertionError("artefatos de encerramento não rastreados por HEAD")
        if git(repo,"ls-tree","HEAD",str(BRIDGE)): raise AssertionError("ponte ainda registrada em HEAD")
        if untracked or diff: raise AssertionError("worktree pós-commit não está limpa")
        state="closed_postcommit"
    else: raise AssertionError(f"modo inválido: {mode}")
    return {"phase":"AP-006F.5","mode":mode,"head":head,"status":"ok","gate_ap006f_commit":"PASS","artifact_count":len(f5),"commit_candidate_path_count":len(COMMIT_PATHS),"bridge":"absent","fallback":"preserved_active_run_legacy","logical_suite":{"passed":636,"xfailed":3},"closure_state":state}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--mode",choices=("auto","precommit","postcommit"),default="auto"); a=p.parse_args(); print(json.dumps(validate(a.repo,a.mode),indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
