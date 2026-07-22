#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
BASELINE_HEAD='4db60736cfb4d2be53af32babdcdbfed84c3e6b4'
BRIDGE_REL=Path('software/academic_pipeline_rc10_7_conformidade')
DISPATCH_REL=Path('software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py')
CANONICAL_REL=Path('software/academic_pipeline_mppg')
LEGACY_REL=Path('software/academic_pipeline_mppg/academic_pipeline/legacy.py')
EXPECTED_LEGACY_SHA256='f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9'
EXPECTED_F2_SHA256='ae2c812bbc9e6c93f61fb1463a4337a705c0aecb337a7ef13d76ab5230fd9e28'
F1=['docs/refactor/academic-pipeline/AP-006/AP-006F1_DEPENDENCY_DECISION_MATRIX.md', 'docs/refactor/academic-pipeline/AP-006/ap006f1_dependency_decision_matrix.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f1_dependency_decision_matrix_contract.py', 'tools/refactor/ap006f1_validate_dependency_decision_matrix.py']
F3=['docs/refactor/academic-pipeline/AP-006/AP-006F3_MINIMAL_MATERIALIZATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f3_minimal_materialization.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f3_minimal_materialization_contract.py', 'tools/refactor/ap006f3_validate_minimal_materialization.py']
F4=['docs/refactor/academic-pipeline/AP-006/AP-006F4_COMPARATIVE_SOURCE_DISTRIBUTION_VALIDATION.md', 'docs/refactor/academic-pipeline/AP-006/ap006f4_comparative_source_distribution_validation.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f4_comparative_source_distribution_contract.py', 'tools/refactor/ap006f4_validate_comparative_source_distribution.py']
F5=['docs/refactor/academic-pipeline/AP-006/AP-006F5_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-006/ap006f5_closure_manifest.json', 'software/academic_pipeline_mppg/tests/characterization/test_ap006f5_closure_contract.py', 'tools/refactor/ap006f5_validate_closure.py']

def git(repo:Path,*args:str)->str: return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(value:str)->list[str]: return [x for x in value.splitlines() if x]
def validate(repo:Path,requested_mode:str="auto")->dict[str,object]:
    repo=repo.resolve(); head=git(repo,"rev-parse","HEAD"); bridge=repo/BRIDGE_REL; canonical=repo/CANONICAL_REL; legacy=repo/LEGACY_REL
    if bridge.exists() or bridge.is_symlink(): raise AssertionError("ponte de compatibilidade ainda existe")
    if not canonical.is_dir() or not legacy.is_file(): raise AssertionError("raiz canônica ou fallback ausente")
    if sha256(legacy)!=EXPECTED_LEGACY_SHA256: raise AssertionError("run_legacy foi alterado")
    names={n.name for n in ast.parse(legacy.read_text(encoding="utf-8")).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if "run_legacy" not in names: raise AssertionError("função run_legacy ausente")
    f1,f3,f4,f5=([Path(x) for x in values] for values in (F1,F3,F4,F5))
    for rel in f3:
        if not (repo/rel).is_file(): raise AssertionError(f"artefato AP-006F.3 ausente: {rel}")
    payload=json.loads((repo/f3[1]).read_text(encoding="utf-8"))
    if payload.get("phase")!="AP-006F.3" or payload.get("status")!="ok" or payload.get("gate_ap006f4")!="PASS": raise AssertionError("contrato F3 divergente")
    if payload.get("decisions")!={"bridge":"removed","fallback":"preserved_active_run_legacy"}: raise AssertionError("decisões F3 divergentes")
    if payload.get("f2_evidence",{}).get("sha256")!=EXPECTED_F2_SHA256: raise AssertionError("evidência F2 divergente")
    untracked=set(lines(git(repo,"ls-files","--others","--exclude-standard"))); tracked=set(lines(git(repo,"ls-files"))); diff=lines(git(repo,"diff","--name-only")); staged=lines(git(repo,"diff","--cached","--name-only"))
    if staged: raise AssertionError("staging não está vazio")
    f4_complete=all((repo/r).is_file() for r in f4); f5_complete=all((repo/r).is_file() for r in f5)
    for values in (f4,f5):
        present=[r for r in values if (repo/r).is_file()]
        if present and len(present)!=len(values): raise AssertionError(f"artefatos descendentes parciais: {present}")
    mode=requested_mode
    if mode=="auto": mode="precommit" if any(str(r) in untracked for r in f3) else "postcommit"
    if mode=="precommit":
        expected={str(r) for r in f1+f3+(f4 if f4_complete else [])+(f5 if f5_complete else [])}
        if untracked!=expected: raise AssertionError(f"não rastreados descendentes divergentes: {sorted(untracked)}")
        if f5_complete:
            if set(diff)!={str(BRIDGE_REL),str(DISPATCH_REL)} or len(diff)!=2: raise AssertionError(f"diff F5 divergente: {diff}")
            state="ap006f5_precommit"
        elif f4_complete:
            if set(diff)!={str(BRIDGE_REL),str(DISPATCH_REL)} or len(diff)!=2: raise AssertionError(f"diff F4 divergente: {diff}")
            state="ap006f4_precommit"
        elif set(diff)=={str(BRIDGE_REL)} and len(diff)==1: state="ap006f3_precommit"
        elif set(diff)=={str(BRIDGE_REL),str(DISPATCH_REL)} and len(diff)==2: state="ap006f4_dispatch_repair_precommit"
        else: raise AssertionError(f"diff precommit divergente: {diff}")
    elif mode=="postcommit":
        if subprocess.run(["git","-C",str(repo),"merge-base","--is-ancestor",BASELINE_HEAD,head],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0: raise AssertionError("HEAD não descende do baseline")
        if not all(str(r) in tracked for r in f3): raise AssertionError("artefatos F3 não rastreados por HEAD")
        if git(repo,"ls-tree","HEAD",str(BRIDGE_REL)): raise AssertionError("ponte ainda registrada em HEAD")
        state="ap006f5_postcommit" if all(str(r) in tracked for r in f5) else ("ap006f4_or_later_postcommit" if all(str(r) in tracked for r in f4) else "ap006f3_or_later_postcommit")
    else: raise AssertionError(f"modo inválido: {mode}")
    return {"phase":"AP-006F.3","mode":mode,"head":head,"status":"ok","gate_ap006f4":"PASS","artifact_count":len(f3),"bridge":"absent","fallback":"preserved_active_run_legacy","descendant_state":state}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--mode",choices=("auto","precommit","postcommit"),default="auto"); a=p.parse_args(); print(json.dumps(validate(a.repo,a.mode),indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
