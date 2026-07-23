#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, json, subprocess, sys
from pathlib import Path
EXPECTED_HEAD='8f30abdcb6bf811f869e09c1fb49ec2d15e0579b'
EXPECTED_PATHS={
'software/academic_pipeline_mppg/academic_pipeline/runtime.py',
'software/academic_pipeline_mppg/academic_pipeline/doctor_runtime.py',
'docs/refactor/academic-pipeline/AP-007/AP-007C2_DOCTOR_NATIVE_ADAPTER.md',
'docs/refactor/academic-pipeline/AP-007/ap007c2_doctor_native_adapter.json',
'software/academic_pipeline_mppg/tests/characterization/test_ap007c2_doctor_native_adapter_contract.py',
'tools/refactor/ap007c2_validate_doctor_native_adapter.py',
'docs/refactor/academic-pipeline/AP-007/AP-007C3_DOCTOR_PUBLIC_INTEGRATION.md',
'docs/refactor/academic-pipeline/AP-007/ap007c3_doctor_public_integration.json',
'software/academic_pipeline_mppg/tests/characterization/test_ap007c3_doctor_public_integration_contract.py',
'tools/refactor/ap007c3_validate_doctor_public_integration.py'}
def git(repo,*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dirty(repo):
    raw=subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'],text=True)
    return {line[3:].strip().strip('"') for line in raw.splitlines() if line}
def validate(repo):
    assert git(repo,'rev-parse','HEAD')==EXPECTED_HEAD; assert dirty(repo)==EXPECTED_PATHS; assert not git(repo,'diff','--cached','--name-only')
    p=json.loads((repo/'docs/refactor/academic-pipeline/AP-007/ap007c3_doctor_public_integration.json').read_text(encoding='utf-8'))
    assert p['phase']=='AP-007C.3' and p['status']=='doctor_publicly_integrated'; assert set(p['candidate_paths'])==EXPECTED_PATHS; assert p['public_route']=={'doctor':'native_doctor','check_config':'legacy_fallback'}
    for rel,expected in p['artifact_sha256'].items(): assert sha(repo/rel)==expected
    software=repo/'software/academic_pipeline_mppg'; sys.path.insert(0,str(software)) if str(software) not in sys.path else None
    from academic_pipeline import runtime
    assert runtime.select_runtime_route(('--doctor',)) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(('--check-config',)) is runtime.RuntimeRoute.LEGACY_FALLBACK
    assert runtime.select_runtime_route(tuple(p['precedence_probe_argv'])) is runtime.RuntimeRoute.LEGACY_FALLBACK
    source=(software/'academic_pipeline/runtime.py').read_text(encoding='utf-8'); ast.parse(source)
    for forbidden in ('globals(','locals(','sys.path','importlib','academic_pipeline_rc10'): assert forbidden not in source
    return {'ok':True,'phase':'AP-007C.3','path_count':10,'doctor_route':'native_doctor','check_config_route':'legacy_fallback'}
def main():
    repo=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd().resolve(); print(json.dumps(validate(repo),ensure_ascii=False,sort_keys=True)); print('AP-007C.3 doctor public integration: OK'); return 0
if __name__=='__main__': raise SystemExit(main())
