#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
BASE='17725a5505eb2f9c0b1a6cfd5899e38d70031f80'; EVID='6e0c24d61054bcb9f111f354e9ce8d13fcf48f49091a17b18032bac85b127c20'; CANDIDATES=['docs/refactor/academic-pipeline/AP-007/AP-007B_CLOSURE_REPORT.md', 'docs/refactor/academic-pipeline/AP-007/AP-007B_NATIVE_RUNTIME_CONTRACT.md', 'docs/refactor/academic-pipeline/AP-007/ap007b_closure_manifest.json', 'docs/refactor/academic-pipeline/AP-007/ap007b_native_runtime_contract.json', 'software/academic_pipeline_mppg/academic_pipeline/cli.py', 'software/academic_pipeline_mppg/academic_pipeline/runtime.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007b_closure_contract.py', 'software/academic_pipeline_mppg/tests/characterization/test_ap007b_native_runtime_contract.py', 'tools/refactor/ap007b_validate_closure.py', 'tools/refactor/ap007b_validate_native_runtime_contract.py']; CLI=Path('software/academic_pipeline_mppg/academic_pipeline/cli.py'); RUNTIME=Path('software/academic_pipeline_mppg/academic_pipeline/runtime.py'); MANIFEST=Path('docs/refactor/academic-pipeline/AP-007/ap007b_closure_manifest.json'); CLI_SHA='79b8b7f58397645b6378bbe29566180850da41a4bd5e1beabcdbcf498c196b19'; RUNTIME_SHA='2c83a7628160b6287e48c97e836f71f5d609cb2402ce1f86d6f5ee181ec6c4f2'
def raw(repo,*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True)
def git(repo,*args): return raw(repo,*args).strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dirty(repo):
 out=set()
 for line in raw(repo,'status','--porcelain=v1','--untracked-files=all').splitlines():
  if line:
   if len(line)<4 or line[2]!=' ': raise AssertionError(f'porcelain inesperado: {line!r}')
   out.add(line[3:])
 return out
def validate(repo,mode='auto'):
 repo=Path(repo).resolve()
 for rel in CANDIDATES:
  if not (repo/rel).is_file(): raise AssertionError(f'ausente: {rel}')
 p=json.loads((repo/MANIFEST).read_text(encoding='utf-8'))
 assert p['phase']=='AP-007B.3' and p['status']=='closure_materialized_precommit' and p['gate_ap007b_commit']=='PASS'
 assert p['formal_evidence']['sha256']==EVID and p['commit_readiness']['candidate_paths']==CANDIDATES and p['commit_readiness']['candidate_path_count']==10
 assert p['validation']['contract']=={'passed':20} and p['validation']['source_tree_regressions']=={'passed':24,'deselected':4}
 assert p['deferred_to_ap007e']['isolated_direct_script_tests']==4
 assert sha(repo/CLI)==CLI_SHA and sha(repo/RUNTIME)==RUNTIME_SHA
 tree=ast.parse((repo/RUNTIME).read_text(encoding='utf-8'))
 bad=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'globals','locals','eval','exec','__import__'}]
 if bad: raise AssertionError(bad)
 if git(repo,'diff','--cached','--name-only'): raise AssertionError('staging não vazio')
 d=dirty(repo)
 if mode=='auto': mode='precommit' if d else 'postcommit'
 if mode=='precommit':
  if d!=set(CANDIDATES): raise AssertionError(sorted(d))
  state='ready_for_explicit_commit_authorization'
 elif mode=='postcommit':
  head=git(repo,'rev-parse','HEAD')
  if subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',BASE,head],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode: raise AssertionError('ancestralidade inválida')
  if d: raise AssertionError(sorted(d))
  tracked=set(raw(repo,'ls-files').splitlines()); missing=set(CANDIDATES)-tracked
  if missing: raise AssertionError(sorted(missing))
  state='closed_postcommit'
 else: raise AssertionError(mode)
 return {'phase':'AP-007B.3','mode':mode,'status':'ok','gate_ap007b_commit':'PASS','candidate_path_count':10,'closure_artifact_count':4,'closure_state':state}
def main():
 a=argparse.ArgumentParser(); a.add_argument('repo',nargs='?',type=Path,default=Path.cwd()); a.add_argument('mode',nargs='?',choices=('auto','precommit','postcommit'),default='auto'); x=a.parse_args(); print(json.dumps(validate(x.repo,x.mode),ensure_ascii=False,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
