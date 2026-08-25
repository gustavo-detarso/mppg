#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE=Path(__file__).resolve()
MODULE=HERE.parents[1]/"mppg_orchestrator.py"

def load_module():
 spec=importlib.util.spec_from_file_location("mppg_orchestrator_checkpoint_test",MODULE)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def rejected(fn,label):
 try:fn()
 except Exception:return
 raise AssertionError(label+" was accepted")

def main():
 with tempfile.TemporaryDirectory(prefix="mppg-proof-continuity-") as td:
  os.environ["MPPG_CHECKPOINT_ROOT"]=td
  m=load_module();scope=["governance/contracts/AUTHORIZATION_GATES.md"]

  rejected(lambda:m.write_gate_checkpoint("missing-predecessor","COMMIT",scope,"commit-token",{},"0"*64),"missing predecessor")

  staging=m.write_gate_checkpoint("scope-check", "STAGING",scope,"staging-token",{"patch_sha256":"a"*64})
  rejected(lambda:m.validate_gate_checkpoint("scope-check","STAGING",expected_scope=["governance/contracts/ORCHESTRATOR_ARCHITECTURE.md"]),"scope mismatch")
  rejected(lambda:m.validate_gate_checkpoint("scope-check","STAGING",expected_token="different-token"),"token mismatch")

  m.write_gate_checkpoint("tamper-check","STAGING",scope,"staging-token",{"patch_sha256":"b"*64})
  tampered=Path(td)/"tamper-check"/"staging.json"
  record=json.loads(tampered.read_text(encoding="utf-8"));record["gate_token"]="tampered"
  tampered.write_text(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
  rejected(lambda:m.validate_gate_checkpoint("tamper-check","STAGING"),"tampered checkpoint")

  first=m.write_gate_checkpoint("restart-check","STAGING",scope,"restart-staging-token",{"patch_sha256":"c"*64})
  code=("import importlib.util,os,sys;"
        "p=sys.argv[1];s=importlib.util.spec_from_file_location('restart_module',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "r=m.validate_gate_checkpoint('restart-check','STAGING',expected_scope=['governance/contracts/AUTHORIZATION_GATES.md'],expected_token='restart-staging-token');"
        "print(r['checkpoint_sha256'])")
  env=os.environ.copy();env["MPPG_CHECKPOINT_ROOT"]=td
  cp=subprocess.run([sys.executable,"-B","-S","-c",code,str(MODULE)],env=env,text=True,capture_output=True,check=True)
  assert cp.stdout.strip()==first["checkpoint_sha256"]

 print("GIT_PROOF_MISSING_PREDECESSOR_REJECTION=PASS")
 print("GIT_PROOF_TAMPER_REJECTION=PASS")
 print("GIT_PROOF_SCOPE_TOKEN_MISMATCH_REJECTION=PASS")
 print("GIT_PROOF_RESTART_RECOVERY=PASS")
 return 0

if __name__=="__main__":raise SystemExit(main())
