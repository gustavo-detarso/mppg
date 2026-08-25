#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,subprocess,tempfile,os
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/"mppg_orchestrator.py";text=SRC.read_text()
with tempfile.TemporaryDirectory(prefix="mppg-autoloop-v7-") as td:
 td=Path(td);repo=td/"repo";repo.mkdir();subprocess.run(["git","init","-b","master",str(repo)],check=True,capture_output=True);subprocess.run(["git","-C",str(repo),"config","user.name","T"],check=True);subprocess.run(["git","-C",str(repo),"config","user.email","t@example.com"],check=True);(repo/"a.txt").write_text("one\n");subprocess.run(["git","-C",str(repo),"add","a.txt"],check=True);subprocess.run(["git","-C",str(repo),"commit","-m","base"],check=True,capture_output=True);os.environ["MPPG_CANONICAL_REPO"]=str(repo);mp=td/"o.py";mp.write_text(text);spec=importlib.util.spec_from_file_location("o",mp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.conf=lambda:{"max_autoremediation_cycles":8,"max_no_progress_cycles":2,"harness_rebuild_threshold":2,"max_ai_external_retries":2,"runtime_root":str(td/"runtime")};ctx={"refs":{"head":"h"},"front":{"kind":"tracked_change"},"profile":{}};m.context=lambda:(ctx,[])
 def D(status="UNRESOLVED",domain="test",finding="no_defect",actions=None,probes=None):return {"status":status,"blocker_domain":domain,"finding_class":finding,"confidence":0.9,"summary":"s","semantic_evidence":[],"automatic_actions":actions or ["NOOP"],"probe_requests":probes or [],"shadow_patch":"","shadow_validation_probes":[],"mutation_required":False,"mutation_scope":[],"proposed_commit_subject":"fix(test): x","next_gate":"CONTINUE_READ_ONLY"}
 class A1:
  def __init__(self):self.n=0
  def resolve(self,*a):self.n+=1;return D("RESOLVED_READ_ONLY") if self.n==2 else D()
 m.AI=A1;rem,dec,con=m.autoremediate(ctx,[{"domain":"test","code":"X","summary":"x"}],td/"r1");assert not rem and len(dec)==2
 class A2:
  def __init__(self):self.n=0
  def resolve(self,ctx,b,h):
   self.n+=1
   if self.n==1:return D(actions=["RERUN_READONLY_PROBE"],probes=["git_status"])
   assert b["code"]=="ORCHESTRATOR_INTERNAL_EXCEPTION",b
   return D("RESOLVED_READ_ONLY","auditor_harness","auditor_harness_defect")
 orig=m.execute_actions
 def boom(d,rd,n):
  if n==1:return [{"action":"RERUN_READONLY_PROBE","ok":False,"exception_blocker":m.exception_blocker(RuntimeError("boom"),"probe:synthetic")}]
  return [{"action":"NOOP","ok":True}]
 m.AI=A2;m.execute_actions=boom;rem,dec,con=m.autoremediate(ctx,[{"domain":"test","code":"E","summary":"e"}],td/"r2");assert not rem and len(dec)>=2;m.execute_actions=orig
 seen=[]
 class A3:
  def __init__(self):self.n=0
  def resolve(self,*a):self.n+=1;return D("RESOLVED_READ_ONLY","auditor_harness","auditor_harness_defect") if self.n==3 else D("UNRESOLVED","auditor_harness","auditor_harness_defect")
 def cap(d,rd,n):seen.append(list(d["automatic_actions"]));return [{"action":d["automatic_actions"][0],"ok":True}]
 m.AI=A3;m.execute_actions=cap;rem,dec,con=m.autoremediate(ctx,[{"domain":"auditor_harness","code":"H","summary":"h"}],td/"r3");assert not rem and any("REBUILD_EPHEMERAL_AUDITOR" in x for x in seen);m.execute_actions=orig
 class A4:
  def resolve(self,*a):return D("RESOLVED_READ_ONLY","auditor_harness","auditor_harness_defect")
 m.AI=A4;m._ACCEPTANCE_CANARY["enabled"]=True;m._ACCEPTANCE_CANARY["calls"]=0;rem,dec,con=m.autoremediate(ctx,[{"domain":"auditor_harness","code":"SYNTHETIC_CLOSED_LOOP_CANARY","summary":"c"}],td/"r4");assert not rem and m._ACCEPTANCE_CANARY["calls"]>=2 and len(dec)>=3;m._ACCEPTANCE_CANARY["enabled"]=False
print("AUTOREMEDIATION_MULTIROUND=PASS");print("ACTION_EXCEPTION_FEEDBACK_LOOP=PASS");print("HARNESS_SECOND_FAILURE_REBUILD=PASS");print("ACCEPTANCE_CANARY_TWO_ACTION_ROUNDS=PASS")
