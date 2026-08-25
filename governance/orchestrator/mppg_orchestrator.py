#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, datetime as dt, hashlib, json, os, re, shutil, stat, subprocess, tempfile, urllib.error, urllib.request, uuid
from pathlib import Path
from typing import Any

ROOT=Path("/home/gustavodetarso/Documentos/mppg")
GOV=ROOT/"governance"
MASTER=GOV/"MPPG_PROMPT_MASTER_CANONICO.md"
SOFTWARE_REL="software/academic_pipeline_mppg"
CFG=GOV/"orchestrator/config/defaults.json"
AI_SCHEMA=GOV/"orchestrator/schemas/ai_resolution.schema.json"
AI_PROMPT=GOV/"orchestrator/prompts/semantic_adjudicator.md"
MASTER_SHA="3f3997c771c0b579089598787b9cedd85ba6749a1dab1c3724e212f50052896d"
PRODUCTION_API="https://api.openai.com/v1/responses"
STATE_ROOT=Path(os.path.expanduser("~/.local/state/mppg-orchestrator"))
RUN_ROOT=STATE_ROOT/"runs"

AUTO_ACTIONS={
 "NOOP","RERUN_READONLY_PROBE","RECLASSIFY_FROM_EVIDENCE",
 "REBUILD_EPHEMERAL_AUDITOR","RECOMPUTE_EPHEMERAL_EVIDENCE",
 "RETRY_EXTERNAL_READONLY","SIMULATE_PATCH_IN_SHADOW"
}
SECRET_PATH=[
 re.compile(r"(^|/)\.env($|[._/-])",re.I),
 re.compile(r"(^|/)(id_rsa|id_ed25519|credentials|secrets?|private[_-]?key)(\.|$|/)",re.I)
]

class OrchestratorError(RuntimeError): pass
def shab(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def shaf(p:Path)->str:return shab(p.read_bytes())

def sanitize(s:str,limit:int=20000)->str:
 key=os.environ.get("OPENAI_API_KEY")
 if key:s=s.replace(key,"[REDACTED_API_KEY]")
 s=re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b","[REDACTED_SECRET]",s)
 return s if len(s)<=limit else s[:limit]+"\n...[TRUNCATED]..."

def run(cmd:list[str],cwd:Path|None=None,check:bool=True,env:dict[str,str]|None=None):
 cp=subprocess.run(cmd,cwd=str(cwd) if cwd else None,env=env,text=True,capture_output=True)
 if check and cp.returncode!=0:
  raise OrchestratorError(f"command_failed rc={cp.returncode} cmd={cmd!r} stdout={sanitize(cp.stdout,3000)!r} stderr={sanitize(cp.stderr,3000)!r}")
 return cp
def git(*a:str,check:bool=True):return run(["git","-C",str(ROOT),*a],check=check)
def remote()->str:
 f=git("ls-remote","origin","refs/heads/master").stdout.strip().split()
 if not f:raise OrchestratorError("origin/master unavailable")
 return f[0]

def verify_governance():
 if not MASTER.is_file() or shaf(MASTER)!=MASTER_SHA:raise OrchestratorError("Prompt Master hash mismatch")
 cp=run(["sha256sum","-c","governance/MANIFEST.sha256"],cwd=ROOT,check=False)
 if cp.returncode!=0:raise OrchestratorError("governance manifest invalid")

def refs()->dict[str,str]:
 return {"branch":git("branch","--show-current").stdout.strip(),"head":git("rev-parse","HEAD").stdout.strip(),"tracking":git("rev-parse","@{upstream}").stdout.strip(),"remote":remote()}

def status_items()->list[dict[str,str]]:
 raw=git("status","--porcelain=v1","-z","--untracked-files=all").stdout
 return [{"status":r[:2],"path":r[3:]} for r in raw.split("\0") if len(r)>=4]

def secret_path(rel:str)->bool:return any(p.search(rel) for p in SECRET_PATH)
def safe_path(rel:str)->Path:
 if not rel or rel.startswith("/") or "\0" in rel:raise OrchestratorError("invalid path")
 p=(ROOT/rel).resolve(strict=False)
 try:p.relative_to(ROOT.resolve())
 except ValueError:raise OrchestratorError("path escapes repository")
 return p

def metadata(rel:str)->dict[str,Any]:
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel)
 if not p.exists() and not p.is_symlink():return {"ok":False,"missing":True,"path":rel}
 st=p.lstat()
 kind="symlink" if stat.S_ISLNK(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "directory" if stat.S_ISDIR(st.st_mode) else "other"
 out={"ok":True,"path":rel,"kind":kind,"size":st.st_size}
 if kind=="file":out["sha256"]=shaf(p);out["suffix"]=p.suffix.lower()
 elif kind=="symlink":out["target"]=os.readlink(p)
 return out

def text_excerpt(rel:str,start:int,count:int)->dict[str,Any]:
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel)
 if not p.is_file():return {"ok":False,"reason":"not_regular_file","path":rel}
 data=p.read_bytes()
 if b"\0" in data[:8192]:return {"ok":False,"reason":"binary_file","path":rel}
 lines=data.decode("utf-8",errors="replace").splitlines()
 start=max(1,int(start));count=max(1,min(int(count),200))
 return {"ok":True,"path":rel,"start_line":start,"text":sanitize("\n".join(lines[start-1:start-1+count]),16000)}

def git_grep(pattern:str,prefix:str)->dict[str,Any]:
 if len(pattern)>200:return {"ok":False,"reason":"pattern_too_long"}
 a=["grep","-n","--full-name","-e",pattern]
 if prefix:a+=["--",prefix]
 cp=git(*a,check=False)
 return {"ok":cp.returncode in (0,1),"matches":sanitize(cp.stdout)}

def ast_summary(rel:str)->dict[str,Any]:
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel)
 if p.suffix!=".py" or not p.is_file():return {"ok":False,"reason":"not_python_file"}
 try:t=ast.parse(p.read_text(encoding="utf-8"))
 except Exception as e:return {"ok":False,"reason":"ast_parse_error","error":type(e).__name__}
 im=[];fn=[];cl=[]
 for n in ast.walk(t):
  if isinstance(n,ast.Import):im += [a.name for a in n.names]
  elif isinstance(n,ast.ImportFrom):im.append(n.module or "")
  elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):fn.append(n.name)
  elif isinstance(n,ast.ClassDef):cl.append(n.name)
 return {"ok":True,"imports":sorted(set(im))[:200],"functions":sorted(set(fn))[:300],"classes":sorted(set(cl))[:200]}

def named_probe(name:str)->dict[str,Any]:
 if name=="git_status":return {"ok":True,"items":status_items()}
 if name=="governance_verify":
  try:verify_governance();return {"ok":True,"refs":refs()}
  except Exception as e:return {"ok":False,"error":sanitize(str(e),1000)}
 if name=="software_status":
  cp=git("status","--porcelain=v1","--untracked-files=all","--",SOFTWARE_REL,check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 if name=="git_diff_name_status":
  cp=git("diff","--name-status",check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 if name=="git_cached_name_status":
  cp=git("diff","--cached","--name-status",check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 return {"ok":False,"reason":"unknown_probe"}

TOOLS=[
 {"type":"function","name":"read_path_metadata","description":"Read non-secret path metadata.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"read_text_excerpt","description":"Read bounded non-secret UTF-8 text.","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"max_lines":{"type":"integer"}},"required":["path","start_line","max_lines"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"git_grep","description":"Read-only discovery. Textual presence alone is not live-edge proof.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path_prefix":{"type":"string"}},"required":["pattern","path_prefix"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"ast_summary","description":"Read-only Python AST summary.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"run_named_probe","description":"Run one deterministic read-only probe.","parameters":{"type":"object","properties":{"name":{"type":"string","enum":["git_status","governance_verify","software_status","git_diff_name_status","git_cached_name_status"]}},"required":["name"],"additionalProperties":False},"strict":True}
]
def execute_tool(name:str,args:dict[str,Any])->dict[str,Any]:
 if name=="read_path_metadata":return metadata(args["path"])
 if name=="read_text_excerpt":return text_excerpt(args["path"],args["start_line"],args["max_lines"])
 if name=="git_grep":return git_grep(args["pattern"],args["path_prefix"])
 if name=="ast_summary":return ast_summary(args["path"])
 if name=="run_named_probe":return named_probe(args["name"])
 return {"ok":False,"reason":"tool_not_allowlisted"}

def loadj(p:Path):return json.loads(p.read_text(encoding="utf-8"))
def outtext(resp:dict[str,Any])->str|None:
 for i in resp.get("output",[]):
  if i.get("type")=="message":
   for c in i.get("content",[]):
    if c.get("type")=="output_text":return c.get("text","")
 return None

def validate_resolution(o:dict[str,Any]):
 s=loadj(AI_SCHEMA)
 if set(o)!=set(s["required"]):raise OrchestratorError("AI resolution key-set mismatch")
 if any(a not in AUTO_ACTIONS for a in o["automatic_actions"]):raise OrchestratorError("AI automatic action outside allowlist")
 if not isinstance(o["mutation_required"],bool):raise OrchestratorError("AI mutation_required invalid")

class AI:
 def __init__(self):
  self.key=os.environ.get("OPENAI_API_KEY")
  if not self.key:raise OrchestratorError("OPENAI_API_KEY missing")
  c=loadj(CFG);self.model=os.environ.get("MPPG_OPENAI_MODEL") or c["model"];self.timeout=int(c["http_timeout_seconds"]);self.rounds=int(c["max_ai_rounds_per_blocker"])
  self.schema=loadj(AI_SCHEMA);self.prompt=AI_PROMPT.read_text(encoding="utf-8");self.master=MASTER.read_text(encoding="utf-8")
 def endpoint(self):
  if os.environ.get("MPPG_ORCHESTRATOR_TEST_MODE")=="1":
   u=os.environ.get("MPPG_ORCHESTRATOR_TEST_ENDPOINT")
   if not u:raise OrchestratorError("test endpoint missing")
   return u
  return PRODUCTION_API
 def post(self,payload):
  body=json.dumps(payload,ensure_ascii=False).encode()
  if self.key.encode() in body:raise OrchestratorError("API key leaked into request body")
  req=urllib.request.Request(self.endpoint(),data=body,method="POST",headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json","User-Agent":"mppg-orchestrator/1"})
  try:
   with urllib.request.urlopen(req,timeout=self.timeout) as r:raw=r.read()
  except urllib.error.HTTPError as e:raise OrchestratorError(f"OpenAI HTTP {e.code}: {sanitize(e.read().decode(errors='replace'),3000)}") from None
  except Exception as e:raise OrchestratorError(f"OpenAI request failure: {type(e).__name__}") from None
  return json.loads(raw)
 def resolve(self,context,blocker):
  items=[{"role":"user","content":"PROMPT MASTER:\n\n"+self.master+"\n\nCURRENT CONTEXT/BLOCKER:\n"+json.dumps({"context":context,"blocker":blocker,"automatic_actions":sorted(AUTO_ACTIONS)},ensure_ascii=False,sort_keys=True)}]
  for _ in range(self.rounds):
   payload={"model":self.model,"instructions":self.prompt,"input":items,"tools":TOOLS,"tool_choice":"auto","parallel_tool_calls":False,"store":False,"text":{"format":{"type":"json_schema","name":"mppg_ai_resolution","strict":True,"schema":self.schema}}}
   resp=self.post(payload);output=resp.get("output",[]);items.extend(output)
   calls=[x for x in output if x.get("type")=="function_call"]
   if calls:
    for c in calls:
     try:a=json.loads(c.get("arguments","{}"))
     except Exception:a={}
     result=execute_tool(c.get("name",""),a)
     items.append({"type":"function_call_output","call_id":c.get("call_id"),"output":sanitize(json.dumps(result,ensure_ascii=False,sort_keys=True))})
    continue
   txt=outtext(resp)
   if not txt:raise OrchestratorError("AI returned no structured output")
   o=json.loads(txt);validate_resolution(o);return o
  raise OrchestratorError("AI tool-loop round limit reached")

def detect():
 items=status_items();staged=[x for x in git("diff","--cached","--name-only").stdout.splitlines() if x];tracked=[x for x in git("diff","--name-only").stdout.splitlines() if x];un=[x["path"] for x in items if x["status"]=="??"]
 sw=[p for p in un if p.startswith(SOFTWARE_REL+"/")];govchg=[x for x in items if x["path"].startswith("governance/")];ext=[p for p in un if not p.startswith("governance/") and not p.startswith(SOFTWARE_REL+"/")]
 if staged:k="preexisting_staging"
 elif sw:k="canonical_software_untracked"
 elif govchg:k="governance_change"
 elif tracked:k="tracked_change"
 elif ext:k="repository_content_ingestion"
 else:k="noop"
 return {"kind":k,"staged":staged,"tracked":tracked,"untracked":un,"software_untracked":sw,"external_untracked":ext,"items":items}

def path_manifest(paths):
 ent=[];bl=[]
 for rel in sorted(paths):
  m=metadata(rel);ent.append(m)
  if secret_path(rel):bl.append({"domain":"environment","code":"SECRET_LIKE_PATH","summary":rel})
  if m.get("kind")=="symlink":bl.append({"domain":"repository_state","code":"SYMLINK_REVIEW_REQUIRED","summary":rel})
 return ent,shab(json.dumps(ent,ensure_ascii=False,sort_keys=True).encode()),bl

def context():
 verify_governance();r=refs();b=[]
 if r["branch"]!="master" or not r["head"]==r["tracking"]==r["remote"]:b.append({"domain":"repository_state","code":"REF_DIVERGENCE","summary":"refs differ"})
 f=detect()
 if f["staged"]:b.append({"domain":"repository_state","code":"PREEXISTING_STAGING","summary":"staging non-empty"})
 if f["software_untracked"]:b.append({"domain":"repository_state","code":"CANONICAL_SOFTWARE_UNTRACKED","summary":"canonical software has untracked content"})
 if f["kind"]=="repository_content_ingestion":
  _,fp,x=path_manifest(f["external_untracked"]);b+=x;p={"front":"repository-content-ingestion","front_class":"other","product_artifact_required":False,"user_acceptance_required":False,"representative_runtime_required":False,"candidate_count":len(f["external_untracked"]),"candidate_manifest_sha256":fp}
 elif f["kind"]=="noop":p={"front":"noop","front_class":"other","product_artifact_required":False,"user_acceptance_required":False,"representative_runtime_required":False}
 else:
  p={"front":f["kind"],"front_class":"mixed","product_artifact_required":True,"user_acceptance_required":True,"representative_runtime_required":True}
  b.append({"domain":"authority_model","code":"SEMANTIC_FRONT_ADJUDICATION_REQUIRED","summary":f["kind"]})
 return {"refs":r,"front":f,"profile":p,"prompt_master_sha256":shaf(MASTER)},b

def adjudicate(ctx,blockers):
 if not blockers:return [],[]
 if not os.environ.get("OPENAI_API_KEY"):return blockers+[{"domain":"environment","code":"OPENAI_API_KEY_NOT_EXPORTED","summary":"AI mandatory"}],[]
 ai=AI();remain=[];dec=[]
 for b in blockers:
  if b["domain"] in {"authorization","product_acceptance","user_acceptance"}:remain.append(b);continue
  try:d=ai.resolve(ctx,b);dec.append(d)
  except Exception as e:remain.append({"domain":"external_dependency","code":"AI_RESOLUTION_FAILED","summary":sanitize(str(e),2000)});continue
  if d["status"]=="RESOLVED_READ_ONLY" and d["mutation_required"] is False:
   print("AI_AUTORESOLVED_BLOCKER="+b["code"]);print("AI_FINDING_CLASS="+d["finding_class"]);print("AI_SUMMARY="+d["summary"])
  else:
   remain.append(b);print("AI_NEXT_GATE="+d["next_gate"])
 return remain,dec

def token(kind,payload):
 r=refs();return shab(json.dumps({"kind":kind,"head":r["head"],"remote":r["remote"],"payload":payload},sort_keys=True,separators=(",",":")).encode())
def gate(name,tok):
 e=f"AUTHORIZE {name} {tok}";print("------------------------------------------------------------");print("AUTHORIZATION_REQUIRED="+name);print("TYPE_EXACTLY: "+e)
 try:x=input("> ")
 except EOFError:x=""
 if x==e:print("AUTHORIZATION_ACCEPTED="+name);return True
 print("AUTHORIZATION_ACCEPTED=false");print("GUIDED_RUN_STOPPED_AT_GATE="+name);return False

def exact_stage(paths,fp):
 if git("diff","--cached","--name-only").stdout.strip():raise OrchestratorError("staging non-empty")
 _,cur,b=path_manifest(paths)
 if b or cur!=fp:raise OrchestratorError("candidate drift/blocker")
 fd,tmp=tempfile.mkstemp(prefix="mppg-index-");os.close(fd);os.unlink(tmp);env=os.environ.copy();env["GIT_INDEX_FILE"]=tmp
 run(["git","-C",str(ROOT),"read-tree","HEAD"],env=env);run(["git","-C",str(ROOT),"add","--",*paths],env=env)
 if run(["git","-C",str(ROOT),"diff","--cached","--check","--no-ext-diff","HEAD"],env=env,check=False).returncode!=0:raise OrchestratorError("candidate diff-check failed")
 cp=run(["git","-C",str(ROOT),"diff","--cached","--binary","--full-index","--no-ext-diff","HEAD"],env=env).stdout;cn=run(["git","-C",str(ROOT),"diff","--cached","--name-only","-z"],env=env).stdout
 ir=git("rev-parse","--git-path","index").stdout.strip();idx=Path(ir);idx=idx if idx.is_absolute() else ROOT/idx;bak=Path(tempfile.mktemp(prefix="mppg-real-index-"));shutil.copy2(idx,bak);before=shaf(idx)
 try:
  git("add","--",*paths)
  if git("diff","--cached","--check","--no-ext-diff",check=False).returncode!=0:raise OrchestratorError("real diff-check failed")
  rp=git("diff","--cached","--binary","--full-index","--no-ext-diff","HEAD").stdout;rn=git("diff","--cached","--name-only","-z").stdout
  if shab(cp.encode())!=shab(rp.encode()) or shab(cn.encode())!=shab(rn.encode()):raise OrchestratorError("candidate/real index mismatch")
 except Exception:
  shutil.copy2(bak,idx)
  if shaf(idx)!=before:raise OrchestratorError("index rollback failed")
  raise
 finally:
  Path(tmp).unlink(missing_ok=True);bak.unlink(missing_ok=True)
 print("CANDIDATE_DIFF_CHECK=PASS");print("REAL_CACHED_DIFF_CHECK=PASS");print("EXACT_STAGING=PASS")

def ingestion(ctx):
 paths=ctx["front"]["external_untracked"];fp=ctx["profile"]["candidate_manifest_sha256"]
 print("FRONT=repository-content-ingestion");print("CANDIDATE_COUNT="+str(len(paths)));print("USER_ACCEPTANCE_REQUIRED=false")
 st=token("STAGING",{"paths":paths,"manifest_sha":fp})
 if not gate("STAGING",st):return 0
 exact_stage(paths,fp)
 patch=git("diff","--cached","--binary","--full-index","--no-ext-diff","HEAD").stdout;names=git("diff","--cached","--name-only","-z").stdout;sub=loadj(CFG)["content_ingestion_commit_subject"]
 ct=token("COMMIT",{"patch":shab(patch.encode()),"paths":shab(names.encode()),"subject":sub})
 if not gate("COMMIT",ct):return 0
 par=git("rev-parse","HEAD").stdout.strip()
 if remote()!=par:raise OrchestratorError("remote moved before commit")
 git("commit","-m",sub);new=git("rev-parse","HEAD").stdout.strip()
 if git("rev-parse","HEAD^").stdout.strip()!=par:raise OrchestratorError("commit parent mismatch")
 print("ISOLATED_COMMIT=PASS");print("NEW_COMMIT="+new)
 pt=token("PUBLICATION",{"old_remote":par,"new_head":new})
 if not gate("PUBLICATION",pt):return 0
 if remote()!=par:raise OrchestratorError("remote moved; non-FF risk")
 if git("push","--porcelain","origin","refs/heads/master:refs/heads/master",check=False).returncode!=0:raise OrchestratorError("push failed")
 r=refs()
 if not r["head"]==r["tracking"]==r["remote"]==new:raise OrchestratorError("postpublication refs diverged")
 print("PUBLICATION=PASS");print("POST_PUBLICATION_AUDIT=PASS");print("FRONT_PROGRESS=100_PERCENT");print("FRONT_CLOSED=true");return 0

def self_test():
 verify_governance();ast.parse(Path(__file__).read_text(encoding="utf-8"));c=loadj(CFG);s=loadj(AI_SCHEMA)
 assert c["api_endpoint"]==PRODUCTION_API and s["additionalProperties"] is False and set(s["required"])==set(s["properties"])
 assert all(t["strict"] and t["parameters"]["additionalProperties"] is False for t in TOOLS)
 print("ORCHESTRATOR_SELF_TEST=PASS")
def main():
 p=argparse.ArgumentParser(prog="mppg-orchestrator");sp=p.add_subparsers(dest="cmd",required=True);sp.add_parser("self-test");sp.add_parser("status");rp=sp.add_parser("run");rp.add_argument("--dry-run",action="store_true");a=p.parse_args()
 try:
  if a.cmd=="self-test":return self_test() or 0
  ctx,b=context()
  if a.cmd=="status":print(json.dumps({"context":ctx,"blockers":b,"openai_api_key_present":bool(os.environ.get("OPENAI_API_KEY"))},ensure_ascii=False,indent=2,sort_keys=True));return 0
  rid=dt.datetime.now().strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8];(RUN_ROOT/rid).mkdir(parents=True,exist_ok=True);print("RUN_ID="+rid);print("FRONT_BASELINE_HEAD="+ctx["refs"]["head"]);print("FRONT_KIND="+ctx["front"]["kind"])
  if b:
   b,d=adjudicate(ctx,b);(RUN_ROOT/rid/"ai_decisions.json").write_text(sanitize(json.dumps(d,ensure_ascii=False,indent=2),50000),encoding="utf-8")
  if b:
   print("TOTAL_BLOCKERS="+str(len(b)))
   for x in b:print("BLOCKER_DOMAIN="+x["domain"]);print("BLOCKER_CODE="+x["code"]);print("BLOCKER_SUMMARY="+x["summary"])
   print("FRONT_CLOSED=false");return 2
  if a.dry_run:print("DRY_RUN=PASS");print(json.dumps(ctx["profile"],ensure_ascii=False,indent=2,sort_keys=True));return 0
  if ctx["front"]["kind"]=="repository_content_ingestion":return ingestion(ctx)
  if ctx["front"]["kind"]=="noop":print("NOOP=true");print("FRONT_CLOSED=true");return 0
  print("NEXT_GATE=MATERIALIZATION_AUTHORIZATION");print("FRONT_CLOSED=false");return 3
 except OrchestratorError as e:print("ORCHESTRATOR_BLOCKED="+sanitize(str(e),4000));return 2
if __name__=="__main__":raise SystemExit(main())
