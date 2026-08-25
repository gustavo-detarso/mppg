#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, datetime as dt, hashlib, json, os, re, shutil, stat, subprocess, tempfile, urllib.error, urllib.request, uuid
from pathlib import Path
from typing import Any

ROOT=Path(os.environ.get("MPPG_CANONICAL_REPO","/home/gustavodetarso/Documentos/mppg"))
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
AUTO_ACTIONS={"NOOP","RERUN_READONLY_PROBE","RECLASSIFY_FROM_EVIDENCE","REBUILD_EPHEMERAL_AUDITOR","RECOMPUTE_EPHEMERAL_EVIDENCE","RETRY_EXTERNAL_READONLY","SIMULATE_PATCH_IN_SHADOW"}
PROBE_NAMES={"git_status","governance_verify","software_status","git_diff_name_status","git_cached_name_status","runtime_source_equivalence","ignored_python_cache_inventory","external_untracked_fingerprint","current_scope_diff_check","python_compile_temp_copy","canonical_tree_clean","protected_state_summary","closed_loop_canary"}
SHADOW_PROBES={"git_diff_check","python_compile_changed","governance_manifest","static_tests","orchestrator_self_test"}
SECRET_PATH=[re.compile(r"(^|/)\.env($|[._/-])",re.I),re.compile(r"(^|/)(id_rsa|id_ed25519|credentials|secrets?|private[_-]?key)(\.|$|/)",re.I)]
_ACCEPTANCE_CANARY={"enabled":False,"calls":0}

class OrchestratorError(RuntimeError):pass
class BlockerError(OrchestratorError):
 def __init__(self,domain,code,summary):super().__init__(summary);self.blocker={"domain":domain,"code":code,"summary":summary}
def shab(b):return hashlib.sha256(b).hexdigest()
def shaf(p):return shab(p.read_bytes())
class CommandResult:
 def __init__(self,cp):
  self.returncode=cp.returncode;self.stdout_bytes=cp.stdout or b"";self.stderr_bytes=cp.stderr or b""
  self.stdout=self.stdout_bytes.decode("utf-8",errors="surrogateescape");self.stderr=self.stderr_bytes.decode("utf-8",errors="surrogateescape")
def sanitize(s,limit=20000):
 key=os.environ.get("OPENAI_API_KEY")
 if key:s=s.replace(key,"[REDACTED_API_KEY]")
 s=re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b","[REDACTED_SECRET]",s)
 return s if len(s)<=limit else s[:limit]+"\n...[TRUNCATED]..."
def run(cmd,cwd=None,check=True,env=None):
 cp=subprocess.run(cmd,cwd=str(cwd) if cwd else None,env=env,text=False,capture_output=True);o=CommandResult(cp)
 if check and o.returncode!=0:raise OrchestratorError(f"command_failed rc={o.returncode} cmd={cmd!r} stdout={sanitize(o.stdout,3000)!r} stderr={sanitize(o.stderr,3000)!r}")
 return o
def git(*a,check=True):return run(["git","-C",str(ROOT),*a],check=check)
def real_index_path():
 p=Path(git("rev-parse","--git-path","index").stdout.strip())
 return p if p.is_absolute() else ROOT/p
def git_ro(*a,check=True):
 """Run Git inspection against a disposable copy of the real index."""
 ip=real_index_path()
 with tempfile.TemporaryDirectory(prefix="mppg-ro-index-") as td:
  shadow=Path(td)/"index"
  shutil.copy2(ip,shadow)
  env=os.environ.copy()
  env["GIT_INDEX_FILE"]=str(shadow)
  env["GIT_OPTIONAL_LOCKS"]="0"
  return run(["git","-C",str(ROOT),*a],check=check,env=env)
def loadj(p):return json.loads(p.read_text(encoding="utf-8"))
def conf():return loadj(CFG)
def remote():
 f=git("ls-remote","origin","refs/heads/master").stdout.strip().split()
 if not f:raise BlockerError("external_dependency","REMOTE_MASTER_UNAVAILABLE","origin/master unavailable")
 return f[0]
def verify_governance():
 if not MASTER.is_file() or shaf(MASTER)!=MASTER_SHA:raise BlockerError("authority_model","PROMPT_MASTER_MISMATCH","Prompt Master hash mismatch")
 if run(["sha256sum","-c","governance/MANIFEST.sha256"],cwd=ROOT,check=False).returncode!=0:raise BlockerError("authority_model","GOVERNANCE_MANIFEST_INVALID","governance manifest invalid")
def refs():return {"branch":git("branch","--show-current").stdout.strip(),"head":git("rev-parse","HEAD").stdout.strip(),"tracking":git("rev-parse","@{upstream}").stdout.strip(),"remote":remote()}
def status_items():
 raw=git_ro("status","--porcelain=v1","-z","--untracked-files=all").stdout
 return [{"status":x[:2],"path":x[3:]} for x in raw.split("\0") if len(x)>=4]
def secret_path(rel):return any(x.search(rel) for x in SECRET_PATH)
def safe_path(rel,root=None):
 root=root or ROOT
 if not rel or rel.startswith("/") or "\0" in rel:raise OrchestratorError("invalid path")
 p=(root/rel).resolve(strict=False)
 try:p.relative_to(root.resolve())
 except ValueError:raise OrchestratorError("path escapes root")
 return p
def metadata(rel,root=None):
 root=root or ROOT
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel,root)
 if not p.exists() and not p.is_symlink():return {"ok":False,"missing":True,"path":rel}
 st=p.lstat();kind="symlink" if stat.S_ISLNK(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "directory" if stat.S_ISDIR(st.st_mode) else "other"
 out={"ok":True,"path":rel,"kind":kind,"size":st.st_size}
 if kind=="file":out["sha256"]=shaf(p);out["suffix"]=p.suffix.lower()
 elif kind=="symlink":out["target"]=os.readlink(p)
 return out
def text_excerpt(rel,start,count):
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel)
 if not p.is_file():return {"ok":False,"reason":"not_regular_file","path":rel}
 data=p.read_bytes()
 if b"\0" in data[:8192]:return {"ok":False,"reason":"binary_file","path":rel}
 lines=data.decode("utf-8",errors="replace").splitlines();start=max(1,int(start));count=max(1,min(int(count),200))
 return {"ok":True,"path":rel,"start_line":start,"text":sanitize("\n".join(lines[start-1:start-1+count]),16000)}
def git_grep(pattern,prefix):
 if len(pattern)>200:return {"ok":False,"reason":"pattern_too_long"}
 a=["grep","-n","--full-name","-e",pattern]
 if prefix:a += ["--",prefix]
 cp=git(*a,check=False);return {"ok":cp.returncode in (0,1),"matches":sanitize(cp.stdout)}
def ast_summary(rel):
 if secret_path(rel):return {"ok":False,"denied":"secret_like_path","path":rel}
 p=safe_path(rel)
 if p.suffix!=".py" or not p.is_file():return {"ok":False,"reason":"not_python_file"}
 try:t=ast.parse(p.read_text(encoding="utf-8"))
 except Exception as e:return {"ok":False,"reason":"ast_parse_error","error":type(e).__name__}
 fn=[];cl=[];im=[]
 for n in ast.walk(t):
  if isinstance(n,ast.Import):im += [a.name for a in n.names]
  elif isinstance(n,ast.ImportFrom):im.append(n.module or "")
  elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):fn.append(n.name)
  elif isinstance(n,ast.ClassDef):cl.append(n.name)
 return {"ok":True,"imports":sorted(set(im))[:200],"functions":sorted(set(fn))[:300],"classes":sorted(set(cl))[:200]}
def tree_rows(root):
 out=[]
 if not root.is_dir():return out
 for p in sorted(root.rglob("*")):
  rel=p.relative_to(root).as_posix()
  if p.is_symlink():out.append(["L",rel,os.readlink(p)])
  elif p.is_file():out.append(["F",rel,shaf(p),stat.S_IMODE(p.stat().st_mode)])
  elif p.is_dir():out.append(["D",rel,stat.S_IMODE(p.stat().st_mode)])
 return out
def external_fingerprint():
 paths=sorted(x["path"] for x in status_items() if x["status"]=="??" and not x["path"].startswith("governance/") and not x["path"].startswith(SOFTWARE_REL+"/") and x["path"]!=".gitattributes")
 ent=[metadata(x) for x in paths];return {"count":len(paths),"sha256":shab(json.dumps(ent,ensure_ascii=True,sort_keys=True).encode()),"paths":paths}
def git_objects_dir():
 p=Path(git("rev-parse","--git-path","objects").stdout.strip());return str((p if p.is_absolute() else ROOT/p).resolve())
def current_scope_diff_check():
 items=status_items();paths=sorted({x["path"] for x in items if x["status"]!="??" or x["path"].startswith("governance/") or x["path"]==".gitattributes"})
 if not paths:return {"ok":True,"returncode":0,"path_count":0,"output_sha256":shab(b"")}
 with tempfile.TemporaryDirectory(prefix="mppg-probe-") as td:
  t=Path(td);idx=t/"idx";obj=t/"objects";obj.mkdir();env=os.environ.copy();env["GIT_INDEX_FILE"]=str(idx);env["GIT_OBJECT_DIRECTORY"]=str(obj);env["GIT_ALTERNATE_OBJECT_DIRECTORIES"]=git_objects_dir()
  run(["git","-C",str(ROOT),"read-tree","HEAD"],env=env);run(["git","-C",str(ROOT),"add","--",*paths],env=env)
  cp=run(["git","-C",str(ROOT),"diff","--cached","--check","--no-ext-diff","HEAD"],env=env,check=False)
  return {"ok":cp.returncode==0,"returncode":cp.returncode,"path_count":len(paths),"output":sanitize(cp.stdout+cp.stderr,3000),"output_sha256":shab(cp.stdout_bytes+cp.stderr_bytes)}
def python_compile_temp_copy():
 src=GOV/"orchestrator/mppg_orchestrator.py"
 with tempfile.TemporaryDirectory(prefix="mppg-pyc-") as td:
  d=Path(td);cpy=d/"x.py";shutil.copy2(src,cpy);cfile=d/"x.pyc";cp=run(["python3","-S","-c","import py_compile,sys;py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)",str(cpy),str(cfile)],check=False)
  return {"ok":cp.returncode==0,"pyc_exists":cfile.is_file(),"canonical_pycache_exists":(src.parent/"__pycache__").exists()}
def named_probe(name):
 if name=="closed_loop_canary":
  if not _ACCEPTANCE_CANARY.get("enabled"):return {"ok":False,"reason":"acceptance_canary_disabled"}
  _ACCEPTANCE_CANARY["calls"]=_ACCEPTANCE_CANARY.get("calls",0)+1
  calls=_ACCEPTANCE_CANARY["calls"]
  return {"ok":calls>=2,"phase":"resolved" if calls>=2 else "unresolved","calls":calls}
 if name=="git_status":return {"ok":True,"items":status_items()}
 if name=="governance_verify":
  try:verify_governance();return {"ok":True,"refs":refs()}
  except Exception as e:return {"ok":False,"error":sanitize(str(e),1000)}
 if name=="software_status":
  cp=git_ro("status","--porcelain=v1","--untracked-files=all","--",SOFTWARE_REL,check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 if name=="git_diff_name_status":
  cp=git_ro("diff","--name-status",check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 if name=="git_cached_name_status":
  cp=git_ro("diff","--cached","--name-status",check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout)}
 if name=="runtime_source_equivalence":
  a=tree_rows(GOV/"orchestrator");b=tree_rows(Path(os.path.expanduser(conf()["runtime_root"])));return {"ok":a==b,"source_sha256":shab(json.dumps(a,ensure_ascii=True).encode()),"runtime_sha256":shab(json.dumps(b,ensure_ascii=True).encode())}
 if name=="ignored_python_cache_inventory":
  rows=[]
  for p in sorted((GOV/"orchestrator").rglob("*")):
   rel=p.relative_to(ROOT).as_posix()
   if "__pycache__" not in rel and not rel.endswith(".pyc"):continue
   rows.append({"path":rel,"ignored":git("check-ignore","-q","--",rel,check=False).returncode==0,"sha256":shaf(p) if p.is_file() else None})
  return {"ok":True,"entries":rows}
 if name=="external_untracked_fingerprint":return {"ok":True,**external_fingerprint()}
 if name=="current_scope_diff_check":return current_scope_diff_check()
 if name=="python_compile_temp_copy":return python_compile_temp_copy()
 if name=="canonical_tree_clean":
  out=git_ro("status","--porcelain=v1","--untracked-files=all","--",SOFTWARE_REL).stdout;return {"ok":not bool(out),"output":sanitize(out)}
 if name=="protected_state_summary":
  e=ROOT/".env";return {"ok":True,"root_env_exists":e.exists(),"root_env_sha256":shaf(e) if e.is_file() else None,"external":external_fingerprint(),"staged":git("diff","--cached","--name-only").stdout.splitlines()}
 return {"ok":False,"reason":"unknown_probe"}
TOOLS=[
 {"type":"function","name":"read_path_metadata","description":"Read non-secret path metadata.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"read_text_excerpt","description":"Read bounded non-secret UTF-8 text.","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"max_lines":{"type":"integer"}},"required":["path","start_line","max_lines"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"git_grep","description":"Read-only discovery; textual presence is not live-edge proof.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path_prefix":{"type":"string"}},"required":["pattern","path_prefix"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"ast_summary","description":"Read-only Python AST summary.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"run_named_probe","description":"Run one deterministic read-only probe.","parameters":{"type":"object","properties":{"name":{"type":"string","enum":sorted(PROBE_NAMES)}},"required":["name"],"additionalProperties":False},"strict":True}]
def execute_tool(name,args):
 if name=="read_path_metadata":return metadata(args["path"])
 if name=="read_text_excerpt":return text_excerpt(args["path"],args["start_line"],args["max_lines"])
 if name=="git_grep":return git_grep(args["pattern"],args["path_prefix"])
 if name=="ast_summary":return ast_summary(args["path"])
 if name=="run_named_probe":return named_probe(args["name"])
 return {"ok":False,"reason":"tool_not_allowlisted"}
def outtext(resp):
 for i in resp.get("output",[]):
  if i.get("type")=="message":
   for c in i.get("content",[]):
    if c.get("type")=="output_text":return c.get("text","")
 return None
def validate_resolution(o):
 s=loadj(AI_SCHEMA)
 if set(o)!=set(s["required"]):raise OrchestratorError("AI resolution key-set mismatch")
 if any(a not in AUTO_ACTIONS for a in o["automatic_actions"]):raise OrchestratorError("AI automatic action outside allowlist")
 if any(x not in PROBE_NAMES for x in o["probe_requests"]):raise OrchestratorError("AI probe outside allowlist")
 if any(x not in SHADOW_PROBES for x in o["shadow_validation_probes"]):raise OrchestratorError("AI shadow probe outside allowlist")
 if len(o["probe_requests"])>int(conf()["max_probe_requests_per_resolution"]):raise OrchestratorError("too many probes")
 if len(o["shadow_patch"].encode())>int(conf()["max_shadow_patch_bytes"]):raise OrchestratorError("shadow patch too large")
 for rel in o["mutation_scope"]:
  if secret_path(rel):raise OrchestratorError("protected mutation scope");safe_path(rel)
class AI:
 def __init__(self):
  self.key=os.environ.get("OPENAI_API_KEY")
  if not self.key:raise BlockerError("environment","OPENAI_API_KEY_NOT_EXPORTED","OPENAI_API_KEY missing")
  c=conf();self.model=os.environ.get("MPPG_OPENAI_MODEL") or c["model"];self.timeout=int(c["http_timeout_seconds"]);self.rounds=int(c["max_ai_rounds_per_blocker"]);self.schema=loadj(AI_SCHEMA);self.prompt=AI_PROMPT.read_text();self.master=MASTER.read_text()
 def endpoint(self):
  if os.environ.get("MPPG_ORCHESTRATOR_TEST_MODE")=="1":
   u=os.environ.get("MPPG_ORCHESTRATOR_TEST_ENDPOINT")
   if not u:raise OrchestratorError("test endpoint missing")
   return u
  return PRODUCTION_API
 def post(self,payload):
  body=json.dumps(payload,ensure_ascii=True).encode()
  if self.key.encode() in body:raise OrchestratorError("API key leaked into request body")
  req=urllib.request.Request(self.endpoint(),data=body,method="POST",headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json","User-Agent":"mppg-orchestrator/2"})
  try:
   with urllib.request.urlopen(req,timeout=self.timeout) as r:raw=r.read()
  except urllib.error.HTTPError as e:raise BlockerError("external_dependency","OPENAI_HTTP_ERROR",f"OpenAI HTTP {e.code}: {sanitize(e.read().decode(errors='replace'),3000)}") from None
  except Exception as e:raise BlockerError("external_dependency","OPENAI_REQUEST_FAILURE",f"OpenAI request failure: {type(e).__name__}") from None
  return json.loads(raw)
 def resolve(self,ctx,blocker,history):
  packet={"context":ctx,"blocker":blocker,"automatic_actions":sorted(AUTO_ACTIONS),"available_probes":sorted(PROBE_NAMES),"shadow_validation_probes":sorted(SHADOW_PROBES),"evidence_history":history[-8:]}
  items=[{"role":"user","content":"PROMPT MASTER:\n\n"+self.master+"\n\nCURRENT BLOCKER PACKET:\n"+json.dumps(packet,ensure_ascii=True,sort_keys=True)}]
  for _ in range(self.rounds):
   payload={"model":self.model,"instructions":self.prompt,"input":items,"tools":TOOLS,"tool_choice":"auto","parallel_tool_calls":False,"store":False,"text":{"format":{"type":"json_schema","name":"mppg_ai_resolution","strict":True,"schema":self.schema}}}
   resp=self.post(payload);output=resp.get("output",[]);items.extend(output);calls=[x for x in output if x.get("type")=="function_call"]
   if calls:
    for c in calls:
     try:a=json.loads(c.get("arguments","{}"))
     except Exception:a={}
     result=execute_tool(c.get("name",""),a);items.append({"type":"function_call_output","call_id":c.get("call_id"),"output":sanitize(json.dumps(result,ensure_ascii=True,sort_keys=True))})
    continue
   txt=outtext(resp)
   if not txt:raise OrchestratorError("AI returned no structured output")
   o=json.loads(txt);validate_resolution(o);return o
  raise OrchestratorError("AI tool-loop round limit reached")
def detect():
 items=status_items();staged=git("diff","--cached","--name-only").stdout.splitlines();tracked=git("diff","--name-only").stdout.splitlines();un=[x["path"] for x in items if x["status"]=="??"];sw=[x for x in un if x.startswith(SOFTWARE_REL+"/")];gov=[x for x in items if x["path"].startswith("governance/") or x["path"]==".gitattributes"];ext=[x for x in un if not x.startswith("governance/") and not x.startswith(SOFTWARE_REL+"/") and x!=".gitattributes"]
 k="preexisting_staging" if staged else "canonical_software_untracked" if sw else "governance_change" if gov else "tracked_change" if tracked else "repository_content_ingestion" if ext else "noop"
 return {"kind":k,"staged":staged,"tracked":tracked,"untracked":un,"software_untracked":sw,"external_untracked":ext,"items":items}
def path_manifest(paths):
 ent=[];bl=[]
 for rel in sorted(paths):
  m=metadata(rel);ent.append(m)
  if secret_path(rel):bl.append({"domain":"environment","code":"SECRET_LIKE_PATH","summary":rel})
  if m.get("kind")=="symlink":bl.append({"domain":"repository_state","code":"SYMLINK_REVIEW_REQUIRED","summary":rel})
 return ent,shab(json.dumps(ent,ensure_ascii=True,sort_keys=True).encode()),bl
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
  p={"front":f["kind"],"front_class":"mixed","product_artifact_required":True,"user_acceptance_required":True,"representative_runtime_required":True};b.append({"domain":"authority_model","code":"SEMANTIC_FRONT_ADJUDICATION_REQUIRED","summary":f["kind"]})
 return {"refs":r,"front":f,"profile":p,"prompt_master_sha256":shaf(MASTER)},b
def savej(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(sanitize(json.dumps(o,ensure_ascii=True,indent=2,sort_keys=True),250000),encoding="utf-8")
def validate_patch_scope(patch,scope):
 touched=[]
 for line in patch.splitlines():
  if line.startswith("+++ b/") or line.startswith("--- a/"):
   rel=line[6:]
   if rel!="/dev/null":touched.append(rel)
 touched=sorted(set(touched))
 if touched!=sorted(scope):raise OrchestratorError(f"patch scope mismatch touched={touched!r} declared={sorted(scope)!r}")
 for rel in touched:
  if secret_path(rel):raise OrchestratorError("patch touches protected path");safe_path(rel)
 return touched
def shadow_probe(root,name,changed):
 def rg(*a,check=True):return run(["git","-C",str(root),*a],check=check)
 if name=="git_diff_check":
  cp=rg("diff","--check","HEAD",check=False);return {"ok":cp.returncode==0,"output":sanitize(cp.stdout+cp.stderr,3000)}
 if name=="python_compile_changed":
  out=[]
  for rel in [x for x in changed if x.endswith(".py") and (root/x).is_file()]:
   with tempfile.TemporaryDirectory(prefix="mppg-shadow-pyc-") as td:
    cp=run(["python3","-S","-c","import py_compile,sys;py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)",str(root/rel),str(Path(td)/"x.pyc")],check=False);out.append({"path":rel,"ok":cp.returncode==0})
  return {"ok":all(x["ok"] for x in out),"files":out}
 if name=="governance_manifest":return {"ok":run(["sha256sum","-c","governance/MANIFEST.sha256"],cwd=root,check=False).returncode==0}
 if name=="static_tests":
  f=root/"governance/orchestrator/tests/test_static.py";cp=run(["python3","-B","-S",str(f)],cwd=root,check=False) if f.is_file() else None;return {"ok":bool(cp and cp.returncode==0),"output":sanitize((cp.stdout+cp.stderr) if cp else "missing",5000)}
 if name=="orchestrator_self_test":return {"ok":True,"reason":"validated during post-materialization runtime acceptance"}
 return {"ok":False,"reason":"unknown_shadow_probe"}
def simulate_patch(decision):
 patch=decision["shadow_patch"]
 if not patch:return {"ok":False,"reason":"empty_patch"}
 touched=validate_patch_scope(patch,decision["mutation_scope"])
 with tempfile.TemporaryDirectory(prefix="mppg-shadow-") as td:
  shadow=Path(td)/"repo";cp=run(["git","clone","--quiet","--no-hardlinks",str(ROOT),str(shadow)],check=False)
  if cp.returncode!=0:return {"ok":False,"reason":"shadow_clone_failed"}
  pp=Path(td)/"p.patch";pp.write_text(patch,encoding="utf-8")
  ck=run(["git","-C",str(shadow),"apply","--check","--binary",str(pp)],check=False)
  if ck.returncode!=0:return {"ok":False,"reason":"patch_check_failed","output":sanitize(ck.stdout+ck.stderr,3000)}
  run(["git","-C",str(shadow),"apply","--binary",str(pp)])
  vals={x:shadow_probe(shadow,x,touched) for x in decision["shadow_validation_probes"]}
  diff=run(["git","-C",str(shadow),"diff","--binary","--full-index","HEAD"])
  targets={rel:metadata(rel,shadow) for rel in touched}
  return {"ok":all(v.get("ok") for v in vals.values()),"touched":touched,"patch":diff.stdout,"patch_sha256":shab(diff.stdout_bytes),"targets":targets,"validation":vals}
def exception_blocker(exc,stage):
 if isinstance(exc,BlockerError):
  b=dict(exc.blocker)
 else:
  b={"domain":"auditor_harness","code":"ORCHESTRATOR_INTERNAL_EXCEPTION","summary":f"{stage}: {type(exc).__name__}: {sanitize(str(exc),2500)}"}
 b["stage"]=stage;return b
def execute_actions(d,run_dir,round_no):
 out=[]
 for action in d["automatic_actions"]:
  try:
   if action=="NOOP":out.append({"action":action,"ok":True});continue
   if action=="SIMULATE_PATCH_IN_SHADOW":out.append({"action":action,**simulate_patch(d)});continue
   if action in {"RERUN_READONLY_PROBE","RECLASSIFY_FROM_EVIDENCE","REBUILD_EPHEMERAL_AUDITOR","RECOMPUTE_EPHEMERAL_EVIDENCE","RETRY_EXTERNAL_READONLY"}:
    probes=d["probe_requests"] or (["governance_verify"] if action=="RETRY_EXTERNAL_READONLY" else ["git_status","governance_verify"]);plan={"format":"declarative_probe_plan_v1","action":action,"probes":probes}
    if action=="REBUILD_EPHEMERAL_AUDITOR":savej(run_dir/f"ephemeral_auditor_{round_no:02d}.json",plan)
    rr={};generated=[]
    for p in probes:
     try:rr[p]=named_probe(p)
     except Exception as e:
      eb=exception_blocker(e,"probe:"+p);rr[p]={"ok":False,"exception_blocker":eb};generated.append(eb)
    out.append({"action":action,"ok":all(x.get("ok",False) for x in rr.values()),"probes":rr,"generated_blockers":generated});continue
   out.append({"action":action,"ok":False,"reason":"unimplemented"})
  except Exception as e:out.append({"action":action,"ok":False,"exception_blocker":exception_blocker(e,"automatic_action:"+action)})
 return out
def _resolve_with_external_retry(ai,ctx,blocker,history):
 attempts=max(1,int(conf().get("max_ai_external_retries",3)));last=None
 for _ in range(attempts):
  try:return ai.resolve(ctx,blocker,history),None
  except BlockerError as e:
   last=exception_blocker(e,"ai.resolve")
   if e.blocker.get("domain")!="external_dependency":return None,last
  except Exception as e:return None,exception_blocker(e,"ai.resolve")
 return None,last or {"domain":"external_dependency","code":"AI_RESOLUTION_RETRY_EXHAUSTED","summary":"AI resolution retry exhausted","stage":"ai.resolve"}
def autoremediate(ctx,blockers,run_dir):
 if not blockers:return [],[],[]
 ai=AI();remain=[];decisions=[];contracts=[];c=conf()
 for original in blockers:
  if original["domain"] in {"authorization","product_acceptance","user_acceptance"}:remain.append(original);continue
  active=dict(original);history=[];last=None;no_progress=0;hcount=0;done=False
  for n in range(1,int(c["max_autoremediation_cycles"])+1):
   d,aierr=_resolve_with_external_retry(ai,ctx,active,history)
   if aierr:
    history.append({"round":n,"active_blocker":active,"decision":None,"results":[{"action":"AI_RESOLVE","ok":False,"exception_blocker":aierr}]});remain.append(aierr);break
   decisions.append(d)
   if active.get("code")=="SYNTHETIC_CLOSED_LOOP_CANARY" and _ACCEPTANCE_CANARY.get("enabled") and _ACCEPTANCE_CANARY.get("calls",0)<2:
    d=dict(d);d["status"]="UNRESOLVED";d["mutation_required"]=False;d["automatic_actions"]=["RERUN_READONLY_PROBE"];d["probe_requests"]=["closed_loop_canary"];d["shadow_patch"]="";d["shadow_validation_probes"]=[];d["mutation_scope"]=[];d["next_gate"]="CONTINUE_READ_ONLY"
   hcount=hcount+1 if d["finding_class"]=="auditor_harness_defect" or d["blocker_domain"] in {"auditor_harness","evidence_packaging","authority_model","scanner_model"} else 0
   if hcount>=int(c["harness_rebuild_threshold"]) and "REBUILD_EPHEMERAL_AUDITOR" not in d["automatic_actions"] and active.get("code")!="SYNTHETIC_CLOSED_LOOP_CANARY":
    d=dict(d);d["automatic_actions"]=["REBUILD_EPHEMERAL_AUDITOR"];d["probe_requests"]=d["probe_requests"] or ["git_status","governance_verify","current_scope_diff_check"]
   results=execute_actions(d,run_dir,n);generated=[]
   for r in results:
    if r.get("exception_blocker"):generated.append(r["exception_blocker"])
    generated.extend(r.get("generated_blockers") or [])
   eh=shab(json.dumps({"active":active,"results":results},ensure_ascii=True,sort_keys=True).encode());event={"round":n,"active_blocker":active,"decision":d,"results":results,"evidence_sha256":eh};history.append(event);savej(run_dir/f"round_{n:02d}_{shab(json.dumps(original,sort_keys=True).encode())[:10]}.json",event)
   no_progress=no_progress+1 if eh==last else 0;last=eh
   if generated:active=dict(generated[-1]);hcount=hcount+1 if active.get("domain") in {"auditor_harness","evidence_packaging","authority_model","scanner_model"} else 0;no_progress=0;continue
   if d["mutation_required"] or d["status"]=="MUTATION_REQUIRED":
    sim=next((x for x in results if x.get("action")=="SIMULATE_PATCH_IN_SHADOW" and x.get("ok")),None);contracts.append({"blocker":active,"decision":d,"shadow":sim});remain.append(active);break
   if d["status"]=="RESOLVED_READ_ONLY" and not d["mutation_required"]:
    if active.get("code")=="SYNTHETIC_CLOSED_LOOP_CANARY" and _ACCEPTANCE_CANARY.get("enabled"):
     if _ACCEPTANCE_CANARY.get("calls",0)<2:active={"domain":"auditor_harness","code":"ACCEPTANCE_CANARY_PREMATURE_RESOLUTION","summary":"AI resolved canary before two executed probe cycles","stage":"acceptance"};continue
     print("AI_AUTORESOLVED_BLOCKER="+active["code"]);done=True;break
    try:newctx,newb=context();still=[x for x in newb if x["code"]==active.get("code") and x["summary"]==active.get("summary")]
    except Exception as e:active=exception_blocker(e,"post_resolution_context");continue
    if active.get("domain")=="repository_state" and still:ctx=newctx
    else:print("AI_AUTORESOLVED_BLOCKER="+active.get("code","UNKNOWN"));print("AI_FINDING_CLASS="+d["finding_class"]);print("AI_SUMMARY="+sanitize(d["summary"],2000));done=True;break
   if no_progress>=int(c["max_no_progress_cycles"]):break
  if not done and active not in remain:remain.append(active)
 return remain,decisions,contracts

def token(kind,payload):
 r=refs();return shab(json.dumps({"kind":kind,"head":r["head"],"remote":r["remote"],"payload":payload},sort_keys=True,separators=(",",":")).encode())
def _interactive_read(prompt):
 if os.environ.get("MPPG_ORCHESTRATOR_TEST_MODE")=="1" and os.environ.get("MPPG_AUTO_APPROVE_TEST_GATES")=="1":
  return os.environ.get("MPPG_EXPECTED_TEST_GATE_INPUT","")
 try:
  with open("/dev/tty","r+",encoding="utf-8",errors="replace") as tty:
   tty.write(prompt);tty.flush();return tty.readline().rstrip("\n")
 except Exception:
  try:return input(prompt)
  except EOFError:return ""
def gate(name,tok,verb="AUTHORIZE"):
 expected=f"{verb} {name} {tok}"
 print("------------------------------------------------------------");print("AUTHORIZATION_REQUIRED="+name);print("TYPE_EXACTLY: "+expected)
 x=_interactive_read("> ")
 if x==expected:print("AUTHORIZATION_ACCEPTED="+name);return True
 print("AUTHORIZATION_ACCEPTED=false");print("GUIDED_RUN_STOPPED_AT_GATE="+name);return False

def backup_scope(scope):
 td=tempfile.TemporaryDirectory(prefix="mppg-materialization-backup-");root=Path(td.name);meta={}
 for rel in scope:
  p=ROOT/rel;meta[rel]={"exists":p.exists() or p.is_symlink()}
  if p.is_file():dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  elif p.exists():raise OrchestratorError("non-regular path in generic materialization scope")
 return td,root,meta
def restore_scope(root,meta):
 for rel,m in meta.items():
  p=ROOT/rel
  if m["exists"]:p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,p)
  elif p.exists() or p.is_symlink():p.unlink()
def _runtime_snapshot(td):
 runtime=Path(os.path.expanduser(conf()["runtime_root"]));snap=Path(td)/"runtime_snapshot";exists=runtime.exists()
 if exists:shutil.copytree(runtime,snap,copy_function=shutil.copy2)
 return runtime,snap,exists
def _restore_runtime(runtime,snap,existed):
 if runtime.exists():shutil.rmtree(runtime)
 if existed:shutil.copytree(snap,runtime,copy_function=shutil.copy2)
def materialize_and_machine_accept(contract):
 d=contract["decision"];sim=contract["shadow"]
 if not sim:return "failed",d,{"domain":"auditor_harness","code":"MISSING_VALIDATED_SHADOW_CANDIDATE","summary":"mutation required without validated shadow candidate","stage":"materialization"}
 scope=d["mutation_scope"];payload={"scope":scope,"patch_sha256":sim["patch_sha256"],"targets":sim["targets"],"rollback_on_machine_failure":True};tok=token("MATERIALIZATION",payload)
 print("MATERIALIZATION_SCOPE="+json.dumps(scope,ensure_ascii=True));print("RUNTIME_SYNC_INCLUDED="+str(any(x.startswith("governance/orchestrator/") for x in scope)).lower());print("ROLLBACK_ON_MACHINE_ACCEPTANCE_FAILURE=true")
 if not gate("MATERIALIZATION",tok):return "gate_stop",d,None
 if git_ro("diff","--cached","--name-only").stdout.strip():return "failed",d,{"domain":"repository_state","code":"STAGING_NOT_EMPTY_BEFORE_MATERIALIZATION","summary":"staging non-empty before materialization","stage":"materialization"}
 td,bak,meta=backup_scope(scope);runtime,snap,runtime_existed=_runtime_snapshot(td.name)
 try:
  pp=Path(td.name)/"candidate.patch";pp.write_text(sim["patch"],encoding="utf-8");run(["git","-C",str(ROOT),"apply","--check","--binary",str(pp)]);run(["git","-C",str(ROOT),"apply","--binary",str(pp)])
  if git_ro("diff","--check",check=False).returncode!=0:raise OrchestratorError("post-materialization diff-check failed")
  if any(x.startswith("governance/orchestrator/") for x in scope):
   if runtime.exists():shutil.rmtree(runtime)
   shutil.copytree(GOV/"orchestrator",runtime,copy_function=shutil.copy2)
  if any(x.startswith("governance/") for x in scope):verify_governance()
  for rel in scope:
   if rel.endswith(".py") and (ROOT/rel).is_file():
    with tempfile.TemporaryDirectory(prefix="mppg-accept-pyc-") as pytd:
     cp=run(["python3","-S","-c","import py_compile,sys;py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)",str(ROOT/rel),str(Path(pytd)/"x.pyc")],check=False)
     if cp.returncode!=0:raise OrchestratorError("python compile acceptance failed")
  if any(x.startswith("governance/orchestrator/") for x in scope):
   cp=run([str(Path(os.path.expanduser(conf()["launcher"]))),"self-test"],cwd=ROOT,check=False)
   if cp.returncode!=0:raise OrchestratorError("runtime self-test failed")
  print("TECHNICAL_POST_MATERIALIZATION=PASS");print("MACHINE_PRODUCT_ACCEPTANCE=PASS");td.cleanup();return "machine_pass",d,None
 except Exception as e:
  try:restore_scope(bak,meta);_restore_runtime(runtime,snap,runtime_existed);print("MATERIALIZATION_ROLLBACK=PASS")
  except Exception as rb:td.cleanup();return "fatal",d,{"domain":"auditor_harness","code":"MATERIALIZATION_ROLLBACK_FAILED","summary":sanitize(str(rb),2500),"stage":"materialization_rollback"}
  td.cleanup();return "failed",d,exception_blocker(e,"machine_product_acceptance")
def user_accept_materialization(scope,decision):
 ua=token("USER_ACCEPTANCE",{"scope":scope,"summary":decision["summary"]});print("USER_ACCEPTANCE_REQUIRED=true");print("REPRESENTATIVE_SUMMARY="+sanitize(decision["summary"],3000))
 if not gate("USER ACCEPTANCE",ua,verb="APPROVE"):return False
 print("USER_ACCEPTANCE=PASS");return True

def exact_stage_scope(scope):
 if git_ro("diff","--cached","--name-only").stdout.strip():raise OrchestratorError("staging non-empty")
 with tempfile.TemporaryDirectory(prefix="mppg-stage-candidate-") as td:
  t=Path(td);idx=t/"idx";obj=t/"objects";obj.mkdir();env=os.environ.copy();env["GIT_INDEX_FILE"]=str(idx);env["GIT_OBJECT_DIRECTORY"]=str(obj);env["GIT_ALTERNATE_OBJECT_DIRECTORIES"]=git_objects_dir();run(["git","-C",str(ROOT),"read-tree","HEAD"],env=env);run(["git","-C",str(ROOT),"add","--",*scope],env=env)
  ck=run(["git","-C",str(ROOT),"diff","--cached","--check","HEAD"],env=env,check=False)
  if ck.returncode!=0:raise OrchestratorError("candidate staged diff-check failed")
  cp=run(["git","-C",str(ROOT),"diff","--cached","--binary","--full-index","HEAD"],env=env);cn=run(["git","-C",str(ROOT),"diff","--cached","--name-only","-z","HEAD"],env=env)
  st=token("STAGING",{"scope":scope,"patch":shab(cp.stdout_bytes),"paths":shab(cn.stdout_bytes)})
  if not gate("STAGING",st):return False,None,None
  ip=Path(git("rev-parse","--git-path","index").stdout.strip());ip=ip if ip.is_absolute() else ROOT/ip;bak=Path(td)/"real-index";shutil.copy2(ip,bak);before=shaf(ip)
  try:
   git("add","--",*scope);rc=git("diff","--cached","--check",check=False)
   if rc.returncode!=0:raise OrchestratorError("real staged diff-check failed")
   rp=git("diff","--cached","--binary","--full-index","HEAD");rn=git("diff","--cached","--name-only","-z","HEAD")
   if shab(rp.stdout_bytes)!=shab(cp.stdout_bytes) or shab(rn.stdout_bytes)!=shab(cn.stdout_bytes):raise OrchestratorError("candidate/real staged mismatch")
  except Exception:shutil.copy2(bak,ip);assert shaf(ip)==before;raise
  print("EXACT_STAGING=PASS");return True,cp.stdout_bytes,cn.stdout_bytes
def commit_publish(scope,patch,names,subject):
 ct=token("COMMIT",{"scope":scope,"patch":shab(patch),"paths":shab(names),"subject":subject})
 if not gate("COMMIT",ct):return False
 par=git("rev-parse","HEAD").stdout.strip()
 if remote()!=par:raise OrchestratorError("remote moved before commit")
 git("commit","-m",subject);new=git("rev-parse","HEAD").stdout.strip()
 if git("rev-parse","HEAD^").stdout.strip()!=par:raise OrchestratorError("commit parent mismatch")
 print("ISOLATED_COMMIT=PASS");print("NEW_COMMIT="+new)
 pt=token("PUBLICATION",{"old_remote":par,"new_head":new})
 if not gate("PUBLICATION",pt):return False
 if remote()!=par:raise OrchestratorError("remote moved before publication")
 if git("push","--porcelain","origin","refs/heads/master:refs/heads/master",check=False).returncode!=0:raise OrchestratorError("push failed")
 r=refs()
 if not r["head"]==r["tracking"]==r["remote"]==new:raise OrchestratorError("postpublication refs diverged")
 print("PUBLICATION=PASS");print("POST_PUBLICATION_CLOSURE=PASS");print("FRONT_CLOSED=true");return True
def ingestion(ctx):
 paths=ctx["front"]["external_untracked"];fp=ctx["profile"]["candidate_manifest_sha256"];print("FRONT=repository-content-ingestion");print("CANDIDATE_COUNT="+str(len(paths)));st=token("STAGING",{"paths":paths,"manifest_sha":fp})
 if not gate("STAGING",st):return 0
 # Existing ingestion exact staging behavior remains intentionally separate.
 _,cur,b=path_manifest(paths)
 if b or cur!=fp:raise OrchestratorError("candidate drift")
 with tempfile.TemporaryDirectory(prefix="mppg-ingestion-") as td:
  t=Path(td);idx=t/"idx";obj=t/"objects";obj.mkdir();env=os.environ.copy();env["GIT_INDEX_FILE"]=str(idx);env["GIT_OBJECT_DIRECTORY"]=str(obj);env["GIT_ALTERNATE_OBJECT_DIRECTORIES"]=git_objects_dir();run(["git","-C",str(ROOT),"read-tree","HEAD"],env=env);run(["git","-C",str(ROOT),"add","--",*paths],env=env);ck=run(["git","-C",str(ROOT),"diff","--cached","--check","HEAD"],env=env,check=False)
  if ck.returncode!=0:raise BlockerError("test","INGESTION_CANDIDATE_DIFF_CHECK_FAILED",sanitize(ck.stdout+ck.stderr,3000))
 git("add","--",*paths);print("EXACT_STAGING=PASS");patch=git("diff","--cached","--binary","--full-index","HEAD").stdout_bytes;names=git("diff","--cached","--name-only","-z","HEAD").stdout_bytes;return 0 if not commit_publish(paths,patch,names,conf()["content_ingestion_commit_subject"]) else 0
def fallback_context():
 r={"branch":"","head":"","tracking":"","remote":""}
 try:r.update(refs())
 except Exception:pass
 try:f=detect()
 except Exception:f={"kind":"unknown","staged":[],"tracked":[],"untracked":[],"software_untracked":[],"external_untracked":[],"items":[]}
 return {"refs":r,"front":f,"profile":{"front":"orchestrator-internal-recovery","front_class":"mixed","product_artifact_required":True,"user_acceptance_required":True,"representative_runtime_required":True},"prompt_master_sha256":shaf(MASTER) if MASTER.is_file() else None}
def safe_context():
 try:return context()
 except Exception as e:return fallback_context(),[exception_blocker(e,"context")]
def live_acceptance_test():
 verify_governance()
 if not os.environ.get("OPENAI_API_KEY"):raise BlockerError("environment","OPENAI_API_KEY_NOT_EXPORTED","OPENAI_API_KEY missing")
 rid="acceptance-"+uuid.uuid4().hex[:10];run_dir=RUN_ROOT/rid;run_dir.mkdir(parents=True,exist_ok=True);_ACCEPTANCE_CANARY["enabled"]=True;_ACCEPTANCE_CANARY["calls"]=0
 blocker={"domain":"auditor_harness","code":"SYNTHETIC_CLOSED_LOOP_CANARY","summary":"Execute closed_loop_canary until phase resolved."};ctx={"refs":refs(),"front":{"kind":"acceptance_canary"},"profile":{"front":"acceptance-canary","front_class":"mixed","product_artifact_required":False,"user_acceptance_required":False,"representative_runtime_required":False},"prompt_master_sha256":shaf(MASTER)}
 try:rem,dec,contracts=autoremediate(ctx,[blocker],run_dir)
 finally:_ACCEPTANCE_CANARY["enabled"]=False
 if rem or contracts:raise OrchestratorError("live closed-loop canary did not resolve")
 if _ACCEPTANCE_CANARY.get("calls",0)<2:raise OrchestratorError("live canary did not execute two probe cycles")
 if len(dec)<2:raise OrchestratorError("live canary did not perform multiple AI rounds")
 print("LIVE_AI_MULTIROUND_AUTOREMEDIATION=PASS");print("LIVE_AI_CANARY_PROBE_CALLS="+str(_ACCEPTANCE_CANARY["calls"]));print("LIVE_AI_DECISION_ROUNDS="+str(len(dec)));print("LIVE_CLOSED_LOOP_ACCEPTANCE=PASS");return 0
def self_test():
 verify_governance();ast.parse(Path(__file__).read_text(encoding="utf-8"));s=loadj(AI_SCHEMA);c=conf();assert s["additionalProperties"] is False and set(s["required"])==set(s["properties"]);assert all(t["strict"] and t["parameters"]["additionalProperties"] is False for t in TOOLS);assert c["api_endpoint"]==PRODUCTION_API;can=CommandResult(subprocess.CompletedProcess(["c"],0,b"\xe2\xff",b""));assert can.stdout.encode("utf-8",errors="surrogateescape")==b"\xe2\xff";pc=python_compile_temp_copy();assert pc["ok"] and not pc["canonical_pycache_exists"];print("ORCHESTRATOR_BYTE_SAFE_IO_CANARY=PASS");print("ORCHESTRATOR_TEMP_PYCOMPILE_CANARY=PASS");print("ORCHESTRATOR_CLOSED_LOOP_ACTION_REGISTRY=PASS");print("ORCHESTRATOR_SELF_TEST=PASS")
def guided_run(dry=False):
 rid=dt.datetime.now().strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8];run_dir=RUN_ROOT/rid;run_dir.mkdir(parents=True,exist_ok=True);print("RUN_ID="+rid);pending=None
 for cycle in range(1,int(conf().get("max_autoremediation_cycles",12))+1):
  ctx,b=safe_context()
  if pending:b=[pending]
  print("FRONT_BASELINE_HEAD="+ctx["refs"].get("head",""));print("FRONT_KIND="+ctx["front"].get("kind","unknown"));print("AUTOREMEDIATION_SUPERVISOR_CYCLE="+str(cycle))
  if b:
   rem,dec,contracts=autoremediate(ctx,b,run_dir);savej(run_dir/"ai_decisions.json",dec);savej(run_dir/"mutation_candidates.json",contracts)
   if contracts:
    if dry:print("DRY_RUN_MUTATION_CANDIDATE=true");print("AI_MUTATION_CANDIDATE_SCOPE="+json.dumps(contracts[0]["decision"]["mutation_scope"],ensure_ascii=True));print("DRY_RUN=PASS");return 0
    state,d,err=materialize_and_machine_accept(contracts[0])
    if state=="gate_stop":return 0
    if state=="fatal":print("TOTAL_BLOCKERS=1");print("BLOCKER_DOMAIN="+err["domain"]);print("BLOCKER_CODE="+err["code"]);print("FRONT_CLOSED=false");return 2
    if state=="failed":pending=err;print("MACHINE_ACCEPTANCE_FAILURE_REENTERED_AI_LOOP=true");continue
    pending=None
    if not user_accept_materialization(d["mutation_scope"],d):return 0
    try:staged,patch,names=exact_stage_scope(d["mutation_scope"])
    except Exception as e:pending=exception_blocker(e,"exact_staging");print("STAGING_FAILURE_REENTERED_AI_LOOP=true");continue
    if not staged:return 0
    subject=d["proposed_commit_subject"].strip() or conf()["generic_commit_subject"];return 0 if commit_publish(d["mutation_scope"],patch,names,subject) else 0
   if rem:
    print("TOTAL_BLOCKERS="+str(len(rem)))
    for x in rem:print("BLOCKER_DOMAIN="+x.get("domain","unknown"));print("BLOCKER_CODE="+x.get("code","UNKNOWN"));print("BLOCKER_SUMMARY="+sanitize(x.get("summary",""),2000))
    print("FRONT_CLOSED=false");return 2
   pending=None;continue
  if dry:print("DRY_RUN=PASS");print(json.dumps(ctx["profile"],ensure_ascii=True,indent=2,sort_keys=True));return 0
  if ctx["front"]["kind"]=="repository_content_ingestion":return ingestion(ctx)
  if ctx["front"]["kind"]=="noop":print("NOOP=true");print("FRONT_CLOSED=true");return 0
  print("NEXT_GATE=MATERIALIZATION_AUTHORIZATION");print("FRONT_CLOSED=false");return 3
 print("AUTOREMEDIATION_CYCLE_BUDGET_EXHAUSTED=true");print("FRONT_CLOSED=false");return 2

def _emit_error_event(exc,stage="main"):
 b=exception_blocker(exc,stage);event={"version":1,"recoverable_by_ai":b.get("domain") in {"auditor_harness","evidence_packaging","authority_model","scanner_model","test"},"blocker":b,"head":None,"staging_nonempty":None}
 try:event["head"]=git("rev-parse","HEAD").stdout.strip();event["staging_nonempty"]=bool(git("diff","--cached","--name-only").stdout.strip())
 except Exception:pass
 print("ORCHESTRATOR_ERROR_EVENT_JSON="+json.dumps(event,ensure_ascii=True,sort_keys=True));return event
def main():
 p=argparse.ArgumentParser(prog="mppg-orchestrator");sp=p.add_subparsers(dest="cmd",required=True);sp.add_parser("self-test");sp.add_parser("status");sp.add_parser("acceptance-test");r=sp.add_parser("run");r.add_argument("--dry-run",action="store_true");a=p.parse_args()
 try:
  if a.cmd=="self-test":return self_test() or 0
  if a.cmd=="acceptance-test":return live_acceptance_test()
  if a.cmd=="status":ctx,b=safe_context();print(json.dumps({"context":ctx,"blockers":b,"openai_api_key_present":bool(os.environ.get("OPENAI_API_KEY"))},ensure_ascii=True,indent=2,sort_keys=True));return 0
  return guided_run(a.dry_run)
 except BlockerError as e:_emit_error_event(e,"main");print("ORCHESTRATOR_BLOCKER_DOMAIN="+e.blocker["domain"]);print("ORCHESTRATOR_BLOCKER_CODE="+e.blocker["code"]);print("ORCHESTRATOR_BLOCKED="+sanitize(e.blocker["summary"],4000));return 2
 except Exception as e:_emit_error_event(e,"main");print("ORCHESTRATOR_BLOCKER_DOMAIN=auditor_harness");print("ORCHESTRATOR_BLOCKER_CODE=UNCLASSIFIED_ORCHESTRATOR_ERROR");print("ORCHESTRATOR_BLOCKED="+sanitize(str(e),4000));return 70

if __name__=="__main__":raise SystemExit(main())
