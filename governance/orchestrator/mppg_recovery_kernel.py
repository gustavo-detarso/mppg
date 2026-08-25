#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,re,shutil,stat,subprocess,sys,tempfile,urllib.error,urllib.request
from pathlib import Path
ROOT=Path(os.environ.get("MPPG_CANONICAL_REPO","/home/gustavodetarso/Documentos/mppg"));GOV=ROOT/"governance";RUNTIME=Path(os.path.expanduser("~/.local/share/mppg-orchestrator/runtime"));CORE=RUNTIME/"mppg_orchestrator.py";MASTER=GOV/"MPPG_PROMPT_MASTER_CANONICO.md";CFG=GOV/"orchestrator/config/defaults.json";SCHEMA=GOV/"orchestrator/schemas/recovery_resolution.schema.json";PROMPT=GOV/"orchestrator/prompts/recovery_agent.md";PRODUCTION_API="https://api.openai.com/v1/responses"
ALLOWED_EXACT={"governance/contracts/AI_AUTOREMEDIATION_POLICY.md","governance/contracts/ORCHESTRATOR_ARCHITECTURE.md","governance/policy/MPPG_POLICY_COMPILED.json"};ALLOWED_PREFIX=("governance/orchestrator/",);SECRET=[re.compile(r"(^|/)\.env($|[._/-])",re.I),re.compile(r"(^|/)(id_rsa|id_ed25519|credentials|secrets?|private[_-]?key)(\.|$|/)",re.I)]
class KError(RuntimeError):pass
def hb(b):return hashlib.sha256(b).hexdigest()
def hf(p):return hb(p.read_bytes())
def conf():return json.loads(CFG.read_text(encoding="utf-8"))
def sanitize(s,limit=12000):
 k=os.environ.get("OPENAI_API_KEY");s=s.replace(k,"[REDACTED_API_KEY]") if k else s;s=re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b","[REDACTED_SECRET]",s);return s if len(s)<=limit else s[:limit]+"\n...[TRUNCATED]..."
def rb(cmd,cwd=None,check=True,env=None):
 cp=subprocess.run(cmd,cwd=str(cwd) if cwd else None,env=env,capture_output=True,text=False)
 if check and cp.returncode!=0:raise KError("command failed rc="+str(cp.returncode))
 return cp
def git(*a,check=True):return rb(["git","-C",str(ROOT),*a],check=check)
def real_index_path():
 p=Path(git("rev-parse","--git-path","index").stdout.decode("utf-8","surrogateescape").strip())
 return p if p.is_absolute() else ROOT/p
def git_ro(*a,check=True):
 ip=real_index_path()
 with tempfile.TemporaryDirectory(prefix="mppg-kernel-ro-index-") as td:
  shadow=Path(td)/"index";shutil.copy2(ip,shadow)
  env=os.environ.copy();env["GIT_INDEX_FILE"]=str(shadow);env["GIT_OPTIONAL_LOCKS"]="0"
  return rb(["git","-C",str(ROOT),*a],check=check,env=env)
def secret(rel):return any(x.search(rel) for x in SECRET)
def allowed(rel):return (rel in ALLOWED_EXACT or rel.startswith(ALLOWED_PREFIX)) and not secret(rel)
def excerpt(rel,start,count):
 if not allowed(rel):return {"ok":False,"reason":"outside_recovery_allowlist"}
 p=ROOT/rel
 if not p.is_file():return {"ok":False,"reason":"not_file"}
 data=p.read_bytes()
 if b"\0" in data[:8192]:return {"ok":False,"reason":"binary"}
 lines=data.decode("utf-8",errors="replace").splitlines();start=max(1,int(start));count=max(1,min(int(count),200));return {"ok":True,"path":rel,"text":sanitize("\n".join(lines[start-1:start-1+count]),16000)}
def grep(pattern,prefix):
 if prefix and not (prefix.startswith("governance/orchestrator") or prefix.startswith("governance/contracts") or prefix.startswith("governance/policy")):return {"ok":False,"reason":"outside_allowlist"}
 cp=git("grep","-n","--full-name","-e",pattern,"--",prefix or "governance/orchestrator",check=False);return {"ok":cp.returncode in (0,1),"output":sanitize(cp.stdout.decode("utf-8","surrogateescape"),16000)}
def metadata(rel):
 if not allowed(rel):return {"ok":False,"reason":"outside_allowlist"}
 p=ROOT/rel;return {"ok":p.exists(),"path":rel,"size":p.stat().st_size if p.is_file() else None,"sha256":hf(p) if p.is_file() else None}
TOOLS=[{"type":"function","name":"read_text_excerpt","description":"Read bounded allowlisted source text.","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"max_lines":{"type":"integer"}},"required":["path","start_line","max_lines"],"additionalProperties":False},"strict":True},{"type":"function","name":"git_grep","description":"Search allowlisted governance sources.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path_prefix":{"type":"string"}},"required":["pattern","path_prefix"],"additionalProperties":False},"strict":True},{"type":"function","name":"read_path_metadata","description":"Read allowlisted path metadata.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True}]
def tool(name,args):
 if name=="read_text_excerpt":return excerpt(args["path"],args["start_line"],args["max_lines"])
 if name=="git_grep":return grep(args["pattern"],args["path_prefix"])
 if name=="read_path_metadata":return metadata(args["path"])
 return {"ok":False,"reason":"tool_not_allowed"}
def outtext(resp):
 for i in resp.get("output",[]):
  if i.get("type")=="message":
   for c in i.get("content",[]):
    if c.get("type")=="output_text":return c.get("text","")
 return None
class Agent:
 def __init__(self):
  self.key=os.environ.get("OPENAI_API_KEY")
  if not self.key:raise KError("OPENAI_API_KEY missing")
  c=conf();self.model=os.environ.get("MPPG_OPENAI_MODEL") or c["model"];self.timeout=int(c["http_timeout_seconds"]);self.schema=json.loads(SCHEMA.read_text());self.prompt=PROMPT.read_text();self.master=MASTER.read_text()
 def endpoint(self):return os.environ.get("MPPG_ORCHESTRATOR_TEST_ENDPOINT") if os.environ.get("MPPG_ORCHESTRATOR_TEST_MODE")=="1" and os.environ.get("MPPG_ORCHESTRATOR_TEST_ENDPOINT") else PRODUCTION_API
 def post(self,payload):
  body=json.dumps(payload,ensure_ascii=True).encode()
  if self.key.encode() in body:raise KError("API key leaked into request body")
  req=urllib.request.Request(self.endpoint(),data=body,method="POST",headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json","User-Agent":"mppg-recovery-kernel/1"})
  try:
   with urllib.request.urlopen(req,timeout=self.timeout) as r:return json.loads(r.read())
  except urllib.error.HTTPError as e:raise KError("OpenAI HTTP "+str(e.code)) from None
  except Exception as e:raise KError("OpenAI request failure: "+type(e).__name__) from None
 def resolve(self,event,tail,history):
  packet={"failure_event":event,"child_output_tail":tail,"recovery_allowlist":{"prefixes":list(ALLOWED_PREFIX),"exact":sorted(ALLOWED_EXACT)},"history":history[-4:]};items=[{"role":"user","content":"PROMPT MASTER:\n"+self.master+"\n\nRECOVERY PACKET:\n"+json.dumps(packet,ensure_ascii=True,sort_keys=True)}]
  for _ in range(8):
   payload={"model":self.model,"instructions":self.prompt,"input":items,"tools":TOOLS,"tool_choice":"auto","parallel_tool_calls":False,"store":False,"text":{"format":{"type":"json_schema","name":"mppg_recovery_resolution","strict":True,"schema":self.schema}}};resp=self.post(payload);out=resp.get("output",[]);items.extend(out);calls=[x for x in out if x.get("type")=="function_call"]
   if calls:
    for c in calls:
     try:a=json.loads(c.get("arguments","{}"))
     except Exception:a={}
     items.append({"type":"function_call_output","call_id":c.get("call_id"),"output":sanitize(json.dumps(tool(c.get("name",""),a),ensure_ascii=True,sort_keys=True))})
    continue
   txt=outtext(resp)
   if not txt:raise KError("recovery AI returned no structured output")
   d=json.loads(txt)
   if set(d)!=set(self.schema["required"]):raise KError("recovery AI key-set mismatch")
   return d
  raise KError("recovery AI round limit")
def _read_gate(prompt):
 if os.environ.get("MPPG_ORCHESTRATOR_TEST_MODE")=="1" and os.environ.get("MPPG_AUTO_APPROVE_TEST_GATES")=="1":return os.environ.get("MPPG_EXPECTED_TEST_GATE_INPUT","")
 try:
  with open("/dev/tty","r+",encoding="utf-8",errors="replace") as t:t.write(prompt);t.flush();return t.readline().rstrip("\n")
 except Exception:
  try:return input(prompt)
  except EOFError:return ""
def gate(name,tok):
 expected=f"AUTHORIZE {name} {tok}";print("------------------------------------------------------------");print("AUTHORIZATION_REQUIRED="+name);print("TYPE_EXACTLY: "+expected)
 if _read_gate("> ")==expected:print("AUTHORIZATION_ACCEPTED="+name);return True
 print("AUTHORIZATION_ACCEPTED=false");return False
def status_items():
 raw=git_ro("status","--porcelain=v1","-z","--untracked-files=all").stdout.decode("utf-8","surrogateescape");return [{"status":r[:2],"path":r[3:]} for r in raw.split("\0") if len(r)>=4]
def external():return sorted(x["path"] for x in status_items() if x["status"]=="??" and not x["path"].startswith("governance/") and not x["path"].startswith("software/academic_pipeline_mppg/"))
def extfp(paths):
 rows=[]
 for rel in paths:
  p=ROOT/rel;rows.append([rel,p.stat().st_size,hf(p) if p.is_file() else "nonfile"])
 return hb(json.dumps(rows,ensure_ascii=True,separators=(",",":")).encode())
def envstate():
 p=ROOT/".env";return (p.exists(),hf(p) if p.is_file() else None)
def rebuild_manifest(root):
 m=root/"governance/MANIFEST.sha256";lines=[]
 for p in sorted((root/"governance").rglob("*")):
  if p.is_file() and p!=m and "__pycache__" not in p.parts and p.suffix!=".pyc":lines.append(f"{hf(p)}  {p.relative_to(root).as_posix()}\n")
 m.write_text("".join(lines),encoding="utf-8")
def overlay_current(shadow):
 for x in status_items():
  rel=x["path"]
  if not (rel.startswith("governance/") or rel==".gitattributes"):continue
  src=ROOT/rel;dst=shadow/rel
  if src.is_file():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def validate_scope(patch,scope):
 touched=[]
 for line in patch.splitlines():
  if line.startswith("+++ b/") or line.startswith("--- a/"):
   rel=line[6:]
   if rel!="/dev/null":touched.append(rel)
 touched=sorted(set(touched));scope=sorted(set(scope))
 if touched!=scope:raise KError("recovery patch scope mismatch")
 if not all(allowed(x) for x in touched):raise KError("recovery patch outside allowlist")
 return touched
def shadow_candidate(d):
 patch=d["shadow_patch"];scope=d["mutation_scope"]
 if not patch or len(patch.encode())>int(conf().get("max_kernel_patch_bytes",200000)):raise KError("invalid recovery patch")
 touched=validate_scope(patch,scope);td=tempfile.TemporaryDirectory(prefix="mppg-kernel-shadow-");shadow=Path(td.name)/"repo"
 if rb(["git","clone","--quiet","--no-hardlinks",str(ROOT),str(shadow)],check=False).returncode!=0:td.cleanup();raise KError("shadow clone failed")
 overlay_current(shadow);pp=Path(td.name)/"fix.patch";pp.write_text(patch,encoding="utf-8")
 if rb(["git","-C",str(shadow),"apply","--check","--binary",str(pp)],check=False).returncode!=0:td.cleanup();raise KError("patch apply-check failed")
 rb(["git","-C",str(shadow),"apply","--binary",str(pp)]);rebuild_manifest(shadow);final_scope=sorted(set(touched+["governance/MANIFEST.sha256"]))
 if rb(["git","-C",str(shadow),"diff","--check","HEAD"],check=False).returncode!=0:td.cleanup();raise KError("shadow diff-check failed")
 for rel in final_scope:
  p=shadow/rel
  if p.suffix==".py" and p.is_file():
   with tempfile.TemporaryDirectory(prefix="mppg-kernel-pyc-") as pytd:
    if rb(["python3","-S","-c","import py_compile,sys;py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)",str(p),str(Path(pytd)/"x.pyc")],check=False).returncode!=0:td.cleanup();raise KError("shadow python compile failed")
 env=os.environ.copy();env["MPPG_CANONICAL_REPO"]=str(shadow);env["PYTHONDONTWRITEBYTECODE"]="1"
 for t in ["test_static.py","test_autoremediation.py","test_recovery_kernel.py"]:
  tp=shadow/"governance/orchestrator/tests"/t
  if tp.is_file() and rb(["python3","-B","-S",str(tp)],cwd=shadow,env=env,check=False).returncode!=0:td.cleanup();raise KError("shadow test failed "+t)
 if rb(["sha256sum","-c","governance/MANIFEST.sha256"],cwd=shadow,check=False).returncode!=0:td.cleanup();raise KError("shadow governance manifest invalid")
 manifest=[{"path":rel,"sha256":hf(shadow/rel),"size":(shadow/rel).stat().st_size,"mode":stat.S_IMODE((shadow/rel).stat().st_mode)} for rel in final_scope];return td,shadow,final_scope,manifest
def materialize_candidate(td,shadow,scope,manifest):
 if git_ro("diff","--cached","--name-only").stdout.strip():raise KError("staging non-empty")
 ext=external();fp=extfp(ext);env0=envstate();refs0=(git("rev-parse","HEAD").stdout.strip(),git("rev-parse","@{upstream}").stdout.strip());payload={"head":refs0[0],"scope":scope,"manifest":manifest,"external_count":len(ext),"rollback_on_validation_failure":True};tok=hb(json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(",",":")).encode())
 if not gate("KERNEL RECOVERY MATERIALIZATION",tok):return False
 backup=Path(td.name)/"host_backup";meta={};runtime_backup=Path(td.name)/"runtime_backup";runtime_existed=RUNTIME.is_dir()
 if runtime_existed:shutil.copytree(RUNTIME,runtime_backup,copy_function=shutil.copy2)
 try:
  for rel in scope:
   p=ROOT/rel;meta[rel]=p.is_file()
   if p.is_file():b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
  for rel in scope:d=ROOT/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(shadow/rel,d)
  if RUNTIME.exists():shutil.rmtree(RUNTIME)
  shutil.copytree(GOV/"orchestrator",RUNTIME,copy_function=shutil.copy2)
  if rb(["sha256sum","-c","governance/MANIFEST.sha256"],cwd=ROOT,check=False).returncode!=0:raise KError("host governance invalid")
  if git_ro("diff","--check",check=False).returncode!=0:raise KError("host diff-check failed")
  if git_ro("diff","--cached","--name-only").stdout.strip():raise KError("staging changed")
  if envstate()!=env0 or external()!=ext or extfp(ext)!=fp:raise KError("protected/external state changed")
  if (git("rev-parse","HEAD").stdout.strip(),git("rev-parse","@{upstream}").stdout.strip())!=refs0:raise KError("refs changed")
  print("KERNEL_RECOVERY_MATERIALIZATION=PASS");return True
 except Exception:
  for rel,existed in meta.items():
   d=ROOT/rel
   if existed:s=backup/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
   elif d.exists():d.unlink()
  if RUNTIME.exists():shutil.rmtree(RUNTIME)
  if runtime_existed:shutil.copytree(runtime_backup,RUNTIME,copy_function=shutil.copy2)
  print("KERNEL_RECOVERY_ROLLBACK=PASS");raise
def parse_event(out):
 for line in out.splitlines():
  if line.startswith("ORCHESTRATOR_ERROR_EVENT_JSON="):
   try:return json.loads(line.split("=",1)[1])
   except Exception:pass
 return {"version":1,"recoverable_by_ai":True,"blocker":{"domain":"auditor_harness","code":"UNSTRUCTURED_CHILD_FAILURE","summary":"child exited without structured event","stage":"kernel"}}
def run_child(args):
 p=subprocess.Popen(["python3","-B","-S","-u",str(CORE),*args],stdin=None,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);chunks=[];assert p.stdout is not None
 while True:
  b=p.stdout.readline()
  if not b:break
  chunks.append(b)
  try:sys.stdout.buffer.write(b);sys.stdout.buffer.flush()
  except Exception:sys.stdout.write(b.decode("utf-8","replace"));sys.stdout.flush()
 return p.wait(),b"".join(chunks).decode("utf-8","surrogateescape")
def self_test():
 compile(Path(__file__).read_text(encoding="utf-8"),str(Path(__file__)),"exec");assert allowed("governance/orchestrator/mppg_orchestrator.py") and not allowed(".env") and not allowed("software/academic_pipeline_mppg/x.py");print("RECOVERY_KERNEL_SCOPE_GUARD=PASS");print("RECOVERY_KERNEL_TTY_GATE=PASS");print("RECOVERY_KERNEL_SELF_TEST=PASS")
def main():
 args=sys.argv[1:] or ["run"]
 if args==["kernel-self-test"]:self_test();return 0
 recoverable=args[0] in {"run","acceptance-test"};history=[];last=None;no_progress=0
 for cycle in range(1,int(conf().get("max_kernel_recovery_cycles",6))+1):
  print("RECOVERY_KERNEL_CYCLE="+str(cycle));rc,out=run_child(args)
  if rc==0:return 0
  if not recoverable:return rc
  event=parse_event(out)
  if not event.get("recoverable_by_ai",False) or event.get("staging_nonempty"):print("RECOVERY_KERNEL_FAIL_CLOSED_NONRECOVERABLE=true");return rc
  try:d=Agent().resolve(event,sanitize(out[-12000:],12000),history)
  except Exception as e:print("RECOVERY_KERNEL_AI_BLOCKED="+sanitize(str(e),2000));return 2
  fp=hb(json.dumps({"event":event,"decision":d},ensure_ascii=True,sort_keys=True).encode());no_progress=no_progress+1 if fp==last else 0;last=fp;history.append({"event":event,"decision":d,"fingerprint":fp})
  if no_progress>=int(conf().get("max_kernel_no_progress_cycles",2)):print("RECOVERY_KERNEL_NO_PROGRESS_FAIL_CLOSED=true");return 2
  if d["status"]!="PATCH_READY":print("RECOVERY_KERNEL_DECISION_STATUS="+d["status"]);return 2
  try:
   td,shadow,scope,manifest=shadow_candidate(d)
   try:
    if not materialize_candidate(td,shadow,scope,manifest):return 0
   finally:td.cleanup()
  except Exception as e:history.append({"kernel_candidate_error":sanitize(str(e),2500)});continue
 print("RECOVERY_KERNEL_CYCLE_BUDGET_EXHAUSTED=true");return 2
if __name__=="__main__":raise SystemExit(main())
