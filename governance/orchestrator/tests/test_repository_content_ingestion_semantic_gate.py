#!/usr/bin/env python3
from __future__ import annotations
import contextlib, importlib.util, io, os, subprocess, tempfile
from pathlib import Path

SRC=Path(__file__).resolve().parents[1]/"mppg_orchestrator.py"
text=SRC.read_text(encoding="utf-8")

with tempfile.TemporaryDirectory(prefix="mppg-ingestion-semantic-gate-") as td:
 td=Path(td);repo=td/"repo";repo.mkdir()
 subprocess.run(["git","init","-b","master",str(repo)],check=True,capture_output=True)
 subprocess.run(["git","-C",str(repo),"config","user.name","T"],check=True)
 subprocess.run(["git","-C",str(repo),"config","user.email","t@example.com"],check=True)
 (repo/"tracked.txt").write_text("baseline\n",encoding="utf-8")
 subprocess.run(["git","-C",str(repo),"add","tracked.txt"],check=True)
 subprocess.run(["git","-C",str(repo),"commit","-m","base"],check=True,capture_output=True)

 os.environ["MPPG_CANONICAL_REPO"]=str(repo)
 mp=td/"o.py";mp.write_text(text,encoding="utf-8")
 spec=importlib.util.spec_from_file_location("o",mp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 fake_master=td/"master.md";fake_master.write_text("master\n",encoding="utf-8");m.MASTER=fake_master
 m.conf=lambda:{"max_autoremediation_cycles":2}

 ext=repo/"disciplinas"/"synthetic_external.pdf";ext.parent.mkdir(parents=True);ext.write_bytes(b"%PDF-synthetic\n")
 f=m.detect()
 assert f["kind"]=="external_untracked_preserved",f
 assert f["external_untracked"]==["disciplinas/synthetic_external.pdf"],f
 m.verify_governance=lambda:None
 oid=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
 m.refs=lambda:{"branch":"master","head":oid,"tracking":oid,"remote":oid}
 ctx,b=m.context()
 assert not b,b
 assert ctx["profile"]["front"]=="external-untracked-preservation",ctx
 assert ctx["profile"]["external_untracked_policy"]=="preserve_out_of_scope",ctx
 m.RUN_ROOT=td/"runs"
 m.safe_context=lambda:(ctx,[])
 m._interactive_read=lambda prompt: (_ for _ in ()).throw(AssertionError("authorization gate must not be reached"))
 out=io.StringIO()
 with contextlib.redirect_stdout(out):
  rc=m.guided_run(False)
 transcript=out.getvalue()
 assert rc==0,transcript
 assert "FRONT_KIND=external_untracked_preserved" in transcript,transcript
 assert "EXTERNAL_UNTRACKED_POLICY=PRESERVE_OUT_OF_SCOPE" in transcript,transcript
 assert "EXTERNAL_UNTRACKED_INTRA_RUN_STABLE=true" in transcript,transcript
 assert "AUTHORIZATION_REQUIRED=" not in transcript,transcript
 assert "FRONT_CLOSED=true" in transcript,transcript
 assert ext.is_file() and ext.read_bytes()==b"%PDF-synthetic\n"
 print("EXTERNAL_UNTRACKED_DISCOVERY_IS_NOT_INGESTION_INTENT=PASS")

 sw=repo/"software"/"academic_pipeline_mppg"/"synthetic.py";sw.parent.mkdir(parents=True);sw.write_text("x=1\n",encoding="utf-8")
 f2=m.detect()
 assert f2["kind"]=="canonical_software_untracked",f2
 ctx2,b2=m.context()
 assert any(x["code"]=="CANONICAL_SOFTWARE_UNTRACKED" for x in b2),b2
 print("CANONICAL_SOFTWARE_UNTRACKED_REMAINS_BLOCKER=PASS")

 try:
  m.ingestion(ctx)
 except m.BlockerError as e:
  assert e.blocker["domain"]=="authority_model",e.blocker
  assert e.blocker["code"]=="EXPLICIT_INGESTION_FRONT_REQUIRED",e.blocker
 else:
  raise AssertionError("ingestion unexpectedly accepted discovery as authority")
 print("INTENTIONAL_INGESTION_REQUIRES_EXPLICIT_SCOPE_AUTHORITY=PASS")
