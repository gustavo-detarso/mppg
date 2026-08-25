\
#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, os, shutil, subprocess, tempfile, time
from pathlib import Path
SRC=Path(__file__).resolve().parents[1]/"mppg_orchestrator.py"
text=SRC.read_text(encoding="utf-8")
with tempfile.TemporaryDirectory(prefix="mppg-index-shadow-test-") as td:
 td=Path(td);repo=td/"repo";repo.mkdir()
 subprocess.run(["git","init","-b","master",str(repo)],check=True,capture_output=True)
 subprocess.run(["git","-C",str(repo),"config","user.name","T"],check=True)
 subprocess.run(["git","-C",str(repo),"config","user.email","t@example.com"],check=True)
 (repo/"f.txt").write_text("alpha\n",encoding="utf-8")
 subprocess.run(["git","-C",str(repo),"add","f.txt"],check=True)
 subprocess.run(["git","-C",str(repo),"commit","-m","base"],check=True,capture_output=True)
 os.environ["MPPG_CANONICAL_REPO"]=str(repo)
 mp=td/"o.py";mp.write_text(text,encoding="utf-8")
 spec=importlib.util.spec_from_file_location("o",mp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ip=m.real_index_path();before=ip.read_bytes()
 baseline=(repo/"f.txt").read_bytes()
 (repo/"f.txt").write_text("temporary\n",encoding="utf-8")
 time.sleep(0.02)
 (repo/"f.txt").write_bytes(baseline)
 # This exact scenario may rewrite the physical stat-cache when normal Git is used.
 assert m.git_ro("diff","--name-only").stdout.strip()==""
 assert m.git_ro("status","--porcelain=v1").stdout.strip()==""
 after=ip.read_bytes()
 assert after==before, "read-only probes modified the real index"
print("REAL_INDEX_PHYSICAL_BYTE_PRESERVATION=PASS")
print("STAT_CACHE_FALSE_POSITIVE_REGRESSION=PASS")
