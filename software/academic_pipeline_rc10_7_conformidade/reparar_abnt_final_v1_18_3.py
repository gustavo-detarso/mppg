#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import shutil
import re

p = Path("corrigir_abnt_final_v1_18_1.py")

if not p.exists():
    raise SystemExit(f"ERRO: não encontrei {p}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = p.with_name(p.name + f".bak_decode_v1_18_3_{stamp}")
shutil.copy2(p, bak)
print(f"[OK] Backup: {bak}")

txt = p.read_text(encoding="utf-8", errors="ignore")

new_run = r'''def run(cmd, cwd=None, env=None, log=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    raw = p.stdout or b""
    try:
        output = raw.decode("utf-8", errors="replace")
    except Exception:
        output = raw.decode("latin-1", errors="replace")

    if log is not None:
        log.append("\n$ " + " ".join(cmd))
        log.append(f"[exit={p.returncode}]")
        log.append(output)

    return p.returncode, output
'''

txt2 = re.sub(
    r'''def run\(cmd, cwd=None, env=None, log=None\):\n    p = subprocess\.run\(\n        cmd,\n        cwd=cwd,\n        env=env,\n        text=True,\n        stdout=subprocess\.PIPE,\n        stderr=subprocess\.STDOUT,\n        check=False,\n    \)\n    if log is not None:\n        log\.append\("\\n\$ " \+ " "\.join\(cmd\)\)\n        log\.append\(f"\[exit=\{p\.returncode\}\]"\)\n        log\.append\(p\.stdout\)\n    return p\.returncode, p\.stdout\n''',
    new_run,
    txt,
    count=1,
)

if txt2 == txt:
    raise SystemExit("ERRO: não consegui localizar a função run() original para substituir.")

p.write_text(txt2, encoding="utf-8")
p.chmod(0o755)

print("[OK] Função run() reparada para capturar saída não UTF-8 do LaTeX.")
