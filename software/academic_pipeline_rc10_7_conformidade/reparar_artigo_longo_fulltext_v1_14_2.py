#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import shutil

p = Path("app_bundle/scripts/pipeline/gerar_artigo_longo_fulltext_secional.py")

if not p.exists():
    raise SystemExit(f"ERRO: arquivo não encontrado: {p}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = p.with_name(p.name + f".bak_openai_tokens_v1_14_2_{stamp}")
shutil.copy2(p, bak)
print(f"[OK] Backup: {bak}")

txt = p.read_text(encoding="utf-8", errors="ignore")

old = '"max_tokens": max_tokens,'
new = '"max_completion_tokens": max_tokens,'

if old not in txt and new not in txt:
    raise SystemExit("ERRO: não encontrei a linha de max_tokens nem max_completion_tokens para reparar.")

txt = txt.replace(old, new)

p.write_text(txt, encoding="utf-8")
p.chmod(0o755)

print(f"[OK] Reparado: {p}")
print("[OK] Parâmetro OpenAI alterado para max_completion_tokens.")
