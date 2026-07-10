#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import shutil

p = Path("app_bundle/scripts/pipeline/gerar_artigo_longo_fulltext_secional.py")

if not p.exists():
    raise SystemExit(f"ERRO: arquivo não encontrado: {p}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = p.with_name(p.name + f".bak_indent_v1_14_1_{stamp}")
shutil.copy2(p, bak)
print(f"[OK] Backup: {bak}")

txt = p.read_text(encoding="utf-8", errors="ignore")
lines = txt.splitlines(True)

fixed = []
for line in lines:
    # Remove um nível global acidental de indentação.
    if line.startswith("    "):
        fixed.append(line[4:])
    else:
        fixed.append(line)

out = "".join(fixed)

# Garante shebang no início real do arquivo.
out = out.lstrip("\ufeff\n\r")
if not out.startswith("#!/usr/bin/env python3"):
    out = "#!/usr/bin/env python3\n" + out

p.write_text(out, encoding="utf-8")
p.chmod(0o755)

print(f"[OK] Arquivo reparado: {p}")
