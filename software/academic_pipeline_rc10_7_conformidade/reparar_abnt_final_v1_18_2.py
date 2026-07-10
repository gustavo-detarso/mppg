#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import shutil

p = Path("corrigir_abnt_final_v1_18_1.py")

if not p.exists():
    raise SystemExit(f"ERRO: não encontrei {p}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = p.with_name(p.name + f".bak_regex_lambda_v1_18_2_{stamp}")
shutil.copy2(p, bak)
print(f"[OK] Backup: {bak}")

txt = p.read_text(encoding="utf-8", errors="ignore")

old1 = 'return pattern.sub(matrix_org_block() + "\\\\n\\\\n", org_txt)'
new1 = 'return pattern.sub(lambda m: matrix_org_block() + "\\\\n\\\\n", org_txt)'

if old1 in txt:
    txt = txt.replace(old1, new1)
else:
    txt = txt.replace(
        'return pattern.sub(matrix_org_block() + "\\n\\n", org_txt)',
        'return pattern.sub(lambda m: matrix_org_block() + "\\n\\n", org_txt)',
    )

# Corrige a substituição do bloco matrix_tex dentro do TEX.
old2 = '''    tex = re.sub(
        r"(?ms)\\\\section\\{Matriz de evid[êe]ncias dos 20 estudos inclu[íi]dos\\}.*?(?=\\\\section\\{S[íi]ntese tem[áa]tica das evid[êe]ncias\\})",
        matrix_tex,
        tex,
        flags=re.I,
    )'''

new2 = '''    tex = re.sub(
        r"(?ms)\\\\section\\{Matriz de evid[êe]ncias dos 20 estudos inclu[íi]dos\\}.*?(?=\\\\section\\{S[íi]ntese tem[áa]tica das evid[êe]ncias\\})",
        lambda m: matrix_tex,
        tex,
        flags=re.I,
    )'''

if old2 in txt:
    txt = txt.replace(old2, new2)
else:
    # Substituição mais simples e segura caso a indentação esteja diferente.
    txt = txt.replace(
        '        matrix_tex,\n        tex,\n        flags=re.I,\n    )',
        '        lambda m: matrix_tex,\n        tex,\n        flags=re.I,\n    )',
    )

p.write_text(txt, encoding="utf-8")
p.chmod(0o755)

print("[OK] Script reparado para usar lambda em substituições com barras invertidas.")

# Verificação mínima
if "pattern.sub(lambda m: matrix_org_block()" not in txt:
    print("[AVISO] Não confirmei a correção do bloco ORG.")
if "lambda m: matrix_tex" not in txt:
    print("[AVISO] Não confirmei a correção do bloco TEX.")
