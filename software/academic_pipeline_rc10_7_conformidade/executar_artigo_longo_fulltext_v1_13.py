
#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ART = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo")
CFG_ART = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo/artigo_final_atestmed_abnt.toml")
PIPE = Path("app_bundle/scripts/pipeline/academic_pipeline_rc10.py")
VALIDATOR = Path("app_bundle/scripts/pipeline/validar_artigo_longo_fulltext.py")

def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)

run([sys.executable, PIPE, "--config", CFG_ART, "--check-config"])
run([sys.executable, PIPE, "--config", CFG_ART])
run([
    sys.executable,
    VALIDATOR,
    "--art-dir", ART,
    "--min-palavras", "8500",
    "--min-referencias", "20",
    "--min-paginas", "18",
])
