
#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ART = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo")
CFG_ART = Path("/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo/artigo_final_atestmed_abnt.toml")
CANONICAL_PIPELINE = [sys.executable, "-m", "academic_pipeline"]
GEN = Path("app_bundle/scripts/pipeline/gerar_artigo_longo_fulltext_secional.py")
VAL = Path("app_bundle/scripts/pipeline/validar_artigo_longo_fulltext.py")

def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)

# 1. Roda o pipeline para manter bibliografia, preâmbulo, arquivos auxiliares e estrutura FGV atualizados.
run([*CANONICAL_PIPELINE, "--config", CFG_ART, "--check-config"])
run([*CANONICAL_PIPELINE, "--config", CFG_ART])

# 2. Sobrescreve o artigo curto por geração secional longa baseada no corpus full text.
run([
    sys.executable,
    GEN,
    "--art-dir", ART,
    "--min-palavras", "8500",
    "--compile",
])

# 3. Validação dura.
run([
    sys.executable,
    VAL,
    "--art-dir", ART,
    "--min-palavras", "8500",
    "--min-referencias", "20",
    "--min-paginas", "18",
])
