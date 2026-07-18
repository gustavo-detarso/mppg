# academic_pipeline rc10.7 — conformidade institucional, prompt bank e rastreabilidade

Esta versão usa `document_model` como fonte canônica do documento e renderiza saídas determinísticas em ORG/PDF/DOCX. Também inclui relatório PRISMA como artefato próprio.

## Fluxo principal

```text
documentos-base.zip + orientacoes.zip + doi_manifest.csv + TOML
        ↓
.bib neutro enriquecido por DOI/bases/IA
        ↓
.document.json canônico
        ↓
.org / .pdf / .docx
        ↓
relatório PRISMA opcional em JSON/ORG/PDF/DOCX/XLSX/SVG
```

## Estabilidade corrigida nesta versão

- `--somente-renderizar` não exige `OPENAI_API_KEY`.
- `[latex].pdf_engine` passa a ser respeitado.
- Atividade usa Ficha Técnica, não capa de paper.
- Dissertação usa macros próprias do `fgv-dissertacao.sty`.
- O relatório PRISMA copia o `.bib` para a pasta do relatório antes de compilar.
- Semantic Scholar e Scopus foram implementados para lookup por DOI quando configurados.
- Deduplicação bibliográfica escolhe a entrada de melhor qualidade.
- DOCX tem renderização manual estável e opção Pandoc/CSL para APA/ABNT mais rigoroso.
- Pacote sem `__pycache__` e sem pasta `_test`.

## Arquivos locais incluídos neste bundle

Este bundle já inclui os arquivos enviados pelo usuário:

```text
app_bundle/misc/academic-writing.el
app_bundle/misc/fgv.png
app_bundle/institutions/fgv/assets/fgv.png
```

O arquivo `.env` não é incluído por conter tokens e deve ser criado ou copiado localmente na raiz do projeto.

Os `.sty` FGV já estão em:

```text
app_bundle/misc/fgv/fgv-paper.sty
app_bundle/misc/fgv/fgv-dissertacao.sty
```

## Instalação

```bash
cd /home/gustavodetarso/Documentos/mppg/software/academic_pipeline
unzip /caminho/academic_pipeline_rc10_7_conformidade_bundle.zip -d /tmp/rc10_4
bash /tmp/rc10_4/academic_pipeline_rc10_7_conformidade/install_rc10.sh
```

## Exemplo de execução

```bash
cd /home/gustavodetarso/Documentos/mppg/software/academic_pipeline
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/config/examples/paper_rc10_exemplo.toml
```

## Apenas renderizar um document.json existente

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/config/examples/paper_rc10_exemplo.toml \
  --somente-renderizar \
  --document-json app_bundle/output/documento/paper_nome_do_tema/paper_nome_do_tema.document.json
```

## Bibliografia APA/ABNT

O `.bib` é neutro. O estilo é escolhido no TOML:

```toml
[bibliografia]
latex_style = "apa"
```

ou:

```toml
[bibliografia]
latex_style = "abnt"
```

Para ABNT no PDF, seu TeX Live precisa ter `biblatex-abnt`.

## DOCX com CSL

Por padrão, o DOCX é gerado por `python-docx`. Para usar Pandoc/CSL:

```toml
[docx]
usar_pandoc = true
csl_path = "../../templates/csl/apa.csl"
falhar_se_pandoc_falhar = false
```

Coloque arquivos CSL em:

```text
app_bundle/templates/csl/
```

## rc10.3 — diagnóstico, validação e rastreabilidade

Antes de executar um paper/atividade/dissertação, recomenda-se:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config caminho.toml --check-config
```

Após a execução, o pipeline gera também:

```text
<prefixo>.run_report.json
<prefixo>.outputs.txt
```

Para recompilar um `.org` já ajustado manualmente, sem chamar a IA:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config caminho.toml \
  --recompile \
  --org caminho/do/documento.org
```

Consulte `docs/DIAGNOSTICO_VALIDACAO_RASTREABILIDADE.md` para o fluxo completo.

## rc10.4 — comandos adicionais de usabilidade

### Criar um novo projeto

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --init-project paper_nome_do_tema \
  --project-type paper
```

Tipos aceitos: `paper`, `atividade`, `prisma`, `atividade_prisma`, `paper_prisma`.

### Gerar `doi_manifest.csv`

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --make-doi-manifest \
  --input-zip app_bundle/projetos/paper_nome_do_tema/documentos-base.zip \
  --output app_bundle/projetos/paper_nome_do_tema/doi_manifest.csv
```

### Inspecionar bibliografia

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --inspect-bib app_bundle/output/documento/paper_nome_do_tema/paper_nome_do_tema.bib
```

### Gerar relatório de qualidade textual

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --quality-report \
  --document-json app_bundle/output/documento/paper_nome_do_tema/paper_nome_do_tema.document.json \
  --org app_bundle/output/documento/paper_nome_do_tema/paper_nome_do_tema.org \
  --bib app_bundle/output/documento/paper_nome_do_tema/paper_nome_do_tema.bib
```

Nas execuções completas, o `quality_report.md` já é gerado automaticamente.

## Configuração rápida do Pipenv

A distribuição inclui `Pipfile`, `requirements.txt`, `.env.template` e `setup_pipenv_env.sh`.

Fluxo recomendado:

```bash
cd /home/gustavodetarso/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade
bash setup_pipenv_env.sh
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --doctor
```

Consulte `docs/SETUP_PIPENV.md` para o passo a passo completo.

## rc10.7 — perfis institucionais

Para usar as especificações da FGV sem repetir todos os caminhos no TOML, inclua:

```toml
[instituicao]
perfil = "fgv"
```

O perfil FGV fica em:

```text
app_bundle/institutions/fgv/
```

Ele define defaults para templates, `.sty`, `reference_docx`, padrão bibliográfico, regras de formatação e validações. Valores explicitamente informados no TOML do trabalho sempre prevalecem sobre o perfil.

Listar perfis disponíveis:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --list-institutions
```

Criar projeto já com perfil FGV:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --init-project meu_paper \
  --project-type paper \
  --institution fgv
```

## rc10.7 — prompt bank e diretivas reutilizáveis

A partir da rc10.6, diretivas gerais, de pesquisa, de triagem, de documento e de instituição podem ficar em arquivos reutilizáveis dentro de `app_bundle/prompts/` ou `app_bundle/institutions/<perfil>/prompts/`.

Estrutura principal:

```text
app_bundle/prompts/
├── global/orientacao_geral_execucao.txt
├── research/triagem_prompt.txt
├── research/diretivas_extras.txt
├── document/paper.txt
├── document/atividade.txt
├── document/dissertacao.txt
└── prisma/relatorio_prisma.txt
```

No TOML:

```toml
[prompts]
ativos = true
global_paths = ["../../prompts/global/orientacao_geral_execucao.txt"]
institution_paths = ["profile://prompts/fgv_geral.txt"]
research_paths = [
  "../../prompts/research/triagem_prompt.txt",
  "../../prompts/research/diretivas_extras.txt"
]
paper_paths = ["../../prompts/document/paper.txt"]
atividade_paths = ["../../prompts/document/atividade.txt"]
dissertacao_paths = ["../../prompts/document/dissertacao.txt"]
prisma_paths = ["../../prompts/prisma/relatorio_prisma.txt"]
```

Ordem de aplicação:

```text
prompt base do pipeline
→ global
→ institucional
→ tarefa/tipo de documento
→ orientações do projeto
→ documentos-base
```

A diretriz geral de execução foi saneada: a formulação antiga que pedia “Chain of Thought/Cadeia de Pensamento” foi substituída por uma exigência de planejamento interno rigoroso e justificativa sintética, verificável e orientada à decisão.

Para verificar quais prompts estão ativos em um TOML:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --show-prompts
```

O `run_report.json` agora registra os prompts efetivamente carregados, com caminho, tamanho, hash SHA-256 e indicação de saneamento quando aplicável.

## rc10.7 — Conformidade institucional e prompt lock

A partir da rc10.7, além de `doctor`, `check-config`, `quality_report` e `prompt bank`, o pipeline possui uma camada de auditoria institucional.

### Explicar perfil institucional

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --explain-profile fgv
```

### Verificar conformidade institucional

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --check-institution-compliance
```

O comando gera:

```text
<prefixo>.compliance_report.md
<prefixo>.compliance_report.json
```

### Gerar prompt lock

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --config app_bundle/projetos/meu_paper/paper_config.toml \
  --write-prompt-lock
```

A execução completa também gera automaticamente:

```text
<prefixo>.prompt_lock.json
<prefixo>.prompt_lock.md
```

## Gerador interativo de TOML

A partir da versão rc10.7.11, o bundle inclui um gerador interativo completo de TOML:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --init-toml
```

Para listar presets:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --list-toml-profiles
```

Para iniciar direto em um preset:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py \
  --init-toml \
  --toml-profile atividade_local_fgv
```

Documentação completa: `app_bundle/docs/TOML_GENERATOR_RC10_7.md`.
