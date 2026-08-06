# Configuração do ambiente Pipenv — academic_pipeline rc10.4

Este arquivo descreve o fluxo recomendado para criar um ambiente Pipenv isolado para a rc10.4.

## 1. Entrar na pasta da rc10.4

```bash
cd /home/gustavodetarso/Documentos/mppg/software/academic_pipeline_mppg
```

## 2. Criar ambiente e instalar dependências

Use o script incluído:

```bash
bash setup_pipenv_env.sh
```

Ou, explicitando seu Python do `pyenv`:

```bash
bash setup_pipenv_env.sh /home/gustavodetarso/.pyenv/versions/3.11.13/bin/python
```

O script instala:

```text
openai
pydantic
python-dotenv
pypdf
python-docx
openpyxl
pytest
```

## 3. Garantir `.env`

O `.env` deve ficar na raiz da rc10.4:

```text
/home/gustavodetarso/Documentos/mppg/software/academic_pipeline_rc10_7_conformidade/.env
```

Você pode copiar seu `.env` existente ou usar um link simbólico:

```bash
ln -s /caminho/do/seu/.env .env
```

Há um arquivo `.env.template` sem segredos como referência.

## 4. Inserir arquivos locais obrigatórios para PDF

```text
app_bundle/misc/academic-writing.el
app_bundle/misc/fgv.png
```

Os estilos FGV já estão em:

```text
app_bundle/misc/fgv/fgv-paper.sty
app_bundle/misc/fgv/fgv-dissertacao.sty
```

## 5. Diagnóstico

```bash
pipenv run python -m academic_pipeline --doctor
```

## 6. Projeto de teste

```bash
pipenv run python -m academic_pipeline \
  --init-project teste_paper_rc10 \
  --project-type paper

pipenv run python -m academic_pipeline \
  --config app_bundle/projetos/teste_paper_rc10/paper_config.toml \
  --check-config
```

Antes de executar o pipeline completo, substitua os arquivos vazios/fictícios do projeto por seus arquivos reais:

```text
app_bundle/projetos/teste_paper_rc10/documentos-base.zip
app_bundle/projetos/teste_paper_rc10/orientacoes.zip
app_bundle/projetos/teste_paper_rc10/doi_manifest.csv
```

## 7. Execução

```bash
pipenv run python -m academic_pipeline \
  --config app_bundle/projetos/teste_paper_rc10/paper_config.toml
```

## 8. Recompilação sem IA

```bash
pipenv run python -m academic_pipeline \
  --config app_bundle/projetos/teste_paper_rc10/paper_config.toml \
  --recompile \
  --org app_bundle/output/documento/teste_paper_rc10/teste_paper_rc10.org
```
