# AP-000 — Baseline legado do Academic Pipeline

## Objetivo

Estabelecer um estado inicial reproduzível para a refatoração do Academic
Pipeline, sem alterações funcionais no software.

## Identificação

- Projeto: Academic Pipeline
- Repositório: `gustavo-detarso/mppg`
- Diretório do software:
  `software/academic_pipeline_rc10_7_conformidade`
- Branch da fase: `ap-refactor/00-baseline`
- Branch de integração: `refactor/academic-pipeline`
- Commit de origem:
  `ef9acfd739274139637fae934c0c5dca4416728e`
- Data da validação: 15 de julho de 2026
- Sistema operacional: Debian GNU/Linux
- Python: 3.11.13
- Gerenciador de ambiente: Pipenv

## Procedimento

O ambiente foi reconstruído por meio do `Pipfile.lock`, incluindo as
dependências de desenvolvimento.

Foram executadas as seguintes verificações:

1. sincronização do ambiente Pipenv;
2. presença das dependências essenciais;
3. compilação sintática dos módulos Python;
4. coleta da suíte existente com pytest;
5. execução integral da suíte existente;
6. importação dos módulos no modo legado;
7. execução de `--help` nos principais pontos de entrada;
8. diagnóstico de importação dos módulos como pacote Python;
9. verificação do estado final da árvore Git.

## Resultado consolidado

```text
pipenv_sync=0
dependencies=0
compileall=0
pytest_collect=0
pytest=0
legacy_imports=0
entrypoints=0
package_imports_informativo=0
git_status_command=0

BASELINE_LEGADO=APROVADO
```

A árvore Git permaneceu limpa após a instalação das dependências e a
execução das validações.

## Comportamento legado confirmado

O software funciona quando o diretório
`app_bundle/scripts/pipeline` é incluído diretamente no caminho de
importação Python.

Esse comportamento deve ser preservado até que testes de caracterização
cubram adequadamente os fluxos que dependem dele.

## Dívida arquitetural identificada

A importação convencional pelo namespace
`app_bundle.scripts.pipeline` ainda não funciona para todos os módulos.

Foram identificadas incompatibilidades nos seguintes componentes:

- `document_builder`;
- `document_validator`;
- `prisma_pipeline`;
- `prisma_validator`;
- `render_org_latex`;
- `render_docx`.

Os módulos utilizam imports locais absolutos, como:

```python
from document_model import AcademicDocument
from bibliography_manager import BibBuildResult
from prisma_model import PrismaReport
```

Esses imports pressupõem a inclusão direta do diretório da pipeline no
`PYTHONPATH`.

A normalização dos imports não faz parte da fase AP-000. Ela somente
deverá ocorrer após a criação de testes de caracterização suficientes.

## Pontos de entrada validados

Foram validados, no modo legado, os seguintes pontos de entrada:

- `academic_pipeline_rc10.py`;
- `academic_pipeline_toml_generator_interativo.py`;
- `academic_pipeline_tui.py`;
- `academic_pipeline_gui.py`;
- `artigo_prisma_workflow.py`;
- `gerar_artigo_final_unificado.py`;
- `render_docx_canonico.py`.

## Critérios de aceite

- [x] Ambiente reconstruído pelo `Pipfile.lock`.
- [x] Dependências de produção e desenvolvimento instaladas.
- [x] Código Python compilável.
- [x] Testes existentes coletados.
- [x] Testes existentes aprovados.
- [x] Pontos de entrada legados executáveis.
- [x] Importações legadas funcionais.
- [x] Incompatibilidades de pacote catalogadas.
- [x] Nenhuma alteração funcional realizada.
- [x] Árvore Git limpa após a validação.

## Próxima fase

A próxima fase é a AP-001 — Testes de caracterização.

Ela deverá ampliar a cobertura do comportamento atual antes de qualquer
alteração estrutural, especialmente nos seguintes fluxos:

1. carregamento e resolução de configurações TOML;
2. criação do modelo canônico de documento;
3. geração de atividades;
4. geração de papers;
5. geração de documentos sem bibliografia;
6. renderização Org/LaTeX;
7. renderização DOCX;
8. fluxo PRISMA;
9. interfaces CLI que coordenam os módulos internos.
