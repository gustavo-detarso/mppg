# AP-005E.2 — Caracterização de build e instalação isolada

## Baseline

- Branch: `ap-refactor/04-consumer-canonicalization`
- Commit: `0d553c975ad7948762f74aa4fcff3903578712de`
- Upstream: `origin/ap-refactor/04-consumer-canonicalization`
- Fingerprint: `73d01999b82d425436e8440ce24eb1757c379fc8c228c8999c0fe116279d3364`

## Artefatos construídos

- Wheel: `academic_pipeline_mppg-0.1.0-py3-none-any.whl` — SHA-256 `cea15ade083c2a0a530693dc04cdabe192de049bc7a4078e2f769cb456ade85c`.
- Sdist: `academic_pipeline_mppg-0.1.0.tar.gz` — SHA-256 `527ce87fb2702ef6906e605e4bcbb93d33a06c56c53059cb5b1f2d1b8c316318`.
- Wheel reproduzido com o mesmo hash em **3** execuções.
- Metadata, nome, versão e console script: corretos.
- Resíduos fortes nos arquivos de distribuição: nenhum.

## Conteúdo instalado

- Arquivos Python no wheel: **66**.
- Arquivos Python da fonte selecionados pelo layout de pacotes: **66**.
- Python rastreado fora do wheel: `app_bundle/templates/make_reference_fgv_docx.py`.
- Arquivos não-Python do pacote no wheel: **0**.
- Arquivos não-Python rastreados sob as raízes do pacote: **184**.
- A contagem rastreada não é uma lista de inclusão: contém projetos, outputs, documentação e resíduos que não devem integrar o wheel.

## Entry points

- `academic-pipeline --help`: aprovado com dependências externas.
- `python -m academic_pipeline --help`: aprovado com dependências externas.
- stdout e stderr dos dois entrypoints: idênticos.
- Os imports foram resolvidos pelo venv temporário, sem checkout ou PYTHONPATH.

## Defeitos confirmados

- `distribution-dependencies-empty` — **confirmed_distribution_metadata_defect**: O wheel declara Requires-Dist vazio; instalação isolada é aceita por pip check, mas o entrypoint falha por ausência de dotenv.
- `operational-package-data-absent` — **confirmed_package_data_defect**: O wheel contém zero arquivos não-Python; perfis FGV não são encontrados e --init-project não localiza seu template.
- `article-workflow-absolute-import` — **confirmed_installed_import_defect**: 64 de 65 módulos passivos importam; artigo_prisma_workflow falha com import absoluto de article_workflow.
- `prisma-helper-sibling-resolution` — **confirmed_installed_layout_defect**: Os helpers existem em app_bundle.scripts.pipeline, mas o módulo instalado os procura como irmãos de academic_pipeline.
- `hardcoded-personal-prompt-path` — **confirmed_portability_defect**: O código produtivo instalado contém caminho absoluto sob /home/gustavodetarso.

## Resultados excluídos como defeito

- `list-layouts-probe-invalid` — A tentativa sem --config falhou conforme o contrato atual.
- `doctor-external-tool-errors` — O doctor detectou dependências Python, mas retornou 2 por lualatex e biber ausentes no ambiente temporário.

## Decisão

A AP-005E.2 é uma subfase de caracterização e não altera código produtivo. Os defeitos reproduzidos tornam a AP-005E.3 obrigatória; ela não poderá ser encerrada como `no-op`.
