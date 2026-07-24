# AP-007E.2 — Build controlado de sdist e wheel

Status: **build_approved_with_classified_findings**. Baseline: `766956710435f1c338d2e0332d24e55106b981b7`.

## Escopo e isolamento

Foram realizados dois builds independentes a partir de snapshots `git archive` do commit publicado, em diretórios descartáveis externos ao worktree, sem `PYTHONPATH`, sem rede, usando diretamente os hooks PEP 517 do backend declarado e sem isolamento que instale dependências e sem instalação dos artefatos produzidos.

## Resíduos rastreados excluídos antes da extração

O arquivo Git continha **435** membros residuais rastreados. Eles foram classificados apenas pelos nomes dos membros do TAR, antes da construção de caminhos de destino e antes da extração. Digest da lista integral: `a4452f3dd6413190c33047a5eeb5fa9821bc1abe6105c85dcc46cee7c69313c8`. A lista exata permanece no JSON.

| Classificação | Membros excluídos |
|---|---|
| backup_tree | 411 |
| patch_backup_tree | 24 |

## Metadata do projeto

| Campo | Valor |
|---|---|
| Nome | academic-pipeline-mppg |
| Versão | 0.1.0 |
| Backend | setuptools.build_meta |
| Console script | academic-pipeline = academic_pipeline.cli:main |
| SOURCE_DATE_EPOCH | 1784896736 |

## Artefatos

| Build | Tipo | Arquivo | Bytes | SHA-256 bruto |
|---|---|---|---|---|
| build_a | wheel | academic_pipeline_mppg-0.1.0-py3-none-any.whl | 654027 | 9bb7508aa1cabe208c5029d2b8e48e8181da5844130c3145fd8ef1907a7a8244 |
| build_a | sdist | academic_pipeline_mppg-0.1.0.tar.gz | 603552 | bf004990cea3edcfa37d3964cc31a4a947597d467606bd8501280a182723ffc1 |
| build_b | wheel | academic_pipeline_mppg-0.1.0-py3-none-any.whl | 654027 | 9bb7508aa1cabe208c5029d2b8e48e8181da5844130c3145fd8ef1907a7a8244 |
| build_b | sdist | academic_pipeline_mppg-0.1.0.tar.gz | 603553 | 01fdb338f5dbf9cfdbddafa1989fbee81eac6d39ae320cde5f680d18835851e2 |

## Reprodutibilidade normalizada

| Artefato | Manifestos equivalentes | Membros A | Membros B |
|---|---|---|---|
| wheel | True | 116 | 116 |
| sdist | True | 147 | 147 |

Campos voláteis nominalmente ignorados: timestamps dos contêineres ZIP/TAR/GZIP. O conteúdo descompactado, modos, caminhos, tamanhos e hashes dos membros foi comparado.

## Conteúdo obrigatório do wheel

| Caminho | Presente |
|---|---|
| academic_pipeline/__init__.py | sim |
| academic_pipeline/__main__.py | sim |
| academic_pipeline/check_config_runtime.py | sim |
| academic_pipeline/cli.py | sim |
| academic_pipeline/doctor_runtime.py | sim |
| academic_pipeline/doi_manifest_runtime.py | sim |
| academic_pipeline/institution_compliance_runtime.py | sim |
| academic_pipeline/list_profiles_runtime.py | sim |
| academic_pipeline/runtime.py | sim |

## Package data

Candidatos E0 dentro dos pacotes: **158**. Ausentes no wheel e ainda sujeitos a classificação: **121**.

| Candidato ausente |
|---|
| app_bundle/.academic_pipeline_tui_state.json |
| app_bundle/clean_institutional_tree_report.json |
| app_bundle/docs/CHANGELOG_rc10_1.md |
| app_bundle/docs/CHANGELOG_rc10_2.md |
| app_bundle/docs/CHANGELOG_rc10_3.md |
| app_bundle/docs/CHANGELOG_rc10_4.md |
| app_bundle/docs/CHANGELOG_rc10_5.md |
| app_bundle/docs/CHANGELOG_rc10_6.md |
| app_bundle/docs/CHANGELOG_rc10_7.md |
| app_bundle/docs/CHANGELOG_rc10_7_1.md |
| app_bundle/docs/CHANGELOG_rc10_7_10.md |
| app_bundle/docs/CHANGELOG_rc10_7_2.md |
| app_bundle/docs/CHANGELOG_rc10_7_4.md |
| app_bundle/docs/CHANGELOG_rc10_7_8.md |
| app_bundle/docs/CHANGELOG_rc10_7_9.md |
| app_bundle/docs/CONFORMIDADE_INSTITUCIONAL.md |
| app_bundle/docs/DEPENDENCIAS.md |
| app_bundle/docs/DIAGNOSTICO_VALIDACAO_RASTREABILIDADE.md |
| app_bundle/docs/README_rc10.md |
| app_bundle/docs/SETUP_PIPENV.md |
| app_bundle/docs/TOML_GENERATOR_RC10_7.md |
| app_bundle/examples/doi/doi_manifest_template.csv |
| app_bundle/examples/doi/doi_manifest_template_com_exemplos.csv |
| app_bundle/institutions/README_INSTITUTIONS.md |
| app_bundle/institutions/fgv/assets/README.md |
| app_bundle/institutions/fgv/manuals/README.md |
| app_bundle/projetos/atividade_3/atividade_config.toml |
| app_bundle/projetos/atividade_4/atividade_config.toml |
| app_bundle/projetos/atividade_case_einstein/.academic_pipeline/cache/atividade_case_einstein/case_einstein_quarta_revolucao_industrial.txt |
| app_bundle/projetos/atividade_case_einstein/atividade_case_einstein.toml |
| app_bundle/projetos/atividade_case_einstein/fontes/case_einstein_quarta_revolucao_industrial.txt |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.bib |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.check_config_report.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.compliance_report.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.compliance_report.md |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.document.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.org |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.outputs.txt |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.prompt_lock.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.prompt_lock.md |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.quality_report.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.quality_report.md |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.rc10_report.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.run_report.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein.tex |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein_bibliografia_diagnostico.json |
| app_bundle/projetos/atividade_case_einstein/output/atividade_case_einstein/atividade_case_einstein_export_pdf.el |
| app_bundle/projetos/atividade_case_einstein/src/case_einstein_quarta_revolucao_industrial.txt |
| app_bundle/projetos/atividade_wellhub/.academic_pipeline/cache/atividade_wellhub/wellhub.txt |
| app_bundle/projetos/atividade_wellhub/atividade_wellhub.toml |
| app_bundle/projetos/atividade_wellhub/fontes/wellhub.txt |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.bib |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.check_config_report.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.compliance_report.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.compliance_report.md |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.document.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.org |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.outputs.txt |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.prompt_lock.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.prompt_lock.md |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.quality_report.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.quality_report.md |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.rc10_report.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.run_report.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub.tex |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub_bibliografia_diagnostico.json |
| app_bundle/projetos/atividade_wellhub/output/atividade_wellhub/atividade_wellhub_export_pdf.el |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.busca_prisma_log.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.candidatos_brutos.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.candidatos_deduplicados.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.check_config_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.outputs.txt |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.pre_triagem_ia.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prisma_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.md |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.protocolo_busca_prisma.md |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.rc10_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.org |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar.tex |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.relatorio_prisma_preliminar_export_pdf.el |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.run_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/execucoes_anteriores/fontes_instaveis_20260630_162614/relatorio_prisma_prisma_fluxo_pmf.triagem_titulo_resumo.csv |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.busca_prisma_log.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.candidatos_brutos.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.candidatos_deduplicados.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.check_config_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_log.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.curadoria_ia_resumo.txt |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.diagrama_prisma.png |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.diagrama_prisma_contagens.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.matriz_estudos_incluidos.csv |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.outputs.txt |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.pre_triagem_ia.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prisma_report.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prisma_report_final.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.json |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.prompt_lock.md |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.protocolo_busca_prisma.md |
| app_bundle/projetos/prisma_fluxo_pmf/output_pesquisa/relatorio_prisma_prisma_fluxo_pmf/relatorio_prisma_prisma_fluxo_pmf.rc10_report.json |

## Achados classificados

| Código | Severidade | Bloqueante | Detalhe |
|---|---|---|---|
| TRACKED_RESIDUAL_SOURCE_MEMBERS_EXCLUDED_BEFORE_EXTRACTION | warning | False | tracked backup/cache/build residue was classified from archive member names and excluded before destination path construction or extraction |
| SDIST_RAW_ARCHIVE_HASH_DIFFERS_NORMALIZED_EQUAL | info | False | raw sdist bytes differ while normalized member manifests are equal |
| PACKAGE_DATA_CANDIDATES_NOT_PRESENT_IN_WHEEL | warning | False | E0 candidates require classification in AP-007E.3/E.4; not all non-Python source files are necessarily distributable package data |

## Gate

`[GATE] AP-007E.2: SDIST E WHEEL CONSTRUÍDOS EM DUAS SANDBOXES, INVENTARIADOS E COM REPRODUTIBILIDADE NORMALIZADA VALIDADA; WORKTREE PRESERVADO, SEM INSTALAÇÃO NO AMBIENTE CANÔNICO, STAGING, COMMIT, TAG OU PUSH.`
