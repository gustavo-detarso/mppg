# AP-004B — inventário de módulos e arquivos (v1.6)

> Levantamento somente preparatório. Nenhum arquivo produtivo foi modificado.

## Estado Git e base canônica

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local/remoto: `6de61fc9741035187836460d97da6d672708998a`.
- Commit AP-004A: `6de61fc9741035187836460d97da6d672708998a`.
- Inventário AP-004A: schema `4`, revisão `4.2`.
- Estado inicial aceito: `ap004b-artifacts-only`.

## Resumo semântico

- Candidatos: **6**.
- Referências brutas: **297**.
- Referências deduplicadas: **269**.
- Consumidores efetivos: **31** em **15** arquivos.
- Produtivos acionáveis: **7**.
- Contratos de compatibilidade: **24**.
- Históricos imutáveis: **166**.
- Operacionais protegidos: **10**.
- Contextuais não acionáveis: **62**.
- Referências ao diretório físico: **441** em **57** arquivos; reservadas à AP-006.
- Manifesto produtivo/compatibilidade: **23** arquivos.
- Manifesto de controle: **2** arquivos.
- Colisões de destino: **1**.
- Código produtivo alterado: **não**.

## Matriz de decisão

| Chave | Caminho atual | Destino proposto/suspenso | Classificação | Acionáveis | Compatibilidade | Excluídos | Política |
|---|---|---|---|---:|---:|---:|---|
| `pipeline_orchestrator` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | `app_bundle/scripts/pipeline/pipeline_orchestrator.py` | renomeação com compatibilidade | 7 | 24 | 221 | wrapper obrigatório no caminho histórico |
| `toml_generator` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py` | renomeação com compatibilidade | 0 | 0 | 5 | wrapper obrigatório no caminho histórico |
| `prisma_ai_prescreen_configurator` | `configurar_pretriagem_ia_prisma_v16.py` | `configurar_pretriagem_ia_prisma.py` | renomeação com compatibilidade | 0 | 0 | 4 | wrapper de script histórico até AP-004E |
| `article_diagnostic_log` | `gerar_log_diagnostico_artigo_v1_18.py` | `gerar_log_diagnostico_artigo.py` | renomeação com compatibilidade | 0 | 0 | 4 | wrapper de script histórico até AP-004E |
| `fulltext_executor_v1_13` | `executar_artigo_longo_fulltext_v1_13.py` | `suspenso: executar_artigo_longo_fulltext.py` | renomeação de alto risco | 0 | 0 | 2 | nenhuma decisão até comparar versões |
| `fulltext_executor_v1_14` | `executar_artigo_longo_fulltext_v1_14.py` | `suspenso: executar_artigo_longo_fulltext.py` | renomeação de alto risco | 0 | 0 | 2 | nenhuma decisão até comparar versões |

## Consumidores efetivos por candidato

### `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`

- SHA-256: `8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977`.
- AST SHA-256: `5104b0e7cb9340076f8eb41cd20d9e1eb01ac784feb24fb233fe085e7b7d12b5`.
- Guarda `__main__`: `['SystemExit', 'main']`.
- Acionáveis: **7**; compatibilidade: **24**.
- Evidências excluídas do aplicador: `{'contextual_non_actionable': 61, 'historical_immutable': 150, 'protected_operational': 10}`.

| Categoria | Tipo | Arquivo:linha | Evidência |
|---|---|---|---|
| `compatibility_contract` | `python_path_assignment` | `academic_pipeline/legacy.py:20` | `academic_pipeline_rc10` |
| `actionable_productive` | `python_string_reference` | `app_bundle/scripts/pipeline/academic_pipeline_gui.py:63` | `academic_pipeline_rc10.py` |
| `actionable_productive` | `python_path_assignment` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4215` | `pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config ` |
| `actionable_productive` | `python_path_assignment` | `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4216` | `pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config ` |
| `actionable_productive` | `python_string_reference` | `app_bundle/scripts/pipeline/academic_pipeline_tui.py:39` | `academic_pipeline_rc10.py` |
| `actionable_productive` | `python_path_assignment` | `app_bundle/scripts/pipeline/prisma_congelar_artigo.py:186` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` |
| `compatibility_contract` | `static_import` | `app_bundle/tests/test_entrypoints_orchestration_characterization.py:13` | `import academic_pipeline_rc10` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_entrypoints_orchestration_characterization.py:20` | `script fragment academic_pipeline_rc10.py document_model canônico academic_pipeline_toml_generator_interativo.py Gerador interativo completo academic_pipeline_tui.py Central operacional visual academic_pipeline_gui.py Interface gráfica FGV artigo_prisma_workflow.py Estado e validação gerar_artigo_fi` |
| `compatibility_contract` | `python_string_literal` | `app_bundle/tests/test_entrypoints_orchestration_characterization.py:23` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_path_assignment` | `app_bundle/tests/test_official_package_entrypoint.py:19` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_official_package_entrypoint.py:50` | `-c import sys, academic_pipeline; print('academic_pipeline_rc10' in sys.modules)` |
| `compatibility_contract` | `python_string_literal` | `app_bundle/tests/test_official_package_entrypoint.py:53` | `import sys, academic_pipeline; print('academic_pipeline_rc10' in sys.modules)` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_official_package_entrypoint.py:75` | `usage: academic_pipeline_rc10.py ` |
| `compatibility_contract` | `python_string_literal` | `app_bundle/tests/test_package_imports_entrypoints.py:16` | `academic_pipeline_rc10` |
| `compatibility_contract` | `python_string_literal` | `app_bundle/tests/test_package_imports_entrypoints.py:23` | `academic_pipeline_rc10` |
| `compatibility_contract` | `python_path_assignment` | `app_bundle/tests/test_package_imports_entrypoints.py:230` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_entrypoints.py:271` | `-m app_bundle.scripts.pipeline.academic_pipeline_rc10 --list-institutions` |
| `compatibility_contract` | `python_string_literal` | `app_bundle/tests/test_package_imports_entrypoints.py:273` | `app_bundle.scripts.pipeline.academic_pipeline_rc10` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_entrypoints.py:277` | `--list-institutions academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_entrypoints.py:278` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `subprocess_or_exec` | `app_bundle/tests/test_package_imports_prisma_core.py:204` | `--list-institutions academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_prisma_core.py:207` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `subprocess_or_exec` | `app_bundle/tests/test_package_imports_rendering.py:178` | `--list-institutions academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_rendering.py:181` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `subprocess_or_exec` | `app_bundle/tests/test_package_imports_support_services.py:247` | `--list-institutions academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_package_imports_support_services.py:250` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_packaging_metadata.py:148` | `--list-institutions academic_pipeline_rc10.py` |
| `compatibility_contract` | `python_string_reference` | `app_bundle/tests/test_packaging_metadata.py:149` | `academic_pipeline_rc10.py` |
| `compatibility_contract` | `static_import` | `app_bundle/tests/test_rc10_configuration_characterization.py:9` | `from academic_pipeline_rc10 import _refs_v6_apply_runtime_policy` |
| `actionable_productive` | `python_string_reference` | `executar_artigo_longo_fulltext_v1_13.py:9` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` |
| `actionable_productive` | `python_string_reference` | `executar_artigo_longo_fulltext_v1_14.py:9` | `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` |

### `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py`

- SHA-256: `3a3d5835ff81671897a44e4cdca588f04176ff03ed1e935d3ed29bf34eb531f1`.
- AST SHA-256: `2bfd168e4e60eb2e50c733b8aaf3fbc6da1d62b8864a196ae285ef423d7cfa68`.
- Guarda `__main__`: `['SystemExit', 'main']`.
- Acionáveis: **0**; compatibilidade: **0**.
- Evidências excluídas do aplicador: `{'contextual_non_actionable': 1, 'historical_immutable': 4}`.
- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.

### `configurar_pretriagem_ia_prisma_v16.py`

- SHA-256: `82d629145538c05120256e07f7567c3de0bb0244822ab5433d1c41d565c9bdc9`.
- AST SHA-256: `6e6b6a2cd44813a0b0f4f6e8310a77ad961c86633c6864ccd7abd19ed440ac0b`.
- Guarda `__main__`: `['SystemExit', 'main', 'print', 'SystemExit']`.
- Acionáveis: **0**; compatibilidade: **0**.
- Evidências excluídas do aplicador: `{'historical_immutable': 4}`.
- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.

### `gerar_log_diagnostico_artigo_v1_18.py`

- SHA-256: `17a83fda712aa3ab63957358282f70aa425d53b587dca8bae4defd36c5f4d757`.
- AST SHA-256: `5d7e36bdeae7c0b01240534367f48593a9334936245ffd52e574195ea37ee0a9`.
- Guarda `__main__`: `['main']`.
- Acionáveis: **0**; compatibilidade: **0**.
- Evidências excluídas do aplicador: `{'historical_immutable': 4}`.
- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.

### `executar_artigo_longo_fulltext_v1_13.py`

- SHA-256: `1694ab48bac702a311a4c64071ad943d366fbd9c511c8decb676230e689955fc`.
- AST SHA-256: `a44eddc7bff19ffb324d02ed09fb93f96c013e41c8c6f0c5090c5d2185db7ba4`.
- Guarda `__main__`: `[]`.
- Acionáveis: **0**; compatibilidade: **0**.
- Evidências excluídas do aplicador: `{'historical_immutable': 2}`.
- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.

### `executar_artigo_longo_fulltext_v1_14.py`

- SHA-256: `dac21bd05dd32cdd1dcae4d94bdf82845ea26751c1e95cb2481262cba2338ba1`.
- AST SHA-256: `6591e7d6d57e18ed7ec2a969af1b31a28be05b0d0c51ffe40718835c45fe0581`.
- Guarda `__main__`: `[]`.
- Acionáveis: **0**; compatibilidade: **0**.
- Evidências excluídas do aplicador: `{'historical_immutable': 2}`.
- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.

## Exclusões obrigatórias

- Documentação, manifestos, snapshots e ferramentas da AP-003 são históricos imutáveis.
- Artefatos finais da AP-004A permanecem históricos e não serão reescritos.
- Menções ao diretório `academic_pipeline_rc10_7_conformidade` pertencem à AP-006.
- Aplicadores, atualizadores, backups, outputs, assets, estados e relatórios operacionais ficam fora do aplicador e do `source_manifest`.
- Textos e docstrings sem consumo operacional comprovado permanecem contextuais.

## Colisão full-text

- Destino suspenso: `executar_artigo_longo_fulltext.py`.
- Origens: `executar_artigo_longo_fulltext_v1_13.py`, `executar_artigo_longo_fulltext_v1_14.py`.
- Bytes idênticos: **não**.
- AST idêntica: **não**.
- Similaridade por linhas: **0.730159**.
- Blocos alterados: **4**.
- Decisão: **suspensa para revisão manual; fora do primeiro aplicador**.

## Validação

- `py_compile`: `passed`.
- `git diff --check`: `passed`.
- Suíte específica: `13 passed`.
- Suíte consolidada: `431 passed, 3 xfailed`.

## Decisão de fase

O aplicador produtivo da AP-004B permanece bloqueado até aprovação expressa deste inventário semântico.
