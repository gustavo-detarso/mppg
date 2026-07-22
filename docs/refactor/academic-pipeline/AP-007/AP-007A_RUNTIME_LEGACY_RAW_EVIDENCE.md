# AP-007A — inventário do runtime legado e decisão preliminar

## FATOS OBSERVADOS

- Referências inventariadas: **7506**.
- Referências ativas: **94**.
- Referências relacionadas ao entrypoint: **61**.
- Comandos/opções descobertos: **138**.
- Arquivos Python analisados por AST: **137**.
- Erros de parse AST: **0**.

### Contagem por padrão

| Padrão | Quantidade |
|---|---:|
| academic_pipeline.legacy | 49 |
| academic_pipeline_rc10 | 7189 |
| dynamic_import | 93 |
| module.main | 3 |
| run_legacy | 79 |
| sys.argv | 93 |

### Contagem por classificação

| Classificação | Quantidade |
|---|---:|
| compatibilidade ativa | 8 |
| histórica | 7255 |
| produtiva interna | 84 |
| pública | 2 |
| teste/contrato | 157 |

### Cadeia pública/distributiva encontrada

- `{'source': 'software/academic_pipeline_mppg/pyproject.toml', 'kind': 'project.scripts', 'name': 'academic-pipeline', 'target': 'academic_pipeline.cli:main'}`

### Referências ativas de alto risco ligadas ao entrypoint

| Padrão | Classe | Arquivo | Linha | Trecho |
|---|---|---|---:|---|
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_1.py` | 1399 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_2.py` | 1424 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_3.py` | 1433 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_4.py` | 1449 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_5.py` | 1451 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_6.py` | 1460 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_8.py` | 1481 | `argv = argv if argv is not None else sys.argv[1:]` |
| sys.argv | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_2_9.py` | 1498 | `argv = argv if argv is not None else sys.argv[1:]` |
| dynamic_import | produtiva interna | `software/academic_pipeline/app_bundle/scripts/pipeline/recompilar_paper.py` | 36 | `spec = importlib.util.spec_from_file_location("academic_pipeline_rc7", script_path)` |
| dynamic_import | produtiva interna | `software/academic_pipeline_mppg/.patch_backups/prisma_pretriagem_ia_semantic_v16_20260630_114301/app_bundle/scripts/pipeline/diagnostics.py` | 150 | `__import__("dotenv" if mod == "dotenv" else mod)` |
| run_legacy | pública | `software/academic_pipeline_mppg/academic_pipeline/cli.py` | 7 | `from .legacy import run_legacy` |
| run_legacy | pública | `software/academic_pipeline_mppg/academic_pipeline/cli.py` | 12 | `return run_legacy(argv)` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 22 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:914-920` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 36 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:922-928` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 50 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:930-937` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 65 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:939-946` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 80 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:948-950` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 91 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:952-967` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 115 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:969-971` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 126 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:973-978` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 140 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:980-990` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 159 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:992-1009` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 184 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1011-1017` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 199 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1045-1046` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 209 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1047-1048` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 219 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1050-1060` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 238 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1062-1074` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 259 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1076-1082` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 274 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1084-1091` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 290 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1093-1094` |
| academic_pipeline_rc10 | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py` | 300 | `# Origem: app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1099-1139` |
| academic_pipeline_rc10 | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 20 | `LEGACY_MODULE_NAME = "academic_pipeline_rc10"` |
| dynamic_import | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 74 | `module = importlib.import_module(LEGACY_MODULE_NAME)` |
| run_legacy | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 97 | `def run_legacy(` |
| sys.argv | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 102 | `"""Executa o ``main`` legado preservando e restaurando ``sys.argv``.` |
| sys.argv | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 108 | `original_argv = sys.argv[:]` |
| sys.argv | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 112 | `sys.argv = [program_name, *forwarded]` |
| module.main | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 115 | `result = module.main()` |
| sys.argv | compatibilidade ativa | `software/academic_pipeline_mppg/academic_pipeline/legacy.py` | 117 | `sys.argv = original_argv` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/prisma_generic_orchestration.py` | 760 | `original_argv = list(sys.argv[1:])` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/prisma_generic_orchestration.py` | 772 | `old_argv = sys.argv[:]` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/prisma_generic_orchestration.py` | 773 | `sys.argv = [sys.argv[0]] + _prisma_artigo_generico_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/prisma_generic_orchestration.py` | 773 | `sys.argv = [sys.argv[0]] + _prisma_artigo_generico_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/academic_pipeline/prisma_generic_orchestration.py` | 777 | `sys.argv = old_argv` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py` | 4785 | `# necessariamente em sys.argv do módulo do wizard.` |
| dynamic_import | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/diagnostics.py` | 158 | `__import__("dotenv" if mod == "dotenv" else mod)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py` | 61 | `old_argv = sys.argv[:]` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py` | 64 | `sys.argv = [module_name.rsplit(".", 1)[-1] + ".py"] + argv` |
| dynamic_import | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py` | 68 | `mod = importlib.import_module(module_name)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/gerar_artigo_final_unificado.py` | 81 | `sys.argv = old_argv` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_flags_v1_3_20260709_114939/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1698 | `argv = list(sys.argv[1:])` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_generico_v1_5_20260709_123336/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1790 | `original_argv = list(sys.argv[1:])` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_generico_v1_5_20260709_123336/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1807 | `old_argv = sys.argv[:]` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_generico_v1_5_20260709_123336/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1808 | `sys.argv = [sys.argv[0]] + _prisma_artigo_toml_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_generico_v1_5_20260709_123336/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1808 | `sys.argv = [sys.argv[0]] + _prisma_artigo_toml_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_generico_v1_5_20260709_123336/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1812 | `sys.argv = old_argv` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_toml_v1_4_20260709_121406/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1751 | `original_argv = list(sys.argv[1:])` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_toml_v1_4_20260709_121406/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1770 | `old_argv = sys.argv[:]` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_toml_v1_4_20260709_121406/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1771 | `sys.argv = [sys.argv[0]] + _prisma_artigo_extra_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_toml_v1_4_20260709_121406/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1771 | `sys.argv = [sys.argv[0]] + _prisma_artigo_extra_strip(original_argv)` |
| sys.argv | produtiva interna | `software/academic_pipeline_mppg/backups/prisma_artigo_toml_v1_4_20260709_121406/app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1775 | `sys.argv = old_argv` |

### Candidatos à primeira onda

| Opção | Evidência | Linha | Prioridade |
|---|---|---:|---|
| `--help` | `argparse (opção implícita, condicionada à presença do parser)` | 0 | onda 1 — inspeção/configuração somente leitura |
| `--list-institutions` | `software/academic_pipeline_mppg/academic_pipeline/cli_parser.py` | 16 | onda 1 — inspeção/configuração somente leitura |
| `--list-layouts` | `software/academic_pipeline_mppg/academic_pipeline/cli_parser.py` | 17 | onda 1 — inspeção/configuração somente leitura |
| `--list-profiles` | `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py` | 4242 | onda 1 — inspeção/configuração somente leitura |
| `--list-toml-profiles` | `software/academic_pipeline_mppg/academic_pipeline/cli_parser.py` | 15 | onda 1 — inspeção/configuração somente leitura |

## PROPOSTAS — NÃO MATERIALIZADAS

**Decisão sobre `run_legacy`:** ADIADA: inventário inicial não autoriza remoção completa de run_legacy.

### Arquitetura-alvo proposta

- academic_pipeline.cli:main como proprietário do parser e do retorno público.
- registro explícito de comandos nativos com metadados de risco e mutabilidade.
- handlers nativos sem manipulação global de sys.argv.
- contexto de execução explícito para configuração, recursos e serviços.
- academic_pipeline.legacy limitado a fallback enumerado para fluxos ainda não migrados.
- remoção do fallback somente após equivalência funcional em fonte, console e wheel isolado.

### Subdivisão proposta

- AP-007A — inventário, caracterização e decisão arquitetural.
- AP-007B — contrato do entrypoint nativo.
- AP-007C — comandos públicos somente leitura/configuração.
- AP-007D — fluxos operacionais em ondas internas pequenas.
- AP-007E — distribuição, isolamento e compatibilidade.
- AP-007F — decisão final sobre run_legacy, encerramento e publicação.

### Critérios de encerramento propostos

- console script e python -m preservam opções públicas e códigos de saída contratados.
- cada comando migrado possui handler nativo e teste de caracterização.
- nenhum comando aprovado depende de mutação global de sys.argv para dispatch.
- fonte e wheel instalado executam com sucesso; falha idêntica não conta como equivalência.
- recursos distributivos usados pelos comandos estão presentes e validados estruturalmente.
- run_legacy é removido ou reduzido a fallback residual explicitamente enumerado.
- suíte produtiva atual e contratos históricos são executados em seus contextos corretos.
- encerramento formal define caminhos exatos e exige autorização antes de Git write operations.

> Esta seção é deliberadamente preliminar. A materialização de contratos ou arquivos da AP-007A depende da análise humana do log e de autorização explícita do escopo.
