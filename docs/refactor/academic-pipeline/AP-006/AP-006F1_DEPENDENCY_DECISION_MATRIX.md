# AP-006F.1 — Matriz de decisão da ponte e dos fallbacks

- Baseline: `4db60736cfb4d2be53af32babdcdbfed84c3e6b4`
- Tree: `6f87372df000f07ea23cb6dfc3250989eef849cb`
- Evidência AP-006F.0: `ap006f0_auditoria_final_dependencias_20260721_111633.json` (`677b1dd3e4d858b066a40254034aa5b7fab82410796bba3a8d10efffa85f30f6`)
- Gate para AP-006F.2: **PASS**

## Conclusão executiva

A ponte simbólica e o adaptador Python devem ser decididos separadamente. A ponte permanece candidata a retirada, mas somente após ensaio sem ponte em árvore descartável. O símbolo `academic_pipeline.legacy:run_legacy` não é código morto: ele integra o caminho ativo `academic-pipeline → academic_pipeline.cli:main → run_legacy`. Portanto, sua retirada não pode ser uma exclusão simples; exige implementação canônica substituta e prova comparativa.

O aviso de AST observado em um arquivo sob `backups/` é não bloqueante. Trata-se de componente explícito de preservação histórica, não de runtime produtivo.

O wheel reconstruído na AP-006F.0 manteve tamanho `641807` e 110 membros, mas apresentou SHA-256 `1ae2ad753dc3426ddad603dba6759873b198dde02139ea0704fb55ff9570db46`, diferente do SHA-256 publicado na AP-006E (`fafbdd3d473220cf6d89e25a5e4cd077c78174293dbf097a7edac064ff9245ba`). Isso sugere que o artefato não é reprodutível byte a byte; a AP-006F.2 deverá comparar o conteúdo normalizado dos membros antes de classificar qualquer diferença como regressão.

## Matriz decisória

| Superfície | Decisão AP-006F.1 | Natureza | Evidência necessária na AP-006F.2 |
|---|---|---|---|
| `bridge_symlink` | `preserve_pending_ap006f2_no_bridge_trial` | `provisional` | remover somente a ponte em árvore descartável e comparar source tree, recursos, TOML, PRISMA, wrappers e suíte distributiva |
| `academic_pipeline.legacy:run_legacy` | `preserve_as_active_runtime_adapter_pending_replacement_trial` | `provisional` | não testar mera exclusão; testar uma implementação canônica substituta que preserve console, python -m, argumentos e códigos de saída |
| `legacy_internal_helpers` | `decide_with_run_legacy_implementation` | `provisional` | comparar comportamento do adaptador atual com a implementação canônica candidata |
| `console_entrypoint` | `retain_public_contract_academic_pipeline_cli_main` | `final_public_contract` | validar o console instalado em todas as variantes sem alterar o nome público |
| `python_module_entrypoint` | `retain_python_m_academic_pipeline_contract` | `final_public_contract` | executar --help e fluxo representativo em ambiente instalado |
| `historical_wrappers` | `classify_individually_no_bulk_removal` | `provisional` | testar diretamente apenas wrappers operacionais e preservar artefatos históricos deliberados |
| `public_packages` | `retain_academic_pipeline_and_app_bundle` | `final_public_contract` | confirmar imports vindos do purelib sem .pth |
| `resource_resolution` | `defer_bridge_decision_until_targeted_trial` | `provisional` | verificar instituições, templates, prompts, misc e recursos empacotados |
| `toml_projects` | `defer_until_representative_project_trials` | `provisional` | executar amostra representativa em baseline e variantes descartáveis |
| `prisma_flows` | `defer_until_representative_prisma_trial` | `provisional` | executar fluxo PRISMA representativo sem rede e com dados locais controlados |
| `wheel_build` | `wheel_independent_of_bridge_but_not_sufficient_for_source_decision` | `final_evidence_interpretation` | reconstruir em árvore separada e comparar manifesto normalizado de membros e SHA-256 por arquivo, sem usar apenas o hash bruto do ZIP |

## Cadeia ativa do runtime

```text
academic-pipeline
  -> academic_pipeline.cli:main
  -> academic_pipeline.legacy:run_legacy
  -> carregamento do módulo histórico de execução
```

Essa cadeia demonstra que `run_legacy` deve ser tratado como adaptador ativo, embora o nome permaneça legado.

## Variantes obrigatórias da AP-006F.2

| Variante | Ponte | Adaptador | Objetivo |
|---|---|---|---|
| `V0_baseline` | present | current_run_legacy | controle positivo e captura de resultados de referência |
| `V1_no_bridge` | removed_in_disposable_tree_only | current_run_legacy | decidir a ponte isoladamente sem misturar alteração do fallback |
| `V2_canonical_runtime_adapter` | present | canonical_candidate_without_academic_pipeline.legacy_public_dependency | decidir a substituição do adaptador separadamente da ponte |
| `V3_no_bridge_plus_canonical_adapter` | removed_in_disposable_tree_only | canonical_candidate | comprovar a topologia terminal combinada somente após V1 e V2 |

## Gate

A AP-006F.2 está autorizada tecnicamente apenas para ensaios em árvores descartáveis. Nenhuma remoção deve ocorrer na worktree real durante a fase experimental.
