# AP-006F.4 — Validação comparativa entre fonte e distribuição

## Situação

A AP-006F.4 foi concluída com gate `GATE_AP006F5=PASS`. A validação foi executada sobre a árvore canônica `software/academic_pipeline_mppg`, com a ponte histórica `software/academic_pipeline_rc10_7_conformidade` removida e o adaptador `academic_pipeline.legacy:run_legacy` preservado.

## Reparo incorporado

A comparação funcional revelou uma falha pública em `--list-toml-profiles`. O dispatch tentava importar módulos inexistentes dentro de `academic_pipeline` e, após o primeiro ajuste, ainda chamava quatro funções por chaves ausentes no dicionário `runtime`. O reparo final alterou oito alvos de importação para `app_bundle.scripts.pipeline` e substituiu quatro chamadas indiretas pelas funções efetivamente importadas.

O arquivo materializado possui SHA-256 `9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe`.

## Validação distributiva

Foram reconstruídos dois wheels no mesmo ambiente: o baseline diretamente do `HEAD` canônico e o candidato com o único overlay autorizado de `academic_pipeline/command_dispatch.py`. Ambos contêm 110 membros. As diferenças ficaram restritas ao dispatch, ao `RECORD` e a metadado técnico opcional do backend de build.

As operações `--list-institutions`, `--list-layouts` com exemplo TOML empacotado e `--list-toml-profiles` produziram resultados equivalentes entre a fonte e o wheel instalado. O console `academic-pipeline` e `python -m academic_pipeline` permaneceram funcionais.

## Suítes

- suíte focada: 95 aprovados e 1 `xfailed`;
- testes atuais não históricos: 603 aprovados e 3 `xfailed`;
- contratos históricos executados no `HEAD` limpo com o Python canônico: 33 aprovados;
- consolidação lógica: 636 aprovados e 3 `xfailed`.

Os contratos históricos foram executados no snapshot ao qual pertencem porque registram hashes, caminhos e a presença da ponte em fases anteriores. O ambiente histórico foi validado como consistente: metadado distributivo ausente e import neutro ausente.

## Integridade

O staging permaneceu vazio. O master operacional manteve exatamente 22 caminhos paralelos, com snapshot `f12b966a7c3e33ea3d1274219529cdb4db58454769fe8448bf1b6c820f318449`. Não houve criação persistente de `.pth`, alteração de `run_legacy`, commit, amend, tag ou push.

## Evidências

- V5: `ap006f4_revalidacao_integrada_comparativa_v5_20260721_185935.log`, SHA-256 `1a16d9a5528d70adc8f879a1d65471cc380aa99bf03eec65c5fbe0d5650aa335`;
- V6: `ap006f4_conclusao_historica_python_canonico_v6_20260721_191039.log`, SHA-256 `091e67b58d9fe5d88c55066723afc9cb34ead840d94df4aafe56fd851bee7661`.

## Decisão

A equivalência fonte/distribuição e a remoção da ponte estão aprovadas. A AP-006F pode avançar para a etapa AP-006F.5 de encerramento formal, permanecendo proibidos staging, commit e publicação sem autorização explícita.
