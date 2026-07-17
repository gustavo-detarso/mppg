# AP-005C — Estratégia para aliases de captura do gerador TOML

## Baseline

- Commit: `9372de8f621c9012a28d4c4a9a64e252a398bdf3`
- Módulo: `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py`
- SHA-256: `7b3ff44794275df2a3470796e78a25c3c87ca2c44f93fac6ec18eee397c89beb`
- Fingerprint do inventário: `6775af14e03baaa9138b0d24fce40625fb7f1fc7293d665b46fdc7f4defd35d8`

## Diagnóstico

Os quatro símbolos adiados pela AP-005B não são aliases redundantes. Cada um captura um binding anterior antes de uma redefinição posterior no mesmo módulo.

A substituição direta pelo nome corrente é proibida porque mudaria a cadeia de patches ou introduziria recursão.

## Estratégia selecionada

A AP-005C.1 deverá introduzir nomes canônicos explícitos para as quatro capturas, manter os aliases históricos apontando para o mesmo objeto anterior e migrar somente os consumidores internos para os novos nomes.

A aplicação deverá ser atômica no único módulo produtivo.

Não haverá novo export público.

A remoção dos aliases históricos permanece proibida.

## Mapeamento

| Alias histórico | Captura canônica planejada | Binding capturado | Usos |
|---|---|---|---:|
| `_original_ensure_reference_policy` | `_captured_wiz_input_ensure_reference_policy` | `_WizInputController._ensure_reference_policy` | 1 |
| `_wiz_disable_references_original` | `_captured_wiz_disable_references` | `_wiz_disable_references` | 1 |
| `_render_toml_original` | `_captured_render_toml` | `render_toml` | 1 |
| `_collect_outputs_and_options_original` | `_captured_collect_outputs_and_options` | `collect_outputs_and_options` | 3 |

## Contratos obrigatórios

1. A captura canônica deve ocorrer na mesma posição relativa.
2. O alias histórico e a captura canônica devem apontar para o mesmo binding anterior.
3. O binding corrente redefinido deve continuar distinto da captura.
4. Todos os seis consumidores internos devem migrar de forma atômica.
5. Nenhum alias histórico pode ser removido nesta fase.
6. Nenhum novo símbolo deve ser exportado publicamente.
7. A suíte canônica deve permanecer sem regressões.

## Bloqueio produtivo

Nenhum aplicador produtivo está autorizado por este documento. A geração do aplicador dependerá da aprovação dos contratos de caracterização e de uma nova auditoria da pré-imagem.
