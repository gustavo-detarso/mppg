# AP-007A — Inventário canônico revisado do runtime legado

## Status

**Materializado como evidência e contrato.** Esta etapa não altera o runtime
produtivo, não remove `run_legacy` e não autoriza staging, commit ou publicação.

## Proveniência e preservação

Os três artefatos da auditoria inicial foram preservados byte a byte:

| Artefato bruto | SHA-256 |
|---|---|
| `ap007a_runtime_legado_inventory_raw.json` | `b2612edfad7bf38be62d498e942350c17a71a2f7cfe023503c2db320d927536e` |
| `ap007a_runtime_legado_references_raw.tsv` | `3d58b0fdcdec0435b5bcef703eb50a8bde0d85a8a06e425160b4066ff94e5c30` |
| `AP-007A_RUNTIME_LEGACY_RAW_EVIDENCE.md` | `7878a953bf1c6e6b2d4b169d0ba25da70b763c8391458afd55f5d79268164923` |

O inventário bruto permanece a evidência lexical completa. A camada canônica
não o reescreve; acrescenta uma revisão semântica separada.

## Resumo bruto

| Indicador | Valor |
|---|---:|
| Referências totais | 7.506 |
| Referências marcadas como ativas/alto risco | 94 |
| Opções descobertas em todos os parsers | 138 |
| Arquivos Python analisados por AST | 137 |
| Erros de parse AST | 0 |

## Correção semântica

As 94 ocorrências brutas não equivalem a 94 dependências produtivas:

| Partição | Ocorrências/linhas |
|---|---:|
| Backups e `.patch_backups` excluídos do runtime canônico | 37 |
| Árvore paralela antiga `software/academic_pipeline/` | 9 |
| Ocorrências brutas na árvore canônica | 48 |
| Linhas canônicas únicas após deduplicação | 47 |
| Linhas com acoplamento operacional efetivo | 20 |
| Linhas relevantes para migração, incluindo instrução gerada | 21 |

Das 47 linhas canônicas, 25 são comentários, docstrings ou proveniência; uma é
um teste dinâmico legítimo de dependências em `diagnostics.py`. Esses itens não
podem ser apresentados como acoplamentos runtime.

## Contrato público corrigido

O número 138 corresponde a opções encontradas em todos os parsers. O parser de
topo declara 62 opções explícitas; com o `--help` implícito do `argparse`, a
superfície pública possui 63 opções.

Primeira onda nativa aprovada para planejamento:

1. `--help`;
2. `--list-toml-profiles`;
3. `--list-institutions`;
4. `--list-layouts`;
5. `--explain-profile`.

`--list-profiles` pertence ao parser interno do gerador TOML e fica excluído da
primeira onda pública.

## Débito arquitetural central

O orquestrador histórico contém 84 chamadas a `globals()` e 84 a `locals()`;
39 pares ocorrem dentro de `_ap003f_pipeline_core()`. A modularização física
existe, mas os módulos extraídos ainda recebem um ambiente implícito construído
pelo monólito.

A AP-007 não deve apenas trocar `run_legacy` por outro wrapper. Deve substituir
a composição implícita por `argv` e dependências explícitas.

## Sete superfícies produtivas

1. `academic_pipeline/cli.py` — ponte pública;
2. `academic_pipeline/legacy.py` — adaptador de compatibilidade;
3. `academic_pipeline_rc10.py` — container implícito de dependências;
4. `prisma_generic_orchestration.py` — reentrada por `sys.argv`;
5. `academic_pipeline_toml_generator_interativo.py` — comandos legados gerados;
6. `gerar_artigo_final_unificado.py` — reentrada dinâmica auxiliar;
7. `pipeline_orchestrator.py` — alias do arquivo histórico.

## Conclusão

O inventário bruto é preservado como evidência. A camada canônica congela 47
linhas revisadas e 7 superfícies produtivas. A remoção imediata de `run_legacy`
fica rejeitada; a migração será progressiva e comprovada por família de
comandos.
