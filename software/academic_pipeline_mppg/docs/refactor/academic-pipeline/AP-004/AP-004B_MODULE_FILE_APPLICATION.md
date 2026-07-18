# AP-004B — aplicação produtiva de módulos e arquivos (v1.4)

> Aplicação vinculada ao inventário AP-004B v1.6 aprovado. Nenhum commit foi criado.

## Base canônica

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD local/remoto: `6de61fc9741035187836460d97da6d672708998a`.
- Baseline: `431 passed, 3 xfailed`.
- Mudança funcional: **não**.
- Mudança de semântica CLI: **não**.

## Módulos canônicos e wrappers

- `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` → `app_bundle/scripts/pipeline/pipeline_orchestrator.py` (`canonical-alias-over-frozen-historical`).
- `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` → `app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py` (`canonical-copy-with-historical-wrapper`).
- `configurar_pretriagem_ia_prisma_v16.py` → `configurar_pretriagem_ia_prisma.py` (`canonical-copy-with-historical-wrapper`).
- `gerar_log_diagnostico_artigo_v1_18.py` → `gerar_log_diagnostico_artigo.py` (`canonical-copy-with-historical-wrapper`).

O orquestrador `academic_pipeline_rc10.py` permanece byte a byte intacto para
preservar os contratos congelados da AP-003G; `pipeline_orchestrator.py` é um
alias canônico que executa essa implementação no próprio namespace. Nos outros
três casos, o caminho canônico é cópia da implementação e o caminho histórico
é um loader transitório. Símbolos privados, monkeypatches, argumentos, código
de saída e guarda `__main__` permanecem preservados.

## Consumidores migrados

- `app_bundle/scripts/pipeline/academic_pipeline_gui.py:63`: `academic_pipeline_rc10.py` → `pipeline_orchestrator.py`.
- `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4215`: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` → `app_bundle/scripts/pipeline/pipeline_orchestrator.py`.
- `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4216`: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` → `app_bundle/scripts/pipeline/pipeline_orchestrator.py`.
- `app_bundle/scripts/pipeline/academic_pipeline_tui.py:39`: `academic_pipeline_rc10.py` → `pipeline_orchestrator.py`.
- `app_bundle/scripts/pipeline/prisma_congelar_artigo.py:186`: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` → `app_bundle/scripts/pipeline/pipeline_orchestrator.py`.

Foram alteradas exatamente cinco ocorrências em quatro arquivos. Os dois usos
internos de `executar_artigo_longo_fulltext_v1_13.py` e
`executar_artigo_longo_fulltext_v1_14.py` permaneceram intocados e continuam
funcionais pelo orquestrador histórico preservado `academic_pipeline_rc10.py`.

## Colisão full-text

- `executar_artigo_longo_fulltext.py`: **não criado**.
- `v1_13` e `v1_14`: preservados byte a byte.
- Decisão: `suspended-manual-review-required`.

## Proteções

- `academic-pipeline` e `python -m academic_pipeline` preservados.
- `academic_pipeline/legacy.py` preservado.
- 24 contratos de compatibilidade preservados.
- Diretório físico reservado à AP-006.
- Três xfails históricos mantidos sem correção.

## Validação

- `py_compile`: `passed`.
- `git diff --check`: `passed`.
- Suíte específica: `157 passed, 1 xfailed in 35.09s`.
- Suíte consolidada: `448 passed, 3 xfailed in 50.54s`.

## Estado

A consolidação permanece bloqueada até revisão do diff e aprovação expressa.
