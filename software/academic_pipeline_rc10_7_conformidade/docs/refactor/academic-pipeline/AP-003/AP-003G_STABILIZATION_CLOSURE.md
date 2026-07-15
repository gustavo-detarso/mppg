# AP-003G — estabilização e encerramento da AP-003

## Resultado

A decomposição do orquestrador foi encerrada com a arquitetura pós-AP-003F preservada e sem novas alterações produtivas na AP-003G.

- Branch: `ap-refactor/03-orchestrator-decomposition`.
- HEAD validado: `7174664e22a941f4a6643d289106f37fa37289b5`.
- Estado da AP-003: **encerrada e estabilizada**.
- Arquivos produtivos modificados na AP-003G: **nenhum**.

## Arquitetura final

- Um único `main()` público.
- Um único núcleo interno `_ap003f_pipeline_core`.
- Alias histórico `_original_main_before_prisma_artigo_generico_wrapper`: ausente.
- Guarda direta chamando `main()`.
- `academic_pipeline.__main__` chamando `main()`.
- Módulo PRISMA referenciando o núcleo interno.

## Módulos congelados

| Módulo | SHA-256 |
|---|---|
| `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | `8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977` |
| `academic_pipeline/__main__.py` | `31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4` |
| `academic_pipeline/cli_parser.py` | `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8` |
| `academic_pipeline/command_dispatch.py` | `42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3` |
| `academic_pipeline/document_orchestration.py` | `3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e` |
| `academic_pipeline/prisma_generic_orchestration.py` | `f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072` |

## Validação

- Novos contratos AP-003G: **7**.
- Suíte específica: **45 passed**.
- Suíte consolidada: **408 passed, 3 xfailed**.
- Falhas: **0**.
- Erros: **0**.

### Comando específico

```bash
pipenv run pytest -q \
    tests/characterization/test_ap003a_orchestrator_contract.py \
    tests/characterization/test_ap003b_parser_contract.py \
    tests/characterization/test_ap003c_dispatch_contract.py \
    tests/characterization/test_ap003d_document_contract.py \
    tests/characterization/test_ap003e_prisma_generic_contract.py \
    tests/characterization/test_ap003f_main_unification_contract.py \
    tests/characterization/test_ap003g_stabilization_contract.py
```

### Comando consolidado

```bash
pipenv run pytest -q -ra \
    app_bundle/tests \
    tests
```

## Xfails preservados

1. `_refs_v6_strip_org` usa `para` em vez de `paren`.
2. `extract_org_abstracts` pode reter palavras-chave.
3. `WorkflowState._normalize` pode reabrir etapa bloqueada.

Esses três comportamentos continuam catalogados como defeitos legados e não foram corrigidos durante a AP-003.

## Encerramento

A AP-003 conclui a decomposição do orquestrador histórico em módulos de parser, despacho, documento e PRISMA/artigo genérico, mantendo as superfícies de entrada compatíveis e o comportamento caracterizado pela suíte histórica.
