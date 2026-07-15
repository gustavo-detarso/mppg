# AP-003G — preparação da estabilização e encerramento

> Inventário pós-unificação, probes de entrada, compilação e suíte contratual. Nenhum módulo produtivo foi alterado.

## Git

- Branch: `ap-refactor/03-orchestrator-decomposition`
- HEAD: `7174664e22a941f4a6643d289106f37fa37289b5`
- Upstream: `origin/ap-refactor/03-orchestrator-decomposition`
- HEAD remoto: `7174664e22a941f4a6643d289106f37fa37289b5`
- Árvore: limpa.
- Sincronização local/remota: confirmada.

## Arquitetura resultante

- `main()` público: linhas 1326–1328.
- Núcleo `_ap003f_pipeline_core`: linhas 498–1243.
- Alias histórico: ausente.
- Guarda direta: preservada e chamando `main()`.
- `academic_pipeline.__main__`: chamando `main()`.
- PRISMA: referenciando o núcleo AP-003F.

## Módulos

| Módulo | Linhas | Bytes | Funções de nível superior | SHA-256 |
|---|---:|---:|---:|---|
| `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` | 1333 | 57269 | 50 | `8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977` |
| `academic_pipeline/__main__.py` | 9 | 197 | 0 | `31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4` |
| `academic_pipeline/cli_parser.py` | 133 | 9097 | 2 | `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8` |
| `academic_pipeline/command_dispatch.py` | 325 | 14483 | 20 | `42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3` |
| `academic_pipeline/document_orchestration.py` | 437 | 28560 | 26 | `3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e` |
| `academic_pipeline/prisma_generic_orchestration.py` | 715 | 39978 | 71 | `f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072` |

## Superfícies de entrada

| Superfície | Resultado | Snapshot |
|---|---|---|
| `direct_script` | equivalente | `tests/characterization/snapshots/ap003a/direct_script_help.txt` |
| `package_module` | equivalente | `tests/characterization/snapshots/ap003a/package_module_help.txt` |
| `console_script` | equivalente | `tests/characterization/snapshots/ap003a/package_module_help.txt` |

## Validações preparatórias

- Arquivos compilados: **6**.
- Suíte contratual AP-003: aprovada.
- Resultado: `......................................                                   [100%]
38 passed in 7.92s`.
- Código produtivo alterado: não.

## Gate produtivo da AP-003G

O aplicador de encerramento deverá:

1. criar o relatório e manifesto finais da AP-003;
2. adicionar contratos finais de arquitetura e entrypoints;
3. executar novamente a suíte específica da AP-003;
4. executar `pytest -q -ra app_bundle/tests tests`;
5. confirmar somente os três `xfail` conhecidos;
6. registrar hashes, contagens e comandos de validação;
7. não modificar o comportamento produtivo;
8. preparar o commit isolado de encerramento da AP-003.
