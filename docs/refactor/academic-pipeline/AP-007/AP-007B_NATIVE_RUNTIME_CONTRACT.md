# AP-007B — Contrato do Runtime Nativo

## Estado de partida

A AP-007B parte do commit `17725a5505eb2f9c0b1a6cfd5899e38d70031f80`.
O entrypoint público ainda encaminhava todas as chamadas diretamente para
`academic_pipeline.legacy:run_legacy`, que importava o monólito
`academic_pipeline_rc10` e acrescentava permanentemente seu diretório ao
`sys.path`.

A auditoria AP-007B.0 confirmou equivalência semântica entre o módulo público,
a chamada direta de `cli.main(argv)` e o script histórico para as cinco
superfícies da primeira onda. Também materializou a dívida observável:
`run_legacy` restaura `sys.argv`, mas não restaura `sys.path`.

## Topologia materializada

```text
academic_pipeline.__main__
    -> academic_pipeline.cli:main(argv)
    -> academic_pipeline.runtime:run(argv)
       |-> primeira onda: parser explícito + dispatch nativo
       `-> demais comandos: fallback legado enumerado
```

`cli.main` continua expondo `run_legacy` como ponto injetável de compatibilidade,
mas não o executa para os comandos nativos. Essa forma preserva os contratos
anteriores enquanto torna a decisão de fallback verificável.

## Primeira onda nativa

- `--help` e `-h`;
- `--list-toml-profiles`;
- `--list-institutions`;
- `--list-layouts`;
- `--explain-profile`.

A primeira onda reutiliza diretamente o contrato canônico
`build_parser(*, pipeline_version: str)` e os
dispatchers `003`, `005`, `006` e `007`. O programa público é definido
explicitamente como `academic-pipeline`, sem depender de `sys.argv[0]`.

## RuntimeContext

`RuntimeContext` é uma dataclass congelada e com slots. Seus campos são
dependências tipadas e enumeradas:

- `path_type`, projetado como a chave histórica `Path`;
- `load_config`;
- `describe_institution_profiles`;
- `available_layouts`;
- `resolve_layout_spec`;
- `explain_profile`.

O adaptador para `command_dispatch` é construído exclusivamente desses campos.
O resultado dos dispatchers preserva o contrato canônico
`DispatchResult(handled, value)`, e `value` é normalizado como código de saída.
O carregamento TOML nativo acrescenta `__config_path__`, `__config_dir__` e
aplica o perfil institucional antes da inspeção dos layouts. Não há uso de
`globals()`, `locals()`, importação dinâmica do monólito ou mutação de
`sys.path`.

## Decisão de fallback

`select_runtime_route` produz somente duas rotas:

- `native_first_wave`;
- `legacy_fallback`.

Uma chamada que contém opção da primeira onda não pode cair silenciosamente no
legado: se o parser aceitar a chamada e nenhum dispatcher a tratar, o runtime
levanta `NativeRuntimeError`. Os demais comandos seguem para o adaptador legado
com `argv` explícito, preservado como `list[str]` para compatibilidade com o
contrato público anterior.

## Política de validação desta subfase

A AP-007B valida equivalência no source tree com o pacote raiz explicitamente
disponível. Os testes que removem `PYTHONPATH` e executam diretamente o script
histórico dependem de instalação ou distribuição isolada e permanecem para a
AP-007E. Nesta subfase, as cinco superfícies são comparadas por probes próprios
com o mesmo source root explícito para o módulo público e o script histórico.

## Não objetivos desta materialização

- remover `run_legacy`;
- migrar fluxos de geração, renderização, PRISMA, GUI, TUI ou diagnóstico;
- reescrever comandos produzidos pelo wizard;
- alterar `legacy.py`, `command_dispatch.py`, `cli_parser.py`, o monólito ou o
  metadado de empacotamento;
- criar staging, commit, tag ou push.

## Evidência

- construtor do parser: `build_parser(*, pipeline_version: str)`;
- opções longas efetivamente registradas: `63`;
- tokens `--...` distintos no texto integral do help:
  `66` — a diferença decorre de opções citadas nas
  descrições;
- chaves explícitas usadas pelos dispatchers: `Path, available_layouts, describe_institution_profiles, explain_profile, load_config, resolve_layout_spec`;
- campos de `DispatchResult`: `handled, value`;
- cinco superfícies comparadas com o script histórico;
- fallback testado por injeção, sem importar o monólito;
- preservação de `sys.argv`, `sys.path` e diretório corrente nos comandos
  nativos.
