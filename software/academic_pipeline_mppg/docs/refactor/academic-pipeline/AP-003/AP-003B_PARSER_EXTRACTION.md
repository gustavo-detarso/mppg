# AP-003B — extração do parser e dos argumentos

## Escopo

A AP-003B extraiu exclusivamente a construção do `ArgumentParser`, as 62 declarações `add_argument` e a chamada `parse_args`. O despacho de comandos e as orquestrações documental, PRISMA e artigo genérico permaneceram no orquestrador histórico.

## Transformação

- Origem: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`, linhas 900–1016 no baseline da AP-003A.
- Destino: `academic_pipeline/cli_parser.py`.
- API criada: `build_parser(pipeline_version=...)` e `parse_args(argv=None, pipeline_version=...)`.
- O primeiro `main()` passou a delegar a leitura dos argumentos a `parse_cli_args`.
- Os dois `main()` e o alias `_original_main_before_prisma_artigo_generico_wrapper` foram preservados.
- A ordem, os nomes, os defaults, os tipos, as escolhas e os textos de ajuda foram mantidos.

## Contratos de caracterização

- 62 argumentos de usuário, além da ação automática de ajuda.
- snapshots de `--help` do script histórico e de `python -m academic_pipeline` preservados byte a byte;
- ausência de `ArgumentParser`, `add_argument` e `parser.parse_args` dentro do primeiro `main()`;
- permanência dos dois `main()` até a AP-003F;
- permanência do wrapper histórico como alias do primeiro `main()`.

## Integridade

- SHA-256 do orquestrador antes: `e0a1b4b80f3cae45c99316223430d2bb6360167ef0b220974cdb0e9b735b87cc`.
- SHA-256 do orquestrador depois: `51af32106184df8fd5810222a8ccdb5cc0818aa3e167ff8bd2e1c96199ef1a0f`.
- Nenhuma alteração foi realizada no despacho ou nos handlers.
