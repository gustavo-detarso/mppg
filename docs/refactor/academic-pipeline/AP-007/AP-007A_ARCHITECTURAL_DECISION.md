# AP-007A — Decisão arquitetural do runtime nativo

## Status

**Aceita e materializada para planejamento.** Esta decisão não modifica código
produtivo e não autoriza operações Git de escrita.

## Decisão

Adotar a estratégia
`native_runtime_with_explicit_argv_and_enumerated_legacy_fallback`.

A cadeia pública futura deverá ser:

```text
academic-pipeline / python -m academic_pipeline
                    │
                    ▼
        academic_pipeline.cli:main(argv)
                    │
                    ▼
        academic_pipeline.runtime:run(argv)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    parser explícito    RuntimeContext mínimo
          │                   │
          └─────────┬─────────┘
                    ▼
            dispatch nativo
```

O `RuntimeContext` não será uma cópia tipada de `globals()` ou `locals()`. Cada
handler deverá declarar apenas as dependências necessárias.

## Política para `run_legacy`

A remoção fica adiada. Durante a transição, `academic_pipeline.legacy` poderá
ser usado somente como fallback enumerado para famílias ainda não migradas.

O caminho público normal será considerado nativo apenas quando não:

- inserir diretório em `sys.path`;
- substituir `sys.argv`;
- importar `academic_pipeline_rc10`;
- chamar `module.main()` por adaptador dinâmico.

## Subfases consolidadas

1. AP-007A — inventário, depuração semântica e decisão arquitetural;
2. AP-007B — contrato do runtime e do entrypoint nativo;
3. AP-007C — comandos públicos de inspeção e configuração;
4. AP-007D — fluxos operacionais em ondas internas pequenas;
5. AP-007E — distribuição, isolamento e compatibilidade;
6. AP-007F — decisão final sobre `run_legacy`, encerramento e publicação.

A AP-007C será iniciada por `--help`, `--list-toml-profiles`,
`--list-institutions`, `--list-layouts` e `--explain-profile`.

## Critérios de encerramento da AP-007

- `cli.main` não depende de `run_legacy` no caminho público normal;
- parser e handlers recebem `argv` explicitamente;
- dispatch público não altera `sys.argv` nem `sys.path`;
- dispatch público não importa `academic_pipeline_rc10`;
- `globals()/locals()` não funcionam como container do runtime público;
- console instalado e `python -m academic_pipeline` têm paridade funcional;
- wheel funciona fora do checkout e sem `PYTHONPATH` herdado;
- o wizard gera comandos pelo entrypoint público;
- scripts auxiliares não reentram pelo monólito histórico;
- `run_legacy` é removido ou reduzido a fallback residual enumerado.

## Restrições para a AP-007B

A AP-007B deve caracterizar o comportamento antes da primeira alteração
produtiva. Ela não poderá migrar todos os fluxos de uma vez, ocultar falhas por
fallback genérico ou considerar duas falhas idênticas como equivalência.
