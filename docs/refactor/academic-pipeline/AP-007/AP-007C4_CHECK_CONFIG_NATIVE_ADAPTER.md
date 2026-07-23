# AP-007C.4 — Adaptador nativo isolado de `--check-config`

## Decisão

A AP-007C.1 determinou que `--doctor` e `--check-config` fossem
migrados em ondas separadas. O doctor já foi integrado publicamente
na AP-007C.3.

Esta subfase materializa apenas o adaptador nativo de
`--check-config`. A rota pública continua no fallback legado até a
AP-007C.5.

## Estratégia

O adaptador não duplica a lógica diagnóstica. Ele:

1. usa o parser canônico;
2. carrega e aplica overrides à configuração;
3. monta um contexto explícito e imutável;
4. chama diretamente `dispatch_stage_017`;
5. devolve o valor de `DispatchResult`.

Assim, impressão, seleção de diretório, escrita do relatório JSON e
códigos semânticos `0/2` permanecem definidos pelo dispatcher
canônico.

## Restrições

- sem `globals()` ou `locals()`;
- sem importação do monólito histórico;
- sem mutação de `sys.path`, `sys.argv` ou cwd;
- sem integração pública nesta subfase;
- sem staging, commit, tag ou push.
