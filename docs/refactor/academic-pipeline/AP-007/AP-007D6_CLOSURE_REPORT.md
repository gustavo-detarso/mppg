# AP-007D.6 — Relatório de encerramento formal

## Resultado

A AP-007D encerra a migração controlada de três fluxos operacionais adicionais para o runtime nativo:

- `--list-profiles`;
- `--check-institution-compliance`;
- `--make-doi-manifest`.

O runtime final preserva o fallback legado explícito para comandos ainda não migrados, mantém precedência conservadora e não introduz `globals()`, `locals()` ou importação dinâmica implícita do monólito.

## Escopo

- baseline publicado: `ab066e68947ac5f33f1c12c9a7db5086d0f93790`;
- runtime final: `b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c`;
- caminhos candidatos: `48`;
- staging, commit, tag e push: não autorizados.

## Contratos públicos

- `--list-profiles`: retorno `0` e saída equivalente à fonte histórica canônica;
- `--check-institution-compliance`: retornos `0`, `1` e `2` preservados;
- `--make-doi-manifest`: diretório e ZIP com retorno `0`; erro de uso com retorno `1`;
- combinações concorrentes permanecem no fallback legado.

## Erros e correções

O manifesto de encerramento registra nominalmente todos os erros observados na AP-007D e a resolução incorporada para prevenir reexecução cega e retrabalho.

## Decisão

O escopo está pronto para decisão humana de commit isolado. Este encerramento não autoriza staging, commit, tag ou push.
