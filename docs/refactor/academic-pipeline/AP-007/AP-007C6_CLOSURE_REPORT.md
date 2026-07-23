# AP-007C.6 — Relatório de encerramento formal

## Escopo encerrado

A frente AP-007C migrou os comandos públicos de inspeção e
configuração selecionados para o runtime nativo:

- `--help`;
- `--list-toml-profiles`;
- `--list-institutions`;
- `--list-layouts`;
- `--explain-profile`;
- `--doctor`;
- `--check-config`.

## Rotas finais

- primeira onda informativa: `native_first_wave`;
- `--doctor`: `native_doctor`;
- `--check-config`: `native_check_config`;
- demais comandos: fallback legado explícito.

A precedência dos dispatchers foi preservada. O doctor antecede
check-config, e comandos de estágios anteriores continuam no fallback
quando combinados com check-config.

## Códigos de processo

- `--check-config` sem `--config`: `1`;
- diagnóstico válido: `0`;
- diagnóstico com problemas: `2`.

## Limites

O monólito histórico permanece como fallback para fluxos operacionais
ainda não migrados. A AP-007D deverá avançar em ondas operacionais
separadas, usando contratos e equivalência antes de cada integração.

## Decisão

A AP-007C está tecnicamente pronta para commit isolado, condicionado
à autorização explícita. Nenhum staging, commit, tag ou push integra
esta materialização de encerramento.
