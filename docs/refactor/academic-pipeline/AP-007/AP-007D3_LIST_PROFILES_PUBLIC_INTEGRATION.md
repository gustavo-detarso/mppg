# AP-007D.3 — Integração pública de `--list-profiles`

## Resultado

O comando `--list-profiles` passou a usar a rota pública
`native_list_profiles`, delegando ao adaptador materializado na AP-007D.2.

## Precedência preservada

1. comandos informativos da primeira onda;
2. `--doctor`;
3. `--check-config`;
4. `--list-profiles`;
5. fallback legado.

A integração é conservadora: combinações de `--list-profiles` com comandos
operacionais não migrados permanecem no fallback. A combinação com rotas
nativas anteriores preserva a prioridade dessas rotas.

## Escopo

- runtime anterior: `7a2fd63ae060c74fe3f06e2eaf7457f176dc5059c0ac7aa95fc23805e810b1e6`;
- runtime integrado: `98f84244f3e447c108f627b9af55ab4782ed20347952b73c63f19e44a2b5371d`;
- adaptador preservado: `9a6bbbd980c0111067af17613b97a30b4bc9c333852cc30279c33999653e333d`;
- registro explícito de `--list-profiles` no parser construído pelo runtime;
- nenhuma alteração em `cli_parser.py`, CLI, dispatcher ou monólito;
- nenhum staging, commit, tag ou push.
