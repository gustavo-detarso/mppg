# AP-003F — unificação do main

## Estratégia aplicada

O primeiro `main()` histórico foi preservado integralmente e renomeado para `_ap003f_pipeline_core`. O segundo `main()`, já reduzido na AP-003E a um wrapper fino do fluxo PRISMA/artigo genérico, permaneceu como a única entrada pública.

O alias `_original_main_before_prisma_artigo_generico_wrapper` foi removido. As referências internas do módulo PRISMA ao alias foram substituídas pelo nome explícito do núcleo. Como o wrapper público envia `globals()` ao entrypoint, o núcleo continua disponível para o fallback legado sem alias.

## Transformações

- Primeiro `main()`: linhas 498–1243, renomeado para `_ap003f_pipeline_core`.
- Alias removido: linha 1325.
- `main()` público preservado: linhas 1327–1329.
- Referências substituídas no módulo PRISMA: **2**.
- Guarda direta preservada.
- `academic_pipeline.__main__` preservado byte a byte.

## Estado estrutural resultante

- Definições públicas `main()`: **1**.
- Núcleos internos `_ap003f_pipeline_core`: **1**.
- Atribuições do alias histórico: **0**.
- O `main()` público delega ao entrypoint AP-003E.
- O fallback PRISMA chama o núcleo interno.

## Integridade

- Orquestrador antes: `431882d57a5a6ed334985b51a04db782ceda8de5cff1aa0e3bf856c4aa2c5b3a`.
- Orquestrador depois: `8516c1b4d55921440905dd4eba84241efde9093a151f9c6ccc33757474bf8977`.
- PRISMA antes: `5b659ef03bee30e7e55dd50ad323edc3d0cdeaa98b1e14d02591396ba5e8d69c`.
- PRISMA depois: `f250487a7787c967a0bad0ac38d5dbe210ff63981078d3c65e1d77655ff5f072`.
- `academic_pipeline.__main__`: `31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4`.
- Parser AP-003B: `f6fd1b98c489e1adf5d8ab61419cab6d78db348b93958ff6d93199df0e5cfbb8`.
- Despacho AP-003C: `42299d4962c9eb97df27f9c5a4ca2f1230746353c2a3a4e777d9e70a623682d3`.
- Documento AP-003D: `3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e`.
- Corpo e assinatura do primeiro `main()` preservados, desconsiderando apenas o nome da função.
- Os três `xfail` conhecidos não foram alterados.
