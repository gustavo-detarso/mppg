# AP-006B — Contrato de compatibilidade

## Superfícies preservadas

A mudança física não poderá alterar:

- distribuição: `academic-pipeline-mppg`;
- versão distributiva inicial: `0.1.0`;
- console: `academic-pipeline`;
- destino do console: `academic_pipeline.cli:main`;
- import público `academic_pipeline`;
- import público `app_bundle`.

## Resolução de recursos

Recursos empacotados deverão usar `importlib.resources`. Recursos dependentes da
árvore do repositório deverão usar uma única função de resolução da raiz. Um
override por variável de ambiente somente será aceito quando explícito,
documentado e coberto por testes.

Fica proibido inferir a raiz a partir do nome
`academic_pipeline_rc10_7_conformidade`.

## Ponte transitória

A ponte será um symlink relativo rastreado do caminho antigo para
`software/academic_pipeline_mppg`, combinado com o resolvedor canônico.

A ponte deverá:

1. preservar consumidores que ainda usam o caminho antigo;
2. não duplicar a árvore produtiva;
3. funcionar em clone limpo;
4. sobreviver à criação de arquivo distribuível;
5. permanecer testada até a decisão de retirada na AP-006F.

## Migração por ondas

A AP-006C cuidará do runtime interno, empacotamento e entrypoints. A AP-006D
refinará e migrará contratos, validadores, consumidores externos, fontes
reutilizáveis e casos ambíguos.

Os números da auditoria são limites superiores, não quantidades automáticas de
edições. Inventários históricos e matrizes geradas podem inflar as contagens.

## Critérios de aceitação

A arquitetura só será considerada estável quando:

- pacote e console funcionarem em instalação limpa;
- suíte canônica permanecer aprovada;
- caminho antigo e novo produzirem comportamento equivalente;
- consumidores ativos estiverem classificados e migrados;
- evidência histórica não tiver sido reescrita em massa;
- não houver árvore duplicada;
- a retirada da ponte tiver decisão explícita.
