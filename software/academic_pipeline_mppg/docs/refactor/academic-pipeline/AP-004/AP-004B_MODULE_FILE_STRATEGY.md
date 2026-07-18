
# AP-004B — estratégia de módulos e arquivos

> Documento preparatório. Nenhuma movimentação ou renomeação produtiva foi realizada.

## Objetivo técnico

Normalizar os quatro nomes aprováveis sem alterar semântica, entrypoints
públicos, conteúdo gerado ou caminhos operacionais. O inventário v1.6 separa
consumo produtivo comprovado de contratos de compatibilidade e de evidências
históricas/contextuais.

## Taxonomia obrigatória

- `actionable_productive`: import, loader, subprocesso ou caminho usado por
  código produtivo; integra o futuro aplicador.
- `compatibility_contract`: camada legada ou teste que deve provar o wrapper
  histórico; integra o manifesto, mas não autoriza remover a compatibilidade.
- `historical_immutable`: documentação, snapshots e ferramentas de fases
  encerradas; nunca atualizar na AP-004B.
- `physical_directory_reference`: menção a
  `academic_pipeline_rc10_7_conformidade`; pertence exclusivamente à AP-006.
- `protected_operational`: aplicadores, backups, outputs, assets, estados e
  relatórios operacionais; fora do aplicador e do manifesto produtivo.
- `contextual_non_actionable`: texto ou documentação sem consumo operacional
  comprovado; fora do primeiro aplicador.

## Estratégia canônica

- Criar o caminho canônico dos quatro módulos aprováveis e manter o caminho
  histórico como wrapper transitório.
- Preservar argumentos, código de saída, `main()` e superfícies públicas.
- Atualizar imports Python por AST ou edição estrutural dirigida.
- Tratar somente registros `actionable_productive`; usar registros
  `compatibility_contract` para caracterizar equivalência.
- Não alterar referências históricas, contextuais, operacionais ou do diretório
  físico.
- Manter `v1_13` e `v1_14` intactos; a colisão full-text permanece fora do
  primeiro aplicador.
- Preservar `academic-pipeline` e `python -m academic_pipeline`.

## Ordem proposta de aplicação futura

1. Orquestrador canônico + wrapper `academic_pipeline_rc10.py`.
2. Gerador TOML canônico + wrapper versionado.
3. Configurador de pré-triagem + wrapper versionado.
4. Gerador de log diagnóstico + wrapper versionado.
5. Caracterização separada da colisão full-text, sem renomeação automática.

## Barreiras obrigatórias do futuro aplicador

- branch e HEAD exatos;
- árvore limpa ou estado preparatório reconhecido;
- remoto sincronizado;
- hashes de `source_manifest` e `control_manifest`;
- contratos AST dos seis candidatos;
- conjunto exato de arquivos permitidos;
- backup externo, escrita atômica e rollback integral;
- `py_compile`, `git diff --check`, suíte específica e suíte consolidada;
- ausência de alterações nos três `xfail` históricos;
- nenhum commit sem aprovação da consolidação.

## Estado

A criação do aplicador produtivo permanece bloqueada até aprovação expressa do
inventário semântico v1.6.
