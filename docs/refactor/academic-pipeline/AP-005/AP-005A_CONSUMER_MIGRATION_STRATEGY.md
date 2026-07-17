# AP-005A — Estratégia de migração de consumidores

## Princípio

A AP-005A apenas descreve dependências e propõe a ordem de migração. Nenhuma superfície pode ser removida, alterada ou depreciada nesta etapa.

## Estado observado

- Superfícies totais: **64**
- Migração prévia: **38**
- Com consumidores internos observados: **45**
- Com consumidores dinâmicos: **1**
- Associadas a ciclos: **0**
- Sem consumidor interno observado: **19**

## Ondas propostas

### Onda 0 — Preservação obrigatória

- entrypoints públicos;
- wrappers históricos congelados;
- superfícies ligadas aos três `xfail`;
- decisões arquiteturais protegidas da AP-004B;
- bridges dinâmicas ainda necessárias.

### Onda 1 — Imports internos diretos

Migrar imports semanticamente resolvidos para módulos e símbolos canônicos, preservando os contratos públicos.

### Onda 2 — Cluster de orquestração PRISMA

Tratar em conjunto os wrappers `*_impl_001` e sua relação com `_invoke_with_runtime`, evitando substituições textuais e verificando ciclos.

### Onda 3 — Aliases do gerador TOML

Migrar os consumidores dos aliases `_original` somente após caracterização focada do fluxo interativo.

### Onda 4 — Reexports e fachadas

Formalizar a API pública em `__init__.py`, fachadas e reexports. Ausência de consumidor interno não autoriza remoção de superfície distribuída.

### Onda 5 — Revisão pós-migração

Somente depois da migração integral, reexecutar o inventário e submeter nominalmente qualquer candidato à preservação, depreciação ou remoção.

## Gates obrigatórios

1. Auditorar todas as evidências de baixa confiança.
2. Confirmar ciclos e imports dinâmicos.
3. Aprovar expressamente o inventário AP-005A.
4. Criar testes de caracterização focados.
5. Criar aplicador apenas quando houver transformação estrutural comprovadamente segura.
6. Executar testes focados e a suíte canônica.
7. Revisar o diff antes de commit ou push.

## Bloqueios

```text
alteração produtiva = bloqueada
aplicador produtivo = bloqueado
remoção = bloqueada
commit = bloqueado até aprovação do inventário
push = bloqueado até aprovação do inventário
```
