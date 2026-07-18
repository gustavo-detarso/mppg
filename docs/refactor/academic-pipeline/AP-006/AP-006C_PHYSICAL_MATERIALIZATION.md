# AP-006C — Materialização física controlada

## Estado

A árvore produtiva foi materializada em
`software/academic_pipeline_mppg`. O caminho anterior
`software/academic_pipeline_rc10_7_conformidade` tornou-se uma ponte de
compatibilidade por symlink relativo.

Esta materialização ainda não está consolidada em commit.

## Evidência de origem

- Baseline: `1b4c71c204a0314aa5bab6db5b49cc1ada86b234`
- Tree OID: `a95de261751e6274a8fa7fbf8010dc4d039507b3`
- Entradas rastreadas movidas: 520
- Resíduos ignorados movidos: 112
- Fingerprint: `7d570f3786f18a2a07b150763cc5e6e1fdf4cc646b394951f85fef9ac3bffe45`

## Topologia

```text
software/academic_pipeline_rc10_7_conformidade
    -> academic_pipeline_mppg

software/academic_pipeline_mppg/
    árvore produtiva canônica
```

Existe uma única árvore produtiva. Duplicação física é proibida.

## Resolvedor canônico

O módulo `academic_pipeline.repository_paths` resolve a raiz por marcadores
estruturais (`pyproject.toml`, `academic_pipeline` e `app_bundle`), sem depender
do nome físico do diretório. A variável `ACADEMIC_PIPELINE_PROJECT_ROOT`
fornece override explícito e validado.

Recursos empacotados continuam sujeitos à política `importlib.resources`.

## Limites

A AP-006C não executa a migração ampla dos 66 consumidores externos
potencialmente acionáveis. Essa migração pertence à AP-006D. A retirada da
ponte somente poderá ser decidida na AP-006F.
