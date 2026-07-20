# AP-006D.4C — Contrato de invalidação e regeneração dos caches CSV

## Decisão

Os quatro CSVs em `dados_prisma` permanecem preservados como fontes da reprodução. Os quatro arquivos homônimos do cache do projeto são cópias derivadas, reproduzíveis pela função produtiva `copy_documents_to_fulltext_cache` definida em `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/corpus_manager.py` e chamada por `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_rc10.py`.

Nenhum dos oito CSVs foi editado na materialização. A revisão do conteúdo autoritativo das fontes pertence à AP-006D.4D.

## Evidência

- Pares fonte-cache: **4**
- Pares idênticos por bytes, CSV normalizado, schema e linhas: **4**
- Linhas de dados totais: **308**
- Referências legadas preservadas nas fontes e cópias: **308**
- Commit comum de introdução: `874e31160822837109d3d8e785f65e7cfadc0335`
- Dry-run: quatro caches removidos e reproduzidos byte a byte em clone descartável
- Estado final do clone: sem diff
- Testes consumidores: **35 passed, 1 xfailed**
- Fingerprint contratual: `c51003d1eae195fddc72b4a5727e7bc4dd45e4a2b2f7f51cab04b22930fda720`

## Política operacional

Para reprodução focada destes quatro arquivos, a primitiva produtiva deve receber somente os quatro `SourceDoc` correspondentes e `clean=False`. Isso evita apagar os demais artefatos do cache. Edição manual dos caches e reescrita automática de `dados_prisma` são proibidas.

Os hashes, schemas, contagens e contratos estão registrados em `docs/refactor/academic-pipeline/AP-006/ap006d4c_cache_regeneration_contract.json`. A ponte `software/academic_pipeline_rc10_7_conformidade -> academic_pipeline_mppg` permanece preservada.
