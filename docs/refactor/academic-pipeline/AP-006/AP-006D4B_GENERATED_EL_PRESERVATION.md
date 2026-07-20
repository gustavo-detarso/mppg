# AP-006D.4B — Preservação dos artefatos `.el`

## Decisão

Os seis artefatos atuais e a evidência histórica são preservados sem reescrita e sem regeneração.

O script `software/academic_pipeline_mppg/app_bundle/scripts/pipeline/gerar_artigo_longo_fulltext_secional.py` foi caracterizado como executor de exportação PDF, não como escritor direto dos arquivos `_export_pdf.el`. A busca rastreada encontrou zero escritores diretos comprovados. Assim, reescrita manual ou execução de um gerador não comprovado apagaria a proveniência em vez de reproduzi-la.

## Contrato

- Artefatos atuais preservados: **6**
- Evidência histórica preservada: **1**
- Referências legadas atuais mantidas em conteúdo gerado: **9**
- Referências legadas históricas mantidas como evidência: **2**
- Fingerprint: `4809c8ff24a5ab6ecee01f1d6deccbb2016e284c0040cf80a98ca63d36624240`

Os hashes registrados em `docs/refactor/academic-pipeline/AP-006/ap006d4b_generated_el_preservation.json` são contratuais. Os seis artefatos atuais só poderão ser regenerados após identificação de um escritor rastreado e reprodução em clone descartável. A evidência histórica não poderá ser regenerada. A ponte `software/academic_pipeline_rc10_7_conformidade -> academic_pipeline_mppg` permanece preservada.
