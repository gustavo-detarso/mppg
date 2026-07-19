# AP-006D.2 — Migração de contratos e validadores

## Baseline

- Commit: `ee33dff7662f9867b7fb642361cf7bd0c89eb822`
- Tree OID: `d9e2a4d9bb1c881c60c5421f61fe2bbaddb2d180`
- Fingerprint do preview: `c44df8c8bb239b8a69e34c83637b57fcb8e5bd9d4f6ef474e0420463ff962735`
- SHA-256 do diff demonstrativo: `502ae8859c1632e86ae06221f1c0f04aad06f2f16bad30668cf95ffe662fa8d3`

## Decisão materializada

Cinco validadores históricos ainda executáveis passaram a usar a raiz física
`software/academic_pipeline_mppg` em suas referências operacionais atuais.

Foram materializadas:

- 19 migrações diretas;
- 3 adaptações contextuais dual-root;
- preservação de 4 registros históricos;
- precedência da raiz canônica;
- fallback explícito para `academic_pipeline_rc10_7_conformidade` enquanto a ponte existir.

## Compatibilidade

A ponte relativa permanece obrigatória durante AP-006D e AP-006E.
Fallback e ponte serão removidos somente na AP-006F.

## Escopo

- 5 arquivos produtivos modificados;
- 4 artefatos de contrato e validação adicionados;
- nenhum staging, commit ou push nesta materialização.
