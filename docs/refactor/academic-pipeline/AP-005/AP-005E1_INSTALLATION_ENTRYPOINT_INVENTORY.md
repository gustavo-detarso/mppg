# AP-005E.1 — Inventário de instalação, metadata e entrypoints

## Baseline

- Branch: `ap-refactor/04-consumer-canonicalization`
- Commit: `ba28822c826c37022581bf88c6a1b488e2c618de`
- Upstream: `origin/ap-refactor/04-consumer-canonicalization`
- Fingerprint: `ffa48ad5a11a2e9872a30c798fb094c4af5d710d44ec1985c51cf3b38268fe1d`

## Metadata pública

- Distribuição: `academic-pipeline-mppg`
- Versão: `0.1.0`
- Python: `>=3.11`
- Backend: `setuptools.build_meta`
- Dependências PEP 621 declaradas: nenhuma
- Autoridade operacional de dependências: `Pipfile` e `Pipfile.lock`
- Console script: `academic-pipeline = academic_pipeline.cli:main`

## Pacotes descobertos

- `academic_pipeline`
- `app_bundle`
- `app_bundle.scripts`
- `app_bundle.scripts.pipeline`
- `app_bundle.scripts.pipeline.article_workflow`

## Censo sob as raízes de pacote

- Arquivos rastreados: **274**
- Python: **90**
- Não Python: **184**
- `__init__.py`: **5**
- Python sob pacotes selecionados: **65**
- Testes Python excluídos da descoberta: **23**

O censo das raízes não representa o manifesto do wheel. Ele inclui testes excluídos, documentos, projetos, outputs e outros arquivos rastreados. A cobertura real somente poderá ser decidida após construção e inspeção do artefato na AP-005E.2.

## Cadeia de entrypoints

1. `academic-pipeline` → `academic_pipeline.cli:main`;
2. `python -m academic_pipeline` → `academic_pipeline.cli:main`;
3. `academic_pipeline.main` → `academic_pipeline.cli:main`;
4. `academic_pipeline.cli:main` → `academic_pipeline.legacy:run_legacy`;
5. o bridge legado carrega `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`.

A cadeia é coerente no código-fonte e está coberta por contratos existentes. A AP-005E.1 não autoriza alteração desses entrypoints.

## Riscos inventariados

- `package-data-coverage` — **unresolved_build_artifact_scope**; decisão: `characterize_in_ap005e2`.
- `runtime-dependency-authority` — **distribution_metadata_gap_by_design**; decisão: `preserve_and_characterize_in_ap005e2`.
- `legacy-relative-path-bridge` — **intentional_compatibility_bridge**; decisão: `preserve`.
- `hardcoded-user-prompt-default` — **portability_risk**; decisão: `characterize_in_ap005e2`.
- `module-self-invocation-by-file` — **installed_layout_risk**; decisão: `characterize_in_ap005e2`.
- `helper-sibling-assumption` — **installed_layout_risk**; decisão: `characterize_in_ap005e2`.

## Decisão

A AP-005E.1 é documental e de caracterização. Nenhuma alteração produtiva é necessária ou autorizada.

Permanecem não demonstrados:

- o manifesto exato do wheel e do sdist;
- a instalação em ambiente virtual realmente novo;
- a suficiência dos arquivos de dados distribuídos;
- a ausência de importação acidental do checkout;
- a portabilidade dos caminhos registrados como risco.

Esses pontos constituem os gates obrigatórios da AP-005E.2.
