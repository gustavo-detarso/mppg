# AP-004E — Estratégia para superfícies de compatibilidade

## Princípio de decisão

A AP-004E preserva por padrão superfícies públicas, distribuídas, protegidas ou dinamicamente resolvidas. Remoção só pode ser proposta quando houver evidência estrutural positiva, migração prévia dos consumidores e aprovação expressa. Ausência de referência interna, isoladamente, não autoriza remoção.

## Ordem proposta

1. Preservar entrypoints públicos, símbolos protegidos, arquivos congelados e decisões da AP-004B.
2. Validar bridges, aliases canônicos e reexports com consumidores produtivos.
3. Investigar registries, `getattr`, imports dinâmicos e strings operacionais.
4. Separar referências exclusivas de testes, documentação e artefatos históricos.
5. Submeter os candidatos privados sem consumidores a aprovação nominal.
6. Bloquear qualquer item com colisão ou conflito de destino.
7. Somente após aprovação, criar aplicador estrutural com AST, pré-validação integral, backup externo e rollback.

## Ondas preparatórias

### fora de remoção

Quantidade: **13**

- `AP004E-9bb395bcbaa2` — `academic_pipeline` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__main__.py:1`: preservar sem alteração. Motivo: entrypoint público explicitamente protegido pela AP-003/AP-004
- `AP004E-999329cd7564` — `WorkflowState._normalize` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0`: preservar. Motivo: símbolo protegido; alteração vedada nesta subfase
- `AP004E-6c490c6d9270` — `_ap003d_impl__refs_v6_strip_org` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0`: preservar. Motivo: símbolo protegido; alteração vedada nesta subfase
- `AP004E-897018b51308` — `_ap003f_pipeline_core` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0`: preservar. Motivo: símbolo protegido; alteração vedada nesta subfase
- `AP004E-2fe302e866ad` — `_refs_v6_strip_org` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0`: preservar. Motivo: símbolo protegido; alteração vedada nesta subfase
- `AP004E-12ef4a00117c` — `extract_org_abstracts` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:0`: preservar. Motivo: símbolo protegido; alteração vedada nesta subfase
- `AP004E-ab00608841d3` — `academic_pipeline_rc10.py` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1`: preservar. Motivo: decisão arquitetural consolidada na AP-004B
- `AP004E-efbe86b407de` — `_refs_original_load_config` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1263`: preservar. Motivo: decisão arquitetural consolidada na AP-004B
- `AP004E-e123da0779ec` — `_refs_original_build_bibliography` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1271`: preservar. Motivo: decisão arquitetural consolidada na AP-004B
- `AP004E-299b7d91c4ee` — `_refs_original_render_org_latex` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_rc10.py:1289`: preservar. Motivo: decisão arquitetural consolidada na AP-004B
- `AP004E-8ffd2daf0e48` — `executar_artigo_longo_fulltext_v1_13` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_13.py:1`: preservar congelado. Motivo: arquivo explicitamente fora do escopo da AP-004E
- `AP004E-ee9cdcce1bde` — `executar_artigo_longo_fulltext_v1_14` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/executar_artigo_longo_fulltext_v1_14.py:1`: preservar congelado. Motivo: arquivo explicitamente fora do escopo da AP-004E
- `AP004E-7e7dc8eface9` — `academic-pipeline` em `software/academic_pipeline_rc10_7_conformidade/pyproject.toml:6`: preservar sem alteração. Motivo: entrypoint público explicitamente protegido pela AP-003/AP-004

### migração prévia

Quantidade: **38**

- `AP004E-e72e9bb23f1e` — `main` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/cli.py:10`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-40f450199df1` — `load_config_impl` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:11`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-a1f507c8ca1e` — `load_existing_document_json_impl` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/document_orchestration.py:59`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-36edca88a8f6` — `stage_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:37`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-0d3fc308271c` — `_json_or_none_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:48`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-f8594e08fa3d` — `make_client_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:58`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-e0be0597d7fe` — `_section_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:65`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-ff473436589c` — `research_output_paths_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:87`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-7a1f70069b96` — `render_external_prisma_outputs_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:123`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-1847360516ee` — `_prisma_curadoria_default_config_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:129`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-edc8203917be` — `_prisma_curadoria_default_out_dir_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:135`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-1b290f50b5e3` — `_prisma_curadoria_default_prompt_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:141`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-1ecfd96dfa69` — `_prisma_curadoria_script_path_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:147`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-0a600a9d1a44` — `_prisma_curadoria_arg_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:153`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-fe333ce8096a` — `_prisma_curadoria_config_from_args_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:159`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-34041cff6510` — `_prisma_curadoria_out_from_args_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:165`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-ebc42e1d4c6c` — `_prisma_curadoria_prompt_from_args_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:171`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-90e1169791e3` — `_prisma_curadoria_input_from_args_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:183`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-760b7614a4e3` — `_prisma_curadoria_run_command_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:199`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-b0edfe05c4e4` — `_prisma_curadoria_build_cmd_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:232`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-9094555eaafc` — `_prisma_curadoria_run_ia_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:239`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-160e1b08feaf` — `_prisma_curadoria_reexportar_xlsx_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:246`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-648d0683722c` — `_prisma_curadoria_pipeline_supports_flag_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:258`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-187e66036dd3` — `_prisma_curadoria_importar_no_pipeline_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:279`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-656eec3d1374` — `_prisma_curadoria_fluxo_completo_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:288`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-6151cb3f2843` — `_prisma_curadoria_mostrar_caminhos_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:311`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-c9c406150ca6` — `_prisma_curadoria_menu_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:349`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-f04bcf304fe1` — `_prisma_curadoria_dispatch_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:366`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-389bb46f68e6` — `_prisma_artigo_generico_get_arg_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:377`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-317f17a716c7` — `_prisma_artigo_generico_strip_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:400`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-bba253a35116` — `_prisma_artigo_generico_out_dir_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:415`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-b81340a16e30` — `_prisma_artigo_generico_run_export_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:444`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-3cddb3d56457` — `_prisma_artigo_generico_run_freeze_impl_001` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:479`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-301f38a187b2` — `run_prisma_generic_entrypoint` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/prisma_generic_orchestration.py:713`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-c3f6df07093a` — `_collect_outputs_and_options_original` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4907`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-936e788786e4` — `_render_toml_original` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4958`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-054764be4586` — `_original_ensure_reference_policy` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4976`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos
- `AP004E-5fa6e68ff3fc` — `_wiz_disable_references_original` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py:4993`: preservar ou migrar consumidores antes. Motivo: há consumidores produtivos internos

### preservação

Quantidade: **13**

- `AP004E-764b3462bb48` — `cli_main` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__init__.py:16`: preservar. Motivo: reexport possui consumidores
- `AP004E-2d5ff25925a0` — `main` em `software/academic_pipeline_rc10_7_conformidade/academic_pipeline/legacy.py:76`: preservar. Motivo: load_legacy_module resolve deliberadamente o main do módulo histórico
- `AP004E-c0a8a6350d64` — `app_bundle.scripts.pipeline.article_workflow` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:1`: preservar. Motivo: bridge ainda consumida
- `AP004E-07fda2a9edec` — `STAGES` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2`: preservar. Motivo: reexport possui consumidores
- `AP004E-ceadf33fae1b` — `StageRecord` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2`: preservar. Motivo: reexport possui consumidores
- `AP004E-0718d5435adb` — `WorkflowState` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:2`: preservar. Motivo: reexport possui consumidores
- `AP004E-3d39381c09de` — `ArticleWorkflow` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3`: preservar. Motivo: reexport possui consumidores
- `AP004E-25545d4eed0d` — `StageValidation` em `software/academic_pipeline_rc10_7_conformidade/app_bundle/scripts/pipeline/article_workflow/__init__.py:3`: preservar. Motivo: reexport possui consumidores
- `AP004E-15bc59c372e4` — `app_bundle/scripts/pipeline/academic_pipeline_rc10.py` em `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:15`: preservar. Motivo: a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova
- `AP004E-fd86eccec8b0` — `app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py` em `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:16`: preservar. Motivo: a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova
- `AP004E-2e67d0de59a4` — `configurar_pretriagem_ia_prisma_v16.py` em `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:17`: preservar. Motivo: a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova
- `AP004E-c5801476fc13` — `gerar_log_diagnostico_artigo_v1_18.py` em `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:18`: preservar. Motivo: a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova
- `AP004E-80cb0eef7050` — `academic_pipeline_rc10.py` em `software/academic_pipeline_rc10_7_conformidade/docs/refactor/academic-pipeline/AP-004/AP-004B_MODULE_FILE_APPLICATION.md:29`: preservar. Motivo: a AP-004E não pode reabrir decisão da AP-004B sem evidência estrutural nova

## Critérios para eventual aplicador

O aplicador futuro não está autorizado nesta preparação. Quando autorizado, deverá:

- validar todos os candidatos e ondas antes da primeira escrita;
- usar AST ou transformação estrutural equivalente em Python;
- tratar strings, comentários e metadados somente após classificação semântica;
- criar backup fora do repositório;
- escrever atomicamente e restaurar integralmente em falha;
- recusar estado parcialmente aplicado;
- executar `py_compile`, `git diff --check`, testes específicos e a suíte consolidada;
- preservar exatamente os três `xfail`, sem `xpass`;
- não criar commit, não publicar e não integrar automaticamente.

## Gate

```text
[BLOQUEIO] Inventário ainda não aprovado.
[BLOQUEIO] Aplicador produtivo não criado.
[BLOQUEIO] Alterações produtivas não autorizadas.
```

Fingerprint contratual: `cee4120c2602bb12e78fe7d41cf22fc261b8a64647c2c2b9d6e256903d5574e3`

Itens totais: **64**; decisões manuais: **0**; candidatos preparatórios à remoção: **0**.
