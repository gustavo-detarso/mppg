# Estrutura organizada de saída

Este bundle foi reorganizado para concentrar os artefatos gerados em `output/`:

- `output/pesquisa/`: artefatos da etapa de pesquisa PRISMA
- `output/documento/`: artefatos da etapa da dissertação/documento
- `output/bundle/`: bundle/handoff consolidado

## Ajustes feitos nesta versão

- removido o diretório duplicado `output/pesquisa/.../documento_bundle`
- incluído o script atualizado `scripts/pipeline/gerador_pesquisa_documento_rc_6_corrigido_v4_4.py`
- mantido o TOML da nova organização em `config/pipeline/toml_pesquisa_dissertacao_ia_governo_federal_bundle_rc20_output_no_bundle.toml`

## Observação

Os arquivos de `output/documento/...` ainda podem conter diagnósticos de tentativas anteriores de compilação de PDF (`.log`, `.tex`, `*_pdf_erro.txt`). Isso foi mantido por utilidade de depuração.
