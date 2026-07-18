# Changelog rc10.1 estável

## Correções de estabilidade

1. `--somente-renderizar` não exige mais OpenAI no carregamento inicial do programa.
2. `[latex].pdf_engine` agora é respeitado por `latex_compile.py`.
3. Atividade passou a renderizar Ficha Técnica em vez de capa de paper.
4. Dissertação passou a renderizar macros próprias do `fgv-dissertacao.sty`.
5. Relatório PRISMA copia o `.bib` para sua própria pasta de saída antes da compilação.
6. Lookup por DOI agora respeita `fontes_metadados` com Crossref, OpenAlex, Semantic Scholar e Scopus.
7. Deduplicação bibliográfica passa a escolher a entrada de melhor qualidade.
8. Se houver `.bib` externo, o pipeline tenta mapear PDF/DOCX para `bib_key` por DOI, título e nome de arquivo.
9. DOCX mantém fallback estável por `python-docx` e acrescenta opção Pandoc/CSL.
10. Exemplos TOML foram corrigidos para caminhos relativos a `app_bundle/config/examples`.
11. Pacote distribuível limpo: sem `_test` e sem `__pycache__`.

## Arquivos locais esperados

O usuário deve inserir manualmente:

- `app_bundle/misc/academic-writing.el`
- `app_bundle/misc/fgv.png`

