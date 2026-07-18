# CHANGELOG rc10.2 — revisão redundante de estabilidade

## Correções adicionais

- Ajuste do modo `--somente-renderizar` para resolver `.bib` relativo ao `document.json` e copiá-lo ao diretório de saída quando necessário.
- A Ficha Técnica da atividade agora é marcada como `:UNNUMBERED: t`, evitando que a Introdução seja deslocada para seção 2.
- Após deduplicação bibliográfica, o pipeline recalcula o mapeamento de documentos para chaves canônicas, evitando que a IA receba/cite chaves removidas do `.bib` final.
- `render_prisma_org.py` agora usa `\printbibliography[heading=none]` quando há heading manual de Referências, evitando título duplicado.
- A validação de `programa == curso` agora se aplica somente a paper/atividade, não a dissertação.
- O DOCX via Pandoc agora recebe `--resource-path` apontando para o diretório de saída, melhorando a resolução de imagens relativas.
- O renderizador de dissertação agora preenche `\palavraschave{}` quando houver palavras-chave no resumo.

## Observação

Os arquivos locais `app_bundle/misc/academic-writing.el` e `app_bundle/misc/fgv.png` continuam fora do pacote e devem ser inseridos localmente.
