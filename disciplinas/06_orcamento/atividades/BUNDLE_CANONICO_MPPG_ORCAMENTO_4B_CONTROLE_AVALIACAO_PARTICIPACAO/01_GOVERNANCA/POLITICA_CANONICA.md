# Política canônica da revisão

- O ORG é a fonte textual principal.
- O BIB é a fonte bibliográfica principal.
- Assets têm nomes estáveis dentro de `03_REVISAO/assets/`.
- `03_REVISAO/compilar.sh` sobrescreve o PDF canônico apenas depois de compilação e validação integral bem-sucedidas.
- Em execução bem-sucedida, os intermediários (`.tex`, `.aux`, `.bbl`, `.bcf`, `.blg`, `.log`, `.out`, `.run.xml`, arquivos de validação etc.) são removidos automaticamente com a pasta `.build/` e o log temporário `compilacao.log`.
- Em caso de falha, `.build/` e `compilacao.log` são preservados para diagnóstico.
- Para depuração voluntária, `MPPG_KEEP_BUILD=1 ./compilar.sh` preserva os intermediários mesmo após sucesso.
- `05_ENTREGAS/` deve conter o conjunto autocontido de entrega: PDF e DOCX da revisão, além de PDF e DOCX do glossário. O PDF da revisão nessa pasta deve ser cópia byte-idêntica do PDF canônico validado em `03_REVISAO/`.
- O DOCX canônico fica em `05_ENTREGAS/` e é regenerado automaticamente a partir do PDF final validado, em modo de fidelidade visual página a página; a sobrescrita só ocorre após a criação integral do arquivo.
- `03_REVISAO/gerar_tudo.sh` gera revisão e glossário em PDF e, em seguida, os respectivos DOCX; só declara sucesso após verificar as quatro saídas em `05_ENTREGAS/` e a identidade byte a byte do PDF principal.
- O DOCX de fidelidade visual preserva o layout do PDF, mas seu conteúdo de página é imagem e não texto Word editável.
- `MPPG_DOCX_DPI` controla a resolução da rasterização, com padrão de 200 dpi.
- Não criar `_vN` nos nomes canônicos internos deste bundle.
- Em mudanças futuras, alterar o arquivo canônico e atualizar `MANIFEST_SHA256.txt` antes de reempacotar o ZIP.


## Glossário canônico
O conteúdo dos 10 verbetes do 4B reside exclusivamente em:
`03_REVISAO/glossario/GLOSSARIO_CONTEUDO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org`.

Esse arquivo alimenta duas materializações:
- glossário autônomo;
- Apêndice A da revisão final.

Nunca editar uma das materializações separadamente. Alterar a fonte canônica e regenerar ambas.
