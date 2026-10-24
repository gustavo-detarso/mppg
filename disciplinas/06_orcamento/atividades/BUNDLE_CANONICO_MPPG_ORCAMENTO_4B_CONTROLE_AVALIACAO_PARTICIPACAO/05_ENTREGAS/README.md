# Entregas materializadas

Este diretório recebe automaticamente **todas as quatro saídas de entrega** quando `03_REVISAO/gerar_tudo.sh` é executado:

- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf`
- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.docx`
- `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf`
- `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.docx`

O PDF principal continua sendo compilado e validado canonicamente em `03_REVISAO/`, mas, após os gates de geração do conjunto de entregas, uma **cópia byte-idêntica** é materializada em `05_ENTREGAS/`. Assim, a pasta de entregas fica autocontida para envio.

Os DOCX são gerados **a partir dos PDFs finais já validados**, página a página, para preservar o mesmo layout, paginação, figuras, quadros, fontes, quebras e posicionamento visual dos PDFs.

Por esse motivo, cada DOCX é uma **materialização de fidelidade visual**: cada página do PDF é incorporada como uma imagem integral na página correspondente do Word. Isso maximiza a paridade visual, mas o texto interno não é editável/selecionável como texto Word.

`gerar_tudo.sh` só informa sucesso depois de verificar a existência das quatro saídas e confirmar que o PDF principal em `05_ENTREGAS/` é byte-idêntico ao PDF canônico de `03_REVISAO/`.
