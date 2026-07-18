# rc10.7.8 — visual da atividade FGV

Correções:

- Atividade FGV não usa mais o `\maketitle` automático do Org, evitando título/autor/data soltos antes da ficha técnica.
- Ficha Técnica renderizada como bloco LaTeX `tcolorbox` + `tabularray`, conforme `template_atividade_fgv_v5_2_7.org`.
- Cabeçalho da atividade passa a exibir logo da FGV e barra fina em degradê azul escuro → azul claro.
- Quebra de página após a Ficha Técnica, antes da seção 1.
- `#+OPTIONS` agora inclui `title:nil` para os documentos renderizados, evitando duplicidade de capa/título.
