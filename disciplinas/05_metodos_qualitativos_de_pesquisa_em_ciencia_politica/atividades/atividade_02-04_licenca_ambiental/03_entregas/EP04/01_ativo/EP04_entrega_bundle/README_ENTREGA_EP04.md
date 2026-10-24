# EP04 — Bundle final de entrega e reprodutibilidade

Este bundle reúne os entregáveis do EP04 e as fontes/evidências de reprodutibilidade após a revisão pós-feedback, o ajuste manual de layout do relatório pelo pesquisador e a reaplicação do QA final.

**Estado:** `QA_FINAL_EP04 = PASS` · **EP04 PRONTO PARA SUBMISSÃO**.

Estado analítico corrente: **5 anchors, 21 temas, 106 códigos substantivos pré-merge, 306 aplicações substantivas, 303 aplicações temáticas pós-merge e 10 casos**.

## Cinco arquivos formais de submissão

1. `relatorio/EP04_relatorio_final_licenciamento_ambiental.docx`
2. `planilhas/EP04_Coding_Frequency_01_PreMerge_QualCoder_Consolidado.xlsx`
3. `planilhas/EP04_Coding_Frequency_02_Temas_PosMerge_QualCoder_Consolidado.xlsx`
4. `apresentacao/EP04_apresentacao_relatorio_final_licenciamento_ambiental_MODELO_FINAL.pptx`
5. `puia/EP04_PUIA_licenciamento_ambiental.docx`

Esses cinco arquivos correspondem aos quatro itens formais do comunicado docente: relatório; duas exportações Coding Frequency; apresentação; PUIA.

## Relatório final

O relatório formal corrente possui **6 páginas de corpo + 1 página de referências**. A largura das colunas da tabela de identificação foi ajustada manualmente pelo pesquisador e **preservada exatamente** durante a intervenção textual final. O documento foi rerenderizado integralmente depois da atualização e passou novamente pelo QA visual.

A pasta `relatorio_simplificado/` permanece apenas como materialização intermediária/histórica do QA anterior e não deve ser usada para submissão.

## Auditoria das planilhas

### Pré-merge — PASS

- 106 códigos substantivos;
- 306 aplicações;
- 10 casos;
- 106/106 `Description` contendo descrição + critério de exclusão;
- `Frequency` e `Cases` reconciliados com as fontes nativas preservadas;
- a aba `Notas` esclarece que Codes-tema históricos com frequência zero na aba nativa de Cases são placeholders e não integram o baseline substantivo.

### Pós-merge — PASS

- 21 Codes-tema `(T)`;
- 303 aplicações temáticas únicas;
- 10 casos;
- 106/106 códigos constituintes preservados exatamente uma vez;
- `Description` reproduz o Memo/QDC pós-feedback, com lista integral de constituintes e 8 contrapontos/exceções;
- `Frequency` e `Cases` reconciliados com `Code_frequencies.xlsx` e `Code_comparisons.xlsx`.

## Derivados e reprodutibilidade

PDFs, fontes Org, scripts, logs, QDC, Journals e exportações nativas são preservados para QA/reprodutibilidade e não integram a submissão formal, salvo solicitação do professor.

A cadeia de compilação histórica permanece no bundle, porém os relatórios não são recompilados por padrão pelo script integrado, para evitar perda do ajuste manual homologado. Uma recompilação forçada exige novo QA.

## Prompt Master

Fonte canônica corrente dentro deste bundle: `PROMPT_MASTER_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.md` — **v2.29**.

Próximo checkpoint: submissão no Eclass até **08/10/2026** e apresentação em **09/10/2026**.
