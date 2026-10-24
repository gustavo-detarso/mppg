# INSTRUÇÕES ESPECÍFICAS DO PROJETO — MPPG / ORÇAMENTO PÚBLICO
## 4B — Controle, avaliação e participação social
### Ênfase recorrente: orçamento participativo e governança orçamentária participativa
**Versão 1.3 — 25/09/2026**

## 1. FUNÇÃO DESTE ARQUIVO
Este arquivo é uma camada específica de instruções para o subtema **4B — Controle, avaliação e participação social** da disciplina Orçamento Público — MPPG/FGV, 2026.2.

Ele deve ser lido em conjunto com:
1. `INSTRUCOES_PROJETO_MPPG_ORCAMENTO_GENERICAS.md`;
2. `PROMPT_MASTER_MPPG_ORCAMENTO_GENERICO.md`;
3. `PROMPT_MASTER_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.md`;
4. `PROMPT_EXECUCAO_REVISAO_ESTRUTURADA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.md`.

Precedência:
**Instruções genéricas → instruções específicas 4B → Master genérico → Master específico 4B → Executor específico → fontes/templates.**

Em caso de conflito material, preserve a regra mais exigente quando seguro e registre a divergência.

## 2. ESCOPO
O subtema 4B reúne:
- controle interno;
- controle externo/TCU;
- auditoria;
- controle social;
- conselho gestor;
- orçamento participativo;
- indicadores de desempenho;
- avaliação de políticas públicas;
- ciclo de políticas públicas;
- risco fiscal.

**Orçamento participativo** é objeto empírico e institucional central recorrente neste projeto, mas não deve ser tratado como sinônimo de todo o subtema 4B. A análise deve conectar participação, controle, avaliação e governança orçamentária.

## 3. PERGUNTAS ORIENTADORAS
Perguntas recorrentes:
- **Transparência produz controle?**
- **Que desenho institucional maximiza participação sem perder responsabilidade fiscal, eficiência e accountability?**

Pergunta de pesquisa corrente da revisão:
> Como os mecanismos de participação social e de controle público se conectam à governança orçamentária e à avaliação da ação estatal no Brasil, e quais fatores condicionam sua efetividade?

## 4. CADEIA ANALÍTICA
Quando pertinente, organizar mecanismos e evidências pela cadeia:
**escuta/abertura → deliberação → prioridade → dotação → execução → monitoramento/avaliação → prestação de contas/accountability**.

A existência formal de canal participativo não deve ser confundida com incidência substantiva. Procure mecanismos verificáveis de consequência ao longo da cadeia.

## 5. CHECKPOINT DO CORPUS
Enquanto não houver reabertura explícita do corpus:
- corpus acadêmico corrente: **12 artigos**;
- cinco obras centrais: **Silva & Carvalho (2006); Gurgel & Justen (2013); Novaes & Santos (2014); Peres (2020); Vieira (2023)**;
- Avritzer (2008) permanece como referência complementar, sem alterar silenciosamente o n=12.

O arquivo `.bib` deste bundle contém o inventário bibliográfico corrente. A ausência do PDF original de uma obra no bundle não autoriza citação literal sem nova conferência do original.

## 6. NORMAS E DADOS PRIORITÁRIOS
Quando pertinentes ao argumento, considerar e verificar em fonte oficial vigente:
- Constituição Federal, especialmente arts. 70–75 e art. 74;
- Lei nº 4.320/1964;
- Estatuto da Cidade, especialmente arts. 43–44;
- LRF e LC nº 131/2009;
- EC nº 109/2021;
- Decreto nº 11.558/2023 (CMAP);
- alterações normativas posteriores pertinentes;
- referenciais e dados do TCU.

Para dados, priorizar quando cabível:
- MUNIC/IBGE;
- dados abertos, fiscalizações e jurisprudência do TCU;
- CMAP/MPO;
- Raio X do Orçamento 2026 apenas como fotografia contextual da LOA 2026 pós-vetos.

## 7. REVISÃO ESTRUTURADA
A revisão segue 8 seções e 12–18 páginas. A Seção 3 deve ser organizada por disputas, nunca por resumos sequenciais de autores. A Seção 5 deve explicitar alcance e limites de cada base. A Seção 7 exige três citações literais verificadas e paginadas por obra central.

Entregas parciais autônomas com fontes terminam com **REFERÊNCIAS em ABNT**. Na consolidação, unificar e deduplicar na Seção 8.


## 8. GLOSSÁRIO OBRIGATÓRIO
O subtema 4B possui **10 verbetes obrigatórios** definidos pelo Roteiro de Produção: controle interno; controle externo (TCU); auditoria; controle social; conselho gestor; orçamento participativo; indicadores de desempenho; avaliação de políticas públicas; ciclo de políticas públicas; risco fiscal.

O glossário deve existir em **duas materializações sincronizadas**:
1. **arquivo autônomo**, com os 10 verbetes e REFERÊNCIAS em ABNT das fontes efetivamente utilizadas;
2. **redação final da revisão**, como **APÊNDICE A — GLOSSÁRIO ORÇAMENTÁRIO DO SUBTEMA 4B**, sem criar uma nona seção da revisão.

Cada verbete deve conter:
- definição em linguagem própria;
- base legal ou conceitual, quando aplicável;
- campo **“Não confundir com”**.

A redação dos verbetes deve ser canônica: manter uma única fonte de conteúdo e reutilizá-la na versão autônoma e no Apêndice A. Qualquer alteração de verbete deve atualizar as duas saídas sem divergência.

Na contagem estrutural, a revisão continua com **oito seções**; o glossário incorporado é apêndice. A declaração de uso de IA permanece ao final do documento.

## 9. PIPELINE DOCUMENTAL HOMOLOGADO
Para a linha ORG/PDF deste trabalho, o fluxo homologado é:
**ORG → TEX → `fgv-paper` → `biblatex-abnt`/Biber → LuaLaTeX → PDF**.

Regras persistentes:
- citações autor-data;
- não aceitar regressão para citações numéricas;
- auditar log LaTeX;
- quadros largos preferencialmente em `tabularx`;
- figuras/gráficos com título e fonte próximos;
- recortar whitespace residual do arquivo de imagem antes da inserção;
- quando a figura estiver em `minipage`, controlar conscientemente o tamanho pela própria `minipage` e/ou `\linewidth`;
- renderizar e inspecionar visualmente o PDF antes de homologar;
- quando a saída DOCX precisar preservar exatamente o layout do PDF, gerar o DOCX **a partir do PDF final validado**, página a página, em modo de fidelidade visual;
- nesse modo, o DOCX deve preservar paginação, quebras, figuras, quadros e posicionamento visual do PDF, registrando explicitamente que o texto interno não é editável como texto Word;
- `gerar_tudo.sh` deve gerar revisão e glossário em PDF e DOCX, sobrescrevendo os nomes canônicos;
- `05_ENTREGAS/` deve receber **as quatro saídas finais** (PDF/DOCX da revisão e PDF/DOCX do glossário);
- o PDF principal em `05_ENTREGAS/` deve ser cópia byte-idêntica do PDF canônico validado em `03_REVISAO/`.

## 10. CHECKPOINT PRELIMINAR E STOP
Fluxo padrão:
**pré-flight → corpus/ledger → Seções 1–8 → preliminar consolidada → auditoria → pendências → STOP**.

Após a versão preliminar, aguardar orientação explícita do usuário antes de produzir a versão final.

## 11. DISCIPLINA DE EVIDÊNCIA
Nunca inventar citação, página, DOI, dispositivo, dado, acórdão ou resultado. Citação literal só após conferência do original e da página exata. Diferenciar conteúdo da fonte, inferência analítica e pesquisa externa atualizada.
