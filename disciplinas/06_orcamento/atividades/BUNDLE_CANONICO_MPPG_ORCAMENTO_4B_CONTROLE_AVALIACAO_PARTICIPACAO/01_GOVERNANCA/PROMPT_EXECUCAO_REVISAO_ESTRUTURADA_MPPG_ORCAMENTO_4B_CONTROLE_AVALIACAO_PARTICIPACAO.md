# PROMPT DE EXECUÇÃO INTEGRAL — REVISÃO ESTRUTURADA DE LITERATURA
## MPPG/FGV — Orçamento Público — 4B: Controle, avaliação e participação social
### Ênfase recorrente: orçamento participativo e governança orçamentária participativa
**Versão específica 1.3 — 25/09/2026**

## 1. OBJETIVO
Execute autonomamente uma revisão estruturada de literatura da disciplina **Orçamento Público — MPPG/FGV, 2026.2**, para o **subtema 4B — Controle, avaliação e participação social**, produzindo todas as seções e artefatos intermediários, consolidando uma **VERSÃO PRELIMINAR COMPLETA**, auditando-a e então **PARANDO** para aguardar orientações para a versão final.

Este executor é específico do 4B e não deve perguntar qual é o tema, salvo se o usuário solicitar explicitamente mudar de subtema.

### Papel deste arquivo na arquitetura
Este é o **executor específico da revisão estruturada**. Ele deve operar **sob as regras** de `INSTRUCOES_PROJETO_MPPG_ORCAMENTO_GENERICAS.md`, `INSTRUCOES_PROJETO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.md`, `PROMPT_MASTER_MPPG_ORCAMENTO_GENERICO.md` e `PROMPT_MASTER_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.md`. Sua função é orquestrar a sequência operacional e os outputs; ele não redefine as regras gerais do projeto.

**Precedência:** Instruções genéricas → instruções específicas 4B → Master genérico → Master específico 4B → este Executor → fontes/templates.

---

## 2. GATE ZERO — FIXAR O SUBTEMA 4B E O CHECKPOINT
Este executor parte do subtema **4B — Controle, avaliação e participação social**. Não pergunte qual é o tema quando o usuário acionar este bundle.

Antes da execução:
- leia o Plano de Ensino e o Roteiro para confirmar fronteiras e vocabulário do 4B;
- leia o Master específico canônico e o checkpoint do bundle;
- trate **orçamento participativo** como objeto central recorrente, sem reduzir todo o 4B a ele;
- preserve a pergunta de pesquisa e o corpus corrente quando o usuário pedir continuidade;
- reabra pergunta/corpus apenas se o usuário solicitar, se a tarefa exigir nova revisão ou se nova evidência justificar mudança explícita.

Se o usuário pedir outro subtema, suspenda este executor e use o bundle/executor genérico.

---

## 3. MODO DE EXECUÇÃO
Com o 4B confirmado, o padrão é **EXECUÇÃO CONTÍNUA ATÉ A VERSÃO PRELIMINAR**.

Sequência:

1. pré-flight das fontes e requisitos;
2. perfil do tema e protocolo;
3. busca/triagem e matriz do corpus;
4. Seção 1;
5. Seção 2;
6. Seção 3;
7. Seção 4;
8. Seção 5;
9. Seção 6;
10. Seção 7;
11. Seção 8;
11A. glossário autônomo do subtema 4B e inclusão sincronizada como Apêndice A;
12. consolidação da versão preliminar;
13. auditoria acadêmica, factual, normativa, bibliográfica e visual;
14. relatório de pendências/decisões;
15. **STOP**.

Não peça aprovação entre etapas, salvo bloqueio real. Se o usuário disser **MODO SUPERVISIONADO**, entregue um output por vez e aguarde aprovação.

---

## 4. HIERARQUIA DAS FONTES
1. Plano de Ensino.
2. Modelo da Revisão de Literatura.
3. Modelo visual mais recente aprovado para a entrega.
4. Roteiro de Produção.
5. Corpus e fontes específicas do tema.
6. Raio X do Orçamento 2026, se pertinente e apenas como fotografia temporal.
7. Material de processo legislativo/eventos, como contexto.
8. Materiais de terceiros, apenas como apoio.

Nunca harmonize conflito material silenciosamente.

---

## 5. PRÉ-FLIGHT E OUTPUT 00
Antes de redigir, produza internamente e materialize como `00_Escopo_Protocolo_e_Checkpoint`:

- subtema exato: 4B — Controle, avaliação e participação social;
- código oficial: 4B;
- objetivo da revisão;
- pergunta de pesquisa provisória;
- recorte temporal, territorial e institucional;
- fronteiras com temas adjacentes;
- vocabulário nuclear derivado do tema e do Roteiro;
- critérios de inclusão/exclusão;
- bases acadêmicas a pesquisar;
- famílias de normas/documentos oficiais;
- bases de dados candidatas;
- pontos que exigem atualização externa;
- modelo visual vigente, se houver;
- requisitos formais e gates da rubrica.

Se uma pergunta de pesquisa já tiver sido explicitamente aprovada pelo usuário para a execução corrente, preserve-a. Caso contrário, formule-a a partir do tema e refine-a depois da leitura inicial do corpus.

---

## 6. CORPUS E MATRIZ AUDITÁVEL
Use busca dirigida e triagem documentada. O piso operacional é:

- **8 obras acadêmicas**;
- **3 normas/documentos oficiais**;
- **2 fontes de dados**.

Quando a literatura disponível e o tema justificarem, use mais fontes.

Crie `00_Matriz_do_Corpus` com, no mínimo:

- ID;
- referência;
- tipo de fonte;
- ano;
- objeto/pergunta;
- método;
- unidade/caso/população, se aplicável;
- tese/achado central;
- evidência mobilizada;
- limitações;
- páginas relevantes verificadas;
- uso previsto nas Seções 1–7;
- status: incluída / complementar / excluída;
- justificativa de exclusão, quando houver.

Registre também:

- bases consultadas;
- strings/termos de busca;
- data da busca;
- período coberto;
- duplicatas;
- número de registros em cada etapa;
- motivos de exclusão.

Se houver números consistentes para a triagem, gere um **fluxograma de seleção**; se não houver, não invente contagens.

---

## 7. DISCIPLINA DE EVIDÊNCIA
Citação literal somente se o original estiver acessível e a página exata tiver sido conferida.

Nunca invente:

- citação;
- página;
- DOI;
- metadado bibliográfico;
- dispositivo legal;
- decisão/acórdão;
- valor ou série;
- resultado empírico.

Se uma fonte sustentar apenas parcialmente uma afirmação, escreva de forma proporcional à evidência.

Separe sempre:

- fonte do projeto;
- pesquisa externa atualizada;
- inferência analítica.

---

## 8. ETAPA 1 — SEÇÃO 1: DELIMITAÇÃO
Produza `01_Secao_1_Delimitacao` com cerca de 1 página:

- objeto;
- pergunta explícita em frase interrogativa;
- recorte;
- fronteiras;
- exclusões justificadas;
- hipótese de leitura, apenas se útil e não tratada como conclusão antecipada.

**Gate:** pergunta explícita; objeto delimitado; exclusões justificadas; aderência ao tema.

Se houver fontes citadas, terminar com REFERÊNCIAS em ABNT.

---

## 9. ETAPA 2 — SEÇÃO 2: CORPUS
Produza `02_Secao_2_Corpus`, 1–2 páginas, em três blocos:

### 2.1 Como o corpus foi montado
Bases, termos, período, critérios, triagem e deduplicação.

### 2.2 O que entrou
Organize as fontes por função analítica — estrutura do debate, norma, evidência/dados — e destaque as cinco obras centrais provisórias.

### 2.3 O que ficou de fora, e por quê
Inclua descartes reais e justificáveis.

**Gate:** mínimo operacional atendido; protocolo rastreável; descartes justificados; sem inflação artificial de fontes.

Terminar com REFERÊNCIAS em ABNT.

---

## 10. ETAPA 3 — SEÇÃO 3: MAPA DAS POSIÇÕES
Produza `03_Secao_3_Mapa_das_Posicoes`, 4–6 páginas.

Organize por **2 a 4 disputas/tensões substantivas**, nunca por autor. Para cada disputa:

- qual é a questão;
- posição/mecanismo A;
- posição/mecanismo B ou alternativa;
- onde colidem;
- que evidência cada lado mobiliza;
- condições e limites;
- implicações para orçamento e política pública.

Encerre com síntese que responda provisoriamente à pergunta da Seção 1 e prepare as lacunas da Seção 6.

**Teste obrigatório:** se os nomes dos autores fossem apagados, o texto ainda mostraria uma disputa? Se não, reorganize.

**Gate:** debate, não catálogo; evidência rastreável; condições/limites explícitos; síntese final.

Terminar com REFERÊNCIAS em ABNT.

---

## 11. ETAPA 4 — SEÇÃO 4: ESTADO DA NORMATIZAÇÃO
Produza `04_Secao_4_Estado_da_Normatizacao`, 2–3 páginas.

Estrutura recomendada:

### 4.1 Base
Constituição, leis gerais e normas estruturantes pertinentes ao tema.

### 4.2 O que mudou
Alterações recentes efetivamente relacionadas ao tema, com data e consequência institucional.

### 4.3 Jurisprudência e órgãos de controle
Decisões, acórdãos, referenciais ou orientações relevantes. Se não houver decisão estruturante identificável, diga isso explicitamente.

Pesquise fonte oficial vigente. Não copie norma em sequência: explique seu efeito sobre o problema da revisão.

**Gate:** atualidade; dispositivos corretos; pertinência ao tema; distinção entre regra, interpretação e inferência.

Terminar com REFERÊNCIAS em ABNT.

---

## 12. ETAPA 5 — SEÇÃO 5: DADOS DISPONÍVEIS
Produza `05_Secao_5_Dados_Disponiveis`, 1–2 páginas.

Escolha **2 a 4 bases realmente pertinentes ao tema**, usando o Roteiro como semente e fontes oficiais atuais quando necessário.

Para cada base, informar:

1. onde está;
2. quem produz;
3. o que contém;
4. cobertura temporal;
5. desagregação/unidade de análise;
6. o que permite investigar;
7. o que não permite afirmar isoladamente;
8. eventuais problemas de comparabilidade ou mudança metodológica.

Esta seção é mapeamento, não análise causal.

**Gate:** fontes pertinentes; alcance e limites explícitos; nenhum uso indevido de fotografia como série ou atividade institucional como efetividade.

Terminar com REFERÊNCIAS em ABNT.

---

## 13. ETAPA 6 — SEÇÃO 6: LACUNAS E CONTROVÉRSIAS ABERTAS
Produza `06_Secao_6_Lacunas_e_Controversias`, cerca de 1 página.

Derive da Seção 3 ao menos duas lacunas reais, preferencialmente 3–4 quando o corpus suportar.

Cada lacuna deve responder:

- o que já sabemos;
- onde a evidência diverge ou termina;
- qual pergunta continua aberta;
- que evidência/desenho seria necessário para avançar.

Não chame de lacuna algo que apenas faltou pesquisar.

**Gate:** todas as lacunas derivam do corpus; perguntas abertas explícitas; sem generalidades vazias.

Terminar com REFERÊNCIAS em ABNT.

---

## 14. ETAPA 7 — SEÇÃO 7: FICHAMENTO DAS CINCO OBRAS CENTRAIS
Produza `07_Secao_7_Fichamento_Cinco_Obras`, 3–4 páginas.

Selecione as cinco obras que mais estruturam a Seção 3. Para cada uma:

- referência completa;
- tese central;
- argumento;
- método/evidência;
- contribuição para a revisão;
- principal limitação/cautela;
- três citações literais com página exata.

Se não for possível verificar três citações com página em uma obra, não invente: substitua a obra central por outra que cumpra o requisito, se substantivamente defensável, ou registre BLOQUEIO na auditoria.

**Gate:** 5 obras; 15 citações verificadas; páginas exatas; correspondência com a Seção 3.

Terminar com REFERÊNCIAS em ABNT.

---

## 15. ETAPA 8 — SEÇÃO 8: REFERÊNCIAS + DECLARAÇÃO DE IA
Produza `08_Secao_8_Referencias_e_IA`.

- Referências em ABNT NBR 6023.
- Apenas fontes efetivamente citadas.
- Toda fonte citada deve constar.
- Deduplicar variantes da mesma referência.
- Verificar autoria, título, periódico/editora, ano, volume/número/páginas, DOI/URL e data de acesso quando aplicável.
- Inserir declaração de uso de IA compatível com o uso efetivo.

**Gate:** correspondência bidirecional texto ↔ referências; sem referência inventada; declaração de IA presente.

---

## 15A. OUTPUT — GLOSSÁRIO ORÇAMENTÁRIO OBRIGATÓRIO
Produza `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO` como **arquivo autônomo** e use o mesmo conteúdo como **APÊNDICE A** da revisão consolidada.

Verbetes obrigatórios, exatamente estes dez:
1. controle interno;
2. controle externo (TCU);
3. auditoria;
4. controle social;
5. conselho gestor;
6. orçamento participativo;
7. indicadores de desempenho;
8. avaliação de políticas públicas;
9. ciclo de políticas públicas;
10. risco fiscal.

Para cada verbete, apresentar:
- definição em linguagem própria;
- base legal ou conceitual, quando aplicável;
- **Não confundir com**.

Regras:
- o glossário autônomo deve terminar com **REFERÊNCIAS em ABNT** correspondentes às fontes efetivamente utilizadas;
- a redação final deve incorporar os mesmos 10 verbetes como **APÊNDICE A — GLOSSÁRIO ORÇAMENTÁRIO DO SUBTEMA 4B**;
- o Apêndice A não é uma nona seção;
- manter uma **única fonte canônica de conteúdo** para ambas as materializações;
- qualquer alteração deve sobrescrever a fonte canônica e regenerar as duas saídas;
- não inventar base legal para termo que não possua regime jurídico único; nesse caso, declarar explicitamente que a base é conceitual ou setorial.

**Gate:** 10/10 verbetes presentes; três componentes por verbete; conteúdo idêntico entre arquivo autônomo e Apêndice A; referências ABNT no autônomo.

---

## 16. OUTPUT 09 — VERSÃO PRELIMINAR CONSOLIDADA
Produza `09_Versao_Preliminar_Consolidada` usando o modelo visual vigente.

Regras:

- incorporar as versões mais recentes de todas as seções;
- manter 12–18 páginas, salvo conflito documentado;
- remover listas de referências intermediárias;
- unificar e deduplicar tudo na Seção 8;
- preservar capa, títulos, paginação, estilos e identidade visual do template;
- inserir apenas gráficos/figuras que tenham função analítica clara;
- toda figura/quadro/gráfico deve ter título e fonte;
- remover placeholders, instruções de modelo e notas internas;
- incorporar o Apêndice A do glossário a partir da fonte canônica única, sem alterar a numeração das oito seções;
- manter a declaração de IA como elemento final da redação consolidada.

---

## 17. OUTPUT 10 — AUDITORIA DA PRELIMINAR
Produza `10_Auditoria_da_Versao_Preliminar` com status **PASS / ATENÇÃO / BLOQUEIO** para:

- Cobertura do corpus — 15%;
- Organização do debate — 25%;
- Precisão normativa — 15%;
- Qualidade do fichamento — 20%;
- Identificação de lacunas — 10%;
- Correção formal e ABNT — 15%;
- consistência da pergunta entre Seções 1, 3 e 6;
- consistência de números/datas/unidades;
- atualidade normativa;
- citações literais e páginas;
- referências cruzadas;
- figuras/tabelas e respectivas fontes;
- aderência visual;
- ausência de resíduos editoriais;
- completude e sincronização do glossário autônomo com o Apêndice A;
- referências ABNT do glossário autônomo.

Não atribua nota final ou score inventado. A auditoria é diagnóstica.

---

## 18. OUTPUT 11 — PENDÊNCIAS E DECISÕES PARA A VERSÃO FINAL
Produza `11_Pendencias_e_Decisoes_para_Versao_Final` contendo apenas:

- questões ainda abertas;
- pontos que dependem de decisão do usuário/docente;
- fontes que precisam de confirmação adicional;
- alternativas editoriais relevantes;
- correções recomendadas antes da versão final.

Não execute essas mudanças sem autorização quando alterarem substantivamente a preliminar.

---

## 19. OUTPUTS AUXILIARES
Quando efetivamente úteis, materialize também:

- matriz de corpus em XLSX/CSV;
- `.bib` ou equivalente bibliográfico;
- fluxograma da seleção;
- gráficos e figuras;
- tabela de rastreabilidade de citações;
- manifesto de arquivos.

Não produza gráficos decorativos nem dados inventados para “preencher” o documento.

---

## 20. STOP CONDITION — OBRIGATÓRIA
Após entregar:

- outputs 00–11;
- versão preliminar consolidada;
- auditoria;
- pendências;
- glossário autônomo e Apêndice A sincronizado;

**PARE.**

Não gere versão final automaticamente.

Aguarde ordem explícita do usuário, como:

- “prossiga para a versão final”;
- “aplique estas correções e finalize”;
- “incorpore o feedback da docente e gere a final”.

---

## 21. FASE FINAL — SOMENTE APÓS AUTORIZAÇÃO
Quando autorizada:

1. incorporar exatamente as orientações recebidas;
2. revalidar fontes alteradas;
3. atualizar normas/dados apenas se necessário;
4. preservar decisões já aprovadas que não foram revogadas;
5. consolidar referências novamente;
6. executar auditoria final completa;
7. regenerar o glossário autônomo e o Apêndice A a partir da fonte canônica única;
8. materializar a versão final em PDF e gerar o DOCX correspondente a partir do PDF validado, em modo de fidelidade visual;
9. materializar também o glossário autônomo em PDF e DOCX;
10. entregar um breve changelog entre preliminar e final.

---

## 22. COMANDO DE INÍCIO
Ao receber este prompt:

1. execute o **GATE ZERO**;
2. se o tema estiver ausente, pergunte qual é e pare;
3. se o tema estiver presente, inicie o pré-flight e prossiga autonomamente até a preliminar, salvo ordem de MODO SUPERVISIONADO.


## CHANGELOG
### v1.2 — 25/09/2026
- passa a exigir geração automática de PDF e DOCX para a revisão e o glossário;
- o DOCX é derivado do PDF final validado para preservar o mesmo layout;
- exige paridade de número de páginas entre PDF e DOCX e sobrescrita canônica.

### v1.1 — 25/09/2026
- acrescenta o glossário obrigatório do 4B como output autônomo e Apêndice A sincronizado;
- fixa os 10 verbetes do Roteiro de Produção e o formato definição + base legal/conceitual + “Não confundir com”;
- exige referências ABNT no glossário autônomo e fonte canônica única para as duas materializações;
- acrescenta gates de completude e sincronização.

### v1.0 — 19/09/2026
- explicita a função deste arquivo como executor operacional da revisão;
- registra dependência e precedência em relação às Instruções e ao Prompt Master.


## CONTRATO DE MATERIALIZAÇÃO DE ENTREGA — v1.3
Ao executar o pipeline completo, `05_ENTREGAS/` deve terminar com quatro artefatos canônicos: PDF e DOCX da revisão e PDF e DOCX do glossário. O PDF da revisão nessa pasta deve ser cópia byte-idêntica do PDF validado em `03_REVISAO/`. O workflow só pode declarar PASS após verificar as quatro saídas.
