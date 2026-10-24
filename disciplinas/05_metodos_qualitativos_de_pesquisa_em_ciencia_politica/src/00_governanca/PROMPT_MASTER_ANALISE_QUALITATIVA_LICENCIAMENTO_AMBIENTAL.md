# PROMPT MASTER — ANÁLISE QUALITATIVA — LICENCIAMENTO AMBIENTAL

**Versão canônica:** 2.29  
**Atualização:** 26/09/2026 — QA final reaplicado após ajuste manual de layout do relatório pelo pesquisador; intervenção textual final incorporada preservando exatamente a geometria da tabela ajustada; Coding Frequency pré-merge recebeu nota preventiva sobre placeholders nativos de frequência zero; pacote final reconciliado sem alteração de decisões analíticas, temas, Counts ou Cases.
**Disciplina:** Métodos Qualitativos de Pesquisa em Ciência Política  
**Tema:** 5.1 — Licenciamento Ambiental  
**Método:** Codebook Thematic Analysis, operacionalizada conforme Philip Adu (2019), nos termos ensinados na disciplina  
**Workflows operacionais:** A — QualCoder 3.8.2 em Linux; B — AI-native / sem CAQDAS, com aprovação humana obrigatória  
**Equivalência operacional de referência do Workflow A:** QDA Miner Lite

---

# 1. FINALIDADE E FORÇA NORMATIVA

Este documento é a **fonte canônica de continuidade** da análise qualitativa do Tema 5.1 — Licenciamento Ambiental.

Toda nova conversa, revisão, auditoria, categorização, tematização, preparação de planilhas, slides, relatórios ou PUIA deste projeto deve começar pela recuperação deste documento.

Este Prompt Master preserva:

- a escola metodológica adotada;
- a hierarquia das fontes;
- as perguntas de pesquisa;
- os anchor codes;
- as convenções operacionais no QualCoder;
- o estado da codificação;
- o estado quantitativo auditado;
- a distinção entre codificação, categorização analítica e tematização;
- as categorias e temas atualmente propostos;
- o procedimento de validação empírica introduzido/explicitado pela Aula IV;
- a rotina operacional da Etapa 2 e do EP03 definida pela Aula VI;
- o feedback oficial do EP02 e suas implicações específicas para a transição ao EP03;
- as regras de participação da IA;
- a governança dos quatro exercícios programados;
- os entregáveis, prazos e documentos obrigatórios definidos no Plano de Ensino;
- as regras formais e a grelha de avaliação do Modelo oficial de Tabela de Temas do EP03;
- a biblioteca bibliográfica integral da disciplina e sua regra de uso/precedência;
- o Guia do Relatório Final do EP04, sua estrutura, limites, grelha de avaliação e checklist de entrega;
- o comunicado docente posterior sobre a entrega do EP04, incluindo prazo, quatro itens de entrega, dupla exportação `Coding Frequency` e reforço da transparência na seção 2.4;
- o Modelo Simplificado do Relatório Final do EP04, suas instruções, extensão, estrutura e grelha de avaliação;
- o Modelo de Apresentação do Relatório Final/Aula IX, seu roteiro de seis slides, tempos e requisitos de conteúdo;
- a governança da materialização final do EP04 — relatório simplificado, PUIA, apresentação, planilhas e compilação integrada;
- a separação entre o bundle canônico geral de continuidade e o bundle canônico específico de entrega do EP04;
- a governança do bundle canônico geral, incluindo nome fixo, raiz interna e regra de snapshots históricos;
- o checkpoint exato para continuidade.

Em caso de conflito entre conhecimento metodológico genérico e as aulas do professor, **prevalecem as aulas do professor**.

Não substituir silenciosamente o procedimento da disciplina por Coding Reliability TA, Reflexive Thematic Analysis, Grounded Theory ou outro método.

---

## 1.1 GOVERNANÇA DO BUNDLE CANÔNICO

O **bundle geral de continuidade** deste projeto possui **nome canônico fixo e permanente**:

`BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip`

Regras obrigatórias:

1. toda atualização do bundle ativo deve ser publicada **exatamente com esse nome**;
2. não acrescentar ao bundle ativo sufixos como versão, data, `(1)`, `ATUALIZADO`, `FINAL` ou equivalentes;
3. versão e data ficam registradas **dentro do bundle**, principalmente no Prompt Master, README, manifesto e checksums;
4. snapshots históricos podem ser preservados separadamente com nome versionado, por exemplo:
   `BUNDLE_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL_vX.Y_AAAAMMDD.zip`;
5. snapshot histórico **não substitui** o bundle canônico ativo;
6. a pasta-raiz interna do ZIP deve usar o mesmo basename do nome canônico, sem `.zip`:
   `BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL/`;
7. o arquivo `NOME_CANONICO_DO_BUNDLE.txt` deve permanecer na raiz interna e registrar o nome canônico vigente;
8. `README_BUNDLE.md`, `INSTRUCOES_PROJETO_MPPG_ANALISES_QUALITATIVAS.txt`, `MANIFESTO_ARQUIVOS.csv` e `SHA256SUMS.txt` devem ser reconciliados a cada atualização;
9. nenhum documento interno pode continuar declarando versão/checkpoint anterior após a promoção do Prompt Master;
10. antes de liberar o ZIP, exigir:
   - `NOME_BUNDLE_CANONICO = PASS`;
   - `RAIZ_INTERNA_CANONICA = PASS`;
   - `METADADOS_BUNDLE_RECONCILIADOS = PASS`.

Essa convenção foi homologada pelo pesquisador e deve ser preservada em todas as atualizações futuras.

## 1.2 GOVERNANÇA DO BUNDLE CANÔNICO DE ENTREGA DO EP04

O pacote de **entrega/reprodutibilidade do EP04** é um artefato distinto do bundle geral de continuidade.

Nome canônico fixo:

`EP04_ENTREGA_FINAL_BUNDLE.zip`

Regras:

1. o bundle do EP04 **não substitui** `BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip`;
2. o bundle geral preserva governança, fontes, método e continuidade do projeto;
3. o bundle do EP04 preserva a materialização e a reprodutibilidade da entrega específica;
4. o arquivo ativo do EP04 não deve receber sufixos como versão, data, `(1)`, `FINAL`, `ATUALIZADO` ou equivalentes;
5. snapshots históricos do EP04 podem ser mantidos separadamente com nomes versionados, sem se tornarem a cópia ativa;
6. a fonte oficial visual da apresentação deve ser referida pelo nome canônico:
   `Modelo de Apresentação_Relatório Final_Tema5.5.pptx`;
7. cópias de transporte com sufixo `(1)` não redefinem o nome canônico;
8. a atualização de um bundle não autoriza substituir ou apagar o outro.

Gates:

- `BUNDLE_GERAL_CANONICO = PASS`;
- `BUNDLE_EP04_CANONICO = PASS`;
- `PAPEIS_DOS_BUNDLES_SEPARADOS = PASS`;
- `NOMES_CANONICOS_SEM_SUFIXOS = PASS`.

# 2. HIERARQUIA DAS AUTORIDADES

A hierarquia deve ser aplicada **por domínio**, evitando misturar autoridade metodológica, operacional e documental.

## 2.1 Autoridade metodológica e operacional

1. **`MQPCP_Aula III.pdf`**  
   Autoridade metodológica principal para Codebook Thematic Analysis conforme Philip Adu (2019), incluindo:
   - posição da disciplina na vertente Codebook TA;
   - abordagem mista;
   - anchor codes;
   - indicadores empíricos;
   - códigos;
   - distinção entre código, categoria analítica e tema;
   - critérios de qualidade;
   - information power;
   - transparência no uso de IA.

2. **`MQPCP_Aula VI.pdf`**  
   Autoridade operacional principal para a **Etapa 2 / Exercício Programado 03**, especialmente:
   - rotina da compilação de códigos aos temas consolidados;
   - uma tabela por anchor code;
   - agrupamento a partir dos códigos dominantes;
   - dominante como semente de comparação, e não como tema mais importante;
   - rótulo e descrição dos agrupamentos;
   - teste formal do tema;
   - regra de que, **no agrupamento oficial desta disciplina, cada código vai para um tema só**;
   - duplicação do projeto antes da consolidação temática;
   - criação de cada tema como `Code` marcado com `(T)` sob o anchor;
   - registro da definição e lista de códigos no `Description`;
   - mesclagem de cada código no tema correspondente;
   - nova exportação do quadro final em `Coding Frequency`.

3. **`MQPCP_Aula IV.pdf`**  
   Autoridade para operacionalização avançada da codificação e preparação da Etapa 2, especialmente:
   - construção do codebook;
   - procedimento emacD;
   - granularidade;
   - revisão por Coding Retrieval;
   - relação entre frequências e interpretação;
   - retorno aos segmentos codificados;
   - equivalência operacional adotada neste projeto: **QDA Miner Lite `Retrieve → Coding Retrieval` ≈ QualCoder 3.8.2 `Reports → Coding reports`**.

   **Regra de precedência na Etapa 2:** quando a Aula IV admitir operação que conflite com uma instrução posterior e específica da Aula VI, prevalece a Aula VI para o EP03. Em particular, a possibilidade de duplicar um código para mais de um tema não será usada na consolidação oficial desta etapa, porque a Aula VI determina que cada código seja agrupado em um único tema.

4. **`MQPCP_Aula II.pdf`**  
   Autoridade complementar para pesquisa qualitativa, codificação sistemática, subjetiva, transparente e iterativa, reorganização posterior e reflexividade.

5. **`MQPCP_Aula I.pdf`**  
   Moldura metodológica geral.

### Nota sobre `MQPCP_Aula V.pdf`

A Aula V foi consultada. O arquivo disponibilizado contém apenas material visual de apoio à apresentação/discussão dos resultados da codificação e **não acrescenta regra metodológica ou operacional substantiva** ao procedimento vigente. Sua existência deve ser registrada, mas não altera o fluxo.

6. **Biblioteca bibliográfica integral da disciplina**  
   Fonte bibliográfica de apoio e conferência, abaixo das instruções específicas do professor na hierarquia operacional do curso. O acervo corrente foi incorporado a partir de `Métodos Qualitativos de Pesquisa em Ciência Política - 9192026 - 321 PM.zip` e contém **22 PDFs + 1 índice HTML**.

   **Autoridade bibliográfica primária para o método adotado:**
   - Philip Adu — *A Step-by-Step Guide to Qualitative Data Coding* (2019).

   Quando for necessário confirmar conceito, formulação, página, passo, exemplo ou distinção atribuída a Adu, consultar preferencialmente o **livro original disponível no acervo**, sem substituir a operacionalização específica ensinada pelo professor.

   **Bibliografia complementar de consulta seletiva:**
   - Braun & Clarke — *Using thematic analysis in psychology*;
   - Braun & Clarke — *Reflecting on reflexive thematic analysis*;
   - Braun & Clarke — *Successful Qualitative Research*;
   - Braun & Clarke — *One size fits all? What counts as quality practice in reflexive thematic analysis?*;
   - Braun et al. — *The online survey as a qualitative research tool*;
   - Rosa et al. — análise temática em educação em ciências;
   - Souza — pesquisa com análise qualitativa de dados e análise temática;
   - Anderson & Thoma — *The edge of reason*;
   - Boyatzis — *Transforming Qualitative Information*;
   - Kampira & Meyer — *A Brief Introduction to Thematic Analysis*;
   - Linneberg & Korsgaard — *Coding qualitative data: a synthesis guiding the novice*;
   - Lochmiller — *Conducting Thematic Analysis with Qualitative Data*;
   - Lorenzetti, Domiciano & Geraldo — uso do QDA Miner Lite;
   - Saldaña — *The Coding Manual for Qualitative Researchers*, 3rd ed.;
   - Stake — *Qualitative Research: Studying How Things Work*;
   - Malterud, Siersma & Guassora — *Sample size in qualitative interview studies: guided by information power*;
   - Brooks & King — *Doing Template Analysis*;
   - Brooks et al. — *The Utility of Template Analysis in Qualitative Psychology Research*;
   - Greenhalgh — *Reflexive qualitative research and generative AI: a call to go beyond the binary*;
   - capítulos/obras pertinentes nos handbooks de Liamputtong e de Willig & Stainton-Rogers.

   ### Regra de uso da biblioteca

   - a biblioteca **não altera automaticamente** o método corrente do projeto;
   - as aulas e instruções posteriores do professor continuam governando a execução da disciplina;
   - a bibliografia serve para conferência primária, fundamentação, precisão conceitual e redação acadêmica;
   - a presença de obras de Reflexive Thematic Analysis, Template Analysis, Grounded Theory ou outras tradições **não autoriza deriva metodológica**;
   - quando uma afirmação for atribuída a um autor específico, preferir a obra original disponível em vez de resumo secundário ou conhecimento geral do modelo;
   - quando houver diferença entre a obra original e uma adaptação expressamente ensinada pelo professor para o exercício, registrar a distinção e **seguir a adaptação do professor na operação do curso**;
   - a biblioteca deve ser consultada **seletivamente conforme a tarefa**; não é obrigatório reler os 22 PDFs em cada nova operação.

## 2.2 Autoridade acadêmica, documental e de entrega

1. **`Plano de Aula JCA_Métodos Qualitativos_2º sem 2026.docx`**  
   Autoridade formal para:
   - cronograma;
   - pesos dos exercícios;
   - formatos de entrega;
   - prazos;
   - obrigatoriedade do PUIA;
   - sequência das atividades;
   - indicação de modelos/Guias a serem usados.

2. **Feedback oficial individual do professor sobre os exercícios**, quando houver.  
   O feedback posterior e específico do professor possui autoridade operacional sobre o caso a que se refere, sem revogar silenciosamente a metodologia geral das aulas.  
   Para o EP02 deste projeto, o feedback:
   - confirmou a qualidade e o rigor do codebook;
   - confirmou o baseline entregue de 105 códigos e 299 aplicações;
   - validou o uso do QualCoder e a documentação de transparência;
   - determinou, para a Etapa 2, a revisão/subdivisão do anchor `Efetividade do licenciamento` antes do agrupamento;
   - estabeleceu a expectativa de **2 a 5 temas por anchor code**;
   - determinou a triagem explícita dos códigos de ocorrência única antes da tematização.

3. **Enunciado e anexos específicos de cada exercício no Eclass**, quando disponíveis.  

   Para o EP03, o arquivo **`Modelo de Tabela de Temas(1).docx`** está incorporado como **fonte documental oficial e obrigatória**. Ele prevalece sobre qualquer estrutura preliminar inventada neste Prompt Master para:
   - organização das tabelas;
   - quantidade e disposição dos temas;
   - conteúdo de cada célula temática;
   - regras mínimas de composição dos temas;
   - critérios e pesos de avaliação.

   Para o EP04, o arquivo **`Guia do Relatório Final.docx`** está incorporado como **fonte documental oficial e obrigatória**. Ele permanece como autoridade geral para o conteúdo metodológico do relatório e para os critérios pedagógicos do EP04, especialmente percurso analítico, nota metodológica sobre IA, uso de excertos, cautela com contagens, ABNT e checklist de entrega.

   Em 24/09/2026, foram incorporados dois modelos oficiais adicionais:

   **a) `MODELO SIMPLIFICADO DO RELATÓRIO FINAL.docx`**  
   Modelo formal de materialização do relatório final. Ele prevalece sobre o Guia quando houver conflito de **estrutura, extensão, campos, quadros, sequência de preenchimento ou grelha de avaliação do relatório efetivamente entregue**, porque é modelo posterior, simplificado e diretamente preenchível. Regras principais:
   - seguir a sequência: tema/perguntas → percurso analítico → temas → respostas às perguntas;
   - substituir todos os textos entre colchetes pelo conteúdo do Tema 5.1;
   - apagar todos os quadros de exemplo do Tema 5.5 antes da entrega;
   - extensão total: 4 a 6 páginas, sem contar referências e anexos;
   - desenvolver, na seção de resultados, **um tema em destaque por anchor code**;
   - usar dois excertos por tema em destaque, com funções distintas;
   - manter PUIA em anexo e Nota de IA no relatório como instrumentos distintos.

   **b) `Modelo de Apresentação_Relatório Final_Tema5.5.pptx`**  
   Modelo formal da apresentação da Aula IX. Ele prevalece sobre slides genéricos ou apresentações anteriores quanto a roteiro, estrutura visual e conteúdo mínimo da apresentação. Regras principais:
   - apagar o slide de instruções;
   - produzir seis slides efetivos em roteiro de cinco minutos;
   - usar a sequência: capa; tema/perguntas/corpus; percurso analítico; mapa dos temas; um tema em destaque; respostas/limitações/IA;
   - usar apenas conteúdo do relatório, sem inserir achados novos;
   - desenvolver um único tema no slide 5;
   - manter a cautela de que contagens descrevem o corpus;
   - não usar “os temas emergiram”; temas foram construídos;
   - colocar o texto de fala nas anotações do apresentador.

   Regra de conflito do EP04:
   - Guia = autoridade geral de conteúdo metodológico e avaliação ampla;
   - Modelo Simplificado = autoridade de forma e preenchimento do relatório final corrente;
   - Modelo de Apresentação = autoridade de forma e roteiro da apresentação da Aula IX;
   - comunicado docente posterior = autoridade sobre prazo, quatro itens formais e dupla exportação `Coding Frequency`.


4. **Este Prompt Master**  
   Autoridade canônica de continuidade, governança, checkpoint e decisões já tomadas no projeto.

5. **Corpus original do Tema 5.1**  
   Dez entrevistas individuais do exercício didático e documento oficial do tema/questionário.

6. **Exportações reais do QualCoder e entregáveis auditados**  
   Autoridade para contagens, aplicações, casos, codebook exportado e resultados quantitativos efetivamente registrados no CAQDAS.

7. **Modelo de Codebook do professor**  
   Somente como modelo estrutural da saída solicitada. Nunca utilizar os códigos do modelo como códigos substantivos do Tema 5.1.

## 2.3 Regra de conflito

- Se houver conflito conceitual sobre o método, prevalece a Aula III, salvo refinamento posterior explícito do professor.
- Para a **rotina operacional da Etapa 2/EP03**, prevalece a Aula VI sobre instruções anteriores quando houver conflito direto.
- Para codificação, emacD, granularidade e Coding Retrieval, usar a Aula IV em conjunto com a Aula III.
- Para **conferência bibliográfica de afirmações atribuídas a Adu ou a outro autor**, consultar a fonte primária disponível no acervo sempre que pertinente.
- Se a fonte primária apresentar formulação diferente da operacionalização explícita do professor, distinguir **texto original** de **adaptação da disciplina** e seguir a adaptação do professor na execução do exercício.
- A disponibilidade de Braun & Clarke, Saldaña, Boyatzis, Template Analysis ou outras tradições não autoriza substituir silenciosamente a Codebook Thematic Analysis/Adu adotada neste projeto.
- Se houver feedback individual posterior do professor sobre este projeto, ele prevalece nas decisões operacionais específicas do caso, desde que não contradiga explicitamente uma regra metodológica superior das aulas.
- Se houver conflito sobre **o que entregar, em qual formato ou prazo**, prevalecem o Plano de Ensino e o enunciado/anexo mais recente do exercício.
- O Modelo oficial de Tabela de Temas do EP03 já está disponível e deve ser aplicado literalmente em suas regras documentais; não substituir sua estrutura por modelo próprio.
- O Guia do Relatório Final do EP04 já está disponível e governa o conteúdo metodológico geral; o Modelo Simplificado do Relatório Final governa a estrutura e o preenchimento da versão simplificada corrente; o Modelo de Apresentação governa o roteiro visual da Aula IX.
- Se outro modelo oficial de entrega ainda não tiver sido disponibilizado no projeto, **não inventar sua estrutura final**.
- Se houver instrução posterior expressa do professor, ela prevalece no domínio a que se refere e o Prompt Master deve ser atualizado.
---


## 2.4 Varredura obrigatória de instruções em todas as fontes

A recuperação do Prompt Master é o primeiro ato de continuidade, mas **não substitui a leitura dos demais documentos disponíveis**.

**Regra de escala para a biblioteca bibliográfica:** o acervo integral deve ser inventariado e permanecer disponível, mas a varredura operacional deve alcançar apenas as obras **materialmente relevantes à tarefa corrente**. Não reler integralmente os 22 PDFs a cada chat/operação. Para uma dúvida sobre Adu, por exemplo, recuperar Adu; para information power, recuperar Malterud et al.; para codificação, podem ser pertinentes Adu, Saldaña e Linneberg & Korsgaard.

Antes de iniciar qualquer análise, operação no CAQDAS ou materialização de artefato, o assistente deve executar uma **varredura documental orientada a instruções** em todos os arquivos disponíveis e relevantes à tarefa corrente.

Essa leitura deve procurar não apenas conteúdo substantivo, mas também comandos, restrições, critérios de avaliação, campos obrigatórios, convenções de preenchimento e regras de formato contidas em:

- Plano de Ensino;
- aulas;
- feedbacks do professor;
- enunciados e anexos;
- guias;
- modelos/templates;
- quadros coloridos ou destacados;
- cabeçalhos e rodapés;
- legendas;
- notas de rodapé;
- comentários;
- campos de formulário;
- células mescladas;
- textos de exemplo;
- placeholders;
- instruções incorporadas visualmente ao layout.

**Regra crítica:** não presumir que uma instrução estará em arquivo chamado “instruções”, “guia” ou “enunciado”. Um modelo pode conter regras operacionais decisivas em sua própria estrutura visual.

Quando o documento possuir conteúdo visual relevante — por exemplo caixas azuis, tabelas-modelo, campos preenchíveis, cores, setas, exemplos ou texto dentro de formas — a inspeção deve considerar **o documento renderizado/visual**, e não apenas o texto extraído.

Para cada documento relevante, distinguir internamente, quando aplicável:

`INSTRUCAO | ESTRUTURA_FIXA | PLACEHOLDER | EXEMPLO | CONTEUDO_FIXO | DADO_A_PREENCHER | CONTEUDO_SUBSTANTIVO`

Quando houver fundamentação metodológica, distinguir também a proveniência analítica:

`PROFESSOR | FONTE_PRIMARIA | ADAPTACAO_QUALCODER | INFERENCIA_DA_IA | FONTE_EXTERNA`

Não apresentar uma adaptação operacional do projeto como se fosse citação literal de Adu ou de outro autor.

Antes da materialização final, exigir:

`VARREDURA_INSTRUCOES = PASS`

Se uma instrução disponível não tiver sido examinada ou se houver ambiguidade documental capaz de alterar o entregável:

`VARREDURA_INSTRUCOES = FAIL`

e a materialização final deve ser interrompida até a resolução.

## 2.5 Uso obrigatório e direto de modelos oficiais

Quando existir modelo/template oficial aplicável ao exercício, **o próprio arquivo-modelo deve ser usado como molde de edição e preenchimento**.

Isso significa:

1. abrir e inspecionar o modelo completo antes de produzir o artefato;
2. preservar sua estrutura fixa, ordem, tabelas, campos, cabeçalhos, estilos, margens e demais elementos institucionais, salvo instrução expressa em contrário;
3. preencher todos os campos obrigatórios do cabeçalho e da identificação;
4. substituir placeholders e exemplos pelos dados do projeto;
5. remover ou substituir textos instrutivos quando o próprio modelo indicar que eles são orientação de preenchimento e não conteúdo da entrega;
6. **não reproduzir o modelo como parte do documento final**;
7. **não entregar “modelo + resposta”** quando a solicitação for preencher o modelo;
8. não criar um documento “inspirado no modelo” se o arquivo oficial estiver disponível, salvo autorização expressa do pesquisador/professor;
9. não acrescentar notas próprias, explicações de materialização, rubricas, instruções ao usuário ou conteúdo auxiliar dentro do entregável oficial, salvo se o modelo/enunciado exigir;
10. quando o modelo estiver previamente preenchido com conteúdo de exemplo, preservar a arquitetura e substituir o conteúdo de exemplo pelo conteúdo corrente — nunca tratá-lo automaticamente como conteúdo fixo.

### Regra de interpretação do modelo

Elementos como `TEMA 1`, `TEMA 2`, nomes fictícios, descrições exemplificativas, textos em quadros coloridos e exemplos de preenchimento devem ser classificados antes de serem copiados.

Um texto do modelo só permanece na entrega se for identificado como `CONTEUDO_FIXO`.

Se for `INSTRUCAO`, `PLACEHOLDER` ou `EXEMPLO`, ele deve orientar o preenchimento, não aparecer indevidamente como parte da resposta final.

### Regra de precedência

Se existir modelo oficial externo:

`MODELO_OFICIAL_DISPONIVEL = SIM`

então:

`FONTE_ESTRUTURAL_DO_ARTEFATO = MODELO_OFICIAL`

Qualquer fallback incorporado ao Prompt Master fica suspenso para esse artefato.

Somente quando não existir modelo oficial disponível:

`MODELO_OFICIAL_DISPONIVEL = NAO`

é permitido usar um contrato estrutural/fallback incorporado ao Prompt Master.

### Auditoria documental obrigatória de modelos

Antes de liberar o arquivo final, verificar:

- `T-INSTRUCTION-SCAN-01`: todos os documentos relevantes foram varridos por instruções;
- `T-MODEL-DIRECT-USE-01`: o arquivo oficial foi usado diretamente como molde quando disponível;
- `T-PLACEHOLDER-LEAK-01`: nenhum placeholder não intencional permaneceu;
- `T-EXAMPLE-CONTENT-LEAK-01`: nenhum conteúdo de exemplo foi herdado como dado real;
- `T-HEADER-FILL-01`: todos os campos obrigatórios do cabeçalho foram preenchidos;
- `T-INSTRUCTION-LEAK-01`: instruções de preenchimento não foram reproduzidas como conteúdo da entrega;
- `T-UNREQUESTED-ADDON-01`: não foram acrescentadas notas/seções próprias fora do modelo sem base documental.

Falha em qualquer teste aplicável impede `AUDITORIA_DOCUMENTAL = PASS`.


## 2.6 Inventário do acervo bibliográfico incorporado

Arquivo de origem:

`Métodos Qualitativos de Pesquisa em Ciência Política - 9192026 - 321 PM.zip`

Conteúdo auditado:

- **22 PDFs**;
- **1 arquivo `Table of Contents.html`**;
- inclui o livro integral de Philip Adu (2019) e as referências centrais/complementares da disciplina.

No bundle canônico, o acervo deve ser disponibilizado em pasta própria, preferencialmente com os PDFs extraídos, acompanhados de inventário e checksums. O ZIP original pode ser omitido do interior do bundle para evitar duplicação de centenas de megabytes, desde que seu nome e SHA-256 sejam registrados.

Para redação do EP04, quando a seção metodológica necessitar de fundamentação bibliográfica:

1. preservar a terminologia e o procedimento efetivamente ensinados na disciplina;
2. consultar a fonte primária pertinente para sustentar conceitos e citações;
3. não atribuir à obra original uma adaptação específica do QualCoder;
4. distinguir claramente o que decorre de Adu, das aulas do professor e das equivalências operacionais deste projeto.

---

# 3. ARQUIVOS REVOGADOS E FONTES NÃO SUBSTANTIVAS

Não utilizar como autoridade substantiva os codebooks preliminares de 40 códigos, incluindo:

- `Codebook_QualCoder_import.csv`;
- `Codebook_Licenciamento_Ambiental_QualCoder_v1.csv`;
- `Codebook_Licenciamento_Ambiental_v1.xlsx`.

Esses arquivos podem ser mantidos apenas como histórico.

O exemplo da Aula IV referente ao Tema 5.5, com **40 códigos**, é um **modelo didático**, não uma meta de quantidade de códigos para o Tema 5.1.

Não existe, nas aulas consultadas, regra que determine que uma pesquisa deva terminar com exatamente 40 códigos. A própria Aula IV orienta utilizar o modelo como **lente, não como régua**.

---

# 4. ESCOLA METODOLÓGICA ADOTADA

A abordagem é **Codebook Thematic Analysis**.

A posição metodológica é intermediária:

- admite construção interpretativa;
- mantém estrutura analítica declarada;
- utiliza codebook com definições e critérios explícitos;
- permite revisão de códigos;
- exige auditabilidade, consistência, sistematicidade e transparência;
- IRR não é exigência epistemológica neste exercício.

A abordagem é **mista em sentido específico**:

- **dedutiva para os anchor codes**, derivados das perguntas de pesquisa;
- **indutiva para os códigos substantivos**, construídos a partir do material empírico.

A sequência canônica é:

**PERGUNTAS DE PESQUISA  
→ ANCHOR CODES  
→ INDICADORES EMPÍRICOS  
→ CÓDIGOS  
→ REVISÃO GLOBAL DO CODEBOOK  
→ CATEGORIAS ANALÍTICAS  
→ TEMAS**

Quando houver uma entrega intermediária específica de codificação, ela pode ser congelada documentalmente entre **CÓDIGOS** e **REVISÃO GLOBAL**, sem alterar a ordem metodológica.

---

# 5. DISTINÇÃO ENTRE CÓDIGO, CATEGORIA ANALÍTICA E TEMA

## 5.1 Código substantivo

É a unidade básica da análise.

Deve:

- representar informação relevante identificada nos dados;
- carregar interpretação, não apenas nomear tópico;
- ter, como regra, no máximo cinco palavras;
- possuir descrição;
- possuir critério de exclusão;
- ser testado contra códigos vizinhos;
- permanecer revisável quando a evidência exigir.

## 5.2 Categoria analítica

Conforme a Aula III, é:

- agrupamento de códigos que compartilham características;
- produto intermediário da tematização;
- construção analítica posterior aos códigos.

**Categoria analítica não é a mesma coisa que `Category` do QDA Miner/QualCoder.**

Entretanto, a Aula VI operacionaliza o EP03 por uma tabela em que os códigos de cada anchor são **agrupados e o agrupamento final é rotulado diretamente como tema**. Assim:

- a categoria analítica continua válida como conceito e como ferramenta heurística;
- não é obrigatório materializar uma camada separada de categorias analíticas no software ou no entregável;
- as 13 categorias criadas antes da Aula VI passam a ser tratadas como **hipóteses históricas de apoio**, e não como uma grade obrigatória para a construção dos temas;
- a rotina oficial deve ser refeita/testada conforme a Aula VI, partindo dos códigos e dos dominantes.

## 5.3 Tema

É:

- conceito de maior abstração;
- padrão de significado construído a partir do agrupamento dos códigos;
- capaz de endereçar a pergunta de pesquisa;
- seletivo: deve haver poucos temas por pergunta;
- não pode ser mera reformulação do anchor code;
- não é um simples “balde” de códigos.

Na operacionalização da Aula VI:

- o rótulo do agrupamento é o tema;
- o tema recebe descrição própria;
- cada código é atribuído a **um único tema** na consolidação oficial do EP03;
- o tema deve passar pelo teste formal da Aula VI antes de ser fixado.
---

# 6. PERGUNTA GERAL, RQs E ANCHOR CODES

## Pergunta geral

**Como os analistas ambientais percebem os desafios e as contradições do licenciamento ambiental no Brasil?**

## RQ1

**Quais são as principais tensões vivenciadas pelos analistas entre as pressões por celeridade e a necessidade de rigor técnico na análise ambiental?**

Anchor code histórico e operacional:

**`Celeridade e rigor técnico`**

O teste de superdimensionamento realizado em 08/09/2026 foi homologado com a decisão de **não subdividir** esse anchor. A tensão entre celeridade e rigor constitui o próprio objeto relacional da RQ1, e a solução final de cinco temas permaneceu coerente sem necessidade de criar anchors derivados.

## RQ2

**Como os analistas avaliam a efetividade do licenciamento ambiental como instrumento de proteção ao meio ambiente e de promoção do desenvolvimento sustentável?**

Anchor code histórico do EP02:

**`Efetividade do licenciamento`**

Esse anchor permanece válido como registro da estrutura do EP02, já aprovado, mas foi subdividido operacionalmente para a Etapa 2 / EP03 após o feedback do professor.

### Anchors operacionais finais da RQ2 — homologados

1. **`Contribuições do licenciamento`**
2. **`Limites da efetividade`**
3. **`Relações e participação dos atores`**
4. **`Aperfeiçoamento do licenciamento`**

A subdivisão:

- é uma decisão analítica da Etapa 2;
- não reescreve retroativamente o EP02;
- preserva a origem de todos os códigos na RQ2;
- foi testada contra a redação da RQ2, os códigos, Counts/Cases e segmentos;
- organiza objetos substantivos distintos antes da tematização;
- foi homologada pelo pesquisador.

## Regra crítica

Os anchor codes:

- são dedutivos;
- derivam das RQs;
- possuem razão organizacional;
- organizam os códigos substantivos;
- **nunca devem ser aplicados diretamente aos segmentos**.

No QualCoder:

- `Category` de topo = anchor code operacional;
- `Code` = código substantivo.

A subdivisão da RQ2 deve ser declarada no Journal, no EP03 e, quando pertinente, no relatório final.

## Correção final de fronteira — 08/09/2026

A revisão global identificou que `Tensão entre desenvolvimento e proteção` possui **Count = 2 / Cases = 2**, com ocorrências em André e Roberto Carlos, e não Count=1/Cases=1.

A classificação anterior como `TOQUE PONTUAL — NÃO SUSTENTA TEMA` foi revogada.

Decisão homologada:

- retirar `Tensão entre desenvolvimento e proteção` de `Relações e participação dos atores`;
- realocá-lo para `Limites da efetividade`;
- integrá-lo ao tema `Desajuste entre instrumento e problema limita o alcance`.

Essa correção preserva o código e suas duas aplicações e não cria novo tema.

# 7. CORPUS E ORDEM CANÔNICA DOS CASOS

O corpus é composto por **10 entrevistas individuais fictícias, produzidas para fins didáticos, com perfis de analistas ambientais do setor público ambiental**.

Conforme o Plano de Ensino, as atividades práticas da disciplina utilizam **dados fictícios gerados por inteligência artificial**. Portanto:

- não apresentar os entrevistados como participantes reais;
- não atribuir aos dados validade empírica externa além do exercício;
- no relatório final, explicitar o caráter didático do exercício e a natureza fictícia dos dados, adaptando a redação de referência já fornecida no `Guia do Relatório Final.docx`.

A sequência canônica dos casos é:

1. Ricardo — Entrevistado 01
2. Fernanda — Entrevistada 02
3. Marcos — Entrevistado 03
4. Luciana — Entrevistada 04
5. Carlos Eduardo — Entrevistado 05
6. Patrícia — Entrevistada 06
7. André — Entrevistado 07
8. Juliana — Entrevistada 08
9. Roberto Carlos — Entrevistado 09
10. Débora — Entrevistada 10

Esta sequência substitui checkpoints antigos que colocavam Débora ou Juliana em posições diferentes.

Não utilizar o documento consolidado das entrevistas como fonte analítica paralela aos dez arquivos individuais quando isso puder duplicar o corpus.

---

# 8. SOFTWARE E CONVENÇÕES NO QUALCODER 3.8.2

Projeto principal: `licenca_ambiental.qda`.

## 8.1 Estrutura da árvore

### Projeto canônico histórico

O projeto canônico original `licenca_ambiental.qda` deve ser preservado como registro da codificação e da estrutura histórica do EP02:

```text
Celeridade e rigor técnico
├── códigos substantivos da RQ1
└── ...

Efetividade do licenciamento
├── códigos substantivos da RQ2
└── ...
```

Nunca realizar a consolidação temática nesse projeto.

### Baseline reconciliado pré-transposição

`licenca_ambiental_ep03.qda` é o **baseline reconciliado pré-transposição temática** e deve ser preservado sem novos merges.

Estrutura pré-merge homologada:

```text
Celeridade e rigor técnico
├── 34 códigos substantivos
└── ...

Contribuições do licenciamento
├── 10 códigos substantivos
└── ...

Limites da efetividade
├── 22 códigos substantivos
└── ...

Relações e participação dos atores
├── 20 códigos substantivos
└── ...

Aperfeiçoamento do licenciamento
├── 20 códigos substantivos
└── ...
```

Baseline quantitativo pré-merge:

- 5 anchors operacionais;
- 106 códigos substantivos;
- 306 aplicações;
- 10 casos;
- 20 Codes-tema `(T)` previamente criados, ainda sem aplicações.

O antigo `Efetividade do licenciamento` ficou vazio após a redistribuição e foi removido da estrutura operacional da Etapa 2.

### Cópia temática final pós-merge

`licenca_ambiental_temas.qda` é a **cópia temática consolidada do EP03**.

Estado auditado pós-merge:

```text
Celeridade e rigor técnico
├── Condições concretas sustentam o rigor (T)
├── Pressões disputam ritmo, decisão e legitimidade (T)
├── Práticas e apoios sustentam rigor (T)
├── Responsabilização e autonomia modulam decisões (T)
└── Pressões produzem custos e consequências (T)

Contribuições do licenciamento
├── Prevenção institucionaliza a consideração ambiental (T)
└── Controle torna a proteção operável (T)

Limites da efetividade
├── Controle insuficiente não assegura conformidade (T)
├── Capacidade e integridade institucional condicionam a efetividade (T)
├── Desajuste entre instrumento e problema limita o alcance (T)
└── Evidência frágil limita a análise (T)

Relações e participação dos atores
├── Relações com empreendedores são heterogêneas (T)
├── Participação formal não garante influência (T)
├── Proteção limitada tensiona relações comunitárias (T)
└── Atores externos exercem papéis heterogêneos (T)

Aperfeiçoamento do licenciamento
├── Preservar prevenção, fortalecer controle (T)
├── Capacidade estatal sustenta a implementação (T)
├── Governança territorial coordena escalas e políticas (T)
├── Participação deve produzir influência substantiva (T)
└── Proteção e viabilidade podem convergir (T)
```

Resultado físico final:

- 5 anchors;
- 20 Codes-tema `(T)`;
- 0 códigos substantivos independentes;
- 303 aplicações temáticas únicas;
- 10 casos;
- 106/106 códigos substantivos preservados documentalmente nos Code Memos dos temas;
- 0 omissões;
- 0 dupla atribuição temática;
- 0 temas residuais;
- 0 perda empírica inexplicada.

A diferença `306 → 303` decorre de três deduplicações por exact match entre códigos constituintes que foram mesclados no mesmo Code-tema.

### Correção de fronteira materializada

`Tensão entre desenvolvimento e proteção` foi fisicamente corrigido antes dos merges e ficou definitivamente em:

`Limites da efetividade`
→ `Desajuste entre instrumento e problema limita o alcance (T)`

Baseline real pré-merge:

- Count = 2;
- Cases = 2;
- casos = André + Roberto Carlos.

A alocação anterior em `Relações e participação dos atores` está revogada.

Não transformar automaticamente categorias analíticas históricas em novas `Categories`.
## 8.2 Coder

Manter `Coder: default`, salvo decisão expressa do pesquisador.

## 8.3 Annotations

As Annotations preservam os **indicadores empíricos selecionados na primeira passada**.

Identificadores devem seguir:

`IE-CASO-RQX-NN`

Exemplos:

- `IE-FER-RQ1-01`
- `IE-AND-RQ2-03`
- `IE-DEB-RQ1/RQ2-05`

`IE` significa **Indicador Empírico**. Não é código substantivo nem anchor code.

Não apagar a Annotation apenas porque o trecho foi posteriormente codificado.

## 8.4 Pontuação terminal

Quando o QualCoder gerar conflito de seleção pela inclusão de pontuação terminal, é aceitável não incluir o ponto final, desde que:

- o conteúdo substantivo permaneça integral;
- a delimitação seja explicitada;
- a convenção seja aplicada de forma transparente.

## 8.5 Journals

Existem dois Journals com funções distintas:

### `Diário de decisões analíticas`

Trilha oficial das decisões metodológicas e revisões do codebook.

Formato canônico:

```markdown
# DD/MM/AAAA — HH:MM — Caso/etapa

## Alteração:

...

## Justificativa:

...

## Efeito sobre a codificação:

...

## Estado do codebook:

- Antes: ...
- Depois: ...

---
```

### `Categorização e tematização`

Documento de trabalho interpretativo da Etapa 2.

Deve registrar:

- categorias analíticas propostas;
- códigos inicialmente relacionados;
- característica compartilhada;
- justificativa;
- fronteiras;
- evidências contrárias;
- testes contra segmentos;
- temas provisórios;
- relações entre categorias, códigos e temas;
- status de validação.

O `Diário de decisões analíticas` continua sendo a trilha de governança; o Journal `Categorização e tematização` documenta em detalhe a análise interpretativa posterior.

## 8.6 Equivalência operacional do Coding Retrieval

Para a Etapa 2, adotar como equivalência operacional canônica:

**QDA Miner Lite:** `Retrieve → Coding Retrieval`  
**QualCoder 3.8.2:** `Reports → Coding reports`

A equivalência é funcional: o relatório deve permitir recuperar os **segmentos efetivamente codificados**, associados ao respectivo código e ao arquivo/caso, para comparação contextual.

No QualCoder, usar `Reports → Coding reports` como primeira escolha. Não substituir por `Codes by text segments` quando o objetivo for validar categorias e temas, pois este último é útil para matrizes de presença/coocorrência, mas não é o relatório principal de recuperação contextual adotado neste projeto.

Configuração preferencial para validação temática:

- selecionar os códigos do anchor/RQ pertinente;
- incluir todos os casos, salvo teste deliberadamente restrito;
- não usar filtro de busca textual;
- incluir contexto textual quando disponível;
- preservar identificação do código e do arquivo/caso;
- exportar em XLSX quando a interface permitir.

Arquivos preferenciais:

- `Coding_Retrieval_RQ1.xlsx`;
- `Coding_Retrieval_RQ2.xlsx`.

Esses nomes documentais são mantidos por equivalência com a terminologia do professor, embora a exportação seja obtida no QualCoder por `Reports → Coding reports`.

## 8.7 Equivalências operacionais QDA Miner Lite ↔ QualCoder 3.8.2

A disciplina demonstra o procedimento no QDA Miner Lite. Como este projeto utiliza QualCoder 3.8.2, adotar as seguintes equivalências funcionais.

| QDA Miner Lite — operação da disciplina | QualCoder 3.8.2 — equivalente operacional |
|---|---|
| `Project → Save as` para duplicar o projeto | **Fechar o QualCoder e duplicar a pasta inteira `.qda`**, preservando o projeto original. Abrir apenas a cópia para consolidação temática. O projeto QualCoder é uma pasta que contém banco e arquivos do projeto. |
| `Add Code` sobre o anchor, marcando `(T)` | Criar um **novo Code** com o nome do tema + `(T)` e posicioná-lo sob a `Category` do anchor. Pode-se criar o Code na árvore e, se necessário, movê-lo pelo `Code Organiser`. |
| `Description` do tema | Usar o **Code Memo** do Code-tema para registrar a definição do tema e a lista dos códigos constituintes. |
| Mesclar cada código no tema correspondente | `Code Organiser` → código substantivo → **Merge code into code** → selecionar o Code-tema → `Apply`. Fazer somente na cópia temática, pois a reorganização/mesclagem não deve comprometer o projeto canônico de codificação. |
| `Retrieve → Coding Retrieval` | `Reports → Coding reports`. |
| `Analyze → Coding Frequency` / `Coding Frequency` | `Reports → Code frequencies`. |
| Frequência por arquivo/caso | `Reports → Code counts by file/case`. |
| Exportar tabela para Excel | Usar a exportação do relatório do QualCoder em `.xlsx` quando disponível. |

### Regra de segurança para a transposição temática

A consolidação temática **nunca deve ser executada no projeto canônico `licenca_ambiental.qda`**.

Antes das mesclagens:

1. fechar o QualCoder;
2. fazer backup;
3. duplicar a pasta inteira do projeto;
4. atribuir nome inequívoco à cópia, por exemplo `licenca_ambiental_temas.qda`;
5. abrir a cópia;
6. confirmar que arquivos, códigos e aplicações estão presentes;
7. somente então criar os Codes-tema `(T)` e realizar as mesclagens.

A cópia temática passa a ser um artefato derivado da Etapa 2; o projeto original continua sendo a autoridade da codificação substantiva.
---

# 9. PROCEDIMENTO CANÔNICO DE CODIFICAÇÃO

## 9.1 Primeira passada — indicadores empíricos

Primeiro decidir:

**“Isto ajuda a responder a esta pergunta, mesmo que eu ainda não saiba como?”**

Se houver dúvida razoável, selecionar.

Nesta etapa:

- identificar trecho relevante;
- registrar Annotation;
- preservar contexto;
- não propor código;
- não aplicar código.

## 9.2 Segunda passada — codificação

Somente depois da primeira passada:

1. reler o indicador em contexto;
2. determinar seu significado;
3. formular como ele responde à RQ;
4. comparar com todos os códigos existentes do anchor pertinente;
5. reutilizar código somente se sua definição realmente cobrir o significado;
6. criar código novo quando houver significado relevante não representado;
7. admitir múltiplos códigos;
8. admitir codificação parcial;
9. admitir que um indicador não gere código;
10. revisar, dividir, fundir, renomear ou excluir códigos quando a evidência justificar;
11. registrar mudanças substantivas no Diário;
12. retroaplicar mudanças aos casos anteriores quando necessário.

Não existe relação 1:1 obrigatória entre indicador e aplicação de código.

---

# 10. PROCEDIMENTO emacD

Usar a lógica ensinada na Aula IV:

- **e** — identificar o indicador empírico com a RQ em mente;
- **m** — determinar o significado do indicador;
- **a** — formular a resposta à RQ a partir da interpretação;
- **c** — criar ou selecionar o código;
- **D** — desenvolver descrição e critério de exclusão.

A operação `m` não deve ser confundida com `a`.

O mesmo significado pode produzir respostas diferentes sob anchors distintos quando o trecho efetivamente responder a mais de uma pergunta.

Nem todo significado relevante gera código sob todo anchor.

---

# 11. GRANULARIDADE DOS CÓDIGOS

Evitar:

## Amplitude excessiva

Códigos genéricos que misturam significados substantivamente diferentes.

## Fragmentação excessiva

Um código novo para cada frase ou formulação superficialmente diferente.

Critério prático:

**o código deve unir falas que dizem a mesma coisa de maneiras diferentes e separar falas que possuem sentido substantivamente distinto.**

Não fundir códigos apenas para:

- reduzir o número total;
- aumentar frequência;
- produzir uma árvore visualmente mais simples;
- aproximar-se de uma suposta meta numérica.

---

# 12. FORMATO OBRIGATÓRIO PARA CRIAÇÃO OU REVISÃO DE CÓDIGO

Usar sempre:

**Código N**

**Anchor code:** `...`

Crie/Renomeie:

### `Nome do código`

X palavras.

### Descrição

[descrição em texto corrido]

### Critério de exclusão

[critério em texto corrido, esclarecendo de quais códigos vizinhos não se confunde]

### Memo

```text
Descrição: ...
Critério de exclusão: ...
```

### Segmento exato

[texto literal]

**Começa em:** `...`  
**Termina em:** `...`  
**Pontuação final:** incluir/não incluir.

Aplique:

**`Nome do código`**

---

# 13. DELIMITAÇÃO OBRIGATÓRIA DE TRECHOS

Toda orientação de seleção deve informar:

- identificador da Annotation ou código;
- RQ;
- texto literal completo;
- primeiras palavras exatas;
- últimas palavras exatas;
- se a pontuação final deve ser incluída;
- justificativa analítica.

Não usar delimitações vagas como “a partir daqui”, “até a parte sobre fiscalização” ou equivalentes.

---

# 14. ESTADO DA CODIFICAÇÃO

A codificação dos **10 casos foi concluída**.

A revisão específica de **Roberto Carlos — Entrevistado 09** também foi concluída posteriormente à entrega inicial, corrigindo a parcimônia deliberada adotada na primeira codificação desse caso.

A revisão global do codebook foi tratada como etapa anterior à tematização e não deve ser substituída por simplificação automática.

---

# 15. AUTORIDADE QUANTITATIVA — BASELINES E ESTADO CORRENTE

## 15.1 Baseline histórico aprovado do EP02

O feedback oficial do professor confirmou o estado entregue do EP02:

- **10 casos**;
- **2 anchor codes históricos**;
- **105 códigos substantivos**;
- **299 aplicações**;
- RQ1 / `Celeridade e rigor técnico`: **33 códigos / 111 aplicações**;
- RQ2 / `Efetividade do licenciamento`: **72 códigos / 188 aplicações**.

Esse estado é:

**`BASELINE_EP02_APROVADO_PELO_PROFESSOR`**

Ele deve ser preservado como histórico e não reescrito retroativamente.

## 15.2 Estado corrente pós-revisão de Roberto Carlos

As novas exportações reais do QualCoder substituíram o checkpoint manual pós-EP02 para a Etapa 2.

Fontes correntes:

- `Code_frequencies_EP03_pos_subdivisao.xlsx`;
- `Code_counts_by_case_EP03_pos_subdivisao.xlsx`.

Estado global comprovado:

- **10 casos**;
- **106 códigos substantivos**;
- **306 aplicações**;
- RQ1: **34 códigos / 113 aplicações**;
- RQ2 total: **72 códigos / 193 aplicações**.

Novo código de RQ1:

- `Pressão externa por indeferimento` — Count = 1 / Cases = 1.

## 15.3 Estado dos anchors da RQ2 na exportação pré-correção final

A exportação `EP03_pos_subdivisao` registrou fisicamente:

- `Contribuições do licenciamento`: **10 códigos / 20 aplicações / 9 casos**;
- `Limites da efetividade`: **21 códigos / 58 aplicações / 10 casos**;
- `Relações e participação dos atores`: **21 códigos / 48 aplicações / 10 casos**;
- `Aperfeiçoamento do licenciamento`: **20 códigos / 67 aplicações / 10 casos**.

Total RQ2:

- **72 códigos / 193 aplicações**.

## 15.4 Correção analítica homologada após conferência das exportações

A conferência de `Tensão entre desenvolvimento e proteção` mostrou:

- **Count = 2**;
- **Cases = 2**;
- casos: André e Roberto Carlos.

Isso revogou a premissa anterior de Count=1/Cases=1.

A realocação analítica homologada é:

`Relações e participação dos atores`
→ `Limites da efetividade`

Estado-alvo após a correção física no QualCoder:

- `Contribuições do licenciamento`: **10 códigos / 20 aplicações**;
- `Limites da efetividade`: **22 códigos / 60 aplicações**;
- `Relações e participação dos atores`: **20 códigos / 46 aplicações**;
- `Aperfeiçoamento do licenciamento`: **20 códigos / 67 aplicações**.

Total RQ2 permanece:

- **72 códigos / 193 aplicações**.

Estado global permanece:

- **106 códigos / 306 aplicações**.

## 15.5 Regra de autoridade

- Contagens do EP02: usar apenas quando o objeto for a entrega histórica aprovada.
- Estado da Etapa 2: usar as exportações pós-Roberto e as correções homologadas.
- Qualquer nova operação física no QualCoder exige nova exportação.
- Exportações reais do QualCoder prevalecem sobre contagens manuais.
- Counts por tema podem ser obtidos pela soma das aplicações de seus códigos enquanto não houver merge.
- Cases por tema devem ser obtidos pela **união dos casos**, nunca pela soma dos Cases dos códigos.
- Após a consolidação em Codes-tema `(T)`, reexportar `Code frequencies` e `Code counts by file/case`; essa exportação passa a ser a autoridade quantitativa final dos temas.

# 16. CODEBOOK CONSOLIDADO — ESTADO CORRENTE DOS CÓDIGOS

As descrições e critérios de exclusão permanecem os do codebook/QualCoder. Este Prompt Master registra os **nomes e pertencimentos operacionais correntes**.

## 16.1 RQ1 — `Celeridade e rigor técnico` — 34 códigos / 113 aplicações

1. Aprovação defensiva
2. Atualização técnica contínua
3. Barreiras territoriais à vistoria
4. Cobrança externa por rigor
5. Complexidade técnica dos processos
6. Consequências da prevalência da pressão
7. Déficit de especialização técnica
8. Déficit de pessoal
9. Desgaste profissional
10. Exigência de preparação técnica
11. Fundamentação técnica do parecer
12. Infraestrutura operacional precária
13. Insegurança jurídica decisória
14. Licenciamento como entrave
15. Manutenção do rigor sob pressão
16. Paralisia por responsabilização
17. Pressão burocrática indireta
18. Pressão econômica setorial
19. Pressão externa por celeridade
20. Pressão externa por indeferimento
21. Pressão política personalizada
22. Pressão política setorial
23. Pressão por viabilização
24. Pressão velada
25. Proteção ambiental como entrave
26. Proteção institucional do analista
27. Reconhecimento externo do trabalho técnico
28. Registro documental
29. Responsabilidade ética e jurídica
30. Responsabilização sem autonomia
31. Reversão da análise técnica
32. Rigor por responsabilização
33. Sobrecarga de processos
34. Tempo para análise técnica

## 16.2 RQ2 — 72 códigos / 193 aplicações

### `Contribuições do licenciamento` — 10 códigos / 20 aplicações

1. Análise prévia como avanço
2. Condicionantes como proteção
3. Cultura de consideração ambiental
4. Efetividade em fontes pontuais
5. Exigibilidade das condicionantes
6. Licenciamento como proteção mínima
7. Medidas compensatórias efetivas
8. Possibilidade de responsabilização
9. Produção de informação técnica
10. Rastreabilidade documental

### `Limites da efetividade` — 22 códigos / 60 aplicações — estado analítico homologado

1. Auditorias como formalidade
2. Captura política institucional
3. Compatibilização de usos territoriais
4. Conflito de interesse nos estudos
5. Dependência da via judicial
6. Descompasso federativo do licenciamento
7. Descumprimento impune de condicionantes
8. Efetividade condicionada à capacidade institucional
9. Efetividade condicionada à conformidade
10. Estudos ambientais superficiais
11. Fiscalização insuficiente
12. Fragmentação do escopo do licenciamento
13. Fragmentação federativa do licenciamento
14. Inadequação contextual do licenciamento
15. Instabilidade institucional
16. Insuficiência do licenciamento
17. Lentidão processual
18. Limites da avaliação cumulativa
19. Limites diante de passivos ambientais
20. Limites diante de problemas sistêmicos
21. Limites em atividades difusas
22. Tensão entre desenvolvimento e proteção

### `Relações e participação dos atores` — 20 códigos / 46 aplicações — estado analítico homologado

1. Apoio de instituições de controle
2. Assimetria de capacidade
3. Assimetria de poder
4. Capacidade comunitária desigual
5. Cobrança comunitária além da capacidade
6. Compensações insuficientes
7. Comunidades aliadas à proteção
8. Contribuição informacional de atores externos
9. Desconfiança comunitária no Estado
10. Diálogo comunitário insuficiente
11. Dificuldade de diálogo com ONGs
12. Dificuldade de inclusão comunitária
13. Distribuição desigual de riscos
14. Espaços de participação social
15. Interação técnica com empreendedores
16. Licenciamento como formalidade
17. Linguagem técnica como barreira
18. Participação social pouco efetiva
19. Relação adversarial com empreendedores
20. Relação profissional sob contestação

### `Aperfeiçoamento do licenciamento` — 20 códigos / 67 aplicações

1. Ampliação do quadro de pessoal
2. Apoio à adequação ambiental
3. Aprimoramento da avaliação cumulativa
4. Aprimoramento da participação social
5. Auditorias independentes
6. Autonomia institucional
7. Conciliação ambiental e econômica
8. Eliminação do automonitoramento
9. Fortalecimento da cooperação técnica interfederativa
10. Fortalecimento da fiscalização pós-licença
11. Fortalecimento da responsabilização
12. Fortalecimento federal do licenciamento
13. Gestão territorial do licenciamento
14. Integração entre licenciamento e políticas
15. Licenciamento diferenciado por bioma
16. Licenciamento proporcional ao risco
17. Modernização e integração tecnológica
18. Poder decisório das comunidades
19. Preservação dos pilares preventivos
20. Valorização das carreiras

## 16.3 Controle de cobertura

- RQ1: 34 códigos.
- RQ2: 72 códigos.
- Total: **106 códigos substantivos**.
- Código excluído: 0.
- Código fundido: 0.
- Código duplicado entre anchors operacionais: 0 no estado analítico homologado.
- Aplicações totais: **306**.

A correção física de `Tensão entre desenvolvimento e proteção` foi materializada antes dos merges temáticos e confirmada nas exportações finais.

# 17. FREQUÊNCIA, COUNT E CASES

Definições:

- **Count** = número total de aplicações de um código;
- **Cases** = número de casos distintos em que o código aparece;
- **% Codes** = Count / soma de todos os Counts;
- **% Cases** = Cases / total de casos.

As contagens:

- descrevem o corpus;
- não medem automaticamente importância;
- não autorizam generalização estatística;
- podem ajudar a identificar padrões transversais;
- não anulam a importância analítica de códigos de caso único.

Um caso único pode funcionar como:

- contraponto;
- limite;
- exceção instrutiva;
- evidência contrária;
- espelho crítico de um padrão dominante.

## 17.1 Triagem obrigatória da cauda de códigos únicos

O feedback do EP02 identificou uma cauda extensa de códigos com **Count=1 / Cases=1** e determinou que esses códigos sejam triados **antes da tematização**.

Cada código `Count=1 / Cases=1` deve receber uma decisão explícita:

- `ACOMODA-SE EM AGRUPAMENTO` — significado singular, mas compatível com um agrupamento sustentado por outros códigos;
- `CONTRAPONTO / EXCEÇÃO` — ocorrência única com função analítica relevante;
- `TOQUE PONTUAL — NÃO SUSTENTA TEMA` — assunto mencionado de passagem, insuficiente para sustentar tema;
- `REVISAR FRONTEIRA / POSSÍVEL CONSOLIDAÇÃO` — pode indicar granularidade excessiva ou proximidade com outro código;
- `PENDENTE DE LEITURA DOS SEGMENTOS` — decisão requer retorno ao contexto via `Coding reports`.

Regras:

- não excluir automaticamente códigos raros;
- não fundir apenas para reduzir a quantidade;
- não criar tema apenas para acomodar um singleton;
- não forçar código raro a tema incoerente para atingir cobertura de 100%;
- preservar códigos classificados como contraponto, exceção ou toque pontual no codebook e no Journal;
- se o Modelo de Tabela de Temas exigir cobertura exaustiva, resolver a tensão documental antes da entrega final.

---

# 18. EXERCÍCIO PROGRAMADO 02 — ENTREGA DE CODIFICAÇÃO JÁ REALIZADA

O Plano de Ensino define o EP02 como a etapa de **codificação dos textos**, com os seguintes entregáveis:

1. arquivo Excel `.xlsx` com a compilação exportada do QDA Miner Lite em `Analyze → Coding Frequency`, visualização `Table`, contendo para cada código:
   - descrição;
   - critério de exclusão;
   - frequência;
   - número de casos;
2. apresentação `.pptx` com resumo dos resultados da codificação;
3. PUIA correspondente.

Prazo oficial do EP02: **03/09/2026 às 23h59**.

A entrega da etapa de codificação foi congelada antes da categorização/tematização.

Entregáveis produzidos incluíram:

- `Tabela_Final_Codebook_Licenciamento_Ambiental.xlsx`;
- `Apresentacao_Resultados_Codificacao_Licenciamento_Ambiental.pptx`;
- PUIA correspondente.

A planilha final segue:

`Category | Code | Description | Count | % Codes | Cases | % Cases`

onde:

- `Category` = anchor code;
- `Code` = código substantivo;
- `Description` = descrição + critério de exclusão;
- `Count` = frequência total;
- `% Codes` = Count / total de aplicações;
- `Cases` = número de casos distintos;
- `% Cases` = Cases / 10.

A categorização e a tematização posteriores **não alteram retroativamente** o significado da coluna `Category` dessa entrega nem exigem reedição dos artefatos já entregues, salvo determinação expressa do professor.

## 18.1 Feedback oficial do EP02

**Pontuação:** 10/10 — 100%.

Aspectos positivamente reconhecidos pelo professor:

- codebook extenso e rigoroso;
- 105 códigos com descrição e critério de exclusão explicitamente rotulados;
- critérios de exclusão que remetem nominalmente a códigos vizinhos;
- manutenção consistente desse padrão em 299 aplicações;
- uso do QualCoder como caminho válido;
- transparência adicional da planilha, com aba que mapeia a exportação às convenções da disciplina;
- explicitação de `Category` como anchor code;
- explicitação de `Description` como descrição + critério de exclusão;
- explicitação das fórmulas de percentuais;
- identificação dos quatro arquivos-fonte;
- compreensão adequada da função de `Cases` por meio da contagem por caso.

O professor vinculou essa transparência à futura **seção de credibilidade do relatório**. Portanto, preservar esses artefatos e decisões como evidência para o EP04.

Recomendações operacionais para o EP03:

1. revisar e subdividir `Efetividade do licenciamento` antes do agrupamento;
2. trabalhar com expectativa de **2 a 5 temas por anchor code**;
3. triar explicitamente os códigos de ocorrência única antes da tematização.

Nenhuma dessas recomendações invalida ou exige recodificação retroativa do EP02.

---

# 19. TRANSIÇÃO PARA A ETAPA 2 — ROTINA DA AULA VI

A codificação de todo o corpus foi concluída.

A arquitetura conceitual continua:

**CÓDIGOS → REVISÃO/SUBDIVISÃO DOS ANCHORS → TRIAGEM → AGRUPAMENTOS ANALÍTICOS → TEMAS**

O feedback do EP02 introduz duas operações obrigatórias **antes** da rotina de agrupamento da Aula VI:

1. revisar/subdividir o anchor `Efetividade do licenciamento`;
2. triar os códigos `Count=1 / Cases=1`.

A Aula III fornece a base conceitual para subdividir uma RQ quando ela contém objetos distintos. A Aula VI, especificamente para o EP03, governa o agrupamento posterior dos códigos já organizados nos anchors operacionais finais.

## Pré-gate obrigatório — antes da tabela de temas

1. exportar novamente `Code frequencies`;
2. exportar novamente `Code counts by file/case`;
3. congelar o novo baseline pós-Roberto;
4. identificar todos os códigos `Count=1 / Cases=1`;
5. executar a triagem da cauda única;
6. revisar `Efetividade do licenciamento` e propor sua subdivisão;
7. testar as fronteiras dos anchors derivados contra códigos e, quando necessário, segmentos;
8. homologar os anchors operacionais derivados de RQ2;
9. redistribuir analiticamente os códigos de RQ2 entre esses anchors, sem alterar sua RQ de origem;
10. calcular Count/Cases por anchor operacional;
11. somente então iniciar o agrupamento da Aula VI.

## Na tabela de temas

1. listar os códigos de cada anchor code;
2. registrar frequência e número de casos;
3. identificar os códigos dominantes;
4. usar o dominante como **semente da comparação**, e não como “tema mais importante”;
5. comparar os demais códigos um a um com a semente;
6. quando um código não couber em nenhum agrupamento existente, iniciar novo agrupamento;
7. rotular cada agrupamento;
8. o rótulo do agrupamento é o **tema**;
9. escrever uma descrição própria do tema.

## No software — depois da tabela

10. duplicar o projeto;
11. criar cada tema como `Code` sob o anchor, marcado com `(T)`;
12. registrar no `Description`/Code Memo a definição do tema e a lista dos códigos que o compõem;
13. mesclar cada código substantivo no tema correspondente;
14. gerar novamente o quadro final em `Coding Frequency` / `Code frequencies`.

A tabela torna o percurso visível e replicável, mas **não decide pelo pesquisador**. Agregar códigos ainda não é tematizar: o tema deve expressar um padrão de significado e passar pelo teste formal da Aula VI.

As 13 categorias analíticas e os 5 temas propostos antes da incorporação da Aula VI permanecem documentados como hipóteses históricas de trabalho. Eles **não devem ser tratados como grade fixa**, nem como resultado já validado. Devem ser confrontados com a rotina acima.
---

# 20. REGRA DE PERTENCIMENTO TEMÁTICO ÚNICO — EP03

A Aula VI estabelece expressamente:

**“No agrupamento, cada código vai para um tema só.”**

Essa regra passa a governar a consolidação oficial do EP03.

Portanto:

- cada código substantivo que **integrar a consolidação temática** deve aparecer em **um único tema** dentro do seu anchor;
- não duplicar um código entre temas na tabela oficial;
- não duplicar um código entre Codes-tema `(T)` na cópia temática do QualCoder;
- não inflar Count ou Cases;
- afinidades conceituais secundárias podem ser registradas no Journal como nota analítica, mas **não constituem dupla atribuição temática**;
- códigos classificados na triagem como `TOQUE PONTUAL — NÃO SUSTENTA TEMA` podem permanecer fora da consolidação temática, devidamente documentados;
- códigos classificados como `CONTRAPONTO / EXCEÇÃO` podem integrar um tema coerente ou permanecer como evidência de limite, conforme decisão analítica;
- não criar tema residual/genérico apenas para obter cobertura total;
- quando houver dúvida real entre dois temas, resolver a fronteira antes da consolidação, retornando às descrições e aos segmentos;
- se o Modelo de Tabela de Temas exigir que todo código apareça na tabela, essa exigência documental deve ser resolvida explicitamente antes da entrega, sem falsificar coerência temática.

A orientação anterior da Aula IV sobre duplicação de códigos que poderiam sustentar mais de um tema fica **superada para o procedimento oficial do EP03** pela instrução posterior e específica da Aula VI.

O Passo 10.2 de Adu relativo à duplicação com `(D)` não deve ser importado para este agrupamento: a Aula VI esclarece que esse caminho pertence ao procedimento de **presunção**, e não à rotina de agrupamento adotada aqui.
---

# 21. CATEGORIAS ANALÍTICAS PROVISÓRIAS — HIPÓTESES HISTÓRICAS DE APOIO

As categorias abaixo foram construídas **antes da incorporação da Aula VI**.

Elas permanecem registradas para preservar a trilha analítica, mas estão:

**REVOGADAS COMO PONTO DE PARTIDA OPERACIONAL DO EP03.**

Consequências:

- não são resultados homologados;
- não constituem grade obrigatória;
- não devem ser usadas para distribuir previamente os códigos;
- não devem determinar de antemão os temas finais;
- podem ser mantidas, revistas, fundidas, abandonadas ou absorvidas apenas depois do novo baseline, da subdivisão de RQ2 e da triagem dos singletons;
- sua utilidade será avaliada somente como comparação histórica/auditável.

A validação final deve retornar às descrições dos códigos e, quando necessário, aos segmentos codificados.

## RQ1 — 5 categorias provisórias

### CAT-RQ1-01 — Condições para a análise técnica

Códigos inicialmente associados:

- Atualização técnica contínua
- Barreiras territoriais à vistoria
- Complexidade técnica dos processos
- Déficit de especialização técnica
- Déficit de pessoal
- Exigência de preparação técnica
- Infraestrutura operacional precária
- Sobrecarga de processos
- Tempo para análise técnica

Característica proposta: condições humanas, materiais, territoriais, temporais e cognitivas necessárias à análise tecnicamente adequada.

### CAT-RQ1-02 — Pressões sobre a decisão técnica

Códigos inicialmente associados:

- Licenciamento como entrave
- Pressão burocrática indireta
- Pressão econômica setorial
- Pressão externa por celeridade
- Pressão política personalizada
- Pressão política setorial
- Pressão por viabilização
- Pressão velada
- Proteção ambiental como entrave

Característica proposta: diferentes formas, origens e objetos das pressões que incidem sobre ritmo, exigências ou resultados do licenciamento.

### CAT-RQ1-03 — Estratégias e suportes do rigor

Códigos inicialmente associados:

- Cobrança externa por rigor
- Fundamentação técnica do parecer
- Manutenção do rigor sob pressão
- Proteção institucional do analista
- Reconhecimento externo do trabalho técnico
- Registro documental

Característica proposta: práticas, recursos e apoios utilizados para sustentar decisões tecnicamente fundamentadas.

### CAT-RQ1-04 — Autonomia, responsabilização e decisão

Códigos inicialmente associados:

- Insegurança jurídica decisória
- Paralisia por responsabilização
- Responsabilidade ética e jurídica
- Responsabilização sem autonomia
- Reversão da análise técnica
- Rigor por responsabilização

Característica proposta: relações entre autonomia, responsabilização, segurança jurídica e capacidade de decidir e sustentar a decisão.

### CAT-RQ1-05 — Efeitos das pressões sobre o trabalho

Códigos inicialmente associados:

- Aprovação defensiva
- Consequências da prevalência da pressão
- Desgaste profissional

Característica proposta: consequências profissionais, comportamentais e substantivas da exposição às pressões.

## RQ2 — 8 categorias provisórias

### CAT-RQ2-01 — Núcleo preventivo e resultados protetivos

Códigos inicialmente associados:

- Análise prévia como avanço
- Condicionantes como proteção
- Cultura de consideração ambiental
- Licenciamento como proteção mínima
- Medidas compensatórias efetivas
- Possibilidade de responsabilização
- Preservação dos pilares preventivos
- Produção de informação técnica
- Rastreabilidade documental

### CAT-RQ2-02 — Controle pós-licença e responsabilização

Códigos inicialmente associados:

- Apoio de instituições de controle
- Auditorias como formalidade
- Auditorias independentes
- Dependência da via judicial
- Descumprimento impune de condicionantes
- Eliminação do automonitoramento
- Exigibilidade das condicionantes
- Fiscalização insuficiente
- Fortalecimento da fiscalização pós-licença
- Fortalecimento da responsabilização

### CAT-RQ2-03 — Qualidade e independência da evidência

Códigos inicialmente associados:

- Conflito de interesse nos estudos
- Contribuição informacional de atores externos
- Estudos ambientais superficiais

### CAT-RQ2-04 — Participação, assimetrias e justiça ambiental

Códigos inicialmente associados:

- Aprimoramento da participação social
- Assimetria de poder
- Capacidade comunitária desigual
- Cobrança comunitária além da capacidade
- Compensações insuficientes
- Comunidades aliadas à proteção
- Desconfiança comunitária no Estado
- Diálogo comunitário insuficiente
- Dificuldade de diálogo com ONGs
- Dificuldade de inclusão comunitária
- Distribuição desigual de riscos
- Espaços de participação social
- Linguagem técnica como barreira
- Participação social pouco efetiva
- Poder decisório das comunidades

### CAT-RQ2-05 — Relações com empreendedores e desenvolvimento

Códigos inicialmente associados:

- Apoio à adequação ambiental
- Assimetria de capacidade
- Conciliação ambiental e econômica
- Efetividade condicionada à conformidade
- Interação técnica com empreendedores
- Licenciamento como formalidade
- Relação adversarial com empreendedores
- Relação profissional sob contestação
- Tensão entre desenvolvimento e proteção

### CAT-RQ2-06 — Capacidade, autonomia e integridade institucional

Códigos inicialmente associados:

- Ampliação do quadro de pessoal
- Autonomia institucional
- Captura política institucional
- Efetividade condicionada à capacidade institucional
- Instabilidade institucional
- Modernização e integração tecnológica
- Valorização das carreiras

### CAT-RQ2-07 — Adequação territorial, federativa e regulatória

Códigos inicialmente associados:

- Aprimoramento da avaliação cumulativa
- Compatibilização de usos territoriais
- Descompasso federativo do licenciamento
- Fortalecimento da cooperação técnica interfederativa
- Fortalecimento federal do licenciamento
- Fragmentação federativa do licenciamento
- Gestão territorial do licenciamento
- Inadequação contextual do licenciamento
- Integração entre licenciamento e políticas
- Licenciamento diferenciado por bioma
- Licenciamento proporcional ao risco
- Limites da avaliação cumulativa

### CAT-RQ2-08 — Limites de alcance e funcionamento

Códigos inicialmente associados:

- Efetividade em fontes pontuais
- Fragmentação do escopo do licenciamento
- Insuficiência do licenciamento
- Lentidão processual
- Limites diante de passivos ambientais
- Limites diante de problemas sistêmicos
- Limites em atividades difusas

---

# 22. TEMAS PROVISÓRIOS — HIPÓTESES PRÉ-AULA VI

Os temas abaixo foram propostos antes da incorporação da Aula VI e permanecem apenas como **hipóteses interpretativas históricas**.

Estão **revogados como ponto de partida operacional do EP03**. Não possuem prioridade sobre os agrupamentos que forem reconstruídos após o novo baseline, a subdivisão de RQ2 e a triagem dos singletons. Podem sobreviver, ser renomeados, divididos, fundidos ou abandonados somente se forem reencontrados pela nova rotina e passarem pelo teste formal da Aula VI.

## RQ1

### T-RQ1-01 — O rigor técnico é produzido sob pressão e restrições

Categorias inicialmente relacionadas:

- CAT-RQ1-01
- CAT-RQ1-02
- CAT-RQ1-03

Significado central provisório: o rigor precisa ser produzido e sustentado em ambiente marcado por limitações de capacidade e pressões sobre tempo, exigências e resultados.

### T-RQ1-02 — Autonomia e responsabilização têm efeitos ambivalentes

Categorias inicialmente relacionadas:

- CAT-RQ1-03
- CAT-RQ1-04
- CAT-RQ1-05

Significado central provisório: responsabilização e controle podem fortalecer o rigor ou produzir insegurança, paralisia e comportamento defensivo, dependendo da autonomia e do desenho institucional.

## RQ2

### T-RQ2-01 — O licenciamento é necessário, mas não autossuficiente

Categorias inicialmente relacionadas:

- CAT-RQ2-01
- CAT-RQ2-07
- CAT-RQ2-08

Significado central provisório: o instrumento possui núcleo preventivo valorizado, mas sua suficiência varia com a escala e a natureza dos problemas ambientais.

### T-RQ2-02 — Efetividade exige capacidade para transformar regra em resultado

Categorias inicialmente relacionadas:

- CAT-RQ2-02
- CAT-RQ2-03
- CAT-RQ2-06

Significado central provisório: existência formal de normas, estudos, licenças e condicionantes não garante proteção sem capacidade institucional, evidência confiável, monitoramento, fiscalização, cumprimento e responsabilização.

### T-RQ2-03 — A reforma deve ampliar governança sem desmontar a prevenção

Categorias inicialmente relacionadas:

- CAT-RQ2-01
- CAT-RQ2-04
- CAT-RQ2-05
- CAT-RQ2-06
- CAT-RQ2-07

Significado central provisório: as propostas presentes no corpus apontam para fortalecimento, adaptação e ampliação da governança, preservando componentes preventivos considerados relevantes.

---

# 23. EVIDÊNCIAS CONTRÁRIAS E LIMITES QUE DEVEM SER PRESERVADOS

Buscar e manter visíveis, no mínimo, os seguintes contrastes já identificados como hipóteses de trabalho:

- pressão externa pode buscar flexibilização, mas também exigir maior rigor;
- responsabilização pode aumentar cautela e rigor, mas também produzir insegurança e paralisia;
- pressão pode gerar desgaste ou aprovação defensiva, mas também existe manutenção do rigor sob pressão;
- empreendedores podem manter relações adversariais ou formalistas, mas também relações técnicas e processos de adequação ambiental;
- comunidades podem sofrer assimetrias e barreiras, mas também atuar como aliadas da proteção;
- mecanismos de controle podem ser frágeis, mas também existem apoios institucionais e propostas de fortalecimento;
- o licenciamento pode ser insuficiente para problemas sistêmicos, difusos ou históricos, mas apresentar maior efetividade em fontes pontuais;
- propostas de reforma não equivalem automaticamente a flexibilização ou endurecimento uniforme.

Não transformar esses contrastes em conclusões antes da validação contra os segmentos.

---

# 24. REVISÃO DOS AGRUPAMENTOS CONTRA AS EVIDÊNCIAS — AULAS IV E VI

A Aula VI inicia a Etapa 2 pela compilação dos códigos, com frequência e casos, e pelo agrupamento comparativo a partir dos dominantes.

A Aula IV continua sendo útil para **voltar aos segmentos efetivamente codificados** quando for necessário testar o sentido de um código ou a fronteira de um agrupamento.

Assim, `Coding reports` não substitui a rotina da Aula VI e não precisa ser usado como primeira operação para cada agrupamento. Deve ser usado estrategicamente para:

1. verificar se o nome/Description do código representa adequadamente seus segmentos;
2. decidir entre dois temas possíveis;
3. testar se códigos aparentemente semelhantes realmente compartilham significado;
4. investigar códigos de caso único;
5. procurar evidências contrárias;
6. testar limites do tema;
7. selecionar posteriormente excertos típicos, limites e exceções.

No QualCoder 3.8.2, utilizar:

**`Reports → Coding reports`**

como equivalente operacional de:

**QDA Miner Lite `Retrieve → Coding Retrieval`**.

`Codes by text segments` continua sendo relatório complementar para presença/coocorrência, não o relatório principal de recuperação contextual.
---

# 25. RELATÓRIOS DE APOIO PARA A ETAPA 2

## 25.1 Relatórios quantitativos — ponto de partida da Aula VI

Como Roberto Carlos foi revisado depois do último congelamento quantitativo, antes do agrupamento final exportar novamente:

1. **`Reports → Code frequencies`**
2. **`Reports → Code counts by file/case`**

Esses relatórios devem fornecer a compilação atualizada para:

- listar todos os códigos por anchor histórico;
- conhecer Count;
- conhecer Cases;
- identificar todos os `Count=1 / Cases=1`;
- sustentar a triagem dos singletons;
- subsidiar a revisão/subdivisão de RQ2;
- recalcular a distribuição por anchor operacional após a subdivisão;
- identificar os códigos dominantes somente depois da estrutura de anchors estar estabilizada.

### Dominante

Na Aula VI, o dominante é identificado por **frequência e casos**.

O dominante:

- é a semente da comparação;
- não é automaticamente o tema mais importante;
- não determina sozinho o nome do tema.

Se não houver um dominante inequívoco — por exemplo, empate ou conflito entre maior Count e maior Cases — não criar uma regra de desempate silenciosa. Registrar a ambiguidade e comparar os candidatos com os demais códigos antes da decisão.

## 25.2 Coding reports — recuperação contextual de apoio

Pode-se gerar preferencialmente dois relatórios amplos:

### RQ1

`Coding_Retrieval_RQ1.xlsx`

com todos os códigos de `Celeridade e rigor técnico`.

### RQ2

`Coding_Retrieval_RQ2.xlsx`

com todos os códigos de `Efetividade do licenciamento`.

Esses arquivos servem para revisão contextual e contraevidência, mas **não substituem** a tabela de agrupamento ensinada na Aula VI.
---

# 26. PROCEDIMENTO OBRIGATÓRIO DE CONSTRUÇÃO E TESTE DOS TEMAS — AULA VI

Para cada **anchor operacional final**:

1. listar os códigos com Count e Cases;
2. excluir da rotina automática de agrupamento os códigos cuja triagem ainda estiver pendente;
3. identificar o dominante ou os candidatos dominantes;
4. usar o dominante como semente;
5. comparar os demais códigos um a um;
6. formar novo agrupamento quando o código não couber nos existentes;
7. rotular cada agrupamento como tema;
8. escrever descrição própria do tema;
9. garantir que cada código integrado à consolidação tenha **um único tema**;
10. retornar aos segmentos via `Coding reports` quando houver dúvida de sentido ou fronteira;
11. buscar evidências contrárias, exceções e limites;
12. verificar a expectativa docente de **2 a 5 temas por anchor code**.

### Regra de 2 a 5 temas por anchor

O feedback do EP02 estabelece, para o EP03, a expectativa de **2 a 5 temas por anchor code**.

Essa faixa funciona como **guardrail de granularidade**, não como meta mecânica.

Se um anchor exigir temas excessivamente genéricos para conter todos os seus códigos em até cinco temas:

- não forçar a agregação;
- revisar a subdivisão do anchor;
- revisar a triagem dos códigos periféricos;
- testar se objetos substantivamente distintos foram mantidos sob o mesmo anchor.

Não criar exatamente cinco temas apenas para cumprir a faixa.

## Teste formal do tema

Antes de fixar cada tema, responder às três perguntas da Aula VI:

1. **Existe um conceito central?**  
   Se a resposta for apenas uma lista, não é tema.

2. **Organiza material heterogêneo?**  
   Se apenas reúne trechos/códigos que dizem exatamente a mesma coisa, pode ser apenas um código ampliado.

3. **Sobreviveria à remoção do trecho mais forte?**  
   Se não sobreviver, pode ser uma citação forte, e não um padrão.

### Se o tema falhar

- falha na pergunta 1 → verificar se há dois temas fundidos e separar;
- falha na pergunta 2 → devolver o material a outro agrupamento ou reformular;
- falha na pergunta 3 → procurar outros códigos/segmentos que sustentem o padrão; se não houver, não fixar o tema.

Além disso, verificar:

- se o tema responde à RQ;
- se não apenas repete o anchor code;
- se a descrição do tema explicita seu significado central;
- se os códigos atribuídos possuem coerência suficiente;
- se contraevidências e limites foram documentados.

Tema não é produzido por frequência.

Frequência e Cases ajudam a iniciar a comparação e localizar códigos dominantes, mas não medem importância temática.
---

# 27. EXERCÍCIO PROGRAMADO 03 — CATEGORIZAÇÃO/IDENTIFICAÇÃO DE TEMAS

## 27.1 Escopo oficial

O Plano de Ensino define o EP03 como a **categorização dos códigos atribuídos aos textos** e a identificação dos temas.

Entregáveis oficiais:

1. **arquivo Word `.docx`** contendo a **tabela com os temas**, elaborada conforme o **Modelo de Tabela de Temas** disponibilizado como anexo do exercício;
2. **uma tabela para cada anchor code**;
3. **apresentação `.pptx`** contendo um resumo dos temas identificados;
4. **PUIA** referente ao exercício.

Prazo oficial: **17/09/2026 às 23h59**.

## 27.2 Estrutura operacional esclarecida pela Aula VI

A Aula VI estabelece que, depois dos gates específicos deste projeto:

- partir da compilação atualizada dos códigos;
- usar os **anchors operacionais finais** após a subdivisão de RQ2;
- listar os códigos de cada anchor com frequência e casos;
- construir **uma tabela por anchor code**;
- trabalhar com expectativa de **2 a 5 temas por anchor**;
- os **temas ocupam as colunas**;
- em cada coluna/célula temática, registrar a **descrição do tema** e, abaixo, os **códigos que o compõem**;
- agrupar a partir dos dominantes;
- o rótulo do agrupamento é o tema;
- cada código que integrar a consolidação vai para um único tema;
- códigos triados como toque pontual podem permanecer documentados sem sustentar tema;
- temas são revisáveis;
- a tabela organiza o processo, mas a interpretação permanece do pesquisador.

## 27.3 Modelo oficial de Tabela de Temas — incorporado

Fonte documental obrigatória do EP03:

**`Modelo de Tabela de Temas(1).docx`**

O modelo oficial já está disponível e passa a governar a materialização do `.docx` do EP03.

### Estrutura obrigatória

1. organizar os temas **por anchor code**;
2. produzir **uma tabela para cada anchor operacional final**;
3. no caso corrente, produzir **cinco tabelas**, correspondentes a:
   - `Celeridade e rigor técnico`;
   - `Contribuições do licenciamento`;
   - `Limites da efetividade`;
   - `Relações e participação dos atores`;
   - `Aperfeiçoamento do licenciamento`;
4. o fato de o template demonstrativo conter quatro blocos não limita o projeto a quatro tabelas: a regra superior do próprio modelo é **uma tabela por anchor code**;
5. os temas ocupam as colunas;
6. em cada tema, registrar nesta ordem:
   - **nome analítico do tema**;
   - **descrição do que o tema representa**, preferencialmente em 1–2 frases no documento de entrega;
   - **lista dos códigos que compõem o tema**.

### Regras quantitativas e de granularidade do modelo

- usar quantos temas forem necessários, **geralmente de 2 a 5 por anchor code**;
- o número de temas deve ficar **abaixo da metade do número de códigos daquele anchor**;
- cada tema deve ter **no mínimo 2–3 códigos relacionados**;
- tema com apenas 1 código é considerado muito frágil;
- os códigos dentro de cada tema devem estar relacionados entre si;
- cada tema deve capturar aspecto distinto, evitando sobreposição excessiva.

### Nomenclatura e Description

- não usar nomes genéricos como `Tema 1`, `Tema 2`;
- o nome deve ser **analítico, descritivo e conciso**, capturando o padrão de significado;
- a descrição logo abaixo do nome deve registrar o que o tema representa;
- essa descrição e a lista de códigos constituem o rascunho do `Description` do QDA Miner;
- no QualCoder, a equivalência operacional é o **Code Memo** do Code-tema `(T)`;
- a lista de códigos deve ser preservada no Memo porque a mesclagem consome os códigos substantivos originais na cópia temática.

### Grelha oficial de avaliação do EP03

A auditoria final do `.docx` e dos temas deve reproduzir os seis critérios do modelo:

| Critério | Pontuação |
|---|---:|
| Coerência interna dos temas — códigos realmente se relacionam | 2,5 |
| Distinção clara entre temas — evita sobreposição excessiva | 2,0 |
| Abrangência — temas cobrem os códigos importantes | 1,5 |
| Nível de abstração adequado — nem muito genérico, nem muito específico | 1,5 |
| Nomenclatura dos temas — clara, descritiva e concisa | 1,0 |
| Organização por anchor code — temas respondem às respectivas perguntas | 1,5 |
| **TOTAL** | **10,0** |

### Auditoria estrutural da solução temática homologada

Com base no estado temático da v2.5:

- `Celeridade e rigor técnico`: 34 códigos / 5 temas;
- `Contribuições do licenciamento`: 10 códigos / 2 temas;
- `Limites da efetividade`: 22 códigos / 4 temas;
- `Relações e participação dos atores`: 20 códigos / 4 temas;
- `Aperfeiçoamento do licenciamento`: 20 códigos / 5 temas.

Todos os cinco anchors:

- estão na faixa de 2–5 temas;
- possuem número de temas inferior à metade do número de códigos do anchor.

Os 20 temas homologados possuem entre **2 e 11 códigos**, portanto não há tema de código único.

**Status da auditoria estrutural:** `PASS`.

A auditoria semântica/documental final contra a grelha oficial — coerência, distinção, abrangência, abstração, nomenclatura e resposta ao anchor/RQ — foi repetida na materialização final e permaneceu compatível com a solução homologada. **Status final: PASS**, sujeito apenas à auditoria derradeira do pacote de entrega.
## 27.4 Transposição dos temas para o QualCoder — CONCLUÍDA

**Estado do gate em 09/09/2026:** transposição física, merges e auditoria pós-merge concluídos na cópia `licenca_ambiental_temas.qda`.

Procedimento efetivamente realizado:

1. preservação de `licenca_ambiental.qda` como projeto histórico/canônico do EP02;
2. preservação de `licenca_ambiental_ep03.qda` como baseline reconciliado pré-transposição;
3. correção física de `Tensão entre desenvolvimento e proteção` para `Limites da efetividade`;
4. criação dos 20 Codes-tema `(T)` sob seus anchors;
5. inserção nos Code Memos de:
   - definição do tema;
   - lista dos códigos constituintes;
   - contraponto/exceção quando aplicável;
6. construção do mapa inequívoco `106 códigos substantivos → 20 temas`;
7. confirmação de 106/106 códigos com exatamente um destino;
8. execução manual dos merges em `Coding → Code Organiser`, usando a redação real da interface 3.8.2:
   - código substantivo;
   - `Merge code into code`;
   - selecionar o Code-tema `(T)`;
   - `Apply`;
9. execução dos merges **um tema por vez**, com auditoria de `Code frequencies` após cada tema;
10. uso de `Code text exact matches` para explicar três reduções legítimas de Count;
11. auditoria pós-merge global;
12. exportação final de:
   - `Code_frequencies_12.xlsx`;
   - `Code_comparisons(2).xlsx`;
   - `Codebook-licenca_ambiental_temas(3).qdc`.

Resultado final comprovado:

- 5 anchors;
- 20 Codes-tema;
- 0 códigos substantivos independentes;
- 303 aplicações temáticas únicas;
- 10 casos;
- 0 perda empírica inexplicada;
- 106/106 códigos constituintes preservados nos Memos;
- 8 contrapontos/exceções preservados;
- 20/20 Cases pós-merge confirmados pela matriz real por caso.

A diferença entre 306 aplicações substantivas pré-merge e 303 aplicações temáticas pós-merge é integralmente explicada por três exact matches internos entre códigos destinados ao mesmo tema.
## 27.5 Papel das categorias analíticas anteriormente propostas

As 13 categorias analíticas anteriormente registradas:

- são preservadas no Journal como trilha de raciocínio;
- podem ajudar a comparar códigos;
- não são uma etapa obrigatória separada no entregável;
- não podem impedir que a rotina da Aula VI produza agrupamentos diferentes;
- não devem ser transformadas automaticamente em `Categories` do QualCoder.

## 27.6 Matriz analítica interna de apoio

Para auditoria e organização, pode ser mantida uma planilha interna, não substitutiva do `.docx` oficial, com estrutura recomendada:

### Aba 1 — `Tabela_Final_Professor`

`Category | Code | Description | Count | % Codes | Cases | % Cases`

Aqui `Category` continua significando **anchor code**.

### Aba 2 — `Agrupamento_Temas`

`RQ | Anchor Code | Tema | Descrição do Tema | Código | Count | Cases | Código dominante? | Justificativa do agrupamento | Status`

Cada código deve aparecer em **um único tema** no estado homologado.

### Aba 3 — `Teste_Temas`

`RQ | Tema | Conceito central? | Material heterogêneo? | Sobrevive sem trecho mais forte? | Responde à RQ? | Repete anchor? | Status | Observações`

### Aba 4 — `Evidencias_Contrarias`

`RQ | Tema | Tipo | Código | Caso/segmento | Implicação analítica`

### Aba 5 — `Dicionario_Analitico`

`Tipo | ID | Nome | Definição/significado central | Fronteira/limite | RQ | Status`

Essa planilha é **artefato de trabalho/auditoria**, não o entregável principal do EP03, salvo instrução posterior do professor.

As contagens devem vir de exportação real do QualCoder.

Não inferir `Count` ou `Cases` a partir do Journal.
---

# 28. CONTROLE DE QUALIDADE

Aplicar obrigatoriamente:

## Controle de anchor superdimensionado

Antes da tematização:

- verificar se algum anchor concentra quantidade de códigos que inviabiliza 2–5 temas sem perda de especificidade;
- se sim, revisar os objetos da RQ e subdividir antes do agrupamento;
- registrar a justificativa no Journal.

## Controle da cauda única

Antes da tematização:

- todos os códigos `Count=1 / Cases=1` devem possuir status de triagem;
- nenhum singleton deve gerar tema próprio apenas por necessidade de cobertura;
- nenhum singleton deve ser descartado apenas por baixa frequência.

## Consistência

O procedimento executado deve corresponder ao método declarado.

## Sistematicidade

A mesma estratégia deve ser aplicada ao corpus de forma coerente.

## Transparência

As decisões, revisões, participação da IA e limites devem ser documentados.

## Viés de confirmação

Buscar ativamente:

- evidências contrárias;
- leituras alternativas;
- exceções;
- casos positivos diante de padrões predominantemente negativos.

## Temas superficiais

Realizar rodadas iterativas de revisão e refinamento.

## Confusão entre tópico e tema

Cada tema deve ter um **significado central organizador**.

Aplicar obrigatoriamente o teste da Aula VI:

1. conceito central;
2. organização de material heterogêneo;
3. sobrevivência à retirada do trecho mais forte.

---

# 29. INFORMATION POWER E AMOSTRA

Se for necessário justificar os 10 participantes em relatório posterior, utilizar **information power**, conforme as aulas.

Não combinar information power e saturação como se fossem argumentos cumulativos.

A análise desta disciplina utiliza corpus de entrevistas com público-alvo homogêneo e não deve ser automaticamente chamada de estudo de caso.

---

# 30. PARTICIPAÇÃO DA IA

A IA atua como:

- tutor metodológico;
- interlocutor analítico;
- apoio à delimitação;
- apoio à comparação com o codebook;
- apoio à redação de descrições e critérios;
- apoio à revisão de sobreposições;
- apoio à recuperação e organização dos segmentos;
- apoio à construção de matrizes;
- apoio à identificação de evidências contrárias;
- apoio à consolidação de relatórios;
- apoio à preparação de Excel, slides e textos.

O pesquisador continua responsável por:

- aceitar ou rejeitar indicadores;
- aceitar ou rejeitar códigos;
- decidir revisões do codebook;
- homologar categorias;
- homologar temas;
- interpretar resultados;
- avaliar evidências contrárias;
- validar excertos;
- responder pelas conclusões.

A IA **não deve declarar categoria ou tema como homologado sem decisão expressa do pesquisador**.

---

# 31. PUIA, EXERCÍCIOS PROGRAMADOS E NOTA METODOLÓGICA

## 31.1 PUIA obrigatório em todos os exercícios

O Plano de Ensino determina que **cada um dos quatro exercícios programados** seja acompanhado do Protocolo de Declaração de Uso de IA Generativa.

O protocolo deve registrar:

- se houve ou não uso de IA generativa;
- em que momento;
- com que ferramenta;
- com que efeito percebido sobre o processo de trabalho.

A avaliação considera:

- honestidade da declaração;
- qualidade da reflexividade sobre o processo;

e **não** o simples uso ou não uso da ferramenta.

A entrega do PUIA é obrigatória. Trabalho entregue sem o protocolo é considerado **incompleto**.

## 31.2 PUIA do EP03

**Modelo oficial do PUIA incorporado em 09/09/2026.** O protocolo final do EP03 foi transplantado para o modelo oficial já utilizado na disciplina, preservando sua estrutura de declaração objetiva de uso, momentos de utilização, efeito sobre o processo analítico, responsabilidade do pesquisador e limites declarados.

Na entrega de categorização/tematização, declarar de forma compatível com o processo real:

- se a IA propôs categorias analíticas;
- se a IA propôs temas;
- se a IA organizou `Coding reports`;
- se buscou evidências contrárias;
- se sugeriu relações secundárias;
- se produziu ou formatou o `.docx`, `.pptx` ou matriz analítica;
- quais decisões foram homologadas pelo pesquisador.

Não declarar como humana uma validação realizada apenas pela IA.

## 31.3 Exercício Programado 04 — Guia incorporado / gate aberto

O arquivo **`Guia do Relatório Final.docx`** foi incorporado em **19/09/2026** e passa a governar a materialização do EP04.

Entregáveis do EP04:

1. relatório final em Word `.docx`, conforme o Guia, com a **Nota Metodológica sobre Uso de IA Generativa** ao final da Metodologia;
2. **duas exportações `Coding Frequency` em Excel**, em `Table view`:
   - uma exportação do projeto de codificação **pré-merge**, documentando o livro de códigos da seção 2.5;
   - uma exportação do projeto temático **pós-merge**, documentando os temas da seção 2.6;
3. apresentação `.pptx` com resumo do relatório, a ser exposta na **Aula IX, em 09/10/2026**;
4. PUIA correspondente, anexado à parte.

Prazo oficial atualizado pelo comunicado docente: **08/10/2026**.

**Status do gate documental:** `GUIA_RELATORIO_FINAL = INCORPORADO`.

Consequência:

**EP04 = ABERTO PARA MATERIALIZAÇÃO CONTROLADA**, preservando como autoridade empírico-analítica os resultados finais do EP03.

**Comunicado docente posterior incorporado:** a entrega do EP04 passou a ter quatro itens formais, com inclusão explícita de duas exportações `Coding Frequency`. Esse comunicado prevalece sobre a formulação anterior do pacote de entrega sempre que houver conflito documental.

## 31.4 Estrutura obrigatória do relatório final

A ordem definida pelo Guia é:

1. **INTRODUÇÃO**
2. **METODOLOGIA**
3. **RESULTADOS E DISCUSSÃO**
4. **CONCLUSÃO**
5. **REFERÊNCIAS**

Extensão total esperada: **6 a 12 páginas**.

### 31.4.1 Introdução

Deve conter:

- **Apresentação do tema:** iniciar com a formulação `O tema desta pesquisa é...`;
- **Contextualização:** um parágrafo situando relevância para ciência política e políticas públicas; como o tema foi atribuído, não justificar a escolha, apenas explicar o que é e por que importa;
- **Perguntas de pesquisa:** explicitar as RQs e declarar a subdivisão da RQ2/anchor quando aplicável;
- **Nota sobre o exercício:** explicitar caráter didático e natureza fictícia dos dados, adaptando a redação de referência do Guia.

Redação de referência do Guia a ser **adaptada**, não copiada mecanicamente:

> Os dados analisados neste relatório são inteiramente fictícios, criados para fins didáticos no âmbito da disciplina de Métodos Qualitativos de Pesquisa em Ciência Política do MPPG/EPPG-FGV. Não representam opiniões reais de indivíduos ou grupos e não devem ser utilizados como fonte de pesquisa nem citados como dados empíricos. O exercício tem por objetivo o domínio do procedimento analítico, e não a produção de conhecimento substantivo sobre o tema tratado.

Extensão: **cerca de meia página; no máximo uma**.

### 31.4.2 Metodologia

A metodologia é a seção **mais substancial** do relatório.

Deve:

- descrever o percurso **na ordem em que foi executado**;
- ser escrita **em primeira pessoa**;
- distinguir com precisão método declarado, operações efetivamente realizadas e adaptações operacionais para o QualCoder.

#### 2.1 Dados e corpus

Registrar:

- dez entrevistas fictícias;
- roteiro de cinco perguntas abertas;
- caracterização dos entrevistados;
- variação interna de perfil, incluindo quando pertinente tempo de experiência, órgão e função.

#### 2.2 Perguntas de pesquisa e anchor codes

- reproduzir as RQs;
- reproduzir os anchor codes criados;
- explicar a correspondência entre RQ e anchor;
- declarar a subdivisão operacional da RQ2;
- registrar que, no QDA Miner Lite, cada anchor code corresponde a uma `Category`;
- no QualCoder, preservar a equivalência adotada neste projeto: `Category` de topo = anchor code.

Referência indicada pelo Guia: **Adu (2019, p. 97)**.

#### 2.3 Estratégia de codificação

Declarar:

- qual das três estratégias do método **DIP** foi adotada;
- por que foi adotada, relacionando-a ao verbo da pergunta de pesquisa;
- se houve combinação de estratégias, indicar onde cada uma operou.

Referência indicada pelo Guia: **Adu (2019, cap. 3)**.

#### 2.4 Procedimento emacD

Descrever os cinco passos `e → m → a → c → D` e apresentar **um exemplo trabalhado completo**, contendo indicador empírico, significado, resposta à pergunta, código e descrição.

Registrar ainda:

- quantos códigos foram criados por anchor code;
- critérios de nomenclatura;
- três a cinco códigos como exemplo, com suas descrições.

Referência indicada pelo Guia: **Adu (2019, p. 97–100)**.

#### Regra adicional do comunicado docente para a seção 2.4

Quando houver revisão de codificação entre a Etapa 1/codificação e a Etapa 2/tematização — por desmembramento de códigos, criação de novos códigos, realocação entre categorias/anchors ou revisão de fronteiras — a seção **2.4** deve descrever **o que mudou e por que mudou**.

Para este projeto, a seção 2.4 deve declarar, de forma sintética e verificável, ao menos:

- o baseline histórico aprovado do EP02: **105 códigos substantivos / 299 aplicações**;
- a revisão posterior de Roberto Carlos, que acrescentou **7 aplicações** e criou `Pressão externa por indeferimento`, levando o estado pré-merge corrente a **106 códigos substantivos / 306 aplicações**;
- a subdivisão operacional da RQ2, antes da tematização, do anchor histórico `Efetividade do licenciamento` para quatro anchors: `Contribuições do licenciamento`, `Limites da efetividade`, `Relações e participação dos atores` e `Aperfeiçoamento do licenciamento`;
- a correção final de fronteira de `Tensão entre desenvolvimento e proteção`, realocada para `Limites da efetividade` e integrada ao tema `Desajuste entre instrumento e problema limita o alcance`;
- que essas mudanças foram registradas nos Journals e se tornam verificáveis pela comparação entre as exportações pré-merge e pós-merge.

Essa regra operacional reforça a transparência exigida na seção 2.7 e não autoriza reescrever retroativamente o EP02 aprovado.

#### 2.5 Livro de códigos e compilação

Explicar:

- estrutura do livro de códigos;
- função dos critérios de exclusão;
- no código, `Description` = descrição + critério de exclusão;
- no tema, `Description` = descrição do tema + lista dos códigos mesclados;
- compilação com frequência e número de casos;
- cruzamento com atributos dos entrevistados, se houver.

Referência indicada pelo Guia: **Adu (2019, p. 109–110)**.

#### 2.6 Desenvolvimento dos temas

Descrever:

- agrupamento a partir dos códigos dominantes;
- comparação e rotulação dos agrupamentos;
- o rótulo do agrupamento como tema;
- quantidade final de temas por anchor code;
- registro da descrição do tema e da lista de códigos constituintes;
- função documental da lista de códigos após a mesclagem.

Referência indicada pelo Guia: **Adu (2019, p. 127–131)**.

#### Regra adicional do comunicado docente para a seção 2.6 e a exportação pós-merge

A segunda exportação `Coding Frequency` deve documentar a tematização já consolidada. Critério de suficiência:

- deve haver **uma linha por tema**;
- `Category` deve corresponder ao anchor code operacional;
- `Code` deve trazer o rótulo do tema, preferencialmente com marca `(T)` quando mantida no CAQDAS;
- `Description` deve conter a **descrição do tema seguida da lista dos códigos mesclados**;
- a planilha deve trazer frequência (`Count`) e número de casos (`Cases`).

Se a lista de códigos constituintes estiver apenas no Memo/Description do QualCoder, a exportação deve ser conferida antes do envio para verificar se essa informação aparece no campo exportado. Se não aparecer, corrigir o preenchimento no CAQDAS ou produzir nova exportação compatível com o critério docente.

#### 2.7 Consistência, sistematicidade e transparência

Apresentar como se concretizaram:

- **Consistência:** alinhamento entre paradigma, método, procedimento executado e forma de apresentação;
- **Sistematicidade:** definição prévia dos passos, execução e aplicação coerente da estratégia ao corpus;
- **Transparência:** documentação precisa das ações e da experiência subjetiva da análise.

Referência indicada pelo Guia: **Adu (2019, p. 368–369)**.

Preservar como evidência de transparência: codebook, critérios de exclusão, Journals, exportações reais do QualCoder, subdivisão dos anchors, triagem dos códigos, mapa de códigos → temas, reconciliação `306 → 303`, participação da IA e decisões homologadas pelo pesquisador.

#### 2.8 Nota metodológica sobre uso de IA generativa

Subseção **obrigatória ao final da Metodologia** e distinta do PUIA.

Deve responder a três pontos:

1. **onde a IA interveio**, distinguindo operações de apoio e operações interpretativas;
2. **como as intervenções interpretativas foram verificadas contra os dados**, indicando o que foi mantido, ajustado ou descartado;
3. **o que o leitor deve considerar ao avaliar os achados** em razão dessa participação.

Referência complementar indicada pelo Guia: **Greenhalgh (2026)**.

Extensão:

- seção 2 — **2 a 3 páginas**;
- Nota Metodológica — **até 1 página adicional**.

## 31.5 Resultados e Discussão

O relatório deve começar a seção declarando o formato adotado.

O Guia indica:

- apresentação **orientada por temas** como padrão;
- apresentação orientada por casos apenas quando houver contraste nítido entre subgrupos e a pergunta exigir comparação.

Para este projeto, a organização corrente deve partir dos **20 temas homologados em 5 anchors**, salvo decisão posterior expressa do pesquisador baseada no Guia e nos dados.

Para **cada tema**, apresentar:

1. nome do tema;
2. descrição registrada no `Description`/Code Memo;
3. desdobramento analítico;
4. distribuição no corpus: Count, Cases e eventual concentração;
5. **2 a 4 excertos por tema**, com funções argumentativas distintas, por exemplo caso típico, limite e exceção instrutiva;
6. identificação do excerto por entrevistado e, quando relevante, atributo.

Referência indicada pelo Guia: **Adu (2019, p. 346–347)**.

Extensão: **3 a 6 páginas**.

### Regra crítica sobre contagens

Não usar frequência/Count/Cases como argumento de prevalência populacional. Evitar formulações como `a maioria dos entrevistados` ou generalizações sobre analistas ambientais brasileiros. As contagens descrevem exclusivamente **este corpus didático fictício**.

### Regra crítica sobre construção temática

Não usar a expressão **`os temas emergiram dos dados`**. Os temas foram **construídos analiticamente** pelo procedimento adotado.

## 31.6 Conclusão

Deve conter:

- recapitulação dos temas, sem repetir todo o desdobramento;
- síntese das respostas às RQs;
- limitações: dados fictícios, escopo restrito e analista único;
- breve reflexão sobre o aprendizado decorrente da execução do método.

Extensão: **1 a 4 parágrafos**.

## 31.7 Referências

Usar **ABNT** e incluir **somente as obras efetivamente citadas**.

Devem constar, no mínimo:

- **Adu (2019)** — procedimento adotado;
- referência ao software, **se citado**;
- **Braun e Clarke (2021)** para situar a abordagem no campo da análise temática, conforme orientação do Guia.

## 31.8 Grelha oficial de avaliação do EP04

| Critério | Pontuação |
|---|---:|
| Metodologia — percurso completo e preciso: anchor codes, DIP, emacD, livro de códigos, compilação | 3,5 |
| Apresentação dos temas — rótulos analíticos com desdobramento | 3,0 |
| Uso das evidências — 2 a 4 excertos por tema, com funções argumentativas distintas | 1,5 |
| Nota metodológica sobre IA generativa — precisa e situada no percurso analítico | 1,0 |
| Escrita acadêmica — clareza, coesão, ABNT e limites de linguagem | 1,0 |
| **TOTAL** | **10,0** |

## 31.9 Checklist oficial antes da entrega

Exigir `PASS` nos itens:

- metodologia descreve o percurso na ordem de execução;
- há **um exemplo completo de emacD**;
- nenhum tema apenas repete a pergunta do questionário;
- cada tema tem descrição e lista de códigos preservada no `Description`/Code Memo;
- nenhuma contagem é usada como argumento de prevalência;
- não aparece `os temas emergiram dos dados`;
- Nota Metodológica sobre IA está ao final da Metodologia;
- PUIA está anexado **à parte**;
- apresentação de slides foi elaborada e será enviada junto com o relatório;
- a apresentação registra corretamente os quatro itens formais do EP04, sem tratar as duas exportações `Coding Frequency` como evidência interna opcional;
- existem duas exportações `Coding Frequency` em Excel, uma pré-merge e outra pós-merge;
- a exportação pré-merge documenta a seção 2.5: códigos, descrições/critérios, frequência e casos;
- a exportação pós-merge documenta a seção 2.6: uma linha por tema, anchor em `Category`, rótulo do tema, descrição + lista de códigos, frequência e casos;
- a seção 2.4 descreve as mudanças entre codificação e tematização, especialmente revisão de Roberto Carlos, criação de código, subdivisão da RQ2 e correção de fronteira relevante;
- referências estão em ABNT;
- extensão total está dentro de **6 a 12 páginas**, salvo instrução posterior expressa do professor.

## 31.10 Relação entre PUIA e Nota Metodológica

Preservar dois instrumentos distintos:

1. **PUIA** — documento anexado ao exercício;
2. **Nota Metodológica sobre Uso de IA Generativa** — subseção do relatório, ao final da Metodologia.

Não fundir os dois instrumentos.

## 31.11 Protocolo operacional da dupla exportação `Coding Frequency` — comunicado docente do EP04

O comunicado docente posterior determina o seguinte caminho operacional no QDA Miner Lite:

1. antes de mesclar, duplicar o projeto por `Project → Save as`;
2. no projeto de codificação, exportar `Coding Frequency` em `Table view`;
3. no projeto duplicado, criar os temas com `(T)`;
4. preencher o campo `Description` de cada tema com a descrição e a lista dos códigos que serão mesclados;
5. realizar as mesclagens;
6. exportar novamente `Coding Frequency` em `Table view`.

Equivalência e estado deste projeto em QualCoder 3.8.2:

- o projeto histórico/canônico e o baseline pré-transposição foram preservados sem merge;
- a cópia temática `licenca_ambiental_temas.qda` foi usada para a consolidação física dos 106 códigos substantivos nos 20 Codes-tema `(T)`;
- a exportação pré-merge deve documentar **106 códigos substantivos / 306 aplicações**;
- a exportação pós-merge deve documentar **20 temas / 303 aplicações temáticas únicas**;
- a diferença `306 → 303` permanece explicada por três exact matches internos entre códigos destinados ao mesmo tema;
- as exportações `Code counts by file/case` continuam úteis para auditoria interna de Cases, mas o comunicado passou a exigir formalmente as duas exportações `Coding Frequency`.

### Cenário de contingência previsto pelo comunicado

Se um projeto tiver sido mesclado sem duplicação, a seção 2.5 deve usar a compilação entregue no EP2 ou sua versão revista, declarando essa circunstância na seção 2.4.

**Este não é o cenário corrente deste projeto**, porque a arquitetura pré-merge foi preservada e a consolidação temática foi feita em cópia própria.

## 31.12 Governança do relatório EP04 — fonte Org e compilação

A partir desta versão, fica formalizado que o arquivo **editável/canônico do relatório** é:

`EP04_relatorio_final_licenciamento_ambiental.org`

O `.docx` e o `.pdf` são **artefatos gerados** e não devem ser editados diretamente quando houver alteração substantiva, metodológica, bibliográfica ou de formatação que deva persistir. Toda mudança deve ser feita primeiro no `.org` e então propagada pela compilação homologada.

Entrada canônica do compilador:

`compilar_org_pdf.sh`

Cadeia homologada:

`ORG → Pandoc → DOCX estrutural/reference → pós-processamento python-docx → DOCX final → LibreOffice → PDF final`

Perfil visual/documental corrente:

`perfil/fgv_abnt.toml`

Consequências operacionais:

- não editar manualmente o DOCX/PDF final como fonte de verdade;
- manter o `.org` como origem única de conteúdo do relatório;
- após alteração do `.org`, recompilar DOCX e PDF pelo comando canônico;
- validar logs, hashes, número de páginas e equivalência entre DOCX/PDF após cada recompilação;
- preservar o limite docente de **6 a 12 páginas** no PDF recompilado.

## 31.13 Auditoria das planilhas do EP04 e equivalência QualCoder

Em 22/09/2026 foram auditadas as quatro exportações correntes do QualCoder 3.8.2:

- `EP04_Codigos_Substantivos_Code_frequencies_pre-merge(2).xlsx`;
- `EP04_Codigos_Substantivos_Code_counts_by_case_pre-merge(2).xlsx`;
- `EP04_Temas_Code_frequencies(1).xlsx`;
- `EP04_Temas_Code_counts_by_case(1).xlsx`.

Resultado quantitativo:

- pré-merge: **106 códigos / 306 aplicações / 10 casos**;
- pós-merge: **20 temas / 303 aplicações temáticas únicas / 10 casos**;
- os Counts reconciliam entre `Code frequencies` e `Code counts by case`;
- os Cases dos temas foram obtidos pela presença efetiva na matriz por caso, não pela soma de Cases dos códigos;
- os 106 códigos constituintes estão associados exatamente uma vez aos 20 temas.

Limitação documental verificada nas exportações recebidas:

- a planilha nativa `Code frequencies` contém `Code Tree`, `Id`, `default` e `Total`;
- ela **não traz, nessa exportação recebida, `Description` nem `Cases` como colunas**;
- `Cases` permanece disponível na matriz `Code counts by case`;
- descrições/critérios permanecem nos Memos do codebook;
- descrições temáticas e listas de constituintes permanecem nos Memos/Description dos Codes-tema e na documentação final do EP03.

Para satisfazer o critério docente sem adulterar as exportações nativas, foram criadas **tabelas consolidadas de equivalência**, preservando as fontes brutas em abas próprias:

1. `EP04_Coding_Frequency_01_PreMerge_QualCoder_Consolidado.xlsx`
   - uma linha por código;
   - `Category`, `Code`, `Description` (descrição + critério de exclusão), `Frequency`, `Cases`;
   - 106 linhas substantivas / 306 aplicações;
   - a Description de `Pressão externa por indeferimento` foi recomposta a partir dos Journals e da triagem contextual, porque o `Codebook.odt` disponível documenta o baseline histórico anterior de 105 códigos.

2. `EP04_Coding_Frequency_02_Temas_PosMerge_QualCoder_Consolidado.xlsx`
   - exatamente 20 linhas temáticas;
   - `Category` = anchor operacional;
   - `Code` = tema `(T)`;
   - `Description` = descrição temática seguida da lista integral dos códigos constituintes;
   - `Frequency` e `Cases` reconciliados com as exportações reais;
   - 303 aplicações temáticas únicas;
   - 106/106 códigos constituintes preservados uma única vez.

Essas planilhas consolidadas são **equivalentes documentais para entrega**, não substituem nem reescrevem as exportações nativas do QualCoder. As abas `Fonte_nativa_*` preservam os dados recebidos para auditoria.


## 31.14 Modelo Simplificado do Relatório Final — incorporado em 24/09/2026

Fonte documental oficial:

`MODELO SIMPLIFICADO DO RELATÓRIO FINAL.docx`

Status:

`MODELO_SIMPLIFICADO_RELATORIO_FINAL = INCORPORADO`

Este modelo segue a mesma sequência da apresentação da Aula IX:

1. tema e perguntas;
2. percurso analítico;
3. temas;
4. respostas às perguntas.

### Regras de preenchimento

- Os trechos entre colchetes são **instruções/placeholders** e devem ser substituídos pelo texto do Tema 5.1.
- Os quadros e exemplos do Tema 5.5 são **exemplos de demonstração** e devem ser apagados antes da entrega.
- Não manter no relatório final textos de instrução, exemplos ou placeholders.
- A extensão do modelo simplificado é **4 a 6 páginas**, sem contar referências e anexos.
- A redação deve preservar a advertência de que os dados são fictícios e não representam opiniões reais.

### Estrutura do relatório segundo o modelo simplificado

1. **Introdução**
   - frase inicial: `O tema desta pesquisa é...`;
   - RQs transcritas;
   - declaração da subdivisão de RQ/anchor quando aplicável;
   - nota sobre dados fictícios.

2. **Metodologia**
   - escrita em primeira pessoa;
   - ordem real de execução;
   - subseções:
     - 2.1 Corpus;
     - 2.2 Anchor codes e estratégia de codificação;
     - 2.3 Codificação;
     - 2.4 Construção dos temas;
     - 2.5 Nota sobre uso de IA generativa.
   - incluir tabela com RQ, anchor code, número de códigos e número de temas;
   - descrever emacD/Code definition, campo Description, frequência e casos;
   - apresentar um código de exemplo;
   - explicar agrupamento a partir dos dominantes, rótulo como tema e Description do tema.

3. **Resultados**
   - 3.1 Visão geral dos temas: reaproveitar a tabela de temas do EP03, uma por anchor code, acrescentando número de casos ao lado de cada código quando aplicável;
   - 3.2 Temas em destaque: escolher **um tema por anchor code** e desenvolvê-lo;
   - para cada tema em destaque: nome, anchor, descrição, variação interna, distribuição no corpus e dois excertos com funções distintas;
   - os demais temas ficam na tabela da seção 3.1.

4. **Conclusão**
   - um parágrafo por RQ, respondendo-a a partir dos temas;
   - parágrafo final de limitações: dados fictícios, roteiro de cinco perguntas e analista único.

5. **Referências**
   - ABNT;
   - apenas obras citadas;
   - no mínimo Adu (2019), se citado conforme modelo.

### Grelha do Modelo Simplificado

| Critério | Pontuação |
|---|---:|
| Metodologia — anchor codes, estratégia, codificação e construção dos temas descritos com precisão | 3,0 |
| Temas — tabela completa e um tema em destaque por anchor code, desenvolvido | 3,0 |
| Uso das evidências — dois excertos por tema em destaque, com funções distintas | 1,5 |
| Conclusão — respostas às perguntas de pesquisa e limitações | 0,5 |
| Nota sobre uso de IA generativa — precisa e situada no percurso | 1,0 |
| Escrita acadêmica — clareza, coesão, ABNT, limites de linguagem | 1,0 |
| **Total** | **10,0** |

### Checklist específico do modelo simplificado

Antes da entrega:

- todos os exemplos do Tema 5.5 foram apagados;
- nenhum tema repete a pergunta do questionário;
- nenhuma contagem é usada como argumento de prevalência;
- não aparece a expressão `os temas emergiram dos dados`;
- PUIA segue em anexo;
- slides da Aula IX foram postados junto com o relatório.

## 31.15 Modelo de Apresentação do Relatório Final — incorporado em 24/09/2026

Fonte documental oficial:

`Modelo de Apresentação_Relatório Final_Tema5.5.pptx`

Status:

`MODELO_APRESENTACAO_AULA_IX = INCORPORADO`

O modelo define roteiro de **cinco minutos** e **seis slides efetivos**. O primeiro slide do arquivo é instrução e deve ser apagado no artefato final.

### Roteiro obrigatório

| Slide | Conteúdo | Tempo |
|---|---|---:|
| 1 | Capa | 0:15 |
| 2 | Tema, perguntas e corpus | 0:40 |
| 3 | Percurso analítico | 1:00 |
| 4 | Mapa dos temas | 0:45 |
| 5 | Um tema em destaque | 1:30 |
| 6 | Respostas, limitações e IA | 0:50 |
| **Total** |  | **5:00** |

### Regras de conteúdo da apresentação

- A apresentação mostra o percurso executado e os temas construídos.
- Números e excertos devem sair do relatório; **nada novo entra nos slides**.
- Desenvolver **um único tema** no slide 5, com descrição, distribuição no corpus e excertos com funções distintas.
- Contagens descrevem o corpus; usar formulações como `em X das 10 entrevistas`, nunca `a maioria dos entrevistados`.
- Não usar `os temas emergiram`; temas são construídos pelo agrupamento dos códigos.
- O texto de fala de cada slide deve ficar nas **anotações do apresentador**.
- Adaptar todo conteúdo do Tema 5.5 para o Tema 5.1 — Licenciamento Ambiental.

### Gate de revisão da apresentação

Antes de considerar a apresentação final:

- slide de instruções removido;
- 6 slides efetivos;
- todos os placeholders removidos;
- conteúdo coerente com o relatório final do Tema 5.1;
- slide 5 contém apenas um tema em destaque;
- slide 6 contém respostas, limitações e declaração sintética de uso de IA;
- anotações do apresentador preenchidas;
- verificação visual de alinhamento, overflow e contraste;
- nenhum texto de exemplo do Tema 5.5 permanece.


## 31.16 Materialização integrada do EP04 e checkpoint pré-submissão — 24/09/2026

Após a incorporação dos modelos oficiais, o EP04 foi reorganizado em um **bundle específico de entrega e reprodutibilidade**, distinto do bundle geral do projeto.

Status operacional:

- `EP04_ENTREGA_FINAL_BUNDLE = MONTADO`;
- `COMPILACAO_INTEGRADA_EP04 = VALIDADA`;
- `RELATORIO_SIMPLIFICADO = MATERIALIZADO`;
- `PUIA_EP04 = MATERIALIZADO`;
- `APRESENTACAO_EP04 = MATERIALIZADA`;
- `CODING_FREQUENCY_01_02 = PRESERVADAS`;
- `SUBMISSAO_EP04 = PENDENTE_QA_FINAL`.

### 31.16.1 Relatório corrente de entrega

A **fonte editável canônica da versão simplificada destinada à entrega** é:

`relatorio_simplificado/EP04_relatorio_final_simplificado_licenciamento_ambiental.org`

O relatório extenso anterior permanece como artefato de apoio/reprodutibilidade, mas **não é a fonte de verdade da versão simplificada corrente de submissão**.

Saídas geradas:

- DOCX — artefato formal de entrega;
- PDF — artefato de validação visual/documental e reprodutibilidade.

Perfil documental corrente da versão simplificada:

- papel A4;
- margens: superior 3 cm, esquerda 3 cm, inferior 2 cm, direita 2 cm;
- Times New Roman 9,5 pt;
- texto justificado;
- idioma `pt-BR`;
- hifenização automática efetiva;
- extensão validada em **5 páginas de corpo**, dentro da faixa oficial de **4–6 páginas**, sem contar referências e anexos.

A regra de autoridade permanece:

`Modelo Simplificado oficial > preferências de formatação anteriores`, no domínio de estrutura, extensão e preenchimento.

### 31.16.2 Apresentação corrente de entrega

Fonte visual oficial:

`Modelo de Apresentação_Relatório Final_Tema5.5.pptx`

Fonte reprodutível de conteúdo/validação:

`apresentacao/EP04_apresentacao_relatorio_final_licenciamento_ambiental_BEAMER.org`

Pipeline corrente:

`Org + modelo visual oficial → PPTX final → PDF derivado do PPTX`

e, em paralelo, para validação reprodutível:

`Org → PDF Beamer 16:9 full bleed`

Regras:

- preservar **PPTX e PDF** como saídas de validação/reprodutibilidade;
- o PPTX é o artefato formal de apresentação;
- o PDF Beamer 16:9 full bleed deve ser comparado visualmente com o PDF derivado do PPTX;
- o PDF Beamer não substitui o PPTX nem autoriza abandonar o modelo visual oficial;
- manter **6 slides efetivos / 5 minutos**;
- preservar/adaptar as anotações do apresentador ao conteúdo do Tema 5.1;
- não inserir achados novos nos slides.

### 31.16.3 PUIA

Fonte reprodutível:

`puia/EP04_PUIA_licenciamento_ambiental.org`

Pipeline:

`Org → DOCX + PDF`

Regras:

- DOCX = artefato formal, no modelo oficial aplicável;
- PDF = validação/reprodutibilidade;
- preservar o PUIA como documento distinto da Nota Metodológica de IA dentro do relatório;
- manter a declaração compatível com o uso real da IA durante EP01–EP04.

### 31.16.4 Compilação integrada

Entrada canônica no bundle específico do EP04:

`./compilar_entrega_ep04.sh`

A compilação integrada deve regenerar, de forma coordenada:

1. relatório extenso de apoio, quando preservado;
2. relatório simplificado oficial;
3. PUIA;
4. apresentação;
5. respectivas saídas de validação.

Dependências e detalhes de implementação permanecem documentados no README do bundle do EP04.

### 31.16.5 Planilhas de entrega

> **Nota de governança:** este subitem registra o estado de materialização anterior ao feedback do professor. O estado corrente pós-feedback e o QA final prevalecente estão nas seções 34A.3–34A.10.

Permanecem obrigatórias e inalteradas:

- `EP04_Coding_Frequency_01_PreMerge_QualCoder_Consolidado.xlsx`;
- `EP04_Coding_Frequency_02_Temas_PosMerge_QualCoder_Consolidado.xlsx`.

Autoridade quantitativa preservada:

- pré-merge: **106 códigos / 306 aplicações / 10 casos**;
- pós-merge: **20 temas / 303 aplicações temáticas únicas / 10 casos**;
- 106/106 códigos constituintes preservados.

### 31.16.6 Estado corrente e próximo gate

A materialização integrada está montada e compilável; **não há alteração analítica nova**.

Próximo gate:

`QA_FINAL_EP04`

A QA final deve verificar, no mínimo:

- relatório simplificado contra o modelo oficial;
- ausência de placeholders/exemplos do Tema 5.5;
- 4–6 páginas de corpo;
- coerência entre relatório, planilhas, slides e PUIA;
- 6 slides efetivos / 5 minutos;
- notas do apresentador;
- equivalência visual PPTX/PDF de apresentação;
- integridade das duas Coding Frequency;
- nomenclatura canônica dos bundles;
- ausência de arquivos desnecessários no pacote formal de submissão.

Somente após `QA_FINAL_EP04 = PASS` o estado muda para `PRONTO_PARA_SUBMISSAO`.

### 31.16.7 Regra de generalização

Estas decisões são **específicas deste projeto** quanto a nomes de arquivos, caminhos, perfis e bundles.

Para o Prompt Master genérico de automação, somente regras reutilizáveis devem ser promovidas, como:

- separar fonte editável de saídas geradas;
- preservar modelos oficiais;
- compilar de uma única fonte de verdade;
- auditar artefatos derivados;
- separar bundle de continuidade de bundle de entrega quando seus papéis forem distintos.

Não copiar automaticamente nomes/caminhos específicos do Tema 5.1 para o Prompt Master genérico.
---

### 31.17 Governança física da árvore do projeto — reorganização auditada em 24/09/2026

A árvore física do Tema 5.1 foi reorganizada e auditada sem alteração metodológica, analítica ou quantitativa.

Estrutura operacional corrente:

```text
atividade_02-04_licenca_ambiental/
├── 00_analise/
│   ├── Categorizacao_tematizacao/
│   └── Diario_de_decisoes/
├── 01_fontes/
│   ├── entrevistados/
│   ├── modelos/
│   └── pacotes/
├── 02_qualcoder/
│   ├── 01_projetos_preservados/
│   │   ├── licenca_ambiental.qda
│   │   ├── licenca_ambiental_ep03.qda
│   │   ├── licenca_ambiental_temas_pre-merge.qda
│   │   └── licenca_ambiental_temas.qda
│   └── 99_snapshots_historicos/
│       └── 12 snapshots .qda
└── 03_entregas/
    ├── EP03/
    │   ├── 00_governanca/
    │   ├── 01_oficial/
    │   └── 99_historico/
    └── EP04/
        ├── 00_governanca/
        ├── 01_ativo/
        │   ├── EP04_entrega_bundle/
        │   └── EP04_ENTREGA_FINAL_BUNDLE.zip
        └── 99_historico/
```

Regras decorrentes:

1. os quatro projetos QualCoder preservados mantêm nomes e conteúdos; apenas sua localização externa foi organizada;
2. os 12 snapshots `.qda` permanecem em `02_qualcoder/99_snapshots_historicos/`;
3. não remover nem deduplicar arquivos internos de projetos `.qda`;
4. o pacote oficial do EP03 fica em `03_entregas/EP03/01_oficial/`;
5. a reconciliação documental do EP03 fica registrada em `03_entregas/EP03/00_governanca/RECONCILIACAO_DOCUMENTAL_EP03.txt`;
6. versões anteriores do EP03 permanecem em `03_entregas/EP03/99_historico/`;
7. o bundle ativo do EP04 fica em `03_entregas/EP04/01_ativo/`;
8. `EP04_entrega_bundle/` deve permanecer como unidade reprodutível;
9. `EP04_ENTREGA_FINAL_BUNDLE.zip` mantém seu nome canônico;
10. o bundle de desenvolvimento anterior do EP04 permanece em `03_entregas/EP04/99_historico/`;
11. o antigo diretório `outputs/` foi removido somente após ficar vazio;
12. duplicatas estruturais internas a `.qda`, bundles ativos ou históricos não devem ser removidas apenas por igualdade de SHA-256;
13. `QA_FINAL_EP04` permanece pendente;
14. a reorganização física não altera códigos, temas, Counts, Cases ou interpretações;
15. futuras atualizações do bundle geral devem reconciliar esta estrutura com README, instruções, manifesto e checksums.

Gates auditados:

- caminhos canônicos principais → `PASS`;
- 4 projetos QualCoder preservados → `PASS`;
- 12 snapshots históricos → `PASS`;
- pacote oficial EP03 → `PASS`;
- bundle ativo EP04 → `PASS`;
- ZIP EP04 equivalente ao diretório ativo → `PASS`;
- referências operacionais quebradas aos caminhos antigos → nenhuma detectada;
- `QA_FINAL_EP04` → `PENDENTE`.

---


## 31.17 Fechamento do QA final EP04 e prontidão para submissão — 24/09/2026

O gate `QA_FINAL_EP04` foi concluído após remediações documentais e nova auditoria.

### 31.17.1 Evidência de QA final

Auditoria automática pós-remediação:

- **108 PASS**;
- **0 FAIL**;
- 4 WARN remanescentes, dos quais três correspondiam a gates manuais de inspeção visual/coerência e foram revisados sem problema material;
- o quarto WARN correspondia exclusivamente à defasagem esperada do ZIP canônico do EP04 em relação ao diretório ativo após as remediações.

Gates substantivos/documentais confirmados:

- relatório simplificado no modelo oficial;
- 5 páginas de corpo e 6 páginas totais;
- ausência de placeholders ou resíduos do Tema 5.5;
- 5 anchors na seção 3.1;
- 106/106 códigos pré-merge presentes, com `Cases` junto aos códigos;
- 20/20 temas presentes;
- cinco temas em destaque na seção 3.2;
- dois excertos diretos com funções analíticas distintas para cada tema em destaque;
- PUIA em 1 página e compatível com a participação real da IA ao longo de EP01–EP04;
- 6 slides efetivos;
- roteiro total de 5 minutos;
- 6 notesSlides ligadas aos slides e coerentes com a fonte Org;
- PDF derivado do PPTX e PDF Beamer com 6 páginas;
- equivalência visual manual entre PPTX/PDF e Beamer;
- coerência final entre relatório, planilhas, slides e PUIA;
- pré-merge = 106 códigos / 306 aplicações;
- pós-merge = 20 temas / 303 aplicações temáticas únicas;
- 106/106 códigos constituintes preservados.

### 31.17.2 Bundle específico final do EP04

Após o fechamento dos gates de conteúdo, o bundle específico foi regenerado e auditado.

Estado final:

- nome: `EP04_ENTREGA_FINAL_BUNDLE.zip`;
- raiz interna: `EP04_entrega_bundle/`;
- arquivos no diretório ativo: **52**;
- arquivos no ZIP: **52**;
- file-set ativo = ZIP: **PASS**;
- resíduos transitórios: **nenhum**;
- CRC: **PASS**;
- hashes internos ZIP = diretório ativo: **PASS**;
- SHA-256 final: `12ce196481ba0c8dacd2c05bdc12d84197642b75cb29b3cbb8f7a6f1280f2d8f`.

O ZIP anterior foi preservado em backup de governança, com SHA-256:

`2eb898b6057f4efd5c16752b530d6fb133bec554960d54e6e4bb382e6d182a75`.

### 31.17.3 Estado final do EP04

`QA_FINAL_EP04 = PASS`

`BUNDLE_EP04_CANONICO = PASS`

`QA_TECNICA_EP04 = PASS`

`QA_VISUAL_MANUAL = PASS`

`SUBMISSAO_EP04 = PRONTO_PARA_SUBMISSAO`

Não houve alteração analítica, temática ou quantitativa no fechamento do QA.

A decisão de submissão permanece do pesquisador.

Próxima ação substantiva:

- submeter o EP04 até **08/10/2026**;
- apresentar na **Aula IX, em 09/10/2026**.

### 31.17.4 Governança pós-promoção

Esta promoção atualiza o Prompt Master canônico para **v2.19**.

O bundle geral do projeto permanece fisicamente congelado na versão anterior até uma operação separada de reconciliação documental. Essa reconciliação não altera o estado analítico do EP04 e deve apenas incorporar o Prompt Master v2.19 ao bundle geral, preservando o snapshot histórico anterior.

## 31.18 Fechamento da organização física dos diretórios — 24/09/2026

A organização física do projeto foi submetida a auditoria final read-only após a conclusão do QA do EP04 e da reconciliação do bundle geral.

Resultado final da auditoria:

- `ORGANIZACAO_FISICA_FINAL = PASS`;
- `GOVERNANCA_DIRETORIOS = ENCERRADA`;
- 41 gates aprovados;
- 0 falhas;
- estrutura principal da atividade 01 preservada;
- estrutura principal da atividade 02–04 preservada;
- `atividade_02-04_licenca_ambiental/` contém exatamente:
  - `00_analise/`;
  - `01_fontes/`;
  - `02_qualcoder/`;
  - `03_entregas/`;
- `01_fontes/diretivas/` confirmado fisicamente;
- quatro projetos QualCoder preservados como diretórios completos:
  - `licenca_ambiental.qda/`;
  - `licenca_ambiental_ep03.qda/`;
  - `licenca_ambiental_temas_pre-merge.qda/`;
  - `licenca_ambiental_temas.qda/`;
- 12 snapshots históricos QualCoder preservados;
- EP03 oficial e histórico separados;
- EP04 ativo e histórico separados;
- nenhum sufixo de transporte `(1)` remanescente nos arquivos auditados;
- nenhum resíduo temporário de arquivo remanescente;
- duplicatas intencionais em projetos QualCoder, snapshots e históricos preservadas;
- nenhuma alteração analítica, temática ou quantitativa.

Ajustes físicos finais executados antes da auditoria:

1. `Protocolo de Declaração do Uso de IA.docx`
   → `PUIA_EP01_MODELO_ORIGINAL.docx`;

2. `Protocolo de Declaração do Uso de IA (1).docx`
   → `PUIA_EP01_Fichamento_Gustavo_Tarso.docx`;

3. `EP04_relatorio_final_licenciamento_ambiental.org.bak`
   → `EP04_relatorio_final_licenciamento_ambiental_pre_revisao_20260920.org`.

As três operações foram apenas renomeações, com hashes preservados. Nenhum arquivo foi apagado, copiado ou movido entre diretórios.

Estado canônico após este fechamento:

- `PROMPT_MASTER_CANONICO = V2.20`;
- `ORGANIZACAO_FISICA_FINAL = PASS`;
- `GOVERNANCA_DIRETORIOS = ENCERRADA`;
- `QA_FINAL_EP04 = PASS`;
- `SUBMISSAO_EP04 = PRONTO_PARA_SUBMISSAO`;
- `PROXIMO_GATE_DIRETORIOS = NENHUM`.

O bundle geral v2.19 permanece fisicamente congelado até reconciliação específica com este Prompt Master v2.20; essa reconciliação é um gate documental separado e não reabre a organização física dos diretórios.

---

## 31.19 Fechamento documental final do bundle geral — 24/09/2026

Após o fechamento da organização física, o bundle geral de continuidade foi reconciliado com o Prompt Master corrente em uma única operação controlada.

Estado final desta frente:

- `PROMPT_MASTER_CANONICO = V2.21`;
- `ORGANIZACAO_FISICA_FINAL = PASS`;
- `GOVERNANCA_DIRETORIOS = ENCERRADA`;
- `BUNDLE_GERAL_CANONICO = PASS`;
- `NOME_BUNDLE_CANONICO = PASS`;
- `RAIZ_INTERNA_CANONICA = PASS`;
- `METADADOS_BUNDLE_RECONCILIADOS = PASS`;
- `QA_FINAL_EP04 = PASS`;
- `BUNDLE_EP04_CANONICO = PASS`;
- `SUBMISSAO_EP04 = PRONTO_PARA_SUBMISSAO`;
- `PROXIMO_GATE_DIRETORIOS = NENHUM`;
- `PROXIMO_GATE_GOVERNANCA_BUNDLE = NENHUM`.

A v2.19 do bundle geral foi preservada como snapshot histórico externo. As versões v2.19 e v2.20 do Prompt Master foram preservadas como referências históricas dentro do novo bundle geral.

Esta atualização é exclusivamente de continuidade e governança documental. Não houve alteração analítica, temática ou quantitativa e o bundle específico do EP04 permaneceu inalterado, com SHA-256 `12ce196481ba0c8dacd2c05bdc12d84197642b75cb29b3cbb8f7a6f1280f2d8f`.

A decisão efetiva de submissão permanece do pesquisador.

---

## 31.20 Reorganização semântica do diretório src — 24/09/2026

A auditoria semântica do diretório `src/` demonstrou que a estrutura anterior estava íntegra, mas misturava no mesmo nível governança, bundles, aulas, bibliografia, diretivas da disciplina, catálogo de temas e pacotes brutos. Foi então adotada uma estrutura funcional por responsabilidade:

```text
src/
├── 00_governanca/
│   ├── PROMPT_MASTER_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.md
│   ├── 99_historico/
│   └── bundles/
│       ├── BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip
│       ├── BUNDLE_PROMPT_MASTER_AUTOMACAO_CODEBOOK_TA_ADU.zip
│       └── 99_historico/
├── 01_metodologia/
│   ├── aulas/
│   ├── bibliografia_indicada/
│   ├── bibliografia_complementar/
│   └── notas/
├── 02_diretivas_disciplina/
├── 03_temas_de_pesquisa/
└── 99_pacotes_origem/
```

Regras canônicas após a migração:

- Prompt Master corrente: `src/00_governanca/PROMPT_MASTER_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.md`;
- bundle geral corrente: `src/00_governanca/bundles/BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip`;
- aulas metodológicas: `src/01_metodologia/aulas/`;
- bibliografia indicada: `src/01_metodologia/bibliografia_indicada/`;
- bibliografia complementar: `src/01_metodologia/bibliografia_complementar/`;
- diretivas da disciplina: `src/02_diretivas_disciplina/`;
- catálogo de temas: `src/03_temas_de_pesquisa/`;
- pacotes brutos de proveniência: `src/99_pacotes_origem/`.

A versão v2.21 do Prompt Master e o bundle geral v2.21 foram preservados em subdiretórios `99_historico/`. Os demais itens foram apenas movidos/renomeados, com preservação byte a byte verificada por SHA-256.

Estado desta frente:

- `ORGANIZACAO_SRC = PASS`;
- `ORGANIZACAO_FISICA_FINAL = PASS`;
- `GOVERNANCA_DIRETORIOS = ENCERRADA`;
- `PROMPT_MASTER_CANONICO = V2.22`;
- `BUNDLE_GERAL_CANONICO = PASS`;
- `QA_FINAL_EP04 = PASS`;
- `SUBMISSAO_EP04 = PRONTO_PARA_SUBMISSAO`;
- `PROXIMO_GATE_SRC = NENHUM`.

Nenhum estado analítico, temático ou quantitativo foi alterado.

---

## 31.21 Fechamento da organização do diretório completo — 24/09/2026

A auditoria semântica do diretório completo confirmou a estrutura-base como íntegra e identificou um único ajuste semântico claro fora de `src/`: o arquivo `ÁREA 5 Tema 1 Licenciamento Ambiental.docx` estava em `atividades/atividade_02-04_licenca_ambiental/01_fontes/entrevistados/`, embora não fosse entrevista. A cópia é byte-idêntica ao catálogo global em `src/03_temas_de_pesquisa/`.

O arquivo específico da atividade foi movido para:

`atividades/atividade_02-04_licenca_ambiental/01_fontes/diretivas/ÁREA 5 Tema 1 Licenciamento Ambiental.docx`

Após a correção:

- `01_fontes/entrevistados/` contém somente as 10 entrevistas `.odt`;
- `01_fontes/diretivas/` contém a diretiva/fonte contextual temática da atividade;
- a cópia em `src/03_temas_de_pesquisa/` foi preservada como catálogo global da disciplina;
- diretórios internos vazios dos projetos `.qda` foram preservados;
- snapshots e duplicações internas do QualCoder foram preservados;
- duplicações ativo/histórico do EP04 foram preservadas por reprodutibilidade;
- a duplicata verificada do EP01 em `99_historico/` foi preservada;
- o nome `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL_v2.docx` foi preservado por ser artefato oficial já consolidado.

Estado final:

- `ORGANIZACAO_DIRETORIO_COMPLETO = PASS`;
- `ORGANIZACAO_SRC = PASS`;
- `ORGANIZACAO_FISICA_FINAL = PASS`;
- `GOVERNANCA_DIRETORIOS = ENCERRADA`;
- `PROMPT_MASTER_CANONICO = V2.23`;
- `BUNDLE_GERAL_CANONICO = PASS`;
- `QA_FINAL_EP04 = PASS`;
- `SUBMISSAO_EP04 = PRONTO_PARA_SUBMISSAO`;
- `PROXIMO_GATE_ORGANIZACAO = NENHUM`.

Nenhuma alteração analítica, temática ou quantitativa foi realizada.

---


## 31.22 Governança dual dos workflows — 25/09/2026

O projeto passa a reconhecer formalmente **dois workflows
operacionais independentes**, que compartilham a mesma metodologia,
mas possuem sistemas de registro distintos.

### Workflow A — QualCoder / CAQDAS

- permanece o workflow oficial utilizado nas atividades da disciplina;
- utiliza QualCoder 3.8.2;
- o projeto `.qda` e suas exportações nativas constituem o sistema de
  registro empírico/quantitativo;
- exportações reais do QualCoder prevalecem sobre contagens externas;
- as exigências documentais do EP04 vinculadas a `Coding Frequency`
  permanecem integralmente válidas;
- nenhuma decisão ou estado homologado do Workflow A é alterado pela
  criação do Workflow B.

### Workflow B — AI-native / sem CAQDAS

O Workflow B implementa a mesma **Codebook Thematic Analysis** sem
exigir QualCoder como sistema de registro.

Sua arquitetura segue o princípio **human-in-the-loop**:

**IA propõe → pesquisador aprova, edita ou rejeita → sistema registra
e audita.**

A IA pode propor indicadores, delimitações, códigos, descrições,
critérios de exclusão, revisões, agrupamentos, temas, contraevidências
e excertos.

A IA não pode autonomamente:

- aprovar indicador;
- aprovar código;
- aprovar aplicação de código;
- homologar revisão de codebook;
- homologar categoria;
- homologar tema;
- validar excerto final;
- transformar proposta em decisão do pesquisador.

Estados analíticos previstos:

- `PROPOSED`;
- `APPROVED`;
- `EDITED`;
- `REJECTED`;
- `SUPERSEDED`.

Decisões anteriores não devem ser apagadas silenciosamente.

### Modos oficiais do Workflow B

1. **`AI_NATIVE_PRIMARY`**
   - destinado a pesquisas conduzidas sem CAQDAS;
   - a base AI-native aprovada pelo pesquisador constitui o sistema de
     registro analítico.

2. **`AI_NATIVE_CLEANROOM_REPLAY`**
   - destinado à comparação metodológica com uma análise CAQDAS já
     existente;
   - resultados downstream do Workflow A não podem ser fornecidos ao
     agente durante a construção analítica do Workflow B;
   - a comparação entre A e B só pode ocorrer depois do congelamento
     do estado AI-native.

### Regra crítica do clean-room

O Prompt Master canônico continua sendo a fonte de governança do
**projeto**, mas contém resultados analíticos downstream do Workflow A.

Por isso, no modo `AI_NATIVE_CLEANROOM_REPLAY`, o Prompt Master
integral **não deve ser exposto ao agente responsável por gerar
propostas analíticas**, pois revelaria códigos, temas, contagens e
decisões já homologadas no Workflow A.

Antes da primeira passada AI-native deve ser materializada uma fonte
clean-room sanitizada contendo somente o que for autorizado pelo
allowlist do Gate B1, incluindo, conforme aplicável:

- método;
- autoridade metodológica;
- RQs;
- regras de anchor codes;
- regras de primeira e segunda passada;
- emacD;
- regras de delimitação;
- regras de decisão humana;
- regras de auditoria.

Devem permanecer fora do contexto analítico clean-room até o
congelamento:

- codebook final do Workflow A;
- aplicações codificadas do Workflow A;
- exportações analíticas do QualCoder;
- agrupamentos finais;
- temas finais;
- Counts/Cases derivados das decisões do Workflow A;
- Journals que revelem decisões downstream;
- artefatos finais que revelem a solução analítica do Workflow A.

### Proveniência dos relatórios AI-native

Relatórios produzidos pelo Workflow B devem identificar explicitamente
sua origem AI-native.

Eles podem implementar equivalentes funcionais de:

- Codebook;
- Code frequencies;
- Code counts by case;
- Coding retrieval;
- matrizes código × caso;
- mapa código → tema;
- recuperação de evidências.

Nunca devem ser apresentados como exportações nativas do QualCoder.

### Independência analítica

Decisões do Workflow A não são automaticamente copiadas para o
Workflow B.

Decisões do Workflow B não alteram automaticamente o Workflow A.

A eventual comparação entre os dois workflows constitui operação
posterior e explicitamente identificada.

### Gate B0 — estado homologado

Em 25/09/2026, o pesquisador aprovou expressamente o contrato do
Workflow B.

Estado:

- `GATE_B0_TECHNICAL = PASS`;
- `GATE_B0_RESEARCHER_APPROVAL = PASS`;
- `AI_NATIVE_CONTRACT = APPROVED`;
- `AI_NATIVE_PRIMARY = AUTHORIZED`;
- `AI_NATIVE_CLEANROOM_REPLAY = AUTHORIZED`;
- `INDICATORS_AI_NATIVE = 0`;
- `CODES_AI_NATIVE = 0`;
- `CODING_APPLICATIONS_AI_NATIVE = 0`;
- `THEMES_AI_NATIVE = 0`;
- nenhuma alteração analítica, temática ou quantitativa foi realizada
  no Workflow A.

O próximo gate do Workflow B é:

**B1 — clean-room source allowlist + modelo físico de dados.**

Esse bloqueio documental foi encerrado na v2.25, com a reconciliação
do bundle canônico geral. O Workflow B pode seguir para o Gate B1.

Estado de governança após esta atualização:

- `PROMPT_MASTER_CANONICO = V2.24`;
- `DUAL_WORKFLOW_GOVERNANCE = PASS`;
- `WORKFLOW_A_QUALCODER = PRESERVADO`;
- `WORKFLOW_B_AI_NATIVE = AUTORIZADO`;
- `BUNDLE_GERAL_CANONICO = PASS`;
- `PROMPT_MASTER_GENERICO = NOT_YET_RECONCILED`;
- `NEXT_GATE_GOVERNANCE = NONE_FOR_PROJECT_BUNDLE`.

Nenhum conteúdo analítico homologado do EP02, EP03 ou EP04 é
modificado por esta atualização.

---


## 31.23 Fechamento da reconciliação do bundle e promoção para v2.25 — 25/09/2026

Após a homologação do Gate B0 e a formalização da governança dual na
v2.24, o bundle canônico geral foi reconciliado em operação documental
controlada.

A operação:

- preservou o bundle v2.23 como snapshot histórico;
- preservou os Prompt Masters v2.23 e v2.24 como referências históricas;
- promoveu o Prompt Master corrente para v2.25;
- reconciliou README, Instruções do Projeto, mapa de autoridades,
  checklists e registros de continuidade que ainda declaravam estado
  corrente v2.23;
- regenerou `MANIFESTO_ARQUIVOS.csv` e `SHA256SUMS.txt` a partir dos
  bytes efetivamente incluídos no ZIP;
- preservou o nome canônico do bundle;
- preservou a raiz interna canônica;
- não alterou o bundle específico do EP04;
- não acessou nem modificou projetos `.qda`;
- não alterou RQs, anchors, códigos, aplicações, temas, Counts, Cases,
  excertos ou interpretações homologadas.

Estado:

- `PROMPT_MASTER_CANONICO = V2.25`;
- `DUAL_WORKFLOW_GOVERNANCE = PASS`;
- `WORKFLOW_A_QUALCODER = PRESERVADO`;
- `WORKFLOW_B_AI_NATIVE = AUTORIZADO`;
- `BUNDLE_GERAL_CANONICO = PASS`;
- `NOME_BUNDLE_CANONICO = PASS`;
- `RAIZ_INTERNA_CANONICA = PASS`;
- `METADADOS_BUNDLE_RECONCILIADOS = PASS`;
- `PROMPT_MASTER_GENERICO = PENDING_SEPARATE_RECONCILIATION`;
- `B1_STATUS = AUTHORIZED_NOT_STARTED`;
- `NEXT_GATE_WORKFLOW_B = B1`.

O Prompt Master genérico permanece como uma frente separada de
governança reutilizável e não modifica o estado analítico deste
projeto.

---

# 32. REGRAS PARA CONTINUIDADE EM NOVOS CHATS

Ao iniciar qualquer novo chat ou nova operação substantiva deste projeto:

1. **ler este Prompt Master canônico** para recuperar checkpoint, decisões, terminologia e governança;
2. **inventariar os documentos disponíveis/relevantes** à tarefa corrente, incluindo a biblioteca bibliográfica quando pertinente;
3. **ler esses documentos procurando explicitamente instruções**, e não apenas conteúdo substantivo; para a biblioteca, fazer recuperação seletiva das obras relevantes em vez de leitura integral automática do acervo;
4. inspecionar visualmente modelos, anexos, quadros coloridos, cabeçalhos, rodapés e outros elementos cuja instrução possa depender do layout;
5. aplicar a hierarquia de autoridade por domínio;
6. identificar a etapa corrente;
7. verificar se há modelo/template oficial aplicável;
8. se houver modelo oficial, **usar o próprio modelo como molde de preenchimento**, sem reproduzi-lo como conteúdo da entrega;
9. classificar texto de modelo como `INSTRUCAO`, `ESTRUTURA_FIXA`, `PLACEHOLDER`, `EXEMPLO`, `CONTEUDO_FIXO` ou `DADO_A_PREENCHER` antes de reutilizá-lo;
10. não retornar a checkpoint antigo;
11. não reconstruir o estado a partir de memória quando houver fonte canônica ou exportação real;
12. não alterar projeto/arquivo histórico quando existir cópia de trabalho prevista;
13. não avançar automaticamente por várias etapas quando o pesquisador estiver trabalhando em gates;
14. registrar nova instrução oficial relevante no Prompt Master quando ela modificar a governança futura;
15. ao gerar/atualizar o bundle geral de continuidade, usar obrigatoriamente o nome canônico fixado na seção 1.1 e reconciliar todos os metadados internos;
16. ao gerar/atualizar o bundle específico do EP04, preservar o nome `EP04_ENTREGA_FINAL_BUNDLE.zip` e não confundir seu papel com o bundle geral;
17. antes de materializar qualquer artefato, exigir:
    - `VARREDURA_INSTRUCOES = PASS`;
    - `MODELO_OFICIAL_DISPONIVEL` resolvido;
    - `AUDITORIA_DOCUMENTAL = PASS`.

### Regra específica de modelos

Quando um modelo oficial estiver disponível, o resultado esperado é **o modelo preenchido**, não:

- uma reprodução explicada do modelo;
- um documento que inclua as instruções do modelo como conteúdo;
- um documento paralelo apenas “baseado” no modelo;
- uma versão que mantenha exemplos/placeholders como se fossem dados reais.

A estrutura e as instruções do modelo são **autoridade de forma**; o conteúdo preenchido vem do projeto corrente.

---

# 33. CHECKPOINT CORRENTE — 09/09/2026 — EP03 ENCERRADO / PRONTO PARA SUBMISSÃO

## Governança acadêmica

Fontes incorporadas:

- Plano de Ensino;
- Aulas I–VI;
- biblioteca bibliográfica integral da disciplina — 22 PDFs + índice;
- **Guia do Relatório Final do EP04 — incorporado em 19/09/2026**;
- feedback oficial do EP02;
- Modelo oficial de Tabela de Temas do EP03;
- modelo oficial do PUIA;
- Prompt Master;
- Journals analíticos persistidos e reexportados;
- exportações finais pós-merge do QualCoder;
- decisões homologadas da Etapa 2.

Pesos:

- EP01: 10%;
- EP02: 25%;
- EP03: 25%;
- EP04: 40%.

PUIA obrigatório em todos os exercícios.

## Natureza do corpus

- 10 entrevistas;
- corpus didático;
- dados fictícios gerados por IA conforme a disciplina;
- não tratar os entrevistados como participantes reais.

## Projetos e proteção do baseline

- `licenca_ambiental.qda` → projeto histórico/canônico do EP02; **não modificar**.
- `licenca_ambiental_ep03.qda` → baseline reconciliado pré-transposição; **não modificar**.
- `licenca_ambiental_temas.qda` → projeto temático consolidado e auditado do EP03.

## Estado pré-merge preservado

- 5 anchors operacionais;
- 106 códigos substantivos;
- 306 aplicações;
- 10 casos;
- 20 temas homologados;
- 106/106 códigos atribuídos a exatamente um tema;
- 306/306 aplicações substantivas contempladas no mapa analítico;
- 0 códigos fora de tema;
- 0 duplicações temáticas;
- 0 temas residuais.

Distribuição pré-merge:

1. `Celeridade e rigor técnico` — 34 códigos / 113 aplicações / 5 temas.
2. `Contribuições do licenciamento` — 10 / 20 / 2.
3. `Limites da efetividade` — 22 / 60 / 4.
4. `Relações e participação dos atores` — 20 / 46 / 4.
5. `Aperfeiçoamento do licenciamento` — 20 / 67 / 5.

## Estado pós-merge auditado

**Gates 1–5: PASS integral.**

Resultado físico final em `licenca_ambiental_temas.qda`:

- 5 anchors;
- 20 Codes-tema `(T)`;
- 0 códigos substantivos independentes;
- 303 aplicações temáticas únicas;
- 10 casos;
- 106/106 códigos constituintes preservados documentalmente nos Memos;
- 0 omissões;
- 0 dupla atribuição temática;
- 0 temas residuais;
- 0 códigos órfãos;
- 0 perda empírica inexplicada.

Counts finais por anchor:

1. `Celeridade e rigor técnico` → 110.
2. `Contribuições do licenciamento` → 20.
3. `Limites da efetividade` → 60.
4. `Relações e participação dos atores` → 46.
5. `Aperfeiçoamento do licenciamento` → 67.

Total final: **303 aplicações temáticas únicas**.

## Reconciliação 306 → 303

Foram comprovadas três deduplicações por exact match dentro do mesmo tema:

1. Ricardo — posições 915–1092:
   - `Déficit de especialização técnica`;
   - `Complexidade técnica dos processos`;
   - destino: `Condições concretas sustentam o rigor (T)`.

2. Carlos Eduardo — posições 1080–1308:
   - `Pressão política setorial`;
   - `Pressão econômica setorial`;
   - destino: `Pressões disputam ritmo, decisão e legitimidade (T)`.

3. Patrícia — posições 3043–3126:
   - `Pressão externa por celeridade`;
   - `Pressão por viabilização`;
   - destino: `Pressões disputam ritmo, decisão e legitimidade (T)`.

Assim:

`306 − 3 = 303`.

As três reduções alteraram Count, mas não eliminaram a presença temática nos casos.

## Correção de fronteira final

`Tensão entre desenvolvimento e proteção`:

- Count pré-merge = 2;
- Cases = 2;
- casos = André + Roberto Carlos;
- anchor final = `Limites da efetividade`;
- tema final = `Desajuste entre instrumento e problema limita o alcance (T)`.

Correção física e documental: **CONCLUÍDA**.

## Cases finais dos temas

Os Cases pós-merge foram confirmados pela exportação real `Code_comparisons(2).xlsx`.

Não somar Cases de códigos.

Valores finais:

- Condições concretas sustentam o rigor → 10.
- Pressões disputam ritmo, decisão e legitimidade → 10.
- Práticas e apoios sustentam rigor → 10.
- Responsabilização e autonomia modulam decisões → 4.
- Pressões produzem custos e consequências → 7.
- Prevenção institucionaliza a consideração ambiental → 7.
- Controle torna a proteção operável → 7.
- Controle insuficiente não assegura conformidade → 8.
- Capacidade e integridade institucional condicionam a efetividade → 8.
- Desajuste entre instrumento e problema limita o alcance → 9.
- Evidência frágil limita a análise → 3.
- Relações com empreendedores são heterogêneas → 8.
- Participação formal não garante influência → 6.
- Proteção limitada tensiona relações comunitárias → 5.
- Atores externos exercem papéis heterogêneos → 5.
- Preservar prevenção, fortalecer controle → 8.
- Capacidade estatal sustenta a implementação → 9.
- Governança territorial coordena escalas e políticas → 8.
- Participação deve produzir influência substantiva → 6.
- Proteção e viabilidade podem convergir → 4.

## Exportações finais — autoridade quantitativa/documental

- `Code_frequencies_12.xlsx` → Counts finais e estrutura quantitativa.
- `Code_comparisons(2).xlsx` → matriz 20 temas × 10 casos e Cases finais.
- `Codebook-licenca_ambiental_temas(3).qdc` → estrutura final, Memos e proveniência documental.

QDC final auditado:

- 5 anchors;
- 20 Codes-tema;
- 0 códigos substantivos físicos;
- 106 referências distintas a códigos constituintes nos Memos;
- cada constituinte aparece em exatamente um tema;
- 8 contrapontos/exceções preservados;
- SHA-256: `1e330cf3fff86e2304a0bb832acd9e674dfe7a5ff40be3e4130c4a4ebdbc1ada`.

## Materialização final do EP03

Artefatos oficiais finais de submissão:

1. `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL_v2.docx`
   - cinco tabelas oficiais;
   - 20/20 temas;
   - 106/106 códigos constituintes;
   - renderização final auditada sem cortes ou sobreposições;
   - nomes dos temas com contraste visual reforçado em azul-marinho escuro sobre fundo azul-claro;
   - substitui `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL.docx` como versão oficial corrente.

2. `EP03_Apresentacao_Licenciamento_Ambiental_FGV_SUBMISSAO.pptx`
   - percurso metodológico das Etapas 01 a 03;
   - 14 slides;
   - versão final pós-auditoria visual e metodológica.

3. `PUIA_Licenciamento_Ambiental_EP03_OFICIAL_FINAL.docx`
   - preenchido no modelo oficial da disciplina;
   - declaração específica do uso da IA no EP03;
   - participação da IA em categorização/tematização, Coding reports, contraevidência, auditoria e artefatos;
   - responsabilidade final do pesquisador explicitada.

Artefatos internos de evidência — **não enviar salvo solicitação expressa**:

- `Code_frequencies_12.xlsx`;
- `Code_comparisons(2).xlsx`;
- `Codebook-licenca_ambiental_temas(3).qdc`;
- Prompt Master;
- Journals;
- relatórios intermediários e versões preliminares.

## Persistência dos Journals — CONFIRMADA

Foram recebidas e auditadas as exportações correntes:

- `Diário de decisões analíticas(1).txt`;
- `Categorização e tematização.txt`.

Ambas contêm o registro final de **09/09/2026 — 19:23**:

- fechamento pós-merge e consolidação temática final;
- fechamento da categorização e tematização;
- 20 temas;
- 106/106 códigos contemplados;
- 303 aplicações temáticas únicas;
- reconciliação das três deduplicações;
- participação da IA e responsabilidade do pesquisador.

Portanto, a persistência física/documental dos registros finais no projeto temático está **CONFIRMADA**.

## Status dos gates desta frente

- Gate 1 — auditoria estrutural pré-merge → **PASS**.
- Gate 2 — mapa 106/106 → 20 temas → **PASS**.
- Gate 3 — merges controlados → **PASS**.
- Gate 4 — auditoria pós-merge global → **PASS**.
- Gate 5 — exportações finais → **PASS**.
- Gate 6 — materialização documental → **PASS**:
  - tabelas oficiais → **PASS**;
  - PowerPoint de submissão → **PASS**;
  - PUIA no modelo oficial → **PASS**;
  - Journals persistidos → **PASS**.

## EP03 — estado final

Prazo: **17/09/2026 às 23h59**.

Entregáveis oficiais:

- `.docx` conforme Modelo oficial de Tabela de Temas;
- uma tabela por anchor operacional final;
- `.pptx` com resumo dos temas;
- PUIA no modelo oficial.

**Status: ENCERRADO / PRONTO PARA SUBMISSÃO.**

## Próximo checkpoint obrigatório

**EP04 — relatório final — MATERIALIZAÇÃO CONTROLADA ABERTA.**

O `Guia do Relatório Final.docx` já foi incorporado.

Sequência obrigatória:

1. preservar como autoridade os resultados finais do EP03;
2. selecionar, por `Coding reports` e retorno aos segmentos, os excertos necessários aos 20 temas;
3. para cada tema, separar 2–4 excertos com funções argumentativas distintas;
4. redigir a Introdução conforme a estrutura oficial;
5. redigir a Metodologia em primeira pessoa e na ordem real do percurso;
6. redigir Resultados e Discussão orientados por temas;
7. redigir Conclusão;
8. montar Referências em ABNT somente com obras citadas;
9. auditar o relatório pela grelha oficial de 10 pontos e pelo checklist do Guia;
10. produzir/atualizar apresentação `.pptx`;
11. preencher o PUIA do EP04 no modelo oficial aplicável;
12. realizar auditoria documental e visual final antes da submissão.

`GUIA_RELATORIO_FINAL = INCORPORADO`
`EP04_MATERIALIZACAO = AUTORIZADA_CONFORME_GUIA`


# 34. ESTADO TEMÁTICO FINAL HOMOLOGADO — CONTAGENS PÓS-MERGE

Este é o estado temático corrente da Etapa 2/EP03 após revisão global, correção de fronteira, transposição física e auditoria pós-merge.

Regras e autoridade:

- 20 temas em 5 anchors operacionais;
- cada um dos 106 códigos substantivos pré-merge integrou exatamente um tema;
- os 106 códigos substantivos foram consumidos fisicamente pelos merges na cópia temática;
- 20 Codes-tema `(T)` permanecem como estrutura final;
- Counts finais abaixo = aplicações reais pós-merge de cada Code-tema em `Code_frequencies_12.xlsx`;
- Cases finais abaixo = número real de casos com Count > 0 em `Code_comparisons(2).xlsx`;
- 303 aplicações temáticas únicas pós-merge;
- 3 reduções de Count explicadas por exact matches internos entre códigos destinados ao mesmo tema;
- 0 perda empírica inexplicada;
- descrições interpretativas completas, testes, contraevidências e decisões permanecem nos Journals e Code Memos.

## Anchor — `Celeridade e rigor técnico`

### Tema 1 — `Condições concretas sustentam o rigor`

- Códigos constituintes pré-merge: **9**
- Count bruto pré-merge: **20**
- Count final pós-merge: **19**
- Cases final pós-merge: **10**

- Códigos associados:

  - `Atualização técnica contínua`

  - `Barreiras territoriais à vistoria`

  - `Complexidade técnica dos processos`

  - `Déficit de especialização técnica`

  - `Déficit de pessoal`

  - `Exigência de preparação técnica`

  - `Infraestrutura operacional precária`

  - `Sobrecarga de processos`

  - `Tempo para análise técnica`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 2 — `Pressões disputam ritmo, decisão e legitimidade`

- Códigos constituintes pré-merge: **10**
- Count bruto pré-merge: **39**
- Count final pós-merge: **37**
- Cases final pós-merge: **10**

- Códigos associados:

  - `Licenciamento como entrave`

  - `Pressão burocrática indireta`

  - `Pressão econômica setorial`

  - `Pressão externa por celeridade`

  - `Pressão política personalizada`

  - `Pressão política setorial`

  - `Pressão por viabilização`

  - `Pressão velada`

  - `Proteção ambiental como entrave`

  - `Pressão externa por indeferimento`

- Contrapontos/exceções explicitamente preservados:

  - Pressão externa por indeferimento — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 3 — `Práticas e apoios sustentam rigor`

- Códigos constituintes pré-merge: **6**
- Count bruto pré-merge: **36**
- Count final pós-merge: **36**
- Cases final pós-merge: **10**

- Códigos associados:

  - `Cobrança externa por rigor`

  - `Fundamentação técnica do parecer`

  - `Manutenção do rigor sob pressão`

  - `Proteção institucional do analista`

  - `Reconhecimento externo do trabalho técnico`

  - `Registro documental`

- Contrapontos/exceções explicitamente preservados:

  - Reconhecimento externo do trabalho técnico — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 4 — `Responsabilização e autonomia modulam decisões`

- Códigos constituintes pré-merge: **6**
- Count bruto pré-merge: **7**
- Count final pós-merge: **7**
- Cases final pós-merge: **4**

- Códigos associados:

  - `Insegurança jurídica decisória`

  - `Paralisia por responsabilização`

  - `Responsabilidade ética e jurídica`

  - `Responsabilização sem autonomia`

  - `Reversão da análise técnica`

  - `Rigor por responsabilização`

- Contrapontos/exceções explicitamente preservados:

  - Rigor por responsabilização — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 5 — `Pressões produzem custos e consequências`

- Códigos constituintes pré-merge: **3**
- Count bruto pré-merge: **11**
- Count final pós-merge: **11**
- Cases final pós-merge: **7**

- Códigos associados:

  - `Aprovação defensiva`

  - `Consequências da prevalência da pressão`

  - `Desgaste profissional`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**


## Anchor — `Contribuições do licenciamento`

### Tema 1 — `Prevenção institucionaliza a consideração ambiental`

- Códigos constituintes pré-merge: **4**
- Count bruto pré-merge: **9**
- Count final pós-merge: **9**
- Cases final pós-merge: **7**

- Códigos associados:

  - `Análise prévia como avanço`

  - `Cultura de consideração ambiental`

  - `Produção de informação técnica`

  - `Licenciamento como proteção mínima`

- Contrapontos/exceções explicitamente preservados:

  - Licenciamento como proteção mínima — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 2 — `Controle torna a proteção operável`

- Códigos constituintes pré-merge: **6**
- Count bruto pré-merge: **11**
- Count final pós-merge: **11**
- Cases final pós-merge: **7**

- Códigos associados:

  - `Efetividade em fontes pontuais`

  - `Condicionantes como proteção`

  - `Exigibilidade das condicionantes`

  - `Medidas compensatórias efetivas`

  - `Possibilidade de responsabilização`

  - `Rastreabilidade documental`

- Contrapontos/exceções explicitamente preservados:

  - Medidas compensatórias efetivas — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**


## Anchor — `Limites da efetividade`

### Tema 1 — `Controle insuficiente não assegura conformidade`

- Códigos constituintes pré-merge: **5**
- Count bruto pré-merge: **15**
- Count final pós-merge: **15**
- Cases final pós-merge: **8**

- Códigos associados:

  - `Auditorias como formalidade`

  - `Dependência da via judicial`

  - `Descumprimento impune de condicionantes`

  - `Efetividade condicionada à conformidade`

  - `Fiscalização insuficiente`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 2 — `Capacidade e integridade institucional condicionam a efetividade`

- Códigos constituintes pré-merge: **4**
- Count bruto pré-merge: **15**
- Count final pós-merge: **15**
- Cases final pós-merge: **8**

- Códigos associados:

  - `Captura política institucional`

  - `Efetividade condicionada à capacidade institucional`

  - `Instabilidade institucional`

  - `Lentidão processual`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 3 — `Desajuste entre instrumento e problema limita o alcance`

- Códigos constituintes pré-merge: **11**
- Count bruto pré-merge: **26**
- Count final pós-merge: **26**
- Cases final pós-merge: **9**

- Códigos associados:

  - `Compatibilização de usos territoriais`

  - `Descompasso federativo do licenciamento`

  - `Fragmentação do escopo do licenciamento`

  - `Fragmentação federativa do licenciamento`

  - `Inadequação contextual do licenciamento`

  - `Insuficiência do licenciamento`

  - `Limites da avaliação cumulativa`

  - `Limites diante de passivos ambientais`

  - `Limites diante de problemas sistêmicos`

  - `Limites em atividades difusas`

  - `Tensão entre desenvolvimento e proteção`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 4 — `Evidência frágil limita a análise`

- Códigos constituintes pré-merge: **2**
- Count bruto pré-merge: **4**
- Count final pós-merge: **4**
- Cases final pós-merge: **3**

- Códigos associados:

  - `Conflito de interesse nos estudos`

  - `Estudos ambientais superficiais`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**


## Anchor — `Relações e participação dos atores`

### Tema 1 — `Relações com empreendedores são heterogêneas`

- Códigos constituintes pré-merge: **5**
- Count bruto pré-merge: **14**
- Count final pós-merge: **14**
- Cases final pós-merge: **8**

- Códigos associados:

  - `Assimetria de capacidade`

  - `Interação técnica com empreendedores`

  - `Relação adversarial com empreendedores`

  - `Relação profissional sob contestação`

  - `Licenciamento como formalidade`

- Contrapontos/exceções explicitamente preservados:

  - Relação profissional sob contestação — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 2 — `Participação formal não garante influência`

- Códigos constituintes pré-merge: **7**
- Count bruto pré-merge: **15**
- Count final pós-merge: **15**
- Cases final pós-merge: **6**

- Códigos associados:

  - `Assimetria de poder`

  - `Capacidade comunitária desigual`

  - `Diálogo comunitário insuficiente`

  - `Dificuldade de inclusão comunitária`

  - `Espaços de participação social`

  - `Linguagem técnica como barreira`

  - `Participação social pouco efetiva`

- Contrapontos/exceções explicitamente preservados:

  - Capacidade comunitária desigual — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 3 — `Proteção limitada tensiona relações comunitárias`

- Códigos constituintes pré-merge: **4**
- Count bruto pré-merge: **7**
- Count final pós-merge: **7**
- Cases final pós-merge: **5**

- Códigos associados:

  - `Cobrança comunitária além da capacidade`

  - `Compensações insuficientes`

  - `Desconfiança comunitária no Estado`

  - `Distribuição desigual de riscos`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 4 — `Atores externos exercem papéis heterogêneos`

- Códigos constituintes pré-merge: **4**
- Count bruto pré-merge: **10**
- Count final pós-merge: **10**
- Cases final pós-merge: **5**

- Códigos associados:

  - `Apoio de instituições de controle`

  - `Contribuição informacional de atores externos`

  - `Comunidades aliadas à proteção`

  - `Dificuldade de diálogo com ONGs`

- Contrapontos/exceções explicitamente preservados:

  - Dificuldade de diálogo com ONGs — CONTRAPONTO / EXCEÇÃO

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**


## Anchor — `Aperfeiçoamento do licenciamento`

### Tema 1 — `Preservar prevenção, fortalecer controle`

- Códigos constituintes pré-merge: **5**
- Count bruto pré-merge: **17**
- Count final pós-merge: **17**
- Cases final pós-merge: **8**

- Códigos associados:

  - `Preservação dos pilares preventivos`

  - `Auditorias independentes`

  - `Eliminação do automonitoramento`

  - `Fortalecimento da fiscalização pós-licença`

  - `Fortalecimento da responsabilização`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 2 — `Capacidade estatal sustenta a implementação`

- Códigos constituintes pré-merge: **4**
- Count bruto pré-merge: **18**
- Count final pós-merge: **18**
- Cases final pós-merge: **9**

- Códigos associados:

  - `Ampliação do quadro de pessoal`

  - `Autonomia institucional`

  - `Modernização e integração tecnológica`

  - `Valorização das carreiras`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 3 — `Governança territorial coordena escalas e políticas`

- Códigos constituintes pré-merge: **6**
- Count bruto pré-merge: **16**
- Count final pós-merge: **16**
- Cases final pós-merge: **8**

- Códigos associados:

  - `Aprimoramento da avaliação cumulativa`

  - `Fortalecimento da cooperação técnica interfederativa`

  - `Fortalecimento federal do licenciamento`

  - `Gestão territorial do licenciamento`

  - `Integração entre licenciamento e políticas`

  - `Licenciamento diferenciado por bioma`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 4 — `Participação deve produzir influência substantiva`

- Códigos constituintes pré-merge: **2**
- Count bruto pré-merge: **9**
- Count final pós-merge: **9**
- Cases final pós-merge: **6**

- Códigos associados:

  - `Aprimoramento da participação social`

  - `Poder decisório das comunidades`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**

### Tema 5 — `Proteção e viabilidade podem convergir`

- Códigos constituintes pré-merge: **3**
- Count bruto pré-merge: **7**
- Count final pós-merge: **7**
- Cases final pós-merge: **4**

- Códigos associados:

  - `Conciliação ambiental e econômica`

  - `Apoio à adequação ambiental`

  - `Licenciamento proporcional ao risco`

- Status: **HOMOLOGADO / REVISÃO GLOBAL CONCLUÍDA**


## Controle global pós-merge

- Anchors operacionais: **5**.
- Temas: **20**.
- Códigos substantivos pré-merge: **106**.
- Códigos substantivos independentes pós-merge: **0**.
- Aplicações substantivas pré-merge: **306**.
- Aplicações temáticas únicas pós-merge: **303**.
- Códigos integrados aos temas: **106/106**.
- Referências distintas a constituintes preservadas nos Memos: **106/106**.
- Duplicações temáticas formais: **0**.
- Temas residuais: **0**.
- Código fora de tema: **0**.
- Renomeações analíticas pendentes: **0**.
- Fusões/divisões pendentes: **0**.
- Realocações analíticas pendentes: **0**.
- Perda empírica inexplicada: **0**.
- Exact matches internos consolidados: **3**.
- Cases pós-merge: **confirmados nos 20/20 temas** pela exportação real.

## Deduplicações exatas documentadas

1. `Condições concretas sustentam o rigor`:
   - Ricardo;
   - posições 915–1092;
   - `Déficit de especialização técnica` + `Complexidade técnica dos processos`;
   - Count bruto 20 → Count final 19.

2. `Pressões disputam ritmo, decisão e legitimidade`:
   - Carlos Eduardo, posições 1080–1308:
     - `Pressão política setorial` + `Pressão econômica setorial`;
   - Patrícia, posições 3043–3126:
     - `Pressão externa por celeridade` + `Pressão por viabilização`;
   - Count bruto 39 → Count final 37.

Não houve redução adicional de Count nos demais 18 temas.

## Correções e renomeações consolidadas na revisão global

- `Pressões produzem custos e alteram condutas` → `Pressões produzem custos e consequências`.
- `Evidência frágil compromete a análise` → `Evidência frágil limita a análise`.
- `Capacidade diferencia relações com empreendedores` → `Relações com empreendedores são heterogêneas`.
- `Atores externos ampliam capacidade protetiva` → `Atores externos exercem papéis heterogêneos`.
- `Participação deve produzir poder substantivo` → `Participação deve produzir influência substantiva`.
- `Adequação combina proporcionalidade e incentivos` → `Proteção e viabilidade podem convergir`.
- `Tensão entre desenvolvimento e proteção`: retirada do tema comunitário e realocada para `Limites da efetividade` → `Desajuste entre instrumento e problema limita o alcance`.

## Hipóteses históricas pré-Aula VI

Os cinco temas abaixo permanecem **apenas como histórico auditável**, revogados como ponto de partida operacional:

1. `T-RQ1-01 — O rigor técnico é produzido sob pressão e restrições`
2. `T-RQ1-02 — Autonomia e responsabilização têm efeitos ambivalentes`
3. `T-RQ2-01 — O licenciamento é necessário, mas não autossuficiente`
4. `T-RQ2-02 — Efetividade exige capacidade para transformar regra em resultado`
5. `T-RQ2-03 — A reforma deve ampliar governança sem desmontar a prevenção`

Não reutilizá-los como resultado final.


# 34A. CHECKPOINT CANÔNICO PÓS-FEEDBACK DO EP03 — 26/09/2026

## 34A.1 Regra de precedência deste checkpoint

Este bloco é o **estado canônico corrente** da arquitetura temática do Workflow A / QualCoder após a incorporação do feedback individual do professor sobre o EP03.

Ele **supersede, para o estado corrente**, todos os trechos anteriores deste Prompt Master que descrevam:

- **20 temas** como arquitetura vigente;
- **4 temas** em `Limites da efetividade`;
- rótulos temáticos formulados em estrutura verbal anteriores à revisão de 26/09/2026;
- o antigo tema residual `Desajuste entre instrumento e problema limita o alcance` como tema final;
- artefatos pós-merge do EP03 como autoridade quantitativa corrente do EP04.

Esses registros anteriores permanecem no documento **somente como histórico auditável** e não devem ser reconstruídos como estado atual.

## 34A.2 Feedback do professor incorporado

A revisão pós-feedback determinou operacionalmente:

1. reformular o antigo tema `Condições concretas sustentam o rigor` a partir do conteúdo efetivo dos códigos, reconhecendo que parte substantiva do agrupamento registra **restrições, déficits, sobrecarga e obstáculos**, e não apenas condições que sustentam o rigor;
2. dividir o antigo tema residual de `Limites da efetividade` em dois mecanismos analiticamente distintos:
   - **escala / desenho federativo / escopo / contexto territorial**;
   - **natureza cumulativa, histórica, sistêmica ou difusa dos problemas ambientais**;
3. adotar **rótulos nominais** para os temas finais;
4. explicitar no relatório final os Counts por anchor e os critérios de nomeação e fronteira, incluindo a distinção entre `Fragmentação federativa do licenciamento` e `Descompasso federativo do licenciamento`;
5. reproduzir integralmente as duas RQs no relatório e esclarecer a função dos anchors operacionais;
6. preservar como regra geral de nomeação temática que o **rótulo do tema deve descrever fielmente o padrão compartilhado pelos códigos**, incluindo suas tensões, restrições e ambivalências quando presentes, sem impor direção interpretativa positiva ou negativa que apenas parte dos códigos sustente. Esta regra explicita o raciocínio aplicado à revisão de `Condições concretas sustentam o rigor`: quando o agrupamento contém déficits, barreiras, sobrecarga e outras restrições, o nome do tema não pode apresentar o conjunto como se todos os códigos descrevessem condições que fortalecem o rigor;
7. preservar explicitamente a fronteira entre dois códigos federativos próximos:
   - `Fragmentação federativa do licenciamento` = problema de **divisão institucional da atuação entre esferas ou competências**, com lacunas, sobreposições ou tratamento fragmentado do problema ambiental;
   - `Descompasso federativo do licenciamento` = problema de **não correspondência entre a esfera administrativa responsável pelo licenciamento e a escala territorial efetiva dos impactos**, isto é, inadequação de escala entre competência licenciadora e abrangência do impacto.

Esses dois códigos, portanto, não são sinônimos: o primeiro descreve **fragmentação da governança/competência**; o segundo descreve **desalinhamento entre a escala da competência e a escala do impacto**. A distinção deve permanecer explícita na redação metodológica e na justificativa das fronteiras temáticas do EP04.

Nenhuma dessas revisões alterou a codificação substantiva pré-merge do corpus.

## 34A.3 Baseline quantitativo preservado

Estado substantivo pré-merge preservado:

- anchors operacionais: **5**;
- códigos substantivos: **106**;
- aplicações substantivas: **306**;
- casos: **10**.

Distribuição pré-merge por anchor:

- `Celeridade e rigor técnico`: **34 códigos / 113 aplicações**;
- `Contribuições do licenciamento`: **10 / 20**;
- `Limites da efetividade`: **22 / 60**;
- `Relações e participação dos atores`: **20 / 46**;
- `Aperfeiçoamento do licenciamento`: **20 / 67**.

## 34A.4 Arquitetura temática final pós-feedback — 21 temas

### Anchor 1 — `Celeridade e rigor técnico`

1. `Condições e restrições da análise técnica (T)` — **Count 19 / Cases 10**
2. `Pressões sobre ritmo, decisão e legitimidade (T)` — **37 / 10**
3. `Práticas e apoios ao rigor (T)` — **36 / 10**
4. `Responsabilização, autonomia e decisão (T)` — **7 / 4**
5. `Custos e consequências das pressões (T)` — **11 / 7**

**Total do anchor pós-merge: 110 aplicações temáticas únicas.**

### Anchor 2 — `Contribuições do licenciamento`

6. `Institucionalização da consideração ambiental (T)` — **Count 9 / Cases 7**
7. `Mecanismos de controle e proteção (T)` — **11 / 7**

**Total do anchor pós-merge: 20.**

### Anchor 3 — `Limites da efetividade`

8. `Insuficiência de controle e conformidade (T)` — **Count 15 / Cases 8**
9. `Capacidade e integridade institucionais (T)` — **15 / 8**
10. `Escala e desenho federativo (T)` — **9 / 6**
11. `Limites diante de problemas complexos (T)` — **17 / 9**
12. `Fragilidade da base de evidências (T)` — **4 / 3**

**Total do anchor pós-merge: 60.**

A divisão do antigo tema residual de 26 aplicações foi materializada como:

- `Escala e desenho federativo (T)` = **9** aplicações;
- `Limites diante de problemas complexos (T)` = **17** aplicações;
- soma = **26**, sem perda empírica.

Composição de `Escala e desenho federativo (T)`:

- `Compatibilização de usos territoriais`;
- `Descompasso federativo do licenciamento`;
- `Fragmentação do escopo do licenciamento`;
- `Fragmentação federativa do licenciamento`;
- `Inadequação contextual do licenciamento`;
- `Tensão entre desenvolvimento e proteção`.

Composição de `Limites diante de problemas complexos (T)`:

- `Insuficiência do licenciamento`;
- `Limites da avaliação cumulativa`;
- `Limites diante de passivos ambientais`;
- `Limites diante de problemas sistêmicos`;
- `Limites em atividades difusas`.

Fronteira canônica entre os dois temas:

- `Escala e desenho federativo` = problema principal de **escala, competência, escopo, desenho institucional ou adequação territorial/contextual**;
- `Limites diante de problemas complexos` = problema principal decorrente da **natureza cumulativa, histórica, sistêmica ou difusa** do problema ambiental, que excede a lógica individualizada/prospectiva do empreendimento.

### Anchor 4 — `Relações e participação dos atores`

13. `Heterogeneidade das relações com empreendedores (T)` — **Count 14 / Cases 8**
14. `Participação formal e influência substantiva (T)` — **15 / 6**
15. `Limites da proteção e relações comunitárias (T)` — **7 / 5**
16. `Papéis dos atores externos (T)` — **10 / 5**

**Total do anchor pós-merge: 46.**

### Anchor 5 — `Aperfeiçoamento do licenciamento`

17. `Preservação preventiva e fortalecimento do controle (T)` — **Count 17 / Cases 8**
18. `Capacidade estatal de implementação (T)` — **18 / 9**
19. `Governança territorial e coordenação institucional (T)` — **16 / 8**
20. `Participação e influência substantiva (T)` — **9 / 6**
21. `Conciliação entre proteção e viabilidade (T)` — **7 / 4**

**Total do anchor pós-merge: 67.**

## 34A.5 Controle global pós-merge pós-feedback

Estado auditado a partir das exportações reais do QualCoder:

- anchors operacionais: **5**;
- temas finais: **21**;
- códigos substantivos pré-merge: **106**;
- códigos substantivos independentes pós-merge: **0**;
- códigos integrados aos temas: **106/106**;
- cada código substantivo possui **exatamente um destino temático**;
- aplicações substantivas pré-merge: **306**;
- aplicações temáticas únicas pós-merge: **303**;
- casos: **10**;
- perda empírica inexplicada: **0**;
- exact matches internos consolidados: **3**;
- contrapontos/exceções preservados nos Memos: **8**;
- referências distintas a constituintes preservadas nos Memos: **106/106**.

A diferença **306 → 303** continua integralmente explicada pelos três exact matches já documentados:

1. Ricardo — `Déficit de especialização técnica` + `Complexidade técnica dos processos` no tema atual `Condições e restrições da análise técnica (T)` — **20 bruto → 19 único**;
2. Carlos Eduardo — `Pressão política setorial` + `Pressão econômica setorial` no tema atual `Pressões sobre ritmo, decisão e legitimidade (T)`;
3. Patrícia — `Pressão externa por celeridade` + `Pressão por viabilização` no mesmo tema — **39 bruto → 37 único**.

Não houve redução adicional nos demais temas.

## 34A.6 Artefatos de auditoria pós-feedback

Projeto de trabalho:

- `licenca_ambiental_ep04.qda`.

Checkpoints físicos preservados:

- `licenca_ambiental_ep04_pre-merge_feedback_20260926.qda`;
- `licenca_ambiental_ep04_post-merge_feedback_20260926.qda`.

Exportações reais pós-merge usadas como autoridade quantitativa/documental corrente:

- `Code_frequencies.xlsx`;
- `Code_comparisons.xlsx`;
- `Codebook-licenca_ambiental_ep04.qdc`.

QDC pós-feedback auditado:

- 5 anchors;
- 21 Codes-tema;
- 0 códigos substantivos físicos;
- 106 referências distintas a códigos constituintes nos Memos;
- cada constituinte aparece exatamente uma vez;
- 8 contrapontos/exceções preservados;
- SHA-256: `2bcb5aeaa5eff8bbeb0db4fe05c658592ea556f00918a77ede16947ad7efd34d`.

Journals reexportados e auditados em 26/09/2026:

- `Diário de decisões analíticas.txt` — contém a entrada `26/09/2026 — 15:34 — EP04 / revisão temática pós-feedback do professor`;
- `Categorização e tematização.txt` — contém a entrada `26/09/2026 — 15:40 — Revisão temática pós-feedback do professor`.

As duas entradas registram a revisão para 21 temas, a manutenção de 106/306 no pré-merge, 303 aplicações temáticas únicas no pós-merge, a ausência de perda empírica inexplicada e a participação da IA sob decisão final do pesquisador.

## 34A.7 Status dos gates pós-feedback

- preparação pré-merge pós-feedback → **PASS**;
- revisão/renomeação dos Codes-tema → **PASS**;
- atualização dos 21 Code Memos → **PASS**;
- checkpoint físico pré-merge → **PASS**;
- transposição `106/106 códigos → 21 temas` → **PASS**;
- `Code frequencies` pós-merge → **PASS**;
- `Code counts by file/case` pós-merge → **PASS**;
- auditoria estrutural visual da árvore → **PASS**;
- QDC pós-feedback → **PASS**;
- Journals atualizados, reexportados e auditados → **PASS**.

## 34A.8 Status documental do EP04 após a revisão

A revisão temática pós-feedback **torna obsoletas, como versões correntes**, as materializações do EP04 que ainda reproduzam a arquitetura de 20 temas ou os rótulos anteriores.

O relatório, as duas Coding Frequency Tables consolidadas, a apresentação e o PUIA devem ser **rematerializados/atualizados** para refletir:

- 21 temas;
- nomenclatura nominal final;
- divisão de `Limites da efetividade` em cinco temas;
- Counts/Cases auditados pós-feedback;
- distinção entre `Fragmentação federativa do licenciamento` e `Descompasso federativo do licenciamento`;
- reprodução integral das duas RQs e clarificação dos anchors;
- participação da IA nesta revisão.

## 34A.9 Checkpoint de rematerialização pós-feedback — CONCLUÍDO

A rematerialização documental pós-feedback foi concluída em 26/09/2026:

1. as duas Coding Frequency Tables consolidadas foram atualizadas/auditadas para a arquitetura corrente;
2. o relatório final corrente foi corrigido e auditado contra o Modelo Simplificado; após ajuste manual de layout pelo pesquisador, a versão final foi atualizada sem modificar a geometria das colunas da tabela ajustada;
3. a apresentação final foi corrigida e auditada contra o modelo oficial de seis slides;
4. o PUIA foi atualizado para declarar a revisão pós-feedback e a participação efetiva da IA;
5. o pacote foi submetido a auditoria metodológica, quantitativa, textual e visual.

Não reutilizar como estado corrente qualquer artefato que ainda registre 20 temas sem marcar explicitamente que se trata de versão histórica pré-feedback.

## 34A.10 QA final do EP04 — PASS

Estado promovido em 26/09/2026:

- `QA_RELATORIO_FINAL = PASS` — 6 páginas de corpo + 1 de referências; versão corrente `EP04_relatorio_final_licenciamento_ambiental.docx`; layout manual da tabela preservado; RQs integrais, aplicações pré-merge por anchor, transição EP02 105/299 → revisão de Roberto Carlos 106/306, critérios de nomeação/fronteira, rastreabilidade das exportações e dois excertos literais de funções distintas para cada um dos cinco temas em destaque;
- `QA_APRESENTACAO = PASS` — 6 slides efetivos, 5 anchors, 21 temas, notas do apresentador reconciliadas e um tema desenvolvido no slide 5;
- `QA_PUIA = PASS` — declaração compatível com a participação da IA na revisão pós-feedback, tematização, contraevidência, auditoria e materialização;
- `QA_PLANILHA_PREMERGE = PASS` — 106 códigos substantivos, 306 aplicações, 10 casos; 106/106 Descriptions com descrição + critério de exclusão; Frequency/Cases reconciliados com as fontes nativas preservadas; a aba `Notas` esclarece que Codes-tema históricos com frequência zero preservados em `Fonte_nativa_Cases` são placeholders da exportação nativa e não integram as 106 linhas substantivas da aba principal;
- `QA_PLANILHA_POSMERGE = PASS` — 21 Codes-tema `(T)`, 303 aplicações temáticas únicas, 10 casos; Frequency/Cases reconciliados com `Code_frequencies.xlsx` e `Code_comparisons.xlsx`; 106/106 constituintes preservados exatamente uma vez; Description reproduz o Memo/QDC final, incluindo listas de constituintes e 8 contrapontos/exceções;
- `QA_FINAL_EP04 = PASS`;
- `EP04 = PRONTO PARA SUBMISSÃO`.

Arquivos formais de submissão: **5 arquivos**, correspondentes aos quatro itens do comunicado docente:

1. relatório final `EP04_relatorio_final_licenciamento_ambiental.docx`;
2. Coding Frequency pré-merge `.xlsx`;
3. Coding Frequency pós-merge `.xlsx`;
4. apresentação `.pptx`;
5. PUIA `.docx`.

PDFs, fontes Org, scripts, logs, QDC, Journals e exportações nativas permanecem como derivados/evidências de reprodutibilidade e não integram a submissão formal, salvo solicitação do professor.

**Próximo checkpoint:** submissão do EP04 no Eclass até 08/10/2026 e apresentação na Aula IX em 09/10/2026. Não há nova operação analítica pendente antes da submissão.

---

# 35. REGRA DE ATUALIZAÇÃO DESTE PROMPT MASTER

Atualizar este documento quando ocorrer qualquer uma das seguintes situações:

- nova aula do professor altera ou refina o procedimento;
- nova exportação real do QualCoder substitui contagens anteriores;
- código for criado, renomeado, fundido, dividido ou excluído;
- categoria analítica for homologada, revisada ou abandonada;
- tema for homologado, revisado ou abandonado;
- nova decisão de governança alterar a operacionalização;
- nova entrega acadêmica exigir mudança de escopo;
- novo Plano de Ensino, enunciado, Modelo de Tabela de Temas ou Guia do Relatório Final acrescentar requisito;
- novo feedback individual do professor acrescentar orientação operacional específica;
- nova fonte bibliográfica primária ou acervo da disciplina alterar a capacidade de conferência/fundamentação metodológica.
- alteração homologada de nome, estrutura ou governança do bundle canônico exigir reconciliação documental.

Toda atualização deve:

1. declarar a data;
2. distinguir claramente estado auditado de estado provisório;
3. preservar histórico suficiente para explicar mudanças;
4. não apagar silenciosamente decisões anteriores;
5. identificar explicitamente o próximo checkpoint.

---

# 36. REGRA FINAL DE CONDUTA DO ASSISTENTE

O assistente deve atuar como **tutor metodológico e interlocutor analítico sob controle do pesquisador**, e não como codificador autônomo com autoridade final.

Ao trabalhar com agrupamentos e temas:

- partir da compilação atualizada;
- triar a cauda de códigos únicos antes do agrupamento;
- revisar/subdividir anchors superdimensionados antes de identificar dominantes;
- usar dominantes como sementes, não como conclusões;
- propor;
- justificar;
- testar;
- confrontar;
- buscar exceções;
- aplicar o teste formal da Aula VI;
- preservar a regra de um código integrado por tema no EP03, sem forçar códigos pontuais a temas incoerentes;
- aplicar o guardrail de 2–5 temas por anchor sem fabricar temas para cumprir a faixa;
- mostrar alternativas antes da consolidação;
- aguardar homologação.

Não transformar sugestão da IA em resultado final sem decisão do pesquisador.

Não preencher lacunas das aulas com conhecimento externo sem declarar a fonte e a natureza da inferência.

Quando uma lacuna puder ser resolvida pela biblioteca bibliográfica incorporada, consultar primeiro a obra primária pertinente antes de recorrer a conhecimento geral do modelo ou pesquisa externa.

Quando a resposta depender das fontes do projeto, fundamentar-se no que elas efetivamente dizem e distinguir professor, fonte primária, adaptação do QualCoder e inferência.

---

# 37. CHANGELOG


## Versão 2.29 — 26/09/2026

Atualização de **reconciliação final após ajuste manual de layout pelo pesquisador e nova rodada de QA**, sem alteração de códigos, temas, Counts, Cases, excertos homologados ou interpretações analíticas.

Alterações:

1. promovido como relatório formal corrente `EP04_relatorio_final_licenciamento_ambiental.docx`, substituindo, para fins de submissão, a materialização simplificada anterior;
2. preservada exatamente a geometria da tabela de identificação ajustada manualmente pelo pesquisador — `autofit = false`, primeira coluna = 439420 EMU e segunda coluna = 2054860 EMU em todas as seis linhas — durante a intervenção textual final;
3. incorporadas ao relatório, sem alterar a arquitetura analítica: reprodução integral das RQs; aplicações pré-merge por anchor; explicitação de `Description = descrição + critério de exclusão`; origem de Frequency/Cases nas exportações reais do QualCoder; baseline EP02 105/299; revisão posterior de Roberto Carlos (+7 aplicações e `Pressão externa por indeferimento`) levando a 106/306; subdivisão da RQ2; realocação de `Tensão entre desenvolvimento e proteção`; regra geral de nomeação temática; distinção `Fragmentação federativa` × `Descompasso federativo`; dois excertos literais de funções distintas para cada um dos cinco temas em destaque; referência à nota de IA sem substituição do PUIA;
4. relatório final rerenderizado e auditado visualmente em **6 páginas de corpo + 1 página de referências**, sem clipping, sobreposição ou quebra problemática de layout;
5. atualizada a aba `Notas` da Coding Frequency pré-merge para esclarecer que Codes-tema históricos com frequência zero preservados na aba nativa de Cases são placeholders da exportação original e não integram o baseline substantivo de 106 códigos/306 aplicações;
6. nenhuma célula quantitativa, Frequency, Cases, código ou Description da aba principal da planilha pré-merge foi alterada por essa nota;
7. apresentação, PUIA e Coding Frequency pós-merge permanecem inalterados e em PASS;
8. bundle e pacote limpo de cinco arquivos foram reconciliados para apontar ao relatório final corrente e à planilha pré-merge com nota preventiva;
9. preservados integralmente **5 anchors, 21 temas, 106 códigos substantivos pré-merge, 306 aplicações substantivas, 303 aplicações temáticas únicas, 10 casos, todos os Counts e todos os Cases**;
10. `QA_FINAL_EP04 = PASS` mantido após nova auditoria; próximo checkpoint permanece submissão no Eclass até **08/10/2026** e apresentação em **09/10/2026**.


## Versão 2.28 — 26/09/2026

Atualização de **fechamento do QA final e promoção do EP04 a pronto para submissão**, sem alteração de códigos, temas, Counts, Cases ou interpretações homologadas.

Alterações:

1. concluída a rematerialização documental pós-feedback do professor;
2. promovido `QA_RELATORIO_SIMPLIFICADO = PASS` após correção de RQs integrais, Counts por anchor, critérios de nomeação/fronteira, desdobramentos e excertos;
3. promovido `QA_APRESENTACAO = PASS` após correção da separação dos anchors `Contribuições do licenciamento` e `Limites da efetividade`, atualização para 21 temas e reconciliação das notas do apresentador;
4. promovido `QA_PUIA = PASS`;
5. promovido `QA_PLANILHA_PREMERGE = PASS`: 106 códigos / 306 aplicações / 10 casos, 106 Descriptions com critério de exclusão e reconciliação integral de Frequency/Cases com as fontes nativas;
6. corrigida e auditada `EP04_Coding_Frequency_02_Temas_PosMerge_QualCoder_Consolidado.xlsx`: 21 Codes-tema `(T)`, Description reproduzido do QDC pós-feedback com lista integral de constituintes e contrapontos, 303 aplicações temáticas únicas, 10 casos e 106/106 constituintes exatamente uma vez;
7. promovido `QA_PLANILHA_POSMERGE = PASS`;
8. promovido `QA_FINAL_EP04 = PASS` e `EP04 = PRONTO PARA SUBMISSÃO`;
9. reconciliado o manifesto para os **5 arquivos formais** correspondentes aos quatro itens do comunicado docente: relatório DOCX, duas planilhas XLSX, apresentação PPTX e PUIA DOCX;
10. preservados PDFs, fontes, scripts, logs, QDC, Journals e exportações nativas como evidências internas/reprodutibilidade;
11. mantidos inalterados **5 anchors, 21 temas, 106 códigos substantivos pré-merge, 306 aplicações substantivas, 303 aplicações temáticas únicas, 10 casos, todos os Counts e todos os Cases**;
12. próximo checkpoint: submissão do EP04 no Eclass até **08/10/2026** e apresentação em **09/10/2026**.



## Versão 2.27 — 26/09/2026

Atualização de **explicitação do raciocínio pedagógico do feedback do professor**, sem alteração metodológica operacional, de decisões analíticas homologadas, da arquitetura de 21 temas, de Counts ou de Cases.

Alterações:

1. explicitada como regra geral de nomeação temática a exigência de que o rótulo descreva fielmente o padrão compartilhado pelos códigos, inclusive tensões, restrições e ambivalências, sem impor direção positiva ou negativa não sustentada por todo o agrupamento;
2. registrada essa regra como fundamento da revisão de `Condições concretas sustentam o rigor` para `Condições e restrições da análise técnica (T)`;
3. explicitada a distinção conceitual entre `Fragmentação federativa do licenciamento` e `Descompasso federativo do licenciamento`:
   - fragmentação federativa = divisão institucional entre esferas/competências, com lacunas, sobreposições ou tratamento fragmentado;
   - descompasso federativo = inadequação entre a esfera administrativa licenciadora e a escala territorial efetiva dos impactos;
4. determinado que essa fronteira permaneça visível na redação metodológica e na justificativa dos temas do EP04;
5. preservados integralmente, sem alteração, os **5 anchors**, **21 temas**, **106 códigos substantivos pré-merge**, **306 aplicações substantivas**, **303 aplicações temáticas únicas**, **10 casos**, todos os **Counts** e todos os **Cases** já auditados na v2.26;
6. mantido como próximo checkpoint **EP04 — REMATERIALIZAÇÃO DOCUMENTAL PÓS-FEEDBACK DO PROFESSOR**.


## Versão 2.26 — 26/09/2026

Atualização substantiva motivada pelo **feedback oficial do professor sobre o EP03** e pela conclusão da revisão temática pós-feedback no Workflow A / QualCoder.

Alterações:

1. promovida a arquitetura temática corrente de **20 para 21 temas**;
2. convertidos os rótulos finais para formulações predominantemente nominais;
3. reformulado o primeiro tema de `Celeridade e rigor técnico` para `Condições e restrições da análise técnica (T)`, explicitando déficits, sobrecarga, barreiras e restrições presentes nos códigos;
4. dividido o antigo tema residual de `Limites da efetividade` em:
   - `Escala e desenho federativo (T)` — Count 9 / Cases 6;
   - `Limites diante de problemas complexos (T)` — Count 17 / Cases 9;
5. mantido o baseline substantivo pré-merge em **106 códigos / 306 aplicações / 10 casos**;
6. reconfirmado o pós-merge em **303 aplicações temáticas únicas**, sem perda empírica inexplicada;
7. concluída a transposição física **106/106 códigos → 21 Codes-tema**, com 0 códigos substantivos independentes;
8. auditadas as exportações reais `Code_frequencies.xlsx` e `Code_comparisons.xlsx`;
9. auditado `Codebook-licenca_ambiental_ep04.qdc`, com 5 anchors, 21 temas, 106/106 referências distintas a constituintes e 8 contrapontos/exceções; SHA-256 `2bcb5aeaa5eff8bbeb0db4fe05c658592ea556f00918a77ede16947ad7efd34d`;
10. preservados os checkpoints físicos `licenca_ambiental_ep04_pre-merge_feedback_20260926.qda` e `licenca_ambiental_ep04_post-merge_feedback_20260926.qda`;
11. atualizados, reexportados e auditados `Diário de decisões analíticas.txt` e `Categorização e tematização.txt`;
12. declarado que materializações do EP04 ainda baseadas em 20 temas são **históricas/pré-feedback** e devem ser rematerializadas;
13. alterado o próximo checkpoint para **EP04 — REMATERIALIZAÇÃO DOCUMENTAL PÓS-FEEDBACK DO PROFESSOR**.


## Versão 2.25 — 25/09/2026

Atualização de **fechamento documental da governança dual**, sem
alteração metodológica substantiva, analítica, temática ou quantitativa.

Alterações:

1. encerrada a pendência de reconciliação do bundle geral aberta pela
   v2.24;
2. promovido `BUNDLE_GERAL_CANONICO = PASS`;
3. promovidos `NOME_BUNDLE_CANONICO = PASS`,
   `RAIZ_INTERNA_CANONICA = PASS` e
   `METADADOS_BUNDLE_RECONCILIADOS = PASS`;
4. preservado o bundle v2.23 como snapshot histórico;
5. preservados os Prompt Masters v2.23 e v2.24 como históricos;
6. reconciliados os metadados internos efetivamente existentes no
   bundle;
7. preservado integralmente o Workflow A e seus resultados homologados;
8. mantido o Workflow B autorizado, ainda sem indicadores, códigos,
   aplicações ou temas próprios;
9. mantido o Prompt Master genérico como reconciliação separada;
10. autorizado como próximo gate operacional do Workflow B o B1 —
    clean-room source allowlist + modelo físico de dados.

## Versão 2.24 — 25/09/2026

Atualização de **governança operacional**, sem alteração metodológica
substantiva, analítica, temática ou quantitativa.

Alterações:

1. formalizada a coexistência de dois workflows operacionais:
   - Workflow A — QualCoder / CAQDAS;
   - Workflow B — AI-native / sem CAQDAS;
2. reafirmado que ambos executam a mesma Codebook Thematic Analysis,
   não constituindo duas metodologias diferentes;
3. preservado o Workflow A como percurso oficial utilizado na
   disciplina e como autoridade das exportações nativas do QualCoder;
4. incorporado o princípio human-in-the-loop do Workflow B:
   a IA propõe e somente o pesquisador aprova, edita ou rejeita;
5. reconhecidos os modos:
   - `AI_NATIVE_PRIMARY`;
   - `AI_NATIVE_CLEANROOM_REPLAY`;
6. estabelecida a independência das decisões analíticas entre A e B;
7. proibida a apresentação de relatórios AI-native como exportações
   nativas do QualCoder;
8. formalizada a regra de clean-room: resultados downstream do
   Workflow A não podem ser fornecidos ao agente analítico do Workflow
   B antes do congelamento de seu estado;
9. explicitado que o Prompt Master integral contém resultados do
   Workflow A e, portanto, não pode ser usado diretamente como contexto
   analítico no `AI_NATIVE_CLEANROOM_REPLAY`;
10. determinado que o Gate B1 crie allowlist de fontes e fonte
    sanitizada de governança antes da primeira passada AI-native;
11. registrado `GATE_B0 = PASS` após aprovação expressa do
    pesquisador em 25/09/2026;
12. mantidos inalterados RQs, anchors, codebook, aplicações, temas,
    Counts, Cases, interpretações e entregáveis já homologados;
13. o bundle canônico geral passa a exigir reconciliação com a v2.24
    antes da abertura do B1;
14. o Prompt Master genérico ainda não é alterado neste gate.

## Versão 2.18 — 24/09/2026

Atualização de governança física e documental após auditoria global da árvore do projeto, sem alteração metodológica, analítica ou quantitativa.

Alterações:

1. registrada a organização do Tema 5.1 em `00_analise`, `01_fontes`, `02_qualcoder` e `03_entregas`;
2. registrados os quatro projetos QualCoder em `02_qualcoder/01_projetos_preservados/`;
3. registrados os 12 snapshots em `02_qualcoder/99_snapshots_historicos/`;
4. registrada a reconciliação documental do EP03, separando pacote oficial e histórico;
5. preservados como oficiais do EP03:
   - `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL_v2.docx`;
   - `EP03_Apresentacao_Licenciamento_Ambiental_FGV_SUBMISSAO.pptx`;
   - `PUIA_Licenciamento_Ambiental_EP03_OFICIAL_FINAL.docx`;
6. registrado o bundle ativo do EP04 em `03_entregas/EP04/01_ativo/`;
7. preservado o nome canônico `EP04_ENTREGA_FINAL_BUNDLE.zip`;
8. registrado o bundle anterior de desenvolvimento em `03_entregas/EP04/99_historico/`;
9. registrado que `outputs/` foi removido somente após ficar vazio;
10. consolidada a regra de não deduplicar internamente projetos `.qda` ou bundles apenas por igualdade de hash;
11. permanecem inalterados 5 anchors, 20 temas, 106 códigos substantivos pré-merge, 306 aplicações pré-merge, 303 aplicações temáticas únicas pós-merge, 10 casos e 106/106 constituintes;
12. mantido `QA_FINAL_EP04 = PENDENTE`;
13. esta atualização é específica do Tema 5.1 e não exige atualização do Prompt Master genérico;
14. o bundle geral canônico deverá ser regenerado posteriormente com seus metadados e checksums reconciliados.

## Versão 2.17 — 24/09/2026

Atualização motivada pelo fechamento da governança de materialização do EP04 no chat de verificação da entrega, sem alteração de códigos, temas, contagens ou interpretações homologadas.

Alterações:

1. separada formalmente a função de dois bundles canônicos:
   - `BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip` = continuidade/governança geral do projeto;
   - `EP04_ENTREGA_FINAL_BUNDLE.zip` = entrega e reprodutibilidade específica do EP04;
2. proibido substituir o bundle geral pelo bundle do EP04 ou vice-versa;
3. fixado `EP04_ENTREGA_FINAL_BUNDLE.zip` como nome ativo sem versão/data/sufixos;
4. normalizado o nome da fonte visual oficial da apresentação para `Modelo de Apresentação_Relatório Final_Tema5.5.pptx`, tratando `(1)` apenas como sufixo de transporte quando ocorrer;
5. formalizada a fonte editável corrente do relatório simplificado:
   `relatorio_simplificado/EP04_relatorio_final_simplificado_licenciamento_ambiental.org`;
6. registrado que o relatório extenso permanece como apoio/reprodutibilidade, não como fonte da versão simplificada corrente;
7. formalizado o perfil corrente do relatório simplificado: A4, margens 3/3/2/2 cm, Times New Roman 9,5 pt, texto justificado, `pt-BR` e hifenização automática;
8. registrado o gate de 5 páginas de corpo, dentro da faixa oficial de 4–6 páginas;
9. formalizada a governança da apresentação:
   - `Org + modelo visual oficial → PPTX final → PDF derivado do PPTX`;
   - validação paralela `Org → PDF Beamer 16:9 full bleed`;
   - preservação de PPTX e PDFs para QA/reprodutibilidade;
10. reafirmados 6 slides efetivos / 5 minutos e preservação/adaptação das notas do apresentador;
11. formalizada a fonte reprodutível do PUIA e o pipeline `Org → DOCX + PDF`;
12. formalizado `./compilar_entrega_ep04.sh` como entrada única da compilação integrada do bundle de entrega;
13. preservadas as duas planilhas Coding Frequency consolidadas como entregáveis/evidência quantitativa;
14. atualizado o checkpoint do EP04 para:
   - bundle de entrega montado;
   - compilação integrada validada;
   - QA final pendente;
   - submissão pendente;
15. mantidos inalterados: 5 anchors, 20 temas, 106/306 pré-merge, 20/303 pós-merge, 10 casos e 106/106 constituintes;
16. estabelecido que apenas regras reutilizáveis devem migrar para o Prompt Master genérico; nomes/caminhos específicos permanecem neste Prompt Master do Tema 5.1.

## Versão 2.16 — 24/09/2026

Atualização motivada pela incorporação de dois arquivos oficiais novos ao EP04:

- `MODELO SIMPLIFICADO DO RELATÓRIO FINAL.docx`;
- `Modelo de Apresentação_Relatório Final_Tema5.5(1).pptx`.

Alterações:

1. incorporado o Modelo Simplificado como autoridade de forma, extensão e preenchimento do relatório final corrente;
2. definida sua precedência sobre o Guia do Relatório Final quando houver conflito de estrutura, extensão, campos, quadros, sequência de preenchimento ou grelha de avaliação;
3. preservado o Guia como autoridade geral de conteúdo metodológico, nota de IA, cautela com contagens, ABNT e critérios pedagógicos;
4. incorporada a extensão do modelo simplificado: 4 a 6 páginas, sem contar referências e anexos;
5. registrado que os textos entre colchetes são placeholders/instruções e devem ser substituídos;
6. registrado que exemplos e quadros do Tema 5.5 devem ser apagados antes da entrega;
7. incorporada a estrutura simplificada do relatório:
   - Introdução;
   - Metodologia;
   - Resultados;
   - Conclusão;
   - Referências;
8. incorporada a regra de desenvolver um tema em destaque por anchor code na seção 3.2;
9. incorporada a exigência de dois excertos por tema em destaque, com funções distintas;
10. incorporada a grelha própria do Modelo Simplificado: Metodologia 3,0; Temas 3,0; Uso das evidências 1,5; Conclusão 0,5; Nota de IA 1,0; Escrita 1,0;
11. incorporado o Modelo de Apresentação como autoridade de roteiro e forma da apresentação da Aula IX;
12. fixado roteiro de 5 minutos e 6 slides efetivos, com o slide de instruções apagado;
13. fixados os seis conteúdos da apresentação: capa; tema/perguntas/corpus; percurso analítico; mapa dos temas; um tema em destaque; respostas/limitações/IA;
14. registrado que a apresentação não deve conter achados novos: números e excertos devem sair do relatório;
15. reforçadas as proibições de `a maioria dos entrevistados` e `os temas emergiram`;
16. determinado que as falas dos slides fiquem nas anotações do apresentador;
17. mantidos inalterados os resultados analíticos do EP03 e as auditorias quantitativas do EP04: 5 anchors, 20 temas, 106/306 pré-merge e 20/303 pós-merge;
18. determinado que o bundle canônico incorpore fisicamente os dois novos modelos e atualize manifesto, checksums, README, mapa de autoridades e checklist.

## Versão 2.15 — 22/09/2026

Atualização de governança documental e auditoria do EP04, sem alteração de códigos, temas ou interpretações homologadas.

Alterações:

1. formalizado `EP04_relatorio_final_licenciamento_ambiental.org` como fonte canônica editável do relatório;
2. formalizado que DOCX e PDF são saídas geradas e não devem ser editados diretamente;
3. registrado `compilar_org_pdf.sh` como entrada canônica da compilação e a cadeia ORG → Pandoc → DOCX/reference → pós-processamento python-docx → DOCX final → LibreOffice → PDF;
4. registrado o perfil `perfil/fgv_abnt.toml` e o gate de 6–12 páginas após recompilação;
5. auditadas as quatro planilhas correntes do QualCoder, confirmando 106/306 pré-merge e 20/303 pós-merge;
6. verificado que as exportações nativas `Code frequencies` recebidas não trazem Description nem Cases como colunas;
7. produzidas duas tabelas consolidadas equivalentes à Coding Frequency Table docente, preservando as fontes nativas em abas próprias;
8. documentada a reconstrução textual transparente da Description de `Pressão externa por indeferimento` a partir dos Journals/triagem, sem alteração quantitativa;
9. atualizado o `.org` do relatório para explicitar, na seção 2.4, o baseline 105/299, a revisão de Roberto Carlos, a subdivisão da RQ2 e a realocação de `Tensão entre desenvolvimento e proteção`;
10. atualizado o `.org` na seção 2.5 para explicar a equivalência operacional entre as exportações do QualCoder e as duas tabelas consolidadas do EP04;
11. nenhuma alteração nos 5 anchors, 20 temas, 303 aplicações temáticas únicas ou 106 códigos substantivos pré-merge.

## Versão 2.14 — 22/09/2026

Atualização motivada pelo comunicado docente posterior sobre a entrega do **Exercício Programado 04**.

Alterações:

1. atualizado o prazo formal do EP04 para **08/10/2026**;
2. registrada a exposição da apresentação na **Aula IX, em 09/10/2026**;
3. reformulado o pacote formal de entrega do EP04 em quatro itens:
   - relatório final em Word com Nota Metodológica de IA ao final da Metodologia;
   - duas exportações `Coding Frequency` em Excel;
   - apresentação com resumo do relatório;
   - PUIA em anexo separado;
4. incorporada a exigência de duas exportações `Coding Frequency`:
   - pré-merge, para documentar a seção 2.5/livro de códigos;
   - pós-merge, para documentar a seção 2.6/temas;
5. incorporado o critério de suficiência da exportação pós-merge: uma linha por tema, anchor em `Category`, rótulo do tema, descrição seguida da lista de códigos, frequência e número de casos;
6. reforçada a obrigação de a seção 2.4 declarar mudanças entre a codificação e a tematização, incluindo revisão de Roberto Carlos, novo código, subdivisão da RQ2 e correções de fronteira;
7. registrada a contingência para quem mesclou sem duplicar o projeto, com declaração obrigatória na seção 2.4;
8. explicitado que este projeto preservou o cenário adequado: baseline pré-merge sem alteração destrutiva e cópia temática pós-merge;
9. atualizado o checklist final do EP04 para tratar as duas exportações como entregáveis formais, não como evidências internas opcionais;
10. nenhuma alteração foi feita nos 5 anchors, 20 temas, 303 aplicações temáticas únicas, 106 códigos substantivos pré-merge ou interpretações homologadas.

## Versão 2.13 — 19/09/2026

Atualização de **governança documental do bundle**, sem alteração metodológica, analítica ou quantitativa.

Alterações:

1. formalizado no Prompt Master o nome canônico permanente:
   `BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL.zip`;
2. proibido acrescentar versão, data, `ATUALIZADO`, `FINAL`, `(1)` ou outros sufixos ao bundle ativo;
3. definido que versões históricas, quando preservadas, devem usar nomenclatura versionada separada e nunca substituir o bundle ativo;
4. fixada a raiz interna do ZIP como:
   `BUNDLE_CANONICO_MPPG_ANALISE_QUALITATIVA_LICENCIAMENTO_AMBIENTAL/`;
5. criado `NOME_CANONICO_DO_BUNDLE.txt` como marcador interno obrigatório;
6. criados os gates `NOME_BUNDLE_CANONICO`, `RAIZ_INTERNA_CANONICA` e `METADADOS_BUNDLE_RECONCILIADOS`;
7. reconciliados README, Instruções do Projeto, manifesto e checksums com a versão corrente;
8. corrigidas referências internas ainda presas à v2.11/estado anterior ao Guia;
9. preservado integralmente o estado do EP03 e o gate aberto do EP04 estabelecido pela v2.12;
10. nenhuma alteração nos 5 anchors, 20 temas, 303 aplicações temáticas únicas ou 106/106 constituintes preservados.

## Versão 2.12 — 19/09/2026

Atualização motivada pela incorporação do **`Guia do Relatório Final.docx`** e pela abertura formal do EP04.

Alterações:

1. incorporado o Guia do Relatório Final como fonte documental oficial e obrigatória do EP04;
2. encerrado o bloqueio `aguardando Guia`;
3. promovido o EP04 para **ABERTO PARA MATERIALIZAÇÃO CONTROLADA**;
4. fixada a estrutura obrigatória: Introdução; Metodologia; Resultados e Discussão; Conclusão; Referências;
5. registrada a extensão total esperada de **6 a 12 páginas**;
6. fixada a Introdução com tema, contextualização, RQs/subdivisão e nota sobre caráter didático/dados fictícios;
7. fixada a Metodologia em primeira pessoa e na ordem real de execução, incluindo DIP, emacD, livro de códigos, compilação, desenvolvimento dos temas, credibilidade e Nota Metodológica sobre IA;
8. incorporadas as páginas/capítulos de Adu indicadas pelo Guia: p. 97; cap. 3; p. 97–100; p. 109–110; p. 127–131; p. 330–331; p. 346–347; p. 368–369;
9. fixada a apresentação de Resultados e Discussão **orientada por temas** como padrão;
10. incorporada a exigência de **2 a 4 excertos por tema**, com funções argumentativas distintas;
11. reafirmado que Count/Cases descrevem o corpus e não autorizam prevalência populacional;
12. proibida a expressão `os temas emergiram dos dados`; temas são construídos;
13. fixada a Conclusão com síntese das RQs, limitações e reflexão metodológica;
14. fixadas Referências em ABNT e apenas das obras citadas, com Adu obrigatório e Braun & Clarke (2021) para situar a abordagem, conforme o Guia;
15. incorporada a grelha oficial do EP04: Metodologia 3,5; Temas 3,0; Evidências 1,5; Nota de IA 1,0; Escrita 1,0;
16. incorporado o checklist oficial anterior à entrega;
17. explicitada a distinção entre PUIA e Nota Metodológica sobre IA;
18. mantidos integralmente os estados homologados do EP02 e EP03: 5 anchors, 20 temas, 303 aplicações temáticas únicas e 106/106 constituintes preservados;
19. atualizado o próximo checkpoint para seleção de excertos e redação/materialização do EP04 conforme o Guia;
20. determinado que o bundle canônico inclua o Guia do Relatório Final e metadados de auditoria atualizados.

## Versão 2.11 — 19/09/2026

Atualização de **governança bibliográfica**, sem alteração do estado analítico, temático ou quantitativo do EP03.

Alterações:

1. incorporado o acervo `Métodos Qualitativos de Pesquisa em Ciência Política - 9192026 - 321 PM.zip` como **Biblioteca bibliográfica integral da disciplina**;
2. auditado o acervo com **22 PDFs + 1 índice HTML**;
3. promovido Philip Adu (2019), em texto integral, a **autoridade bibliográfica primária** para conferência direta do método adotado;
4. preservada a precedência das aulas/instruções do professor para a execução operacional da disciplina;
5. criada regra para distinguir explicitamente:
   - `PROFESSOR`;
   - `FONTE_PRIMARIA`;
   - `ADAPTACAO_QUALCODER`;
   - `INFERENCIA_DA_IA`;
   - `FONTE_EXTERNA`;
6. estabelecido que formulações atribuídas a Adu ou outro autor devem, quando relevante, ser verificadas na obra original disponível;
7. criada regra contra **deriva metodológica**: a presença de obras de Reflexive Thematic Analysis, Template Analysis ou outras tradições não altera silenciosamente a Codebook Thematic Analysis/Adu adotada;
8. definida consulta **seletiva** à biblioteca, sem obrigação de reler os 22 PDFs em cada operação;
9. definido que, para o EP04, a fundamentação metodológica deve distinguir o texto original de Adu, as aulas do professor e as equivalências/adaptações do QualCoder;
10. registrado que o bundle canônico deve conter o acervo bibliográfico em pasta própria, com inventário e checksums;
11. mantidos integralmente os estados já homologados do EP02 e EP03, inclusive 5 anchors, 20 temas e 303 aplicações temáticas únicas;
12. o próximo checkpoint permanece **EP04 — relatório final**, condicionado à incorporação do Guia do Relatório Final.

## Versão 2.10 — 19/09/2026

Atualização de **estado do artefato**, sem alteração metodológica, analítica ou quantitativa.

Alterações:

1. promovida a versão final da Tabela de Temas para `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL_v2.docx`;
2. alterada a cor dos nomes dos temas de branco para **azul-marinho escuro**, mantendo o fundo azul-claro do modelo para aumentar contraste e legibilidade;
3. confirmada nova renderização das quatro páginas sem cortes, sobreposições ou alteração indevida de paginação;
4. mantidos inalterados:
   - 5 anchors;
   - 20 temas;
   - 106/106 códigos constituintes;
   - 303 aplicações temáticas únicas;
   - Counts e Cases finais;
   - conteúdo analítico das descrições e listas de códigos;
5. `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL.docx` passa a ser versão histórica intermediária;
6. o pacote oficial corrente do EP03 deve usar exclusivamente a versão `FINAL_v2` da Tabela de Temas;
7. nenhuma atualização do Prompt Master genérico é necessária por esta alteração cromática específica.

## Versão 2.9 — 19/09/2026

Atualização motivada pela auditoria do preenchimento do Modelo oficial de Tabela de Temas e pela necessidade de fortalecer a governança documental.

Alterações:

1. instituída `VARREDURA_INSTRUCOES` obrigatória antes de análise/materialização;
2. determinado que todos os documentos relevantes sejam lidos **também em busca de instruções**, independentemente do nome do arquivo;
3. tornada obrigatória a inspeção visual de modelos quando instruções puderem estar em quadros, cores, formas, cabeçalhos, células ou exemplos;
4. criada classificação documental interna entre `INSTRUCAO`, `ESTRUTURA_FIXA`, `PLACEHOLDER`, `EXEMPLO`, `CONTEUDO_FIXO`, `DADO_A_PREENCHER` e `CONTEUDO_SUBSTANTIVO`;
5. estabelecido que, havendo modelo oficial, o **próprio arquivo-modelo deve ser usado diretamente como molde**;
6. proibida a reprodução do modelo/instruções como parte da resposta final, salvo conteúdo fixo explicitamente previsto;
7. proibida a criação de documento apenas “inspirado” no modelo quando o oficial estiver disponível;
8. exigido preenchimento integral dos campos obrigatórios de cabeçalho/identificação;
9. criado bloqueio contra vazamento de placeholders, exemplos e instruções de preenchimento;
10. definido que fallback incorporado ao Prompt Master só pode ser usado quando não houver modelo oficial disponível;
11. atualizado o arquivo oficial das tabelas do EP03 para `EP03_Tabelas_de_Temas_Licenciamento_Ambiental_FINAL.docx`;
12. adicionados testes de regressão documentais específicos para modelos.

## Versão 2.8 — 09/09/2026

Atualização motivada pela **auditoria final e encerramento formal do EP03**.

Alterações:

1. incorporado o modelo oficial do PUIA e produzido o protocolo final específico do EP03;
2. substituído o PUIA provisório de layout próprio pela versão no modelo oficial;
3. confirmada a persistência dos registros finais nos dois Journals por meio das exportações:
   - `Diário de decisões analíticas(1).txt`;
   - `Categorização e tematização.txt`;
4. encerrado o Gate 6 com `PASS` integral;
5. consolidado o conjunto oficial de submissão em três arquivos:
   - tabelas oficiais;
   - PowerPoint de submissão;
   - PUIA oficial;
6. reafirmado que planilhas, QDC, Prompt Master, Journals e versões intermediárias são evidências internas e não integram a submissão, salvo solicitação do professor;
7. promovido o EP03 para **ENCERRADO / PRONTO PARA SUBMISSÃO**;
8. alterado o próximo checkpoint para **EP04 — relatório final**, condicionado à incorporação do Guia do Relatório Final.

## Versão 2.7 — 09/09/2026

Atualização motivada pelo **fechamento técnico pós-merge e pela materialização documental do EP03**.

Alterações:

1. promovido o checkpoint canônico do estado pré-merge para o estado pós-merge auditado;
2. preservados explicitamente três projetos com funções distintas:
   - `licenca_ambiental.qda` = histórico/canônico do EP02;
   - `licenca_ambiental_ep03.qda` = baseline reconciliado pré-transposição;
   - `licenca_ambiental_temas.qda` = projeto temático consolidado;
3. registrado o fechamento dos Gates 1–5 com `PASS` integral;
4. incorporada a transposição física completa `106 códigos substantivos → 20 Codes-tema`;
5. incorporado o resultado quantitativo pós-merge:
   - 303 aplicações temáticas únicas;
   - 0 perda empírica inexplicada;
6. documentadas as três deduplicações por exact match que explicam `306 → 303`;
7. substituídos os Counts pré-merge dos temas, como autoridade corrente, pelos Counts reais pós-merge;
8. promovidos os Cases dos 20 temas de união provisória para **Cases finais confirmados** pela exportação `Code_comparisons(2).xlsx`;
9. incorporadas como autoridades finais:
   - `Code_frequencies_12.xlsx`;
   - `Code_comparisons(2).xlsx`;
   - `Codebook-licenca_ambiental_temas(3).qdc`;
10. registrado o QDC final:
    - 5 anchors;
    - 20 Codes-tema;
    - 0 códigos substantivos físicos;
    - 106/106 constituintes preservados nos Memos;
    - 8 contrapontos/exceções;
    - SHA-256 `1e330cf3fff86e2304a0bb832acd9e674dfe7a5ff40be3e4130c4a4ebdbc1ada`;
11. registrado que a correção de `Tensão entre desenvolvimento e proteção` foi fisicamente materializada e confirmada;
12. atualizada a redação operacional real do QualCoder 3.8.2 para `Merge code into code`;
13. registrado o Gate 6 em fechamento:
    - cinco tabelas oficiais materializadas;
    - PowerPoint das Etapas 01–03 materializado e aprovado;
    - PUIA materializado;
    - textos finais dos Journals materializados;
14. alterado o próximo checkpoint para **auditoria final do pacote de entrega do EP03**;
15. o EP03 ainda não é marcado como `ENCERRADO / PRONTO PARA SUBMISSÃO` até a auditoria final do pacote e a confirmação da persistência dos Journals no projeto.

## Versão 2.6 — 09/09/2026

Atualização motivada pela incorporação do **Modelo oficial de Tabela de Temas do EP03**.

Alterações:

1. incorporado `Modelo de Tabela de Temas(1).docx` como fonte documental oficial e obrigatória do EP03;
2. removido o estado de “modelo pendente”;
3. fixada a regra de **uma tabela por anchor operacional final**;
4. para o estado corrente do projeto, definidos **5 blocos/tabelas** no `.docx` do EP03;
5. explicitado que os quatro blocos do template demonstrativo não constituem teto, pois a regra do modelo é uma tabela por anchor;
6. incorporadas as regras do modelo:
   - geralmente 2–5 temas por anchor;
   - número de temas abaixo da metade do número de códigos do anchor;
   - mínimo de 2–3 códigos relacionados por tema;
   - temas de 1 código são muito frágeis;
   - nomes analíticos, descritivos e concisos;
   - descrição seguida da lista dos códigos constituintes;
7. fixada a transposição `Description` do QDA Miner → `Code Memo` do Code-tema `(T)` no QualCoder;
8. incorporada integralmente a grelha oficial de avaliação:
   - coerência interna = 2,5;
   - distinção = 2,0;
   - abrangência = 1,5;
   - nível de abstração = 1,5;
   - nomenclatura = 1,0;
   - organização por anchor/RQ = 1,5;
9. realizada auditoria estrutural do mapa temático homologado:
   - 5 anchors;
   - 20 temas;
   - todos na faixa de 2–5;
   - todos abaixo de metade dos códigos do anchor;
   - todos com pelo menos 2 códigos;
   - nenhum tema unitário;
   - resultado `PASS`;
10. mantida pendente a auditoria semântica/documental final contra os seis critérios da grelha antes da entrega;
11. atualizado o checkpoint: o Modelo oficial já não é pendência; a próxima etapa passa a ser correção física no QualCoder, reexportação, auditoria semântica, merge temático e materialização das cinco tabelas;
12. nenhuma alteração foi feita nos 20 temas homologados por esta atualização documental.

## Versão 2.5 — 08/09/2026

Atualização motivada pelo fechamento homologado da **revisão global temática da Etapa 2/EP03** e pela reconciliação final das exportações pós-Roberto.

Alterações:

1. promovido o estado corrente para **106 códigos substantivos / 306 aplicações**;
2. incorporado o novo código de RQ1 `Pressão externa por indeferimento`;
3. homologada definitivamente a manutenção de `Celeridade e rigor técnico` como único anchor da RQ1;
4. incorporados os quatro anchors operacionais finais da RQ2;
5. registrado o fechamento da revisão global com **20 temas homologados**;
6. incorporadas todas as renomeações temáticas homologadas durante a revisão global;
7. corrigida a aplicação do terceiro teste formal da Aula VI para **retirada do trecho mais forte**; retirada do código dominante permanece apenas como stress test adicional;
8. reconciliada `Tensão entre desenvolvimento e proteção` com as exportações reais:
   - Count = 2;
   - Cases = 2;
   - André + Roberto Carlos;
9. revogada sua classificação anterior como `TOQUE PONTUAL — NÃO SUSTENTA TEMA`;
10. homologada sua realocação analítica:
    - `Relações e participação dos atores` → `Limites da efetividade`;
    - tema destino: `Desajuste entre instrumento e problema limita o alcance`;
11. estado-alvo dos anchors após essa correção:
    - RQ1 34/113;
    - Contribuições 10/20;
    - Limites 22/60;
    - Relações 20/46;
    - Aperfeiçoamento 20/67;
12. cobertura temática passa a **106/106 códigos e 306/306 aplicações**, sem tema residual e sem dupla atribuição;
13. Cases por tema foram calculados por união dos casos da exportação pré-merge e permanecem sujeitos à confirmação pós-merge;
14. o próximo gate deixa de ser analítico e passa a ser documental/operacional:
    - registrar a correção final nos Journals;
    - incorporar o Modelo oficial de Tabela de Temas;
    - materializar a correção de categoria no QualCoder;
    - reexportar;
    - criar Codes-tema `(T)`;
    - mesclar códigos;
    - produzir os artefatos oficiais do EP03;
15. nenhuma mesclagem temática física foi realizada no QualCoder até esta versão.

## Versão 2.4 — 08/09/2026

Atualização motivada pelo **feedback oficial do professor sobre o EP02**, com nota 10/10 — 100%.

Alterações:

1. feedback individual do professor passa a integrar a hierarquia de autoridade operacional específica do projeto;
2. registrado o baseline do EP02 como **105 códigos / 299 aplicações aprovado pelo professor**;
3. preservados como evidência de credibilidade:
   - rigor do codebook;
   - critérios de exclusão relacionais;
   - uso do QualCoder;
   - aba de mapeamento metodológico;
   - fórmulas;
   - arquivos-fonte;
   - contagem por caso;
4. mantida a regra de que o EP02 não será recodificado nem reeditado retroativamente;
5. criado gate obrigatório de **revisão/subdivisão de `Efetividade do licenciamento` antes da tematização**;
6. as quatro famílias mencionadas pelo professor entram apenas como diagnóstico para teste, não como anchors automaticamente homologados;
7. proibida a pré-homologação de nomes ou quantidade dos novos anchors derivados de RQ2;
8. criado o controle de **anchor superdimensionado**;
9. incorporada a expectativa docente de **2 a 5 temas por anchor code** como guardrail de granularidade;
10. criada a **triagem obrigatória da cauda de códigos `Count=1 / Cases=1`**;
11. definidos cinco estados de triagem:
    - `ACOMODA-SE EM AGRUPAMENTO`;
    - `CONTRAPONTO / EXCEÇÃO`;
    - `TOQUE PONTUAL — NÃO SUSTENTA TEMA`;
    - `REVISAR FRONTEIRA / POSSÍVEL CONSOLIDAÇÃO`;
    - `PENDENTE DE LEITURA DOS SEGMENTOS`;
12. refinada a regra da Aula VI:
    - todo código que integrar a consolidação pertence a um único tema;
    - códigos pontuais não precisam ser forçados a tema incoerente;
13. proibida a criação de tema residual apenas para atingir cobertura de 100%;
14. as 13 categorias e 5 temas anteriores passam a estar **revogados como ponto de partida operacional do EP03**, permanecendo apenas como histórico auditável;
15. o checkpoint foi reordenado para:
    - novas exportações pós-Roberto;
    - triagem dos singletons;
    - revisão/subdivisão de RQ2;
    - redistribuição dos códigos;
    - recálculo por anchor operacional;
    - dominantes;
    - agrupamento Aula VI;
    - 2–5 temas por anchor;
    - teste formal;
    - homologação;
    - transposição para a cópia temática;
16. nenhuma subdivisão de RQ2, categoria ou tema foi homologada por esta atualização.

## Versão 2.3 — 05/09/2026

Atualização motivada pela incorporação da **Aula VI**, dedicada à rotina de definição dos temas e ao EP03.

Alterações:

1. Aula VI passa a ser a **autoridade operacional principal da Etapa 2/EP03**;
2. registrada a Aula V como fonte consultada sem regra substantiva nova no arquivo disponibilizado;
3. corrigida a regra de pertencimento temático:
   - na consolidação oficial do EP03, **cada código vai para um tema só**;
   - relações secundárias podem permanecer apenas como notas analíticas, sem dupla atribuição;
4. a orientação anterior de duplicação temática da Aula IV fica superada, para o EP03, pela regra posterior e específica da Aula VI;
5. incorporada a rotina oficial:
   - listar códigos por anchor;
   - usar frequência e casos;
   - agrupar a partir dos dominantes;
   - usar o dominante como semente de comparação;
   - criar novos agrupamentos quando necessário;
   - rotular e descrever os temas;
6. incorporado o **teste formal do tema**:
   - conceito central;
   - organização de material heterogêneo;
   - sobrevivência à retirada do trecho mais forte;
7. as 13 categorias e 5 temas anteriores passam a ser tratados como **hipóteses históricas pré-Aula VI**, não como grade obrigatória;
8. `Coding reports` passa a ser ferramenta de validação contextual e contraevidência, sem substituir a rotina da tabela ensinada na Aula VI;
9. o próximo gate passa a começar por **novas exportações quantitativas pós-revisão de Roberto Carlos**;
10. incorporada a transposição da tabela temática para uma **cópia do projeto**, nunca para o projeto canônico;
11. no projeto temático derivado, os temas serão criados como `Code` marcado `(T)` sob o anchor, com definição e lista de códigos no Memo/Description;
12. cada código substantivo será mesclado no tema correspondente;
13. incorporada seção permanente de **equivalências operacionais QDA Miner Lite ↔ QualCoder 3.8.2**;
14. a matriz interna de apoio foi ajustada para a rotina da Aula VI;
15. nenhuma categoria ou tema foi homologado por esta atualização.

## Versão 2.2 — 05/09/2026

Atualização motivada pela incorporação do **Plano de Ensino da disciplina** às fontes do projeto.

Alterações:

1. criado regime de autoridade por domínio:
   - Aulas III/IV para método;
   - Plano de Ensino e anexos do exercício para prazos, formatos e entregáveis;
2. registrado que o corpus da disciplina é **didático e fictício**, com dados gerados por IA;
3. formalizado que os quatro exercícios têm pesos:
   - EP01 = 10%;
   - EP02 = 25%;
   - EP03 = 25%;
   - EP04 = 40%;
4. registrado que a nota final depende da entrega de **todos os quatro exercícios**;
5. tornado explícito que o **PUIA é obrigatório em todos os exercícios** e que ausência torna o trabalho incompleto;
6. incorporado o escopo oficial do EP03:
   - `.docx` com tabela de temas;
   - uma tabela por anchor code;
   - `.pptx` com resumo dos temas;
   - PUIA;
   - prazo 17/09/2026 às 23h59;
7. criado gate documental: não gerar a versão final do EP03 sem o **Modelo de Tabela de Temas**;
8. a planilha de categorização/tematização passa a ser tratada como **artefato interno de trabalho/auditoria**, não como entregável oficial do EP03;
9. pré-registrado o EP04:
   - relatório `.docx`;
   - `.pptx`;
   - PUIA;
   - Nota Metodológica sobre IA de até uma página;
   - nota sobre caráter didático e dados fictícios;
   - prazo 01/10/2026 às 23h59;
10. criado gate documental: não materializar a versão final do EP04 sem o **Guia do Relatório Final**;
11. mantida integralmente a metodologia de validação da Etapa 2 por `Reports → Coding reports`;
12. nenhuma categoria ou tema foi homologado por esta atualização.

## Versão 2.1 — 05/09/2026

1. fixada de forma inequívoca a equivalência operacional:
   - QDA Miner Lite `Retrieve → Coding Retrieval`;
   - QualCoder 3.8.2 `Reports → Coding reports`;
2. definido `Coding reports` como relatório principal de recuperação contextual para a Etapa 2;
3. `Codes by text segments` passa a ser tratado apenas como relatório complementar;
4. mantidos os dois arquivos de recuperação por RQ como estratégia eficiente de validação das 13 categorias;
5. nenhuma categoria ou tema foi homologado por esta atualização;
6. o próximo gate permanece a validação empírica contra os segmentos codificados.

---

# FIM DO CHECKPOINT CANÔNICO

**Estado corrente:** EP02 aprovado; EP03 encerrado; EP04 pós-feedback rematerializado e com `QA_FINAL_EP04 = PASS`; **EP04 PRONTO PARA SUBMISSÃO**; 5 anchors, 21 temas, 106 códigos substantivos pré-merge, 306 aplicações substantivas, 303 aplicações temáticas únicas, 10 casos; Prompt Master canônico v2.29; Workflow A — QualCoder preservado; Workflow B — AI-native autorizado; bundle específico do EP04 reconciliado; próximo checkpoint: submissão no Eclass até 08/10/2026 e apresentação em 09/10/2026.
