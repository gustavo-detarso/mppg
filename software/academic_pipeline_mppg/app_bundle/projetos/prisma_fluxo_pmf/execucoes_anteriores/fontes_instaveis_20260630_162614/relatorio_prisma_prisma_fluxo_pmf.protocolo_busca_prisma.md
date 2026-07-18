# Protocolo e registro de busca PRISMA

- Execução: 2026-06-30T19:00:33+00:00
- Estratégia: blocos_tematicos
- Bases selecionadas: crossref, openalex, semantic_scholar, scopus, wos, pubmed, europe_pmc, scielo, core
- Meta de estudos incluídos: 15
- Limite por bloco e base: 20
- Limite seguro do Scopus por bloco: 10
- Unpaywall: ativado; registros enriquecidos: 38
- Pré-triagem por IA: ativada; decisão final de elegibilidade permanece humana.

## Blocos de busca
- **Benefícios por incapacidade e avaliação médico-pericial** (`beneficios_incapacidade_avaliacao_medico_pericial`): `disability benefits AND medical assessment`
- **Benefícios por incapacidade e avaliação médico-pericial** (`beneficios_incapacidade_avaliacao_medico_pericial`): `incapacity benefit AND medical evaluation`
- **Benefícios por incapacidade e avaliação médico-pericial** (`beneficios_incapacidade_avaliacao_medico_pericial`): `work disability AND medical certification`
- **Análise documental e elegibilidade** (`analise_documental_elegibilidade`): `documentary assessment AND disability`
- **Análise documental e elegibilidade** (`analise_documental_elegibilidade`): `medical certificate AND disability benefits`
- **Análise documental e elegibilidade** (`analise_documental_elegibilidade`): `medical documentation AND work incapacity`
- **Teleperícia e avaliação remota** (`telepericia_avaliacao_remota`): `remote medical assessment AND disability`
- **Teleperícia e avaliação remota** (`telepericia_avaliacao_remota`): `telemedicine AND disability assessment`
- **Teleperícia e avaliação remota** (`telepericia_avaliacao_remota`): `telehealth AND medical certification`
- **Capacidade, filas e alocação de serviços** (`capacidade_filas_alocacao`): `disability benefits AND waiting time`
- **Capacidade, filas e alocação de serviços** (`capacidade_filas_alocacao`): `medical assessment AND service capacity`
- **Capacidade, filas e alocação de serviços** (`capacidade_filas_alocacao`): `case allocation AND disability services`
- **Equidade territorial, qualidade e controle** (`equidade_qualidade_controle`): `disability assessment AND equity of access`
- **Equidade territorial, qualidade e controle** (`equidade_qualidade_controle`): `remote assessment AND quality assurance`
- **Equidade territorial, qualidade e controle** (`equidade_qualidade_controle`): `medical certification AND audit`

## Critérios de inclusão
- Estudos que analisem benefícios por incapacidade, afastamento por doença, seguro por incapacidade, certificação médica ou avaliação médico-pericial.
- Estudos que abordem triagem documental, elegibilidade, análise de documentos médicos, validação de atestados ou processos de decisão sobre incapacidade.
- Estudos que examinem teleperícia, telemedicina aplicada à avaliação, avaliação médica remota, videoperícia ou modalidades híbridas de avaliação.
- Estudos que analisem capacidade operacional, filas, tempo de espera, produtividade, priorização de casos, alocação de profissionais ou gestão de demanda em serviços periciais ou sistemas de benefícios.
- Estudos que discutam acesso territorial, equidade, barreiras geográficas, inclusão de populações remotas ou desigualdades de acesso à avaliação médico-pericial.
- Estudos que tratem de qualidade decisória, confiabilidade, auditoria, controle, integridade, prevenção de fraude, revisão de decisões ou riscos associados à avaliação remota ou documental.
- Estudos empíricos, avaliações de política pública, estudos de caso institucionais, revisões sistemáticas, revisões de escopo ou relatórios técnicos com evidência útil para a pergunta de pesquisa.
- Publicações em português, inglês ou espanhol.

## Critérios de exclusão
- Telemedicina exclusivamente assistencial, sem relação substantiva com avaliação de incapacidade, certificação, elegibilidade, triagem ou gestão de fluxos.
- Estudos clínicos que não contribuam para compreender processos de avaliação médico-pericial, decisão administrativa, capacidade, acesso, qualidade ou controle.
- Textos sem relação substantiva com a pergunta de pesquisa ou com ao menos um eixo da matriz analítica.
- Editorial, comentário, notícia, apresentação, resumo de congresso ou opinião sem análise, dados, revisão ou contribuição institucional verificável.
- Duplicidades entre bases ou consultas.
- Registros sem título, resumo ou metadados suficientes para triagem humana.
- Publicações em idioma diferente de português, inglês ou espanhol quando não houver informação suficiente para avaliação confiável.

## Contagens
- registros identificados: 1474
- duplicatas removidas: 174
- registros apos deduplicacao: 1300
- registros pre triagem ia avaliados: 800
- registros enviados para triagem: 250
- triagem titulo resumo concluida: 0
- textos completos avaliados: 0
- estudos incluidos: 0

## Registro por fonte e bloco
- **Crossref** — recuperados: 256
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 18
    - expressão: `disability benefits AND medical assessment`
    - URL: `https://api.crossref.org/works?query.bibliographic=disability+benefits+AND+medical+assessment&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `incapacity benefit AND medical evaluation`
    - URL: `https://api.crossref.org/works?query.bibliographic=incapacity+benefit+AND+medical+evaluation&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `work disability AND medical certification`
    - aviso/erro: HTTP 500: {"status":"error","message-type":"exception","message-version":"1.0.0","message":{"name":"class clojure.lang.ExceptionInfo","description":"clojure.lang.ExceptionInfo: Response Exception qbits.spandex.Response@3751da1","message":"Response Exception","stack":["qbits.spandex$response_ex__GT_ex_info.inv
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `documentary assessment AND disability`
    - URL: `https://api.crossref.org/works?query.bibliographic=documentary+assessment+AND+disability&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://api.crossref.org/works?query.bibliographic=medical+certificate+AND+disability+benefits&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://api.crossref.org/works?query.bibliographic=medical+documentation+AND+work+incapacity&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://api.crossref.org/works?query.bibliographic=remote+medical+assessment+AND+disability&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telemedicine AND disability assessment`
    - aviso/erro: HTTP 500: {"status":"error","message-type":"exception","message-version":"1.0.0","message":{"name":"class clojure.lang.ExceptionInfo","description":"clojure.lang.ExceptionInfo: Response Exception qbits.spandex.Response@b304da8c","message":"Response Exception","stack":["qbits.spandex$response_ex__GT_ex_info.in
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telehealth AND medical certification`
    - URL: `https://api.crossref.org/works?query.bibliographic=telehealth+AND+medical+certification&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `disability benefits AND waiting time`
    - URL: `https://api.crossref.org/works?query.bibliographic=disability+benefits+AND+waiting+time&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `medical assessment AND service capacity`
    - URL: `https://api.crossref.org/works?query.bibliographic=medical+assessment+AND+service+capacity&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 18
    - expressão: `case allocation AND disability services`
    - URL: `https://api.crossref.org/works?query.bibliographic=case+allocation+AND+disability+services&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `disability assessment AND equity of access`
    - URL: `https://api.crossref.org/works?query.bibliographic=disability+assessment+AND+equity+of+access&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://api.crossref.org/works?query.bibliographic=remote+assessment+AND+quality+assurance&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://api.crossref.org/works?query.bibliographic=medical+certification+AND+audit&rows=20&select=DOI%2Ctitle%2Cauthor%2Cpublished-print%2Cpublished-online%2Cissued%2Ccontainer-title%2Cabstract%2Ctype%2CURL&mailto=REDACTED`
  - resumo de falhas: Benefícios por incapacidade e avaliação médico-pericial: HTTP 500: {"status":"error","message-type":"exception","message-version":"1.0.0","message":{"name":"class clojure.lang.ExceptionInfo","description":"clojure.lang.ExceptionInfo: Response Exception qbits.spandex.Response@3751da1","message":"Response Exception","stack":["qbits.spandex$response_ex__GT_ex_info.inv; Teleperícia e avaliação remota: HTTP 500: {"status":"error","message-type":"exception","message-version":"1.0.0","message":{"name":"class clojure.lang.ExceptionInfo","description":"clojure.lang.ExceptionInfo: Response Exception qbits.spandex.Response@b304da8c","message":"Response Exception","stack":["qbits.spandex$response_ex__GT_ex_info.in
- **OpenAlex** — recuperados: 300
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `disability benefits AND medical assessment`
    - URL: `https://api.openalex.org/works?search=disability+benefits+AND+medical+assessment&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `incapacity benefit AND medical evaluation`
    - URL: `https://api.openalex.org/works?search=incapacity+benefit+AND+medical+evaluation&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `work disability AND medical certification`
    - URL: `https://api.openalex.org/works?search=work+disability+AND+medical+certification&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `documentary assessment AND disability`
    - URL: `https://api.openalex.org/works?search=documentary+assessment+AND+disability&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://api.openalex.org/works?search=medical+certificate+AND+disability+benefits&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://api.openalex.org/works?search=medical+documentation+AND+work+incapacity&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://api.openalex.org/works?search=remote+medical+assessment+AND+disability&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telemedicine AND disability assessment`
    - URL: `https://api.openalex.org/works?search=telemedicine+AND+disability+assessment&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telehealth AND medical certification`
    - URL: `https://api.openalex.org/works?search=telehealth+AND+medical+certification&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `disability benefits AND waiting time`
    - URL: `https://api.openalex.org/works?search=disability+benefits+AND+waiting+time&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `medical assessment AND service capacity`
    - URL: `https://api.openalex.org/works?search=medical+assessment+AND+service+capacity&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `case allocation AND disability services`
    - URL: `https://api.openalex.org/works?search=case+allocation+AND+disability+services&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `disability assessment AND equity of access`
    - URL: `https://api.openalex.org/works?search=disability+assessment+AND+equity+of+access&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://api.openalex.org/works?search=remote+assessment+AND+quality+assurance&per-page=20&mailto=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://api.openalex.org/works?search=medical+certification+AND+audit&per-page=20&mailto=REDACTED&api_key=REDACTED`
- **Semantic Scholar** — recuperados: 256
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `disability benefits AND medical assessment`
    - aviso/erro: HTTP 500: {"message": "Internal Server Error"}

  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `incapacity benefit AND medical evaluation`
    - aviso/erro: HTTP 500: {"message": "Internal Server Error"}

  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `work disability AND medical certification`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=work+disability+AND+medical+certification&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `documentary assessment AND disability`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=documentary+assessment+AND+disability&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=medical+certificate+AND+disability+benefits&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Análise documental e elegibilidade — recuperados: 16
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=medical+documentation+AND+work+incapacity&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=remote+medical+assessment+AND+disability&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telemedicine AND disability assessment`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=telemedicine+AND+disability+assessment&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telehealth AND medical certification`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=telehealth+AND+medical+certification&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `disability benefits AND waiting time`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=disability+benefits+AND+waiting+time&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `medical assessment AND service capacity`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=medical+assessment+AND+service+capacity&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `case allocation AND disability services`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=case+allocation+AND+disability+services&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `disability assessment AND equity of access`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=disability+assessment+AND+equity+of+access&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=remote+assessment+AND+quality+assurance&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://api.semanticscholar.org/graph/v1/paper/search?query=medical+certification+AND+audit&limit=20&fields=title%2Cabstract%2Cauthors%2Cyear%2Cvenue%2CexternalIds%2Curl%2CopenAccessPdf%2CcitationCount%2CpublicationTypes`
  - resumo de falhas: Benefícios por incapacidade e avaliação médico-pericial: HTTP 500: {"message": "Internal Server Error"}
; Benefícios por incapacidade e avaliação médico-pericial: HTTP 500: {"message": "Internal Server Error"}

- **Scopus** — recuperados: 15
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 1
    - expressão: `disability benefits AND medical assessment`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22disability+benefits+AND+medical+assessment%22%29&count=10&httpAccept=application%2Fjson`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 1
    - expressão: `incapacity benefit AND medical evaluation`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22incapacity+benefit+AND+medical+evaluation%22%29&count=10&httpAccept=application%2Fjson`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 1
    - expressão: `work disability AND medical certification`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22work+disability+AND+medical+certification%22%29&count=10&httpAccept=application%2Fjson`
  - Análise documental e elegibilidade — recuperados: 1
    - expressão: `documentary assessment AND disability`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22documentary+assessment+AND+disability%22%29&count=10&httpAccept=application%2Fjson`
  - Análise documental e elegibilidade — recuperados: 1
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22medical+certificate+AND+disability+benefits%22%29&count=10&httpAccept=application%2Fjson`
  - Análise documental e elegibilidade — recuperados: 1
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22medical+documentation+AND+work+incapacity%22%29&count=10&httpAccept=application%2Fjson`
  - Teleperícia e avaliação remota — recuperados: 1
    - expressão: `remote medical assessment AND disability`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22remote+medical+assessment+AND+disability%22%29&count=10&httpAccept=application%2Fjson`
  - Teleperícia e avaliação remota — recuperados: 1
    - expressão: `telemedicine AND disability assessment`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22telemedicine+AND+disability+assessment%22%29&count=10&httpAccept=application%2Fjson`
  - Teleperícia e avaliação remota — recuperados: 1
    - expressão: `telehealth AND medical certification`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22telehealth+AND+medical+certification%22%29&count=10&httpAccept=application%2Fjson`
  - Capacidade, filas e alocação de serviços — recuperados: 1
    - expressão: `disability benefits AND waiting time`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22disability+benefits+AND+waiting+time%22%29&count=10&httpAccept=application%2Fjson`
  - Capacidade, filas e alocação de serviços — recuperados: 1
    - expressão: `medical assessment AND service capacity`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22medical+assessment+AND+service+capacity%22%29&count=10&httpAccept=application%2Fjson`
  - Capacidade, filas e alocação de serviços — recuperados: 1
    - expressão: `case allocation AND disability services`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22case+allocation+AND+disability+services%22%29&count=10&httpAccept=application%2Fjson`
  - Equidade territorial, qualidade e controle — recuperados: 1
    - expressão: `disability assessment AND equity of access`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22disability+assessment+AND+equity+of+access%22%29&count=10&httpAccept=application%2Fjson`
  - Equidade territorial, qualidade e controle — recuperados: 1
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22remote+assessment+AND+quality+assurance%22%29&count=10&httpAccept=application%2Fjson`
  - Equidade territorial, qualidade e controle — recuperados: 1
    - expressão: `medical certification AND audit`
    - URL: `https://api.elsevier.com/content/search/scopus?query=TITLE-ABS-KEY%28%22medical+certification+AND+audit%22%29&count=10&httpAccept=application%2Fjson`
- **Web of Science** — recuperados: 0
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `disability benefits AND medical assessment`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `incapacity benefit AND medical evaluation`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `work disability AND medical certification`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `documentary assessment AND disability`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `medical certificate AND disability benefits`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `medical documentation AND work incapacity`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `remote medical assessment AND disability`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telemedicine AND disability assessment`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telehealth AND medical certification`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `disability benefits AND waiting time`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `medical assessment AND service capacity`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `case allocation AND disability services`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `disability assessment AND equity of access`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `remote assessment AND quality assurance`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `medical certification AND audit`
    - aviso/erro: WOS_API_KEY não configurada; a fonte foi ignorada.
  - resumo de falhas: Benefícios por incapacidade e avaliação médico-pericial: WOS_API_KEY não configurada; a fonte foi ignorada.; Benefícios por incapacidade e avaliação médico-pericial: WOS_API_KEY não configurada; a fonte foi ignorada.; Benefícios por incapacidade e avaliação médico-pericial: WOS_API_KEY não configurada; a fonte foi ignorada.; Análise documental e elegibilidade: WOS_API_KEY não configurada; a fonte foi ignorada.; Análise documental e elegibilidade: WOS_API_KEY não configurada; a fonte foi ignorada.; Análise documental e elegibilidade: WOS_API_KEY não configurada; a fonte foi ignorada.; Teleperícia e avaliação remota: WOS_API_KEY não configurada; a fonte foi ignorada.; Teleperícia e avaliação remota: WOS_API_KEY não configurada; a fonte foi ignorada.; Teleperícia e avaliação remota: WOS_API_KEY não configurada; a fonte foi ignorada.; Capacidade, filas e alocação de serviços: WOS_API_KEY não configurada; a fonte foi ignorada.; Capacidade, filas e alocação de serviços: WOS_API_KEY não configurada; a fonte foi ignorada.; Capacidade, filas e alocação de serviços: WOS_API_KEY não configurada; a fonte foi ignorada.; Equidade territorial, qualidade e controle: WOS_API_KEY não configurada; a fonte foi ignorada.; Equidade territorial, qualidade e controle: WOS_API_KEY não configurada; a fonte foi ignorada.; Equidade territorial, qualidade e controle: WOS_API_KEY não configurada; a fonte foi ignorada.
- **PubMed/NCBI** — recuperados: 289
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `disability benefits AND medical assessment`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=disability+benefits+AND+medical+assessment&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `incapacity benefit AND medical evaluation`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=incapacity+benefit+AND+medical+evaluation&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `work disability AND medical certification`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=work+disability+AND+medical+certification&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `documentary assessment AND disability`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=documentary+assessment+AND+disability&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=medical+certificate+AND+disability+benefits&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Análise documental e elegibilidade — recuperados: 9
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=medical+documentation+AND+work+incapacity&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=remote+medical+assessment+AND+disability&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telemedicine AND disability assessment`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=telemedicine+AND+disability+assessment&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telehealth AND medical certification`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=telehealth+AND+medical+certification&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `disability benefits AND waiting time`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=disability+benefits+AND+waiting+time&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `medical assessment AND service capacity`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=medical+assessment+AND+service+capacity&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `case allocation AND disability services`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=case+allocation+AND+disability+services&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `disability assessment AND equity of access`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=disability+assessment+AND+equity+of+access&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=remote+assessment+AND+quality+assurance&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=medical+certification+AND+audit&retmax=20&retmode=json&sort=relevance&tool=AcademicPipelinePRISMA&email=REDACTED&api_key=REDACTED`
- **Europe PMC** — recuperados: 300
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `disability benefits AND medical assessment`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=disability+benefits+AND+medical+assessment&format=json&resultType=core&pageSize=20`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `incapacity benefit AND medical evaluation`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=incapacity+benefit+AND+medical+evaluation&format=json&resultType=core&pageSize=20`
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 20
    - expressão: `work disability AND medical certification`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=work+disability+AND+medical+certification&format=json&resultType=core&pageSize=20`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `documentary assessment AND disability`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=documentary+assessment+AND+disability&format=json&resultType=core&pageSize=20`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical certificate AND disability benefits`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=medical+certificate+AND+disability+benefits&format=json&resultType=core&pageSize=20`
  - Análise documental e elegibilidade — recuperados: 20
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=medical+documentation+AND+work+incapacity&format=json&resultType=core&pageSize=20`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=remote+medical+assessment+AND+disability&format=json&resultType=core&pageSize=20`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telemedicine AND disability assessment`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=telemedicine+AND+disability+assessment&format=json&resultType=core&pageSize=20`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `telehealth AND medical certification`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=telehealth+AND+medical+certification&format=json&resultType=core&pageSize=20`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `disability benefits AND waiting time`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=disability+benefits+AND+waiting+time&format=json&resultType=core&pageSize=20`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `medical assessment AND service capacity`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=medical+assessment+AND+service+capacity&format=json&resultType=core&pageSize=20`
  - Capacidade, filas e alocação de serviços — recuperados: 20
    - expressão: `case allocation AND disability services`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=case+allocation+AND+disability+services&format=json&resultType=core&pageSize=20`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `disability assessment AND equity of access`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=disability+assessment+AND+equity+of+access&format=json&resultType=core&pageSize=20`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `remote assessment AND quality assurance`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=remote+assessment+AND+quality+assurance&format=json&resultType=core&pageSize=20`
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=medical+certification+AND+audit&format=json&resultType=core&pageSize=20`
- **SciELO** — recuperados: 0
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `disability benefits AND medical assessment`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `incapacity benefit AND medical evaluation`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `work disability AND medical certification`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `documentary assessment AND disability`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `medical certificate AND disability benefits`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `medical documentation AND work incapacity`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `remote medical assessment AND disability`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telemedicine AND disability assessment`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telehealth AND medical certification`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `disability benefits AND waiting time`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `medical assessment AND service capacity`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `case allocation AND disability services`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `disability assessment AND equity of access`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `remote assessment AND quality assurance`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `medical certification AND audit`
    - aviso/erro: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
  - resumo de falhas: Benefícios por incapacidade e avaliação médico-pericial: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Benefícios por incapacidade e avaliação médico-pericial: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Benefícios por incapacidade e avaliação médico-pericial: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Análise documental e elegibilidade: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Análise documental e elegibilidade: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Análise documental e elegibilidade: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Teleperícia e avaliação remota: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Teleperícia e avaliação remota: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Teleperícia e avaliação remota: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Capacidade, filas e alocação de serviços: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Capacidade, filas e alocação de serviços: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Capacidade, filas e alocação de serviços: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Equidade territorial, qualidade e controle: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Equidade territorial, qualidade e controle: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"; Equidade territorial, qualidade e controle: HTTP 403: <!DOCTYPE html>
<html>
<head>
    <title>Establishing a secure connection ...</title>
    <meta name="viewport" content="width=device-width, initial-scale=0.8">
    <link href="/.bunny-shield/assets/challenge-styles.css" rel="stylesheet" />
    <script src="/.bunny-shield/assets/shield-challenge.js"
- **CORE** — recuperados: 58
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `disability benefits AND medical assessment`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `incapacity benefit AND medical evaluation`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Benefícios por incapacidade e avaliação médico-pericial — recuperados: 0
    - expressão: `work disability AND medical certification`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `documentary assessment AND disability`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Análise documental e elegibilidade — recuperados: 0
    - expressão: `medical certificate AND disability benefits`
    - aviso/erro: HTTP 500: {"message":"Azure search failed with status code: 503. Error context: message=Failed to execute request because the request rate has caused your service to exceed the limits of its provisioned capacity. Reduce the rate of requests, or adjust the number of replicas\/partitions. See http:\/\/aka.ms\/a
  - Análise documental e elegibilidade — recuperados: 18
    - expressão: `medical documentation AND work incapacity`
    - URL: `https://api.core.ac.uk/v3/search/works?q=medical+documentation+AND+work+incapacity&limit=20`
  - Teleperícia e avaliação remota — recuperados: 20
    - expressão: `remote medical assessment AND disability`
    - URL: `https://api.core.ac.uk/v3/search/works?q=remote+medical+assessment+AND+disability&limit=20`
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telemedicine AND disability assessment`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Teleperícia e avaliação remota — recuperados: 0
    - expressão: `telehealth AND medical certification`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `disability benefits AND waiting time`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `medical assessment AND service capacity`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Capacidade, filas e alocação de serviços — recuperados: 0
    - expressão: `case allocation AND disability services`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `disability assessment AND equity of access`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Equidade territorial, qualidade e controle — recuperados: 0
    - expressão: `remote assessment AND quality assurance`
    - aviso/erro: Tempo esgotado na consulta bibliográfica.
  - Equidade territorial, qualidade e controle — recuperados: 20
    - expressão: `medical certification AND audit`
    - URL: `https://api.core.ac.uk/v3/search/works?q=medical+certification+AND+audit&limit=20`
  - resumo de falhas: Benefícios por incapacidade e avaliação médico-pericial: Tempo esgotado na consulta bibliográfica.; Benefícios por incapacidade e avaliação médico-pericial: Tempo esgotado na consulta bibliográfica.; Benefícios por incapacidade e avaliação médico-pericial: Tempo esgotado na consulta bibliográfica.; Análise documental e elegibilidade: Tempo esgotado na consulta bibliográfica.; Análise documental e elegibilidade: HTTP 500: {"message":"Azure search failed with status code: 503. Error context: message=Failed to execute request because the request rate has caused your service to exceed the limits of its provisioned capacity. Reduce the rate of requests, or adjust the number of replicas\/partitions. See http:\/\/aka.ms\/a; Teleperícia e avaliação remota: Tempo esgotado na consulta bibliográfica.; Teleperícia e avaliação remota: Tempo esgotado na consulta bibliográfica.; Capacidade, filas e alocação de serviços: Tempo esgotado na consulta bibliográfica.; Capacidade, filas e alocação de serviços: Tempo esgotado na consulta bibliográfica.; Capacidade, filas e alocação de serviços: Tempo esgotado na consulta bibliográfica.; Equidade territorial, qualidade e controle: Tempo esgotado na consulta bibliográfica.; Equidade territorial, qualidade e controle: Tempo esgotado na consulta bibliográfica.

## Pré-triagem assistida por IA
- Modelo: gpt-5.4.
- Registros avaliados: 800 de 1300.
- Lotes concluídos/falhos: 40/0.
- A IA apenas prioriza e justifica a ordem de revisão; ela não inclui nem exclui estudos definitivamente.

## Próxima etapa obrigatória
Revise a planilha ordenada por aderência. A decisão humana continua obrigatória para inclusão, exclusão e elegibilidade de texto completo.
