# Métodos Quantitativos — MPPG/FGV

## Organização

- `aulas/`: materiais de correlação e regressão OLS.
- `monitorias/`: exercícios, resoluções e históricos.
- `casos/caso1_fgv/`: Caso 1, código R e relatório.
- `casos/caso2_fgv/`: Caso 2, dados do Inep e relatório.
- `bibliografia/`: bibliografia complementar.
- `materiais_complementares/`: documentos e links.
- `protocolo/`: Prompt Master e registros de governança.
- `pacotes/`: arquivos de entrega preservados.
- `a_classificar/`: documentos pendentes de classificação.
- `.organizacao/`: manifestos locais de migração.

## Ambiente

- Debian 12
- R 4.6.1 (versão verificada em 30/10/2026)
- Emacs, ESS e R Markdown

Pacotes utilizados incluem rmarkdown, knitr, readxl,
ggplot2 e openxlsx, conforme as necessidades dos trabalhos.

## Renderização

Executar o R a partir da pasta correspondente ao trabalho.

Caso 1:
`rmarkdown::render("relatorio_caso1.Rmd")`

Caso 2:
`rmarkdown::render("caso2_FGV.Rmd")`

Monitoria 1:
`rmarkdown::render("Monitoria_1_resolvida_final.Rmd")`

## Reprodutibilidade

Em 30/10/2026, os três relatórios foram renderizados
com sucesso em cópias isoladas de seus diretórios.

A renderização depende dos arquivos auxiliares e das
bibliotecas R disponíveis no ambiente.

O Caso 2 utiliza fonte oficial do Inep preservada
localmente. O repositório público pode não incluir
todos os dados necessários à reprodução integral.

## Versionamento

Materiais didáticos de terceiros, documentos de
distribuição restrita, backups e pacotes de entrega
não devem ser publicados sem revisão.

A publicação do código-fonte não implica homologação
científica integral dos resultados.
