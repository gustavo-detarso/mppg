# -*- coding: utf-8 -*-
# Caso 1 | Antes do turno da tarde | MPPG — Métodos Quantitativos
# Executável no GNU Emacs + ESS (ou pelo Rscript).
# O relatório HTML completo é produzido por:
# rmarkdown::render("relatorio_caso1.Rmd", output_file = "relatorio_caso1.html")
#
# Pacotes para toda a solução:
# install.packages(c("ggplot2", "dplyr", "patchwork", "scales", "knitr", "rmarkdown"),
#                  repos = "https://cloud.r-project.org")

pacotes_necessarios <- c("ggplot2", "dplyr", "patchwork", "scales")
faltantes <- pacotes_necessarios[
  !vapply(pacotes_necessarios, requireNamespace, logical(1), quietly = TRUE)
]
if (length(faltantes) > 0L) {
  stop("Instale os pacotes faltantes com install.packages(c(",
       paste(sprintf('"%s"', faltantes), collapse = ", "),
       '), repos = "https://cloud.r-project.org")')
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(scales)
})

# 1. Transcrição conferida com a tabela da página 2 do documento.
dados <- data.frame(
  turno = c("D02", "D11", "D14", "D15", "D16",
            "D18", "D20", "D21", "D22", "D23"),
  temperatura = c(33, 28, 33, 35, 31, 32, 34, 34, 26, 32),
  atendimentos = c(1, 2, 3, 2, 2, 3, 1, 3, 2, 1),
  trabalhadores = rep(100L, 10L),
  stringsAsFactors = FALSE
)
stopifnot(nrow(dados) == 10L,
          all(dados$trabalhadores == 100L),
          sum(dados$atendimentos) == 20L,
          all(dados$atendimentos >= 0L))

# 2. Estatísticas descritivas: unidade de análise = turno, não pessoa única.
n_turnos <- nrow(dados)
total_atendimentos <- sum(dados$atendimentos)
total_turnos_trabalhador <- sum(dados$trabalhadores)
media_atendimentos <- mean(dados$atendimentos)
proporcao_turnos_trabalhador <- total_atendimentos / total_turnos_trabalhador
turnos_por_temperatura <- dados |>
  group_by(temperatura) |>
  summarise(n_turnos = n(),
            media_atendimentos = mean(atendimentos),
            .groups = "drop")

# 3. Correlação, regressão linear e intervalo de confiança da inclinação.
teste_pearson <- cor.test(dados$temperatura, dados$atendimentos,
                          method = "pearson")
modelo_lm <- lm(atendimentos ~ temperatura, data = dados)
ic_lm <- confint(modelo_lm, level = 0.95)

# 4. Modelo para contagens por turno (sem interpretar associação como causal).
# O offset seria log(trabalhadores), caso as equipes variassem entre turnos.
modelo_poisson <- glm(atendimentos ~ temperatura,
                      family = poisson(link = "log"),
                      offset = log(trabalhadores),
                      data = dados)
ic_poisson <- confint.default(modelo_poisson, level = 0.95)
razao_taxas <- unname(exp(coef(modelo_poisson)["temperatura"]))
ic_razao_taxas <- exp(ic_poisson["temperatura", ])

# 4.1. Dispersão de Pearson do GLM de Poisson e sensibilidade leave-one-out.
variancia_amostral_atendimentos <- var(dados$atendimentos)
X2_pearson_poisson <- sum(residuals(modelo_poisson, type = "pearson")^2)
gl_residuais_poisson <- df.residual(modelo_poisson)
dispersao_pearson_poisson <- X2_pearson_poisson / gl_residuais_poisson
sensibilidade <- data.frame(
  turno_omitido = dados$turno,
  temperatura_omitida = dados$temperatura,
  atendimentos_omitidos = dados$atendimentos,
  r_sem_turno = vapply(seq_len(nrow(dados)), function(i) {
    cor(dados$temperatura[-i], dados$atendimentos[-i])
  }, numeric(1)), stringsAsFactors = FALSE
)
stopifnot(abs(dispersao_pearson_poisson - 0.375) < 1e-9,
          abs(min(sensibilidade$r_sem_turno) + 0.135854244832) < 1e-8,
          abs(max(sensibilidade$r_sem_turno) - 0.135854244832) < 1e-8)

# 5. Intervalo binomial SOMENTE para apêndice didático: pressupostos NÃO verificados.

# Exige suposições fortes e não representa uma previsão clínica para hoje.
dados_35 <- subset(dados, temperatura == 35)
stopifnot(nrow(dados_35) == 1L)
ic_35_ilustrativo <- binom.test(dados_35$atendimentos,
                              dados_35$trabalhadores)$conf.int

# 6. Predição da média do GLM e IC95% do valor médio (NÃO intervalo preditivo).
faixa <- data.frame(
  temperatura = seq(min(dados$temperatura), max(dados$temperatura),
                    length.out = 180L),
  trabalhadores = 100L
)
pred_link <- predict(modelo_poisson, newdata = faixa,
                     type = "link", se.fit = TRUE)
faixa$estimativa <- exp(pred_link$fit)
faixa$inferior <- exp(pred_link$fit - qnorm(0.975) * pred_link$se.fit)
faixa$superior <- exp(pred_link$fit + qnorm(0.975) * pred_link$se.fit)

# 7. Custos conhecidos; NÃO atribuir cobertura ou custo que o caso não informa.
alternativas_custo <- data.frame(
  alternativa = c("Turno original", "Reorganização", "Serviço substituto"),
  custo = c(0, 32000, 45000),
  situacao = c("Sem gasto adicional programado",
               "Dentro da autonomia",
               "Requer autorização adicional"),
  stringsAsFactors = FALSE
)
teto_orcamento <- 40000
saldo_reorganizacao <- teto_orcamento - 32000

# 8. Estética de publicação.
cor_azul <- "#003B71"
cor_turquesa <- "#006BB6"
cor_laranja <- "#C67931"
cor_cinza <- "#566B7F"
cor_claro <- "#E3F1FC"

tema_caso <- function() {
  theme_minimal(base_size = 12, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", color = cor_azul, size = 14),
      plot.subtitle = element_text(color = cor_cinza, size = 10),
      axis.title = element_text(color = cor_azul, size = 10),
      axis.text = element_text(color = cor_cinza),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 12, 12, 10)
    )
}

# Figura 1: reproduz o achado original sem ocultar dispersão.
g1 <- ggplot(dados, aes(x = temperatura, y = atendimentos)) +
  geom_hline(yintercept = media_atendimentos,
             linetype = "dashed", color = cor_turquesa, linewidth = 0.85) +
  geom_vline(xintercept = 35, linetype = "dotted",
             color = cor_laranja, linewidth = 0.75) +
  geom_point(size = 3.5, color = cor_azul, alpha = 0.95) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              linewidth = 0.75, color = cor_turquesa) +
  scale_x_continuous(breaks = seq(26, 35, 1)) +
  scale_y_continuous(breaks = 0:4, limits = c(0, 4.25)) +
  labs(title = "01 · Temperatura × atendimentos",
       subtitle = "10 turnos; linha ajustada sem inclinação",
       x = "Temperatura máxima (°C)", y = "Atendimentos por turno") +
  tema_caso()

# Figura 2: tamanho da evidência por faixa de temperatura.
g2 <- ggplot(turnos_por_temperatura,
             aes(x = factor(temperatura), y = n_turnos)) +
  geom_col(width = 0.65, fill = cor_turquesa) +
  geom_text(aes(label = n_turnos), vjust = -0.45,
            size = 4.1, color = cor_azul, fontface = "bold") +
  scale_y_continuous(breaks = 0:3, limits = c(0, 3.1)) +
  labs(title = "02 · Quantos turnos por temperatura?",
       subtitle = "Apenas um turno observado a 35°C",
       x = "Temperatura máxima (°C)", y = "Número de turnos") +
  tema_caso()

# Figura 3: incerteza na média sob hipóteses do GLM.
g3 <- ggplot() +
  geom_ribbon(data = faixa, aes(x = temperatura,
              ymin = inferior, ymax = superior),
              fill = cor_claro, alpha = 0.97) +
  geom_line(data = faixa, aes(x = temperatura, y = estimativa),
            color = cor_turquesa, linewidth = 1.05) +
  geom_point(data = dados,
             aes(x = temperatura, y = atendimentos),
             color = cor_azul, size = 2.7, alpha = 0.9) +
  geom_vline(xintercept = 35, linetype = "dotted",
             color = cor_laranja, linewidth = 0.75) +
  scale_x_continuous(breaks = seq(26, 35, 1)) +
  labs(title = "03 · Modelo de Poisson com IC95%",
       subtitle = "Faixa: incerteza da MÉDIA estimada, não do risco individual",
       x = "Temperatura máxima (°C)", y = "Atendimentos estimados / 100 trabalhadores") +
  tema_caso()

# Figura 4: somente custos adicionais explicitamente informados.
g4 <- ggplot(alternativas_custo,
             aes(x = custo, y = reorder(alternativa, custo))) +
  geom_col(width = 0.62, fill = cor_azul) +
  geom_vline(xintercept = teto_orcamento, linetype = "dashed",
             color = cor_laranja, linewidth = 0.95) +
  geom_text(aes(label = paste0("R$ ",
                              format(custo, big.mark = ".", scientific = FALSE,
                                     trim = TRUE))),
            hjust = -0.12, color = cor_azul, fontface = "bold", size = 3.8) +
  scale_x_continuous(limits = c(0, 53500),
                     breaks = c(0, 10000, 20000, 30000, 40000, 50000),
                     labels = function(x) paste0("R$ ", x / 1000, " mil")) +
  labs(title = "04 · Desembolso adicional conhecido",
       subtitle = "Linha laranja: autonomia de R$ 40 mil",
       x = "Custo adicional informado", y = NULL) +
  tema_caso() +
  theme(axis.text.y = element_text(face = "bold"))

painel <- (g1 | g2) / (g3 | g4) +
  plot_annotation(
    title = "ANTES DO TURNO DA TARDE",
    subtitle = "O que os dados mostram — e o que não permitem afirmar",
    caption = "Fonte: Caso 1, relatório da direção (p. 2). Custos: enunciado (p. 1).",
    theme = theme(
      plot.title = element_text(size = 19, face = "bold", color = cor_azul),
      plot.subtitle = element_text(size = 11, color = cor_cinza),
      plot.caption = element_text(size = 9, color = cor_cinza)
    )
  )

# 9. Saídas automáticas. Executar a partir da pasta do projeto.
dir.create("saidas", showWarnings = FALSE, recursive = TRUE)
ggsave(filename = file.path("saidas", "painel_analitico.png"),
       plot = painel, width = 14, height = 9.8, dpi = 185, bg = "white")
write.csv2(dados, file.path("saidas", "dados_digitados.csv"),
           row.names = FALSE, fileEncoding = "UTF-8")

resumo_resultados <- data.frame(
  indicador = c("Turnos", "Turnos-trabalhador", "Atendimentos",
                "Média de atendimentos por turno", "Proporção de turnos-trabalhador",
                "Correlação de Pearson", "Valor-p de Pearson",
                "Inclinação linear (atendimentos por °C)",
                "IC95% inferior da inclinação", "IC95% superior da inclinação",
                "Razão de taxas Poisson por °C",
                "IC95% inferior Poisson", "IC95% superior Poisson",
                "Dispersão de Pearson / GLM",
                "Sensibilidade leave-one-out (r mínimo)",
                "Sensibilidade leave-one-out (r máximo)",
                "Custo reorganização", "Saldo da autonomia"),
  valor = c(n_turnos, total_turnos_trabalhador, total_atendimentos,
            media_atendimentos, proporcao_turnos_trabalhador,
            unname(teste_pearson$estimate), teste_pearson$p.value,
            unname(coef(modelo_lm)["temperatura"]),
            ic_lm["temperatura", 1], ic_lm["temperatura", 2],
            razao_taxas, ic_razao_taxas[1], ic_razao_taxas[2],
            dispersao_pearson_poisson, min(sensibilidade$r_sem_turno),
            max(sensibilidade$r_sem_turno), 32000, saldo_reorganizacao)
)
write.csv2(resumo_resultados,
           file.path("saidas", "resultados_estatisticos.csv"),
           row.names = FALSE, fileEncoding = "UTF-8")

# 10. Resumo do console/ESS.
cat("\n--- CASO 1 | RESUMO ---\n")
cat("Turnos:", n_turnos, "| atendimentos:", total_atendimentos,
    "| média:", sprintf("%.2f", media_atendimentos), "\n")
cat("Correlação de Pearson:", sprintf("%.3f", unname(teste_pearson$estimate)),
    "| p:", sprintf("%.3f", teste_pearson$p.value), "\n")
cat("IC95% da inclinação linear:",
    paste(sprintf("%.3f", ic_lm["temperatura", ]), collapse = " a "), "\n")
cat("Razão de taxas Poisson/°C:", sprintf("%.3f", razao_taxas),
    "| IC95%:", paste(sprintf("%.3f", ic_razao_taxas), collapse = " a "), "\n")
cat("Gráfico em: saidas/painel_analitico.png\n")
cat("ATENÇÃO: estes registros não provam a inexistência de risco pelo calor.\n\n")

# Para exibir o painel na janela gráfica do R/ESS, execute no console: print(painel)
