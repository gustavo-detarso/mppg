# -*- coding: utf-8 -*-
# Diagnóstico isolado do chunk "codigo-poisson-conferencia"
# Execução na pasta caso1_FGV: source("diagnostico_poisson.R", encoding="UTF-8")
# Usa somente as bibliotecas estatísticas da instalação R.
T <- c(33, 28, 33, 35, 31, 32, 34, 34, 26, 32)
Y <- c(1, 2, 3, 2, 2, 3, 1, 3, 2, 1)
N <- rep(100L, 10)
dados <- data.frame(temperatura=T, atendimentos=Y, trabalhadores=N)
modelo <- glm(atendimentos ~ temperatura, data=dados,
              family=poisson(link="log"), offset=log(trabalhadores))
mu <- as.numeric(fitted(modelo))
I_beta <- sum(mu * (T - weighted.mean(T, mu))^2)
ep_manual <- 1 / sqrt(I_beta)
ep_vcov <- sqrt(as.numeric(vcov(modelo)["temperatura", "temperatura"]))
beta <- unname(coef(modelo)["temperatura"])
IC_manual <- as.numeric(exp(beta + c(-1,1)*qnorm(.975)*ep_manual))
IC_vcov <- as.numeric(exp(beta + c(-1,1)*qnorm(.975)*ep_vcov))
IC_default <- as.numeric(exp(confint.default(modelo)["temperatura",]))
cat("Versão R:", R.version.string, "\n")
cat("beta:", beta, "\n")
cat("EP informação Fisher:", ep_manual, "\n")
cat("EP vcov(glm):", ep_vcov, "\n")
cat("IC informação Fisher:", paste(IC_manual, collapse=" / "), "\n")
cat("IC vcov(glm):", paste(IC_vcov, collapse=" / "), "\n")
cat("IC confint.default:", paste(IC_default, collapse=" / "), "\n")
cat("Diferença máxima IC manual × confint.default:",
    max(abs(IC_manual - IC_default)), "\n")
stopifnot(is.finite(ep_manual), is.finite(ep_vcov),
          abs(ep_manual-ep_vcov)<1e-6,
          max(abs(IC_vcov-IC_default))<1e-6)
cat("Conferência de Poisson: APROVADA.\n")
