# -*- coding: utf-8 -*-
# Memoria de calculo independente: Caso 1 — Antes do turno da tarde.
# Executar no ESS com source("verificar_calculos_passo_a_passo.R")
# Usa apenas R base (stats); nao precisa de pacotes adicionais.

turno <- c("D02","D11","D14","D15","D16","D18","D20","D21","D22","D23")
T <- c(33,28,33,35,31,32,34,34,26,32)
Y <- c(1,2,3,2,2,3,1,3,2,1)
N <- rep(100L, length(T))
n <- length(T)
stopifnot(n == 10L, sum(Y) == 20L)

cat("\n1. SOMATORIOS E MEDIAS\n")
stat <- c(n=n, soma_T=sum(T), soma_Y=sum(Y), soma_T2=sum(T^2),
          soma_Y2=sum(Y^2), soma_TY=sum(T*Y), media_T=mean(T), media_Y=mean(Y))
print(stat)

cat("\n2. TABELA DE PARCELAS\n")
dT <- T - mean(T)
dY <- Y - mean(Y)
memoria <- data.frame(turno,T,Y,desvio_T=dT,desvio_Y=dY,
                     quad_T=dT^2, quad_Y=dY^2, produto=dT*dY)
print(memoria, row.names=FALSE)
Sxx <- sum(dT^2); Syy <- sum(dY^2); Sxy <- sum(dT*dY)
r_manual <- Sxy / sqrt(Sxx*Syy)
cat(sprintf("Sxx=%.1f; Syy=%.1f; Sxy=%.1f; r=%.5f\n",Sxx,Syy,Sxy,r_manual))

cat("\n3. REGRESSAO LINEAR, ERRO-PADRAO E INTERVALO\n")
beta1 <- Sxy/Sxx
beta0 <- mean(Y)-beta1*mean(T)
residuos <- Y-(beta0+beta1*T)
sigma2 <- sum(residuos^2)/(n-2)
EP <- sqrt(sigma2/Sxx)
tcrit <- qt(.975, df=n-2)
IC_lm <- beta1 + c(-1,1)*tcrit*EP
print(c(intercepto=beta0,inclinacao=beta1,SQE=sum(residuos^2),
        variancia_residual=sigma2,EP_inclinacao=EP,t_critico=tcrit,
        IC_inferior=IC_lm[1],IC_superior=IC_lm[2]))

cat("\n4. POISSON COM OFFSET\n")
# Verificar analiticamente escore no ponto beta=0 e mu=2.
mu <- rep(mean(Y),n)
escores <- c(intercepto=sum(Y-mu),temperatura=sum(T*(Y-mu)))
Ibeta <- sum(mu*(T-mean(T))^2)
EP_pois <- 1/sqrt(Ibeta)
IC_beta <- c(-1,1)*qnorm(.975)*EP_pois
RT <- exp(0)
IC_RT <- exp(IC_beta)
print(escores)
print(c(alpha=log(mean(Y)/100),beta=0, informacao=Ibeta,
        EP_beta=EP_pois,razao_taxas=RT,RT_inferior=IC_RT[1],RT_superior=IC_RT[2]))

cat("\n4.1. DIAGNOSTICO DE DISPERSAO E LEAVE-ONE-OUT\n")
variancia_amostral <- var(Y)
pearson_X2 <- sum((Y - mu)^2 / mu)
razao_disp <- pearson_X2 / (n-2)
sem_um <- vapply(seq_len(n), function(i) cor(T[-i], Y[-i]), numeric(1))
print(data.frame(turno_omitido = turno, r_sem_turno = sem_um), row.names = FALSE)
print(c(media=mean(Y), variancia_amostral=variancia_amostral,
        Pearson_X2=pearson_X2, gl=n-2, razao_dispersao=razao_disp,
        r_min=min(sem_um), r_max=max(sem_um)))
stopifnot(abs(razao_disp-0.375)<1e-12,
          max(abs(sem_um))<0.136,
          min(sem_um)<0, max(sem_um)>0)

cat("\n5. BINOMIAL EXATO — APENDICE DIDATICO: HIPOTESES NAO VERIFICADAS\n")
i35 <- which(T==35)
stopifnot(length(i35)==1L)
IC_bin <- c(qbeta(.025,Y[i35],N[i35]-Y[i35]+1L),
            qbeta(.975,Y[i35]+1L,N[i35]-Y[i35]))
print(c(observado=Y[i35]/N[i35], IC_inferior=IC_bin[1],IC_superior=IC_bin[2]))

cat("\n6. ORCAMENTO\n")
print(c(limite=40000,custo_reorganizacao=32000,saldo=40000-32000,
        fracao_autonomia=32000/40000,excedente_substituto=45000-40000))

cat("\n7. CONFERENCIA COM FUNCOES R\n")
stopifnot(isTRUE(all.equal(unname(cor(T,Y)), r_manual, tolerance=1e-12)))
lm_fit <- lm(Y ~ T)
stopifnot(abs(unname(coef(lm_fit)[2])-beta1)<1e-11)
stopifnot(max(abs(confint(lm_fit)[2,]-IC_lm))<1e-10)
glm_fit <- glm(Y ~ T, poisson(link="log"), offset=log(N))
stopifnot(abs(unname(coef(glm_fit)[2]))<1e-10)
stopifnot(max(abs(exp(confint.default(glm_fit)[2,])-IC_RT))<1e-9)
stopifnot(max(abs(binom.test(Y[i35],N[i35])$conf.int-IC_bin))<1e-12)
cat("Conferencia aprovada: contas manuais = R base, dentro das tolerancias.\n")
cat("Nota: nenhuma dessas estimativas comprova seguranca ocupacional.\n")
cat("O intervalo binomial exige eventos individuais e independencia, nao documentados no caso.\n")
