#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
dados_dir <- if (length(args) >= 1) args[[1]] else "/home/gustavodetarso/Documentos/mppg/disciplinas/04_decisoes_baseadas_em_evidencia/atividades/artigo/dados_prisma"
out_dir <- if (length(args) >= 2) args[[2]] else dados_dir

dados_dir <- normalizePath(dados_dir, mustWork = TRUE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

csvs <- list.files(dados_dir, pattern = "referencias_incluidas_seminario[.]csv$", full.names = TRUE)
if (length(csvs) == 0) {
  csvs <- list.files(dados_dir, pattern = "referencias_incluidas.*[.]csv$", full.names = TRUE)
}
if (length(csvs) == 0) {
  stop("Nenhum CSV de referências incluídas encontrado em: ", dados_dir)
}
input_csv <- csvs[[1]]

read_any_csv <- function(path) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  sample <- paste(head(txt, 5), collapse = "\n")
  sep <- if (grepl(";", sample) && !grepl(",", sample)) ";" else ","
  tryCatch(
    read.csv(path, sep = sep, stringsAsFactors = FALSE, check.names = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(e) read.csv(path, sep = ifelse(sep == ";", ",", ";"), stringsAsFactors = FALSE, check.names = FALSE, fileEncoding = "UTF-8-BOM")
  )
}

norm_names <- function(x) {
  x <- iconv(x, from = "", to = "ASCII//TRANSLIT")
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_|_$", "", x)
}

pick_col <- function(df, candidates) {
  nn <- norm_names(names(df))
  cc <- norm_names(candidates)
  idx <- match(cc, nn)
  idx <- idx[!is.na(idx)]
  if (length(idx) == 0) return(rep("", nrow(df)))
  as.character(df[[idx[[1]]]])
}

first_nonempty <- function(...) {
  cols <- list(...)
  if (length(cols) == 0) return(character(0))
  out <- cols[[1]]
  if (length(cols) > 1) {
    for (i in 2:length(cols)) {
      repl <- is.na(out) | trimws(out) == ""
      out[repl] <- cols[[i]][repl]
    }
  }
  out[is.na(out)] <- ""
  out
}

contains_any <- function(x, pats) {
  x <- tolower(iconv(x, from = "", to = "ASCII//TRANSLIT"))
  Reduce(`|`, lapply(pats, function(p) grepl(p, x, ignore.case = TRUE, perl = TRUE)))
}

df <- read_any_csv(input_csv)
n <- nrow(df)

titulo <- first_nonempty(pick_col(df, c("titulo", "título", "title", "article_title")), rep("", n))
autores <- first_nonempty(pick_col(df, c("autores", "authors", "author", "criadores", "creators")), rep("", n))
ano_raw <- first_nonempty(pick_col(df, c("ano", "year", "publication_year", "published_year", "published", "data_publicacao")), rep("", n))
periodico <- first_nonempty(pick_col(df, c("periodico", "periódico", "journal", "journaltitle", "venue", "source", "container_title", "publication")), rep("", n))
doi <- first_nonempty(pick_col(df, c("doi")), rep("", n))
url <- first_nonempty(pick_col(df, c("url", "link", "source_url", "landing_page")), rep("", n))
resumo <- first_nonempty(pick_col(df, c("resumo", "abstract", "summary")), rep("", n))

texto <- paste(titulo, resumo, periodico)
ano <- suppressWarnings(as.integer(ano_raw))
miss <- is.na(ano)
if (any(miss)) {
  mm <- regmatches(texto[miss], regexpr("(19|20)[0-9]{2}", texto[miss]))
  ano[miss] <- suppressWarnings(as.integer(ifelse(mm == "", NA, mm)))
}

eixo_map <- list(
  "Telemedicina, teleperícia e consulta remota" = c("telemed", "telehealth", "video", "remote", "virtual", "telessa", "teleper"),
  "Incapacidade funcional, ICF e capacidade laboral" = c("disability", "functioning", "icf", "incapacidade", "funcional", "capacidade", "work disability"),
  "Certificação médica, benefícios e seguridade social" = c("sickness", "benefit", "social security", "certification", "previd", "seguridade", "beneficio"),
  "Instrumentos, escalas e padronização da avaliação" = c("scale", "instrument", "assessment", "evaluation", "avaliacao", "classific", "mini-icf", "instrumento"),
  "IA, digitalização e apoio documental" = c("artificial intelligence", "\\bai\\b", "digital", "ipad", "application", "app", "document", "automat")
)

eixos_long <- data.frame()
for (eixo in names(eixo_map)) {
  hit <- contains_any(texto, eixo_map[[eixo]])
  if (any(hit)) {
    eixos_long <- rbind(
      eixos_long,
      data.frame(referencia = which(hit), eixo = eixo, titulo = titulo[hit], ano = ano[hit], stringsAsFactors = FALSE)
    )
  }
}
if (nrow(eixos_long) == 0) {
  eixos_long <- data.frame(referencia = integer(), eixo = character(), titulo = character(), ano = integer(), stringsAsFactors = FALSE)
}

refs <- data.frame(
  id = seq_len(n),
  autores = autores,
  ano = ano,
  titulo = titulo,
  periodico = periodico,
  doi = doi,
  url = url,
  eixo_inferido = sapply(seq_len(n), function(i) {
    e <- unique(eixos_long$eixo[eixos_long$referencia == i])
    if (length(e) == 0) "Não classificado automaticamente" else paste(e, collapse = "; ")
  }),
  stringsAsFactors = FALSE
)

por_ano <- as.data.frame(table(ano = refs$ano, useNA = "no"), stringsAsFactors = FALSE)
names(por_ano) <- c("ano", "n_referencias")
por_ano$ano <- as.integer(as.character(por_ano$ano))
por_ano <- por_ano[order(por_ano$ano), , drop = FALSE]

por_eixo <- as.data.frame(table(eixo = eixos_long$eixo), stringsAsFactors = FALSE)
names(por_eixo) <- c("eixo", "n_ocorrencias")
por_eixo <- por_eixo[order(-por_eixo$n_ocorrencias, por_eixo$eixo), , drop = FALSE]

por_periodico <- as.data.frame(sort(table(refs$periodico[trimws(refs$periodico) != ""]), decreasing = TRUE), stringsAsFactors = FALSE)
names(por_periodico) <- c("periodico", "n_referencias")

write.csv(refs, file.path(out_dir, "insumos_r_artigo_referencias_enriquecidas.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(por_ano, file.path(out_dir, "insumos_r_artigo_referencias_por_ano.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(por_eixo, file.path(out_dir, "insumos_r_artigo_eixos_tematicos.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(por_periodico, file.path(out_dir, "insumos_r_artigo_periodicos.csv"), row.names = FALSE, fileEncoding = "UTF-8")

png(file.path(out_dir, "insumos_r_artigo_grafico_referencias_por_ano.png"), width = 1400, height = 900, res = 140)
if (nrow(por_ano) > 0) {
  barplot(por_ano$n_referencias, names.arg = por_ano$ano, las = 2,
          main = "Referências incluídas por ano de publicação",
          xlab = "Ano", ylab = "Número de referências")
} else {
  plot.new(); title("Referências incluídas por ano de publicação"); text(0.5, 0.5, "Ano não identificado nos metadados")
}
dev.off()

png(file.path(out_dir, "insumos_r_artigo_grafico_eixos_tematicos.png"), width = 1600, height = 900, res = 140)
if (nrow(por_eixo) > 0) {
  par(mar = c(5, 14, 4, 2))
  barplot(rev(por_eixo$n_ocorrencias), names.arg = rev(por_eixo$eixo), horiz = TRUE, las = 1,
          main = "Eixos temáticos inferidos nas referências incluídas",
          xlab = "Número de ocorrências")
} else {
  plot.new(); title("Eixos temáticos inferidos nas referências incluídas"); text(0.5, 0.5, "Eixos não classificados automaticamente")
}
dev.off()

md <- file.path(out_dir, "insumos_r_artigo_resumo_estatistico.md")
sink(md)
cat("# Insumos estatísticos em R para o artigo\n\n")
cat("Arquivo analisado: `", input_csv, "`\n\n", sep = "")
cat("## Estatísticas descritivas do corpus selecionado\n\n")
cat("- Referências incluídas no corpus final: ", n, "\n", sep = "")
cat("- Referências com DOI identificado: ", sum(trimws(refs$doi) != ""), " (", round(100 * mean(trimws(refs$doi) != ""), 1), "%)\n", sep = "")
if (any(!is.na(refs$ano))) {
  cat("- Intervalo de anos identificado: ", min(refs$ano, na.rm = TRUE), "–", max(refs$ano, na.rm = TRUE), "\n", sep = "")
  cat("- Mediana do ano de publicação: ", median(refs$ano, na.rm = TRUE), "\n", sep = "")
}
cat("\n## Tabela 1 — Matriz das referências incluídas\n\n")
cat("| ID | Ano | Autores | Título | Periódico | Eixo inferido |\n")
cat("|---:|---:|---|---|---|---|\n")
for (i in seq_len(nrow(refs))) {
  cat("| ", refs$id[i], " | ", ifelse(is.na(refs$ano[i]), "", refs$ano[i]), " | ",
      gsub("\\|", "/", refs$autores[i]), " | ",
      gsub("\\|", "/", refs$titulo[i]), " | ",
      gsub("\\|", "/", refs$periodico[i]), " | ",
      gsub("\\|", "/", refs$eixo_inferido[i]), " |\n", sep = "")
}
cat("\n## Tabela 2 — Referências por ano\n\n")
cat("| Ano | N referências |\n")
cat("|---:|---:|\n")
if (nrow(por_ano) > 0) {
  for (i in seq_len(nrow(por_ano))) cat("| ", por_ano$ano[i], " | ", por_ano$n_referencias[i], " |\n", sep = "")
}
cat("\n## Tabela 3 — Eixos temáticos inferidos\n\n")
cat("| Eixo temático | N ocorrências |\n")
cat("|---|---:|\n")
if (nrow(por_eixo) > 0) {
  for (i in seq_len(nrow(por_eixo))) cat("| ", por_eixo$eixo[i], " | ", por_eixo$n_ocorrencias[i], " |\n", sep = "")
}
cat("\n## Figuras geradas em R\n\n")
cat("- `insumos_r_artigo_grafico_referencias_por_ano.png`\n")
cat("- `insumos_r_artigo_grafico_eixos_tematicos.png`\n")
cat("\n## Nota metodológica\n\n")
cat("As estatísticas acima são descritivas do conjunto de referências incluídas, não medidas de efeito clínico ou causal. Elas devem ser usadas para caracterizar o corpus, orientar a síntese narrativa e estruturar tabelas do artigo, sem inferir impactos quantitativos que não tenham sido extraídos diretamente dos estudos.\n")
sink()

cat("[OK] Insumos estatísticos gerados em: ", out_dir, "\n", sep = "")
cat("[OK] Resumo Markdown: ", md, "\n", sep = "")
