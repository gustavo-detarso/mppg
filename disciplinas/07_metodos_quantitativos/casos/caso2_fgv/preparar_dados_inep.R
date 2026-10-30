#!/usr/bin/env Rscript
# Regeneração determinística a partir do ZIP OFICIAL LOCAL (sem rede).
# A função encapsula os arquivos temporários e não confia em CSV já existente.
preparar_ideb_local <- function() {
  origem <- "dados/original/divulgacao_anos_iniciais_municipios_2025.zip"
  destino <- "dados/ideb_municipal_2019_2023.csv"
  hash_esperado <- "89100286327ab07a4b1318ba51974d62da90af334f663bedf6374cb67298a3e7"
  if (!file.exists(origem)) stop("ZIP oficial ausente: ", origem, call. = FALSE)
  if (!requireNamespace("readxl", quietly = TRUE))
    stop('Instale readxl: install.packages("readxl")', call. = FALSE)
  if (!requireNamespace("readr", quietly = TRUE))
    stop('Instale readr: install.packages("readr")', call. = FALSE)

  # No Debian, sha256sum fornece verificação de integridade sem pacote adicional.
  verificador <- Sys.which("sha256sum")
  if (!nzchar(verificador)) stop("Programa sha256sum não localizado.", call. = FALSE)
  retorno <- suppressWarnings(system2(verificador, shQuote(normalizePath(origem)), stdout = TRUE, stderr = TRUE))
  if (length(retorno) < 1L || !identical(strsplit(retorno[1L], " ")[[1L]][1L], hash_esperado))
    stop("O ZIP não corresponde ao SHA-256 do arquivo oficial homologado.", call. = FALSE)

  membros <- utils::unzip(origem, list = TRUE)$Name
  xlsx <- membros[grepl("\\.xlsx$", membros, ignore.case = TRUE)]
  if (length(xlsx) != 1L) stop("Esperado exatamente um XLSX no ZIP.", call. = FALSE)
  temporarios <- tempfile("inep_zip_")
  dir.create(temporarios)
  on.exit(unlink(temporarios, recursive = TRUE), add = TRUE)
  utils::unzip(origem, files = xlsx, exdir = temporarios)
  planilha <- file.path(temporarios, xlsx)
  tabela <- readxl::read_excel(planilha, sheet = "IDEB_AI_MUNICÍPIOS", skip = 9,
                              col_names = TRUE, .name_repair = "minimal", col_types = "text")
  colunas <- c("SG_UF", "CO_MUNICIPIO", "NO_MUNICIPIO", "REDE",
               "VL_OBSERVADO_2019", "VL_OBSERVADO_2023")
  if (!all(colunas %in% names(tabela)))
    stop("Cabeçalhos do Inep diferentes dos documentados.", call. = FALSE)
  tabela <- tabela[!is.na(tabela$REDE) & trimws(tabela$REDE) == "Municipal", colunas]
  converter <- function(x) {
    x <- trimws(as.character(x))
    x[x %in% c("", "-", "--", "*", "NA")] <- NA_character_
    suppressWarnings(as.numeric(gsub(",", ".", x, fixed = TRUE)))
  }
  base <- data.frame(
    codigo_municipio = trimws(tabela$CO_MUNICIPIO),
    municipio = trimws(tabela$NO_MUNICIPIO),
    uf = trimws(tabela$SG_UF),
    ideb_2019 = converter(tabela$VL_OBSERVADO_2019),
    ideb_2023 = converter(tabela$VL_OBSERVADO_2023),
    stringsAsFactors = FALSE)
  if (anyNA(base$codigo_municipio) || anyDuplicated(base$codigo_municipio) ||
      any(!grepl("^[0-9]{7}$", base$codigo_municipio)))
    stop("Códigos municipais ausentes, inválidos ou duplicados.", call. = FALSE)
  for (col in c("ideb_2019", "ideb_2023"))
    if (any(!is.na(base[[col]]) & (base[[col]] < 0 | base[[col]] > 10)))
      stop("Ideb fora da escala [0,10]: ", col, call. = FALSE)
  pares <- sum(stats::complete.cases(base[, c("ideb_2019", "ideb_2023")]))
  if (nrow(base) != 5450L || sum(!is.na(base$ideb_2019)) != 5139L ||
      sum(!is.na(base$ideb_2023)) != 5241L || pares != 5049L)
    stop("As contagens diferem da base oficial congelada: investigue a fonte.", call. = FALSE)
  base <- base[order(base$codigo_municipio), ]
  dir.create(dirname(destino), recursive = TRUE, showWarnings = FALSE)
  csv_tmp <- tempfile("ideb_", tmpdir = dirname(destino), fileext = ".csv")
  on.exit(unlink(csv_tmp), add = TRUE)
  readr::write_csv(base, csv_tmp, na = "")
  # A saída original só é substituída depois de todas as verificações.
  if (!file.copy(csv_tmp, destino, overwrite = TRUE))
    stop("Falha ao substituir CSV reconstruído.", call. = FALSE)
  list(arquivo = destino, municipios = nrow(base), pares = pares,
       zip_sha256 = hash_esperado, origem = origem)
}
resultado_preparacao <- preparar_ideb_local()
cat("CSV reconstruído do ZIP oficial:", resultado_preparacao$municipios,
    "municípios;", resultado_preparacao$pares, "pares completos.\n")
