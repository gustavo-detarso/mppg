# Academic Pipeline — Manual de uso e configuração TOML

Manual completo para distribuição junto com o programa `bundle_projeto_pesquisa_documento_rc_20`.

Este projeto automatiza a produção de pesquisas acadêmicas e documentos em formato Org-mode/PDF a partir de configurações TOML. Ele pode operar em três grandes cenários:

1. **Pesquisa acadêmica/PRISMA**: busca, triagem, seleção, análise e síntese de literatura.
2. **Documento acadêmico**: geração de paper, dissertação, atividade, resposta discursiva, resumo, resumo expandido, fichamento ou ensaio.
3. **Pipeline integrado**: execução encadeada de pesquisa, organização de bundle/corpus e geração do documento final.

> Recomendação: execute os comandos sempre a partir da raiz do bundle, para que os caminhos relativos do TOML funcionem corretamente.

---

## 1. Visão geral rápida

### Scripts principais

| Finalidade | Script |
|---|---|
| Pesquisa acadêmica / PRISMA / projeto empírico | `scripts/research/gerador_pesquisa_rc_2.py` |
| Pipeline integrado pesquisa + documento | `scripts/pipeline/gerador_pesquisa_documento_rc_6_refatorado_v5_2_10.py` |
| Documento acadêmico standalone | `scripts/document/gerador_documento_academico_rc_3.py` |
| Núcleo auxiliar de geração textual | `scripts/document/gerar_documento_org_ai_interativo_rc_1.py` |

### TOMLs principais

| Finalidade | TOML recomendado |
|---|---|
| Pesquisa isolada | `config/research/template_toml_unificado_rc_2.toml` |
| Documento standalone | `config/document/template_toml_documento_academico_rc_3.toml` |
| Pipeline completo e mais flexível | `config/pipeline/toml_pipeline_completo_v5_2_0.toml` |
| Simulação sem custo de API | `config/pipeline/toml_mock_run_pesquisa_dissertacao_ia_governo_publico_federal_rc_5.toml` |
| Dissertação robusta/longa | `config/pipeline/toml_pesquisa_dissertacao_ia_governo_federal_bundle_rc20_v5_longa.toml` |
| Dissertação com metas mais exigentes | `config/pipeline/toml_pesquisa_dissertacao_ia_governo_federal_bundle_rc20_excelencia.toml` |
| Organização de saída em `output/pesquisa` e `output/documento` | `config/pipeline/toml_pesquisa_dissertacao_ia_governo_federal_bundle_rc20_output_no_bundle.toml` |

### Templates principais

| Tipo | Template |
|---|---|
| Pesquisa / PRISMA | `templates/template_research.org` |
| Paper | `templates/template_paper.org` |
| Dissertação FGV | `templates/template_dissertacao_fgv_apa_v2.org` ou `templates/template_dissertacao_fgv_apa_v3_corrigido.org` |
| Atividade FGV | `templates/template_atividade_fgv_v5_2_0.org`, `v5_2_4`, `v5_2_6` ou `v5_2_7` |

---

## 2. Instalação

### 2.1. Requisitos de sistema

Recomendado:

- Python 3.11 ou superior.
- `pip` ou `pipenv`.
- Emacs com Org-mode, caso queira exportar `.org` para PDF via Org/LaTeX.
- LuaLaTeX e Biber, caso use bibliografia BibLaTeX e PDF acadêmico.
- `pdftotext`/Poppler, caso use documentos locais em PDF.

Em Debian/Ubuntu, uma base possível é:

```bash
sudo apt update
sudo apt install -y python3 python3-pip pipenv emacs texlive-full biber poppler-utils
```

O `texlive-full` é grande. Em ambiente controlado, pode-se instalar um conjunto menor, desde que inclua LuaLaTeX, BibLaTeX/Biber e os pacotes usados pelos templates.

### 2.2. Instalação das dependências Python

Com `pipenv`:

```bash
cd bundle_projeto_pesquisa_documento_rc_20
pipenv install
```

Ou com `requirements.txt`:

```bash
cd bundle_projeto_pesquisa_documento_rc_20
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências principais do bundle:

```text
openai
requests
python-dotenv
pydantic
pypdf
python-docx
```

---

## 3. Configuração do `.env`

O projeto espera as chaves de API em variáveis de ambiente ou em um arquivo `.env` na raiz do bundle.

Exemplo:

```env
OPENAI_API_KEY=coloque_sua_chave_aqui
OPENAI_MODEL=gpt-5.4

# Opcionais, conforme as bases usadas
SEMANTIC_SCHOLAR_API_KEY=
SCOPUS_API_KEY=
SCOPUS_INSTTOKEN=
WOS_API_KEY=
NCBI_API_KEY=
NCBI_EMAIL=seu_email@exemplo.com
OPENALEX_API_KEY=
OPENALEX_EMAIL=seu_email@exemplo.com
CROSSREF_EMAIL=seu_email@exemplo.com
EUROPEPMC_EMAIL=seu_email@exemplo.com
CORE_API_KEY=
UNPAYWALL_EMAIL=seu_email@exemplo.com

# Opcionais para exportação PDF
ORG_LATEX_CLASS_INIT=/home/usuario/.emacs.d/lisp/academic-writing.el
LATEX_EXTRA_PATH=/home/usuario/texmf/tex/latex/fgv/fgv-paper.sty
```

### Variáveis mais importantes

| Variável | Uso |
|---|---|
| `OPENAI_API_KEY` | Obrigatória para as etapas de IA. |
| `OPENAI_MODEL` | Modelo padrão quando não definido no TOML. |
| `SEMANTIC_SCHOLAR_API_KEY` | Opcional, melhora limites de uso do Semantic Scholar. |
| `SCOPUS_API_KEY` | Necessária para Scopus. |
| `SCOPUS_INSTTOKEN` | Opcional/necessária em ambientes institucionais Scopus. |
| `WOS_API_KEY` | Necessária para Web of Science. |
| `NCBI_API_KEY` e `NCBI_EMAIL` | PubMed/NCBI. |
| `OPENALEX_EMAIL` | Recomendado para OpenAlex. |
| `CROSSREF_EMAIL` | Recomendado para Crossref. |
| `EUROPEPMC_EMAIL` | Recomendado para Europe PMC. |
| `CORE_API_KEY` | Necessária para uso ampliado do CORE. |
| `UNPAYWALL_EMAIL` | Usado para localização de PDFs/DOIs quando aplicável. |
| `ORG_LATEX_CLASS_INIT` | `.el` que registra classes Org/LaTeX, como `fgv-paper`. |
| `LATEX_EXTRA_PATH` | Caminho para `.sty`, `.cls` ou pasta com arquivos LaTeX extras. |

---

## 4. Estrutura de diretórios

Estrutura esperada do bundle:

```text
bundle_projeto_pesquisa_documento_rc_20/
├── config/
│   ├── document/
│   ├── pipeline/
│   └── research/
├── docs/
│   ├── modelos/
│   ├── diagrama_arquitetura.md
│   ├── estrutura_output_nova.md
│   └── manual_unificado_rc_18.md
├── misc/
│   ├── academic-writing.el
│   └── fgv/
│       ├── fgv-paper.sty
│       ├── fgv-dissertacao.sty
│       └── fgv.png
├── prompts/
│   ├── diretivas_extras.txt
│   ├── orientacao_geral_execucao.txt
│   └── triagem_prompt.txt
├── scripts/
│   ├── document/
│   ├── pipeline/
│   └── research/
├── templates/
└── output/
```

Saídas recomendadas:

```text
output/
├── pesquisa/       # resultados da busca/triagem/PRISMA
├── documento/      # documento final: ORG, BIB, PDF, auditoria
├── corpus_local/   # corpus criado a partir de ZIP/pasta local
└── bundle/         # bundle consolidado/handoff, se habilitado
```

---

## 5. Como executar

### 5.1. Pesquisa isolada

```bash
cd bundle_projeto_pesquisa_documento_rc_20
pipenv run python scripts/research/gerador_pesquisa_rc_2.py \
  --config config/research/template_toml_unificado_rc_2.toml
```

Sem `pipenv`:

```bash
python scripts/research/gerador_pesquisa_rc_2.py \
  --config config/research/template_toml_unificado_rc_2.toml
```

### 5.2. Pipeline completo

```bash
cd bundle_projeto_pesquisa_documento_rc_20
pipenv run python scripts/pipeline/gerador_pesquisa_documento_rc_6_refatorado_v5_2_10.py \
  --config config/pipeline/toml_pipeline_completo_v5_2_0.toml
```

### 5.3. Documento standalone

```bash
cd bundle_projeto_pesquisa_documento_rc_20
pipenv run python scripts/document/gerador_documento_academico_rc_3.py \
  --config config/document/template_toml_documento_academico_rc_3.toml
```

### 5.4. Simulação/mock sem custo de API

Use quando quiser testar estrutura, saídas e templates sem fazer uma busca real:

```bash
pipenv run python scripts/pipeline/gerador_pesquisa_documento_rc_6_refatorado_v5_2_10.py \
  --config config/pipeline/toml_mock_run_pesquisa_dissertacao_ia_governo_publico_federal_rc_5.toml
```

---

## 6. Matriz de decisão: qual modo usar?

| Necessidade | Use |
|---|---|
| Quero apenas buscar e selecionar textos | Pesquisa isolada (`scripts/research/...`) |
| Quero buscar literatura e depois gerar documento | Pipeline integrado com `modo_entrada = "pesquisa"` |
| Já tenho uma pesquisa pronta e quero gerar o documento | Pipeline com `modo_entrada = "pesquisa_existente"` ou documento standalone |
| Tenho PDFs/DOCX/TXT/ORG fornecidos pelo professor | Pipeline com `modo_entrada = "documentos_locais"` |
| Quero uma atividade/resposta com base exclusiva nos textos anexados | `documentos_locais` + `tipo_documento = "atividade"` ou `"resposta_discursiva"` |
| Quero uma dissertação FGV longa | Pipeline completo + `tipo_documento = "dissertacao"` + template de dissertação |
| Quero um paper curto | Pipeline ou standalone + `tipo_documento = "paper"` |
| Quero reescrever um `.org` anterior | `modo_escrita = "reescrever"` + `documento_org_existente` |
| Quero expandir um texto anterior | `modo_escrita = "expandir"` + `preservar_estrutura_do_org_anterior = true` |
| Quero só validar caminhos | `[controle].dry_run = true` |

---

## 7. Referência completa do TOML

Esta seção descreve as possibilidades de configuração. Nem todo campo precisa ser usado em todas as execuções. Alguns campos são específicos da pesquisa; outros só fazem sentido no pipeline ou na geração de documento.

### 7.1. `[pipeline]`

Controla o fluxo geral do pipeline integrado.

```toml
[pipeline]
modo_entrada = "pesquisa"
executar_pesquisa = true
executar_bundle = false
executar_documento = false
pesquisa_dir_existente = ""
script_pesquisa = "./scripts/research/gerador_pesquisa_rc_2.py"
bundle_dir = ""
```

| Campo | Tipo | Valores/Exemplos | Função |
|---|---:|---|---|
| `modo_entrada` | string | `"pesquisa"`, `"pesquisa_existente"`, `"documentos_locais"`, `"local"`, `"zip"`, `"pasta_local"` | Define a origem principal dos dados. |
| `executar_pesquisa` | bool | `true`/`false` | Roda ou não a etapa de pesquisa. |
| `executar_bundle` | bool | `true`/`false` | Gera bundle/handoff consolidado. |
| `executar_documento` | bool | `true`/`false` | Gera documento final. |
| `pesquisa_dir_existente` | path | `"output/pesquisa/meu_tema"` | Usado quando a pesquisa já existe. |
| `script_pesquisa` | path | `"scripts/research/gerador_pesquisa_rc_2.py"` | Script chamado pela etapa de pesquisa. |
| `bundle_dir` | path | `"output/bundle"` | Diretório de bundle já existente ou destino. |

#### Modos típicos

Só pesquisa:

```toml
[pipeline]
modo_entrada = "pesquisa"
executar_pesquisa = true
executar_bundle = false
executar_documento = false
```

Pesquisa + documento:

```toml
[pipeline]
modo_entrada = "pesquisa"
executar_pesquisa = true
executar_bundle = true
executar_documento = true
```

Documento a partir de pesquisa pronta:

```toml
[pipeline]
modo_entrada = "pesquisa_existente"
executar_pesquisa = false
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = "./output/pesquisa/dissertacao_ia_governo_publico_federal"
```

Documentos locais:

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true
```

---

### 7.2. `[documentos_locais]`

Transforma um ZIP ou pasta de arquivos locais em corpus compatível com o fluxo de pesquisa. É ideal para atividades com textos fornecidos pelo professor.

```toml
[documentos_locais]
ativos = false
modo_entrada = "documentos_locais"
input_zip = ""
input_dir = ""
recursive = true
tipos = ["pdf", "docx", "txt", "md", "org"]
prefixo = ""
output_dir = "./output/corpus_local"
criar_subdiretorio = true
limpar_extracao_anterior = true
limpar_cache_anterior = true
sobrescrever_corpus_local = true
copiar_para_fulltext_cache = true
gerar_debug_sintetico = true
gerar_bib_sintetico = true
usar_nome_arquivo_como_chave = true
autor_padrao = "Material fornecido pelo professor"
ano_padrao = "s.d."
gerar_resumo_individual = true
gerar_contexto_consolidado = true
exigir_uso_de_todos_os_documentos = true
responder_exclusivamente_com_base_nos_documentos = true
permitir_fontes_externas = false
ignorar_arquivos_ocultos = true
min_caracteres_documento_valido = 500
```

| Campo | Função |
|---|---|
| `ativos` | Ativa o modo de documentos locais mesmo que `[pipeline].modo_entrada` não esteja preenchido. |
| `input_zip` | Caminho para ZIP com PDFs/DOCX/TXT/MD/ORG. Tem prioridade sobre `input_dir`. |
| `input_dir` | Pasta com documentos locais. |
| `recursive` | Procura arquivos em subpastas. |
| `tipos` | Extensões permitidas no corpus local. |
| `prefixo` | Nome-base do corpus local. |
| `output_dir` | Onde salvar o corpus convertido. |
| `copiar_para_fulltext_cache` | Copia os documentos para `*_fulltext_cache`, preservando a lógica da pesquisa. |
| `gerar_debug_sintetico` | Cria JSON de debug sintético. |
| `gerar_bib_sintetico` | Cria `.bib` sintético se não houver bibliografia externa. |
| `autor_padrao`/`ano_padrao` | Metadados usados para documentos sem metadados. |
| `exigir_uso_de_todos_os_documentos` | Orienta o documento a usar todo o corpus local. |
| `responder_exclusivamente_com_base_nos_documentos` | Impede uso substantivo de fontes externas. |
| `permitir_fontes_externas` | Quando `true`, admite complementação externa. |
| `min_caracteres_documento_valido` | Ignora extrações muito curtas. |

#### Exemplo — atividade com ZIP de textos do professor

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true

[documentos_locais]
ativos = true
input_zip = "./inputs/textos_aula_1.zip"
recursive = true
tipos = ["pdf", "docx", "txt", "md", "org"]
prefixo = "atividade_politicas_publicas_aula_1"
output_dir = "./output/corpus_local"
exigir_uso_de_todos_os_documentos = true
responder_exclusivamente_com_base_nos_documentos = true
permitir_fontes_externas = false
```

---

### 7.3. `[atividade]` e `[atividade.metadados]`

Guarda o tipo de atividade e os metadados acadêmicos.

Formato simples usado pelo script de pesquisa:

```toml
[atividade]
modo = "revisao_sistematica"
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"
```

Formato aninhado usado nos TOMLs mais completos:

```toml
[atividade]
modo = "revisao_sistematica"
tema = ""

[atividade.metadados]
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"
data = ""
```

Valores possíveis ou úteis para `modo`:

| Valor | Uso |
|---|---|
| `revisao_sistematica` | Revisão PRISMA/revisão de literatura com busca e triagem. |
| `pesquisa_empirica` | Projeto/roteiro de pesquisa empírica. |
| `resposta_discursiva` | Resposta estruturada a uma pergunta. |
| `fichamento` | Fichamento acadêmico. |
| `resumo` | Resumo. |
| `resumo_expandido` | Resumo expandido. |
| `ensaio` | Ensaio acadêmico. |
| `atividade_fgv` | Atividade formatada em modelo FGV. |

---

### 7.4. `[pesquisa]`

Define a tríade substantiva do trabalho e os parâmetros de pesquisa.

```toml
[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização do poder e resultados de políticas públicas"
objetivo = "Investigar como a literatura explica a relação entre capacidades estatais, centralização e desempenho de políticas públicas"
pergunta_pesquisa = ""
trabalho = ""
tipo_estudo = "Revisão de literatura"
periodo = "2019-2026"
idiomas = ["inglês", "português"]
palavras_chave = []
bases = ["semantic_scholar", "scopus", "pubmed", "openalex", "crossref", "europe_pmc", "core"]
```

| Campo | Função |
|---|---|
| `tema` | Assunto geral. |
| `recorte` | Delimitação temática, temporal, institucional ou empírica. |
| `objetivo` | Objetivo geral. |
| `pergunta_pesquisa` | Pergunta explícita, quando houver. |
| `trabalho` | Título manual ou tipo nominal, como `"Dissertação"`. Pode ficar vazio. |
| `tipo_estudo` | Natureza do estudo. Ex.: revisão, estudo de caso, survey. |
| `periodo` | Recorte temporal da busca. |
| `idiomas` | Idiomas desejados. |
| `palavras_chave` | Lista manual; deixe `[]` para a IA sugerir. |
| `bases` | Bases bibliográficas a consultar. |

Bases reconhecidas:

```toml
bases = [
  "semantic_scholar",
  "scopus",
  "web_of_science",
  "pubmed",
  "openalex",
  "crossref",
  "europe_pmc",
  "core"
]
```

Tipos de estudo úteis:

```toml
tipo_estudo = "Revisão sistemática"
tipo_estudo = "Revisão de literatura"
tipo_estudo = "Scoping review"
tipo_estudo = "Metanálise"
tipo_estudo = "Estudo de caso"
tipo_estudo = "Estudo qualitativo"
tipo_estudo = "Estudo quantitativo"
tipo_estudo = "Métodos mistos"
tipo_estudo = "Survey"
```

---

### 7.5. `[bibliografia]`

Define o estilo de citação/referência.

```toml
[bibliografia]
estilo_citacao = "APA"
```

Valores comuns:

```toml
estilo_citacao = "APA"
estilo_citacao = "ABNT"
estilo_citacao = "Chicago"
estilo_citacao = "MLA"
estilo_citacao = "Vancouver"
```

No documento standalone, também é aceito via CLI:

```bash
--citation-style apa
--citation-style abnt
```

---

### 7.6. `[busca]`

Controla a amplitude da busca, da triagem e das análises geradas.

```toml
[busca]
sugerir_palavras_chave_ia = true
query_bilingue = true
quantidade_triagem = 25
quantidade_selecionados = 3
salvar_busca_bruta_json = true
incluir_analise_detalhada_ia = true
incluir_sintese_integradora_ia = true
```

| Campo | Função |
|---|---|
| `sugerir_palavras_chave_ia` | Usa IA para sugerir palavras-chave e termos correlatos. |
| `query_bilingue` | Monta queries em português e inglês quando possível. |
| `quantidade_triagem` | Quantos registros seguem para triagem. |
| `quantidade_selecionados` | Quantos textos finais serão selecionados. |
| `salvar_busca_bruta_json` | Salva respostas brutas das bases. |
| `incluir_analise_detalhada_ia` | Inclui análise individual dos selecionados. |
| `incluir_sintese_integradora_ia` | Inclui síntese integradora quando há múltiplos textos. |

Sugestão de escala:

| Uso | `quantidade_triagem` | `quantidade_selecionados` |
|---|---:|---:|
| Teste rápido | 10–25 | 1–3 |
| Atividade curta | 25–60 | 3–8 |
| Paper | 60–120 | 8–20 |
| Dissertação | 120–300+ | 20–60 |

---

### 7.7. `[triagem]`

Controla o rigor da seleção e as orientações específicas de ranking.

```toml
[triagem]
rigor = "moderado"
usar_score_hibrido = true
orientacoes_paths = [
  "./prompts/triagem_prompt.txt",
  "./prompts/diretivas_extras.txt"
]
orientacao_inline = ""
permitir_textos_nao_publicos = false
```

| Campo | Valores | Função |
|---|---|---|
| `rigor` | `"estrito"`, `"moderado"`, `"exploratorio"` | Define tolerância temática da seleção. |
| `usar_score_hibrido` | bool | Combina score local + avaliação da IA. |
| `orientacoes_paths` | lista de paths | Arquivos com instruções específicas de triagem. |
| `orientacao_inline` | string | Instrução escrita diretamente no TOML. |
| `permitir_textos_nao_publicos` | bool | Permite manter registros sem PDF local quando houver URL/landing page verificável. |

#### Quando usar cada rigor

| Rigor | Uso recomendado |
|---|---|
| `estrito` | Tema muito delimitado; evite textos genéricos. |
| `moderado` | Equilíbrio entre precisão e abrangência. |
| `exploratorio` | Mapeamento inicial, scoping review, tema novo ou pouco consolidado. |

---

### 7.8. `[queries]`

Permite substituir as queries automáticas por consultas manuais.

```toml
[queries]
query_geral = ""
query_semantic = ""
query_scopus = ""
query_wos = ""
query_pubmed = ""
query_openalex = ""
query_crossref = ""
query_europepmc = ""
query_core = ""
```

Se todos os campos estiverem vazios, o script monta as queries a partir de `tema`, `recorte`, `objetivo`, `tipo_estudo`, `idiomas` e `palavras_chave`.

Exemplo com query geral:

```toml
[queries]
query_geral = '"artificial intelligence" AND "public administration" AND (governance OR ethics OR compliance OR risk)'
```

Exemplo com queries por base:

```toml
[queries]
query_semantic = '"AI governance" "public administration" ethics compliance'
query_scopus = 'TITLE-ABS-KEY("artificial intelligence" AND "public administration" AND governance)'
query_pubmed = '("artificial intelligence"[Title/Abstract]) AND (government[Title/Abstract] OR "public sector"[Title/Abstract])'
query_openalex = 'artificial intelligence public administration governance ethics'
query_crossref = 'artificial intelligence public administration governance compliance'
query_europepmc = 'artificial intelligence public administration governance'
query_core = 'artificial intelligence public administration governance ethics compliance'
```

---

### 7.9. `[saida]`

Controla a saída da pesquisa ou do corpus equivalente.

```toml
[saida]
prefixo = "atividade_modelo"
output_dir = "./output/pesquisa"
criar_subdiretorio = true
org_modelo = "./templates/template_research.org"
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]
orientacao_inline = ""
exportar_pdf = true
gerar_env_example = false
remover_auxiliares = true
```

| Campo | Função |
|---|---|
| `prefixo` | Nome-base dos arquivos de pesquisa. |
| `output_dir` | Diretório de saída. |
| `criar_subdiretorio` | Cria subpasta com o prefixo. |
| `org_modelo` | Template `.org` da pesquisa. |
| `orientacoes_paths` | Orientações gerais da etapa de pesquisa. |
| `orientacao_inline` | Orientação geral escrita no TOML. |
| `exportar_pdf` | Compila PDF da pesquisa, se possível. |
| `gerar_env_example` | Gera `.env.example`. |
| `remover_auxiliares` | Remove auxiliares de compilação, preservando artefatos principais. |

---

### 7.10. `[prompts]`

Permite trocar prompts externos sem alterar o código.

```toml
[prompts]
prompt_sistema_path = ""
prompt_resumo_individual_path = ""
prompt_contexto_consolidado_path = ""
prompt_documento_final_path = ""
prompt_dissertacao_etapas_path = ""
prompt_atividade_path = ""
prompt_fichamento_path = ""
prompt_resposta_discursiva_path = ""
prompt_resumo_expandido_path = ""
prompt_inline = ""
```

| Campo | Função |
|---|---|
| `prompt_sistema_path` | Prompt de sistema ou orientação global. |
| `prompt_resumo_individual_path` | Prompt para resumir documentos individualmente. |
| `prompt_contexto_consolidado_path` | Prompt para consolidar o corpus. |
| `prompt_documento_final_path` | Prompt geral do documento final. |
| `prompt_dissertacao_etapas_path` | Prompt específico para dissertação gerada em etapas. |
| `prompt_atividade_path` | Prompt específico para atividade. |
| `prompt_fichamento_path` | Prompt específico para fichamento. |
| `prompt_resposta_discursiva_path` | Prompt específico para resposta discursiva. |
| `prompt_resumo_expandido_path` | Prompt específico para resumo expandido. |
| `prompt_inline` | Orientação adicional direta no TOML. |

---

### 7.11. `[interpretacao]`

Controla como o corpus será usado substantivamente pela IA.

```toml
[interpretacao]
usar_textos_completos = true
usar_resumos_individuais = true
usar_contexto_consolidado = true
exigir_cobertura_total_fulltext_cache = true
criar_secao_integracao_documentos_faltantes = true
permitir_fontes_externas = false
responder_exclusivamente_com_base_nos_documentos = true
comparar_textos_entre_si = true
identificar_convergencias = true
identificar_divergencias = true
identificar_lacunas = true
extrair_conceitos_centrais = true
extrair_argumentos_centrais = true
```

| Campo | Função |
|---|---|
| `usar_textos_completos` | Usa extrações completas dos documentos. |
| `usar_resumos_individuais` | Usa resumos intermediários. |
| `usar_contexto_consolidado` | Usa síntese consolidada do corpus. |
| `exigir_cobertura_total_fulltext_cache` | Exige uso dos textos disponíveis no cache. |
| `criar_secao_integracao_documentos_faltantes` | Cria seção integradora quando há textos pouco usados. |
| `permitir_fontes_externas` | Autoriza ou bloqueia complementação externa. |
| `responder_exclusivamente_com_base_nos_documentos` | Força base exclusiva nos documentos fornecidos. |
| `comparar_textos_entre_si` | Pede comparação entre documentos. |
| `identificar_convergencias` | Pede convergências. |
| `identificar_divergencias` | Pede divergências. |
| `identificar_lacunas` | Pede lacunas. |
| `extrair_conceitos_centrais` | Pede conceitos centrais. |
| `extrair_argumentos_centrais` | Pede argumentos centrais. |

---

### 7.12. `[geracao]`

Controla a geração textual por IA, especialmente em documentos longos.

```toml
[geracao]
modo = "etapas"
gerar_por_subsecoes = true
max_tentativas_por_etapa = 2
salvar_checkpoints = true
reutilizar_checkpoints = true
timeout_segundos = 300
max_caracteres_por_documento = 120000
max_caracteres_por_chunk = 30000
reparar_secao_curta = true
reparar_citacoes_faltantes = true
remover_headings_duplicados = true
limpar_empty_citation = true
salvar_prompts_auditoria = true
salvar_uso_referencias = true
salvar_limites_secoes = true
```

| Campo | Função |
|---|---|
| `modo` | `"etapas"` para textos longos; `"unica"` para saída única. |
| `gerar_por_subsecoes` | Gera por subseções para reduzir truncamento. |
| `max_tentativas_por_etapa` | Número de tentativas por etapa. |
| `salvar_checkpoints` | Salva pontos intermediários. |
| `reutilizar_checkpoints` | Retoma etapas já concluídas. |
| `timeout_segundos` | Timeout operacional de chamadas longas. |
| `max_caracteres_por_documento` | Limite de leitura por documento. |
| `max_caracteres_por_chunk` | Limite por pedaço de texto. |
| `reparar_secao_curta` | Tenta corrigir seção abaixo da extensão esperada. |
| `reparar_citacoes_faltantes` | Tenta corrigir ausência de citações. |
| `remover_headings_duplicados` | Limpa títulos repetidos. |
| `limpar_empty_citation` | Remove/corrige citações vazias. |
| `salvar_prompts_auditoria` | Salva prompts usados. |
| `salvar_uso_referencias` | Salva mapa de uso de referências. |
| `salvar_limites_secoes` | Salva diagnóstico de extensão por seção. |

---

### 7.13. `[extracao]`

Controla extração de texto de documentos locais.

```toml
[extracao]
pdf_engine = "pdftotext"
docx_engine = "python-docx"
usar_ocr_pdf = false
idioma_ocr = "por+eng"
normalizar_espacos = true
remover_cabecalhos_rodapes_repetidos = false
min_caracteres_documento_valido = 500
max_caracteres_extraidos_por_arquivo = 250000
```

| Campo | Função |
|---|---|
| `pdf_engine` | Motor de extração de PDF. Recomendado: `pdftotext`. |
| `docx_engine` | Motor de leitura de DOCX. Recomendado: `python-docx`. |
| `usar_ocr_pdf` | Ative apenas para PDFs escaneados. |
| `idioma_ocr` | Idiomas para OCR, se habilitado. |
| `normalizar_espacos` | Corrige espaços e quebras excessivas. |
| `remover_cabecalhos_rodapes_repetidos` | Tenta remover ruído repetitivo. |
| `min_caracteres_documento_valido` | Descarta extração muito curta. |
| `max_caracteres_extraidos_por_arquivo` | Truncamento defensivo por arquivo. |

---

### 7.14. `[documento]`

Controla o documento final.

```toml
[documento]
tipo_documento = "paper"
prefixo = ""
output_dir = "./output/documento"
criar_subdiretorio = true
template_org = "./templates/template_paper.org"
exportar_pdf = true

institution_name = "Fundação Getúlio Vargas"
school_name = ""
program_name = "Mestrado Profissional em Políticas Públicas e Governo"
area_de_concentracao = "Políticas Públicas e Governo"
ano = "2026"
linha_pesquisa = "Governança, Estado e Políticas Públicas"
coorientador = ""
data_aprovacao = "A definir"
banca = []

curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
aluno = "Nome do aluno"
data = ""
titulo_trabalho = ""

min_palavras_total = 22000
alvo_palavras_total = 28000
min_palavras_introducao = 1800
min_palavras_referencial = 7000
min_palavras_metodologia = 3000
min_palavras_resultados = 8500
min_palavras_conclusao = 1800

min_palavras_atividade = 1200
alvo_palavras_atividade = 2500
min_palavras_resumo = 600
alvo_palavras_resumo = 1200
min_palavras_fichamento = 1200
alvo_palavras_fichamento = 2500

base_docs_paths = []
orientacoes_paths = []
orientacao_inline = ""
usar_artigos_selecionados_pesquisa = true
artigos_extras_paths = []
modo_escrita = "novo"
perfil_redacao = "academico_equilibrado"
usar_contexto_consolidado_da_pesquisa = true
reformular_tema_recorte_objetivo = false
permitir_busca_correlata_extra = false
priorizar_citacoes_dos_selecionados = true
extras_so_complementam = true
minimo_citacoes_dos_selecionados = 3
usar_bib_da_pesquisa = true
incluir_artigos_extras_no_bib = true
reescrever_a_partir_do_org_atual = false
documento_org_existente = ""
preservar_estrutura_do_org_anterior = false
bib_paths = []
tema = ""
recorte = ""
objetivo = ""
geracao_em_etapas = true
```

Tipos de documento reconhecidos pelo pipeline:

| Valor | Aliases | Uso |
|---|---|---|
| `paper` | `artigo`, `papel`, `documento` | Artigo/paper acadêmico. |
| `dissertacao` | `dissertação`, `thesis` | Dissertação. |
| `atividade` | `atividade_fgv` | Atividade acadêmica. |
| `resposta_discursiva` | `resposta` | Resposta a questão discursiva. |
| `resumo` | — | Resumo. |
| `resumo_expandido` | — | Resumo expandido. |
| `fichamento` | — | Fichamento. |
| `ensaio` | — | Ensaio. |
| `ensaio_curto` | — | Ensaio curto. |

#### Banca dinâmica para dissertação

```toml
[documento]
banca = []

[[documento.banca]]
funcao = "Orientador"
nome = "Prof. Dr. Nome do Orientador"
instituicao = "Fundação Getúlio Vargas"

[[documento.banca]]
funcao = "Membro interno"
nome = "Prof. Dr. Nome do Professor"
instituicao = "Fundação Getúlio Vargas"

[[documento.banca]]
funcao = "Membro externo"
nome = "Prof. Dr. Nome do Professor"
instituicao = "Universidade de Brasília"
```

#### Modos de escrita

| Valor | Efeito |
|---|---|
| `novo` | Gera documento novo. |
| `reescrever` | Reescreve documento anterior. |
| `expandir` | Expande documento anterior preservando a base. |

#### Reescrita a partir de ORG anterior

```toml
[documento]
modo_escrita = "reescrever"
reescrever_a_partir_do_org_atual = true
documento_org_existente = "./output/documento/meu_texto/meu_texto.org"
preservar_estrutura_do_org_anterior = true
```

#### Artigos extras

```toml
[documento]
artigos_extras_paths = [
  "./inputs/artigos_extras",
  "./inputs/artigo_especifico.pdf"
]
extras_so_complementam = true
incluir_artigos_extras_no_bib = true
```

#### Bibliografias extras

```toml
[documento]
bib_paths = [
  "./inputs/referencias_extra.bib"
]
```

---

### 7.15. `[latex]`

Controla a exportação PDF via Org/LaTeX.

```toml
[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-paper.sty"
comando_exportacao_pdf = ""
fgv_logo_path = "./misc/fgv/fgv.png"
emacs_init = ""
```

| Campo | Função |
|---|---|
| `org_latex_class_init` | Arquivo `.el` que registra classes como `fgv-paper` e `fgv-dissertacao` no Org. |
| `latex_extra_path` | `.sty`, `.cls` ou pasta que deve entrar no caminho do LaTeX. |
| `comando_exportacao_pdf` | Comando customizado de exportação. Se vazio, usa rotina padrão. |
| `fgv_logo_path` | Logo usado em templates/cabeçalhos, quando aplicável. |
| `emacs_init` | Arquivo de inicialização Emacs opcional. |

Placeholders aceitos em `comando_exportacao_pdf` no script de pesquisa:

```text
{org}
{org_dir}
{org_stem}
{pdf}
{pdf_dir}
{class_init}
{latex_path}
{latex_dir}
{bib}
```

Exemplo:

```toml
[latex]
comando_exportacao_pdf = "cd {org_dir} && TEXINPUTS={latex_dir}//: emacs --batch -Q -l {class_init} {org} --funcall org-latex-export-to-pdf"
```

### Observação importante sobre PDF FGV

Se o `.org` usa `#+LATEX_CLASS: fgv-paper` ou `fgv-dissertacao`, o Emacs batch precisa conhecer essa classe. Por isso, configure:

```toml
[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-paper.sty"
```

Para dissertação:

```toml
[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-dissertacao.sty"
```

---

### 7.16. `[openai]`

Configura modelo e parâmetros opcionais.

```toml
[openai]
model = "gpt-5.4"
timeout_segundos = 300
max_output_tokens = 8192
temperature = 0.2
```

| Campo | Função |
|---|---|
| `model` | Modelo usado nas chamadas de IA. |
| `timeout_segundos` | Timeout desejado para chamadas longas. |
| `max_output_tokens` | Teto de tokens de saída, quando implementado. |
| `temperature` | Criatividade/variação, quando implementado. |

---

### 7.17. `[controle]`

Controla execução, logs e simulação.

```toml
[controle]
nao_interativo = true
dry_run = false
mock_run = false
mock_seed = 42
mock_quantidade_registros = 40
mock_gerar_pdf = true
salvar_config = true
config_output = ""
verbose = true
salvar_logs = true
```

| Campo | Função |
|---|---|
| `nao_interativo` | Não pergunta nada; usa TOML/CLI. Recomendado para automação. |
| `dry_run` | Valida caminhos/configurações sem executar geração real. |
| `mock_run` | Simula execução do pipeline. |
| `mock_seed` | Semente da simulação. |
| `mock_quantidade_registros` | Quantidade de registros sintéticos. |
| `mock_gerar_pdf` | Tenta gerar PDF no mock, se aplicável. |
| `salvar_config` | Salva TOML final resolvido. |
| `config_output` | Caminho do TOML final salvo. |
| `verbose` | Logs mais detalhados. |
| `salvar_logs` | Salva logs em arquivo, quando implementado. |

---

## 8. Exemplos completos de configuração

### 8.1. Pesquisa PRISMA simples

```toml
[atividade]
modo = "revisao_sistematica"
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"

[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização do poder e resultados de políticas públicas"
objetivo = "Identificar como a literatura recente explica a relação entre capacidades estatais, centralização e desempenho de políticas públicas"
trabalho = ""
tipo_estudo = "Revisão de literatura"
periodo = "2019-2026"
idiomas = ["inglês", "português"]
bases = ["semantic_scholar", "scopus", "pubmed", "openalex", "crossref", "europe_pmc", "core"]
palavras_chave = []

[bibliografia]
estilo_citacao = "APA"

[busca]
sugerir_palavras_chave_ia = true
query_bilingue = true
quantidade_triagem = 60
quantidade_selecionados = 8
salvar_busca_bruta_json = true
incluir_analise_detalhada_ia = true
incluir_sintese_integradora_ia = true

[triagem]
rigor = "moderado"
usar_score_hibrido = true
orientacoes_paths = ["./prompts/triagem_prompt.txt", "./prompts/diretivas_extras.txt"]
orientacao_inline = "Priorize aderência direta ao recorte."
permitir_textos_nao_publicos = false

[queries]
query_geral = ""
query_semantic = ""
query_scopus = ""
query_wos = ""
query_pubmed = ""
query_openalex = ""
query_crossref = ""
query_europepmc = ""
query_core = ""

[saida]
prefixo = "capacidades_estatais_desempenho"
output_dir = "./output/pesquisa"
criar_subdiretorio = true
org_modelo = "./templates/template_research.org"
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]
orientacao_inline = ""
exportar_pdf = true
gerar_env_example = false
remover_auxiliares = true

[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-paper.sty"
comando_exportacao_pdf = ""
fgv_logo_path = "./misc/fgv/fgv.png"

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
salvar_config = true
config_output = ""
```

Comando:

```bash
pipenv run python scripts/research/gerador_pesquisa_rc_2.py \
  --config config/research/template_toml_unificado_rc_2.toml
```

---

### 8.2. Pesquisa empírica sem PRISMA

```toml
[atividade]
modo = "pesquisa_empirica"
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"

[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização político-administrativa e resultados de políticas públicas"
objetivo = "Elaborar uma proposta de pesquisa empírica sobre capacidades estatais, coerção, centralização e desempenho estatal"
trabalho = ""
tipo_estudo = "Métodos mistos"
idiomas = ["português", "inglês"]
palavras_chave = []

[busca]
sugerir_palavras_chave_ia = true
query_bilingue = true

[saida]
prefixo = "projeto_empirico_capacidades_estatais"
output_dir = "./output/pesquisa"
criar_subdiretorio = true
org_modelo = "./templates/template_research.org"
exportar_pdf = true

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
```

---

### 8.3. Pipeline: pesquisa + dissertação

```toml
[pipeline]
modo_entrada = "pesquisa"
executar_pesquisa = true
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = ""
script_pesquisa = "./scripts/research/gerador_pesquisa_rc_2.py"
bundle_dir = ""

[atividade]
modo = "revisao_sistematica"

[atividade.metadados]
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"
data = ""

[pesquisa]
tema = "Uso de inteligência artificial no governo público federal"
recorte = "Governança, ética, compliance e gestão de riscos na administração pública federal brasileira, 2019-2026"
objetivo = "Investigar como a literatura trata a adoção de IA no governo federal brasileiro, com foco em governança, ética, compliance e riscos."
trabalho = "Dissertação"
tipo_estudo = "Revisão de literatura"
periodo = "2019-2026"
idiomas = ["inglês", "português"]
palavras_chave = []
bases = ["semantic_scholar", "scopus", "pubmed", "openalex", "crossref", "europe_pmc", "core"]

[bibliografia]
estilo_citacao = "APA"

[busca]
sugerir_palavras_chave_ia = true
query_bilingue = true
quantidade_triagem = 140
quantidade_selecionados = 40
salvar_busca_bruta_json = true
incluir_analise_detalhada_ia = true
incluir_sintese_integradora_ia = true

[triagem]
rigor = "moderado"
usar_score_hibrido = true
orientacoes_paths = ["./prompts/triagem_prompt.txt", "./prompts/diretivas_extras.txt"]
orientacao_inline = "Priorize estudos diretamente conectados a IA, governança, ética, compliance, riscos, accountability, transparência e controle institucional."
permitir_textos_nao_publicos = false

[saida]
prefixo = "dissertacao_ia_governo_publico_federal"
output_dir = "./output/pesquisa"
criar_subdiretorio = true
org_modelo = "./templates/template_research.org"
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt", "./prompts/diretivas_extras.txt"]
exportar_pdf = true

[documento]
tipo_documento = "dissertacao"
prefixo = "dissertacao_ia_governo_publico_federal"
output_dir = "./output/documento"
criar_subdiretorio = true
template_org = "./templates/template_dissertacao_fgv_apa_v2.org"
exportar_pdf = true
institution_name = "Fundação Getúlio Vargas"
program_name = "Mestrado Profissional em Políticas Públicas e Governo"
area_de_concentracao = "Políticas Públicas e Governo"
ano = "2026"
linha_pesquisa = "Governança, Estado e Políticas Públicas"
banca = []
min_palavras_total = 22000
alvo_palavras_total = 28000
orientacoes_paths = [
  "./prompts/orientacao_geral_execucao.txt",
  "./docs/modelos/fgv/formatacao-de-trabalhos-academicos-manual-fgv-impressao-2025.pdf",
  "./docs/modelos/fgv/modelo-de-dissertacao-2025.docx"
]
orientacao_inline = "Produza uma dissertação, não um paper curto. Respeite o template e use os textos selecionados como base principal."
usar_artigos_selecionados_pesquisa = true
modo_escrita = "novo"
perfil_redacao = "academico_equilibrado"
usar_contexto_consolidado_da_pesquisa = true
priorizar_citacoes_dos_selecionados = true
usar_bib_da_pesquisa = true
geracao_em_etapas = true

[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-dissertacao.sty"
fgv_logo_path = "./misc/fgv/fgv.png"

[openai]
model = "gpt-5.4"
timeout_segundos = 300
max_output_tokens = 8192
temperature = 0.2

[controle]
nao_interativo = true
dry_run = false
mock_run = false
verbose = true
salvar_logs = true
```

---

### 8.4. Documento a partir de pesquisa existente

```toml
[pipeline]
modo_entrada = "pesquisa_existente"
executar_pesquisa = false
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = "./output/pesquisa/dissertacao_ia_governo_publico_federal"
script_pesquisa = "./scripts/research/gerador_pesquisa_rc_2.py"

[documento]
tipo_documento = "paper"
prefixo = "paper_ia_governo_publico_federal"
output_dir = "./output/documento"
criar_subdiretorio = true
template_org = "./templates/template_paper.org"
exportar_pdf = true
usar_artigos_selecionados_pesquisa = true
usar_bib_da_pesquisa = true
modo_escrita = "novo"
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]
orientacao_inline = "Gere um paper analítico com introdução, referencial, metodologia, discussão e conclusão."

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
```

---

### 8.5. Atividade FGV com documentos locais

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true

[documentos_locais]
ativos = true
input_zip = "./inputs/textos_da_aula.zip"
input_dir = ""
recursive = true
tipos = ["pdf", "docx", "txt", "md", "org"]
prefixo = "atividade_politicas_publicas_aula_1"
output_dir = "./output/corpus_local"
criar_subdiretorio = true
copiar_para_fulltext_cache = true
gerar_debug_sintetico = true
gerar_bib_sintetico = true
autor_padrao = "Material fornecido pelo professor"
ano_padrao = "s.d."
exigir_uso_de_todos_os_documentos = true
responder_exclusivamente_com_base_nos_documentos = true
permitir_fontes_externas = false

[atividade]
modo = "atividade_fgv"

[atividade.metadados]
disciplina = "Teorias da Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado Profissional em Políticas Públicas e Governo"
turma = "2026.01"
polo = "Brasília"
aluno = "Nome do aluno"

[pesquisa]
tema = "Políticas públicas e teorias de análise"
recorte = "Discussão dos textos da aula 1"
objetivo = "Responder à atividade com base exclusiva nos textos fornecidos."

[documento]
tipo_documento = "atividade"
prefixo = "atividade_politicas_publicas_aula_1"
output_dir = "./output/documento"
criar_subdiretorio = true
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
exportar_pdf = true
titulo_trabalho = "Atividade — Políticas Públicas e Teorias de Análise"
min_palavras_atividade = 1200
alvo_palavras_atividade = 2500
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]
orientacao_inline = "Responda exclusivamente com base nos textos fornecidos. Compare convergências, divergências, conceitos centrais e implicações para a administração pública."
modo_escrita = "novo"
usar_artigos_selecionados_pesquisa = true
usar_bib_da_pesquisa = true

[interpretacao]
usar_textos_completos = true
usar_resumos_individuais = true
usar_contexto_consolidado = true
exigir_cobertura_total_fulltext_cache = true
permitir_fontes_externas = false
responder_exclusivamente_com_base_nos_documentos = true
comparar_textos_entre_si = true
identificar_convergencias = true
identificar_divergencias = true
extrair_conceitos_centrais = true
extrair_argumentos_centrais = true

[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-paper.sty"
fgv_logo_path = "./misc/fgv/fgv.png"

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
dry_run = false
```

---

### 8.6. Fichamento com documentos locais

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true

[documentos_locais]
ativos = true
input_dir = "./inputs/textos_fichamento"
recursive = true
tipos = ["pdf", "docx", "txt", "md", "org"]
prefixo = "fichamento_aula_2"
output_dir = "./output/corpus_local"
responder_exclusivamente_com_base_nos_documentos = true
permitir_fontes_externas = false

[atividade]
modo = "fichamento"

[documento]
tipo_documento = "fichamento"
prefixo = "fichamento_aula_2"
output_dir = "./output/documento"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
exportar_pdf = true
min_palavras_fichamento = 1200
alvo_palavras_fichamento = 2500
orientacao_inline = "Produza fichamento analítico com referência ao argumento central, conceitos, método, contribuições, limites e relação com a disciplina."

[interpretacao]
comparar_textos_entre_si = true
identificar_convergencias = true
identificar_divergencias = true
extrair_conceitos_centrais = true
extrair_argumentos_centrais = true

[controle]
nao_interativo = true
```

---

### 8.7. Resposta discursiva

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true

[documentos_locais]
ativos = true
input_zip = "./inputs/base_resposta_discursiva.zip"
prefixo = "resposta_discursiva_capacidades_estatais"
output_dir = "./output/corpus_local"
responder_exclusivamente_com_base_nos_documentos = true
permitir_fontes_externas = false

[atividade]
modo = "resposta_discursiva"

[pesquisa]
tema = "Capacidades estatais"
recorte = "Coerção, centralização e desempenho de políticas públicas"
objetivo = "Responder à questão discursiva com base nos textos fornecidos."
pergunta_pesquisa = "Como a relação entre coerção e centralização do poder pode afetar o desempenho estatal em políticas públicas?"

[documento]
tipo_documento = "resposta_discursiva"
prefixo = "resposta_discursiva_capacidades_estatais"
output_dir = "./output/documento"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
exportar_pdf = true
orientacao_inline = "Estruture a resposta em introdução, desenvolvimento argumentativo e conclusão. Use conceitos dos textos e evite extrapolações não sustentadas."
min_palavras_atividade = 1000
alvo_palavras_atividade = 1800

[controle]
nao_interativo = true
```

---

### 8.8. Resumo expandido

```toml
[pipeline]
modo_entrada = "documentos_locais"
executar_pesquisa = false
executar_bundle = true
executar_documento = true

[documentos_locais]
ativos = true
input_dir = "./inputs/textos_resumo"
prefixo = "resumo_expandido_aula_3"
output_dir = "./output/corpus_local"

[atividade]
modo = "resumo_expandido"

[documento]
tipo_documento = "resumo_expandido"
prefixo = "resumo_expandido_aula_3"
output_dir = "./output/documento"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
exportar_pdf = true
min_palavras_resumo = 800
alvo_palavras_resumo = 1500
orientacao_inline = "Produza resumo expandido com problema, objetivo, argumentos centrais, conceitos, contribuições e síntese crítica."

[controle]
nao_interativo = true
```

---

### 8.9. Reescrever ou expandir um documento existente

```toml
[pipeline]
modo_entrada = "pesquisa_existente"
executar_pesquisa = false
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = "./output/pesquisa/minha_pesquisa"

[documento]
tipo_documento = "paper"
prefixo = "paper_reescrito"
output_dir = "./output/documento"
template_org = "./templates/template_paper.org"
exportar_pdf = true
modo_escrita = "expandir"
reescrever_a_partir_do_org_atual = true
documento_org_existente = "./output/documento/paper_antigo/paper_antigo.org"
preservar_estrutura_do_org_anterior = true
usar_artigos_selecionados_pesquisa = true
artigos_extras_paths = ["./inputs/artigos_extras"]
orientacao_inline = "Preserve a estrutura geral do documento anterior, mas aprofunde a fundamentação teórica e melhore a articulação entre autores."

[controle]
nao_interativo = true
```

---

### 8.10. Execução sem PDF, apenas ORG/BIB/JSON

```toml
[saida]
exportar_pdf = false

[documento]
exportar_pdf = false

[controle]
nao_interativo = true
```

Use este modo quando o ambiente LaTeX/Emacs ainda não estiver configurado.

---

### 8.11. Execução de diagnóstico (`dry_run`)

```toml
[controle]
nao_interativo = true
dry_run = true
```

O `dry_run` é recomendado antes de uma execução longa. Ele ajuda a verificar:

- se os caminhos existem;
- se o TOML está coerente;
- se o template foi encontrado;
- se a pesquisa existente foi localizada;
- se os documentos locais estão acessíveis.

---

## 9. Saídas geradas

### 9.1. Pesquisa

Saídas típicas em `output/pesquisa/<prefixo>/`:

```text
<prefixo>.org
<prefixo>.bib
<prefixo>.pdf
<prefixo>_debug.json
<prefixo>_semantic_scholar_raw.json
<prefixo>_scopus_raw.json
<prefixo>_pubmed_raw.json
<prefixo>_openalex_raw.json
<prefixo>_crossref_raw.json
<prefixo>_europe_pmc_raw.json
<prefixo>_core_raw.json
<prefixo>_prisma.svg
<prefixo>_prisma.pdf
<prefixo>_fulltext_cache/
pipeline_research_config.toml
```

### 9.2. Corpus local

Saídas típicas em `output/corpus_local/<prefixo>/`:

```text
<prefixo>.org
<prefixo>.bib
<prefixo>_debug.json
<prefixo>_contexto_local.json
<prefixo>_fulltext_cache/
extracted/
pipeline_research_config.toml
```

### 9.3. Documento final

Saídas típicas em `output/documento/<prefixo>/`:

```text
<prefixo>.org
<prefixo>.bib
<prefixo>.pdf
<prefixo>_contexto.json
<prefixo>_prompts_auditoria.txt
<prefixo>_proveniencia.json
<prefixo>_uso_referencias.json
<prefixo>_uso_referencias.md
<prefixo>_limites_secoes.json
entrega_final/
```

### 9.4. Entrega final

A pasta `entrega_final/` pode conter:

```text
README_entrega.md
manifest.json
<prefixo>.org
<prefixo>.bib
<prefixo>.pdf
<prefixo>_contexto.json
<prefixo>_debug.json
<prefixo>_proveniencia.json
<prefixo>_prompts_auditoria.txt
<prefixo>_uso_referencias.md
```

---

## 10. Boas práticas de uso

### 10.1. Caminhos

Prefira caminhos relativos à raiz do bundle:

```toml
template_org = "./templates/template_paper.org"
output_dir = "./output/documento"
```

Evite caminhos absolutos quando for distribuir o programa para outra pessoa.

### 10.2. Separação entre pesquisa, documento e corpus local

Use:

```text
output/pesquisa/
output/documento/
output/corpus_local/
```

Evite salvar tudo em uma única pasta, pois isso dificulta depuração e reaproveitamento.

### 10.3. Orientações

Use:

```toml
[saida]
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]

[triagem]
orientacoes_paths = ["./prompts/triagem_prompt.txt", "./prompts/diretivas_extras.txt"]

[documento]
orientacoes_paths = ["./prompts/orientacao_geral_execucao.txt"]
```

Regra prática:

| Seção | Coloque aqui |
|---|---|
| `[saida]` | Orientações gerais da pesquisa. |
| `[triagem]` | Critérios de seleção, exclusão e ranking. |
| `[documento]` | Estilo, estrutura, profundidade e tipo documental. |
| `[prompts]` | Prompts substituíveis por tarefa. |

### 10.4. Não misture artigos extras com o cache da pesquisa

Use:

```toml
[documento]
artigos_extras_paths = ["./inputs/artigos_extras"]
```

Não coloque manualmente artigos extras dentro de `*_fulltext_cache`, salvo se souber exatamente o efeito bibliográfico desejado.

### 10.5. Para documento com apenas um texto

Quando `quantidade_selecionados = 1` ou quando o corpus local tem apenas um documento, prefira orientar o documento para:

- análise individual;
- resposta ao problema;
- comentário crítico;
- identificação de argumento central;
- aplicação ao tema da disciplina.

Evite pedir “síntese integradora dos textos” se só houver um texto.

Exemplo:

```toml
[documento]
orientacao_inline = "Como há apenas um texto-base, não produza síntese integradora comparativa. Faça análise individual aprofundada do texto, seguida de conclusão crítica."
```

---

## 11. Solução de problemas

### 11.1. `OPENAI_API_KEY não encontrada`

Verifique se existe `.env` na raiz do bundle:

```bash
cat .env
```

Ou exporte manualmente:

```bash
export OPENAI_API_KEY="sua_chave"
```

### 11.2. `Unknown LaTeX class 'fgv-paper'` ou `fgv-dissertacao`

O Emacs batch não conhece a classe LaTeX. Configure:

```toml
[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-paper.sty"
```

Para dissertação:

```toml
[latex]
org_latex_class_init = "./misc/academic-writing.el"
latex_extra_path = "./misc/fgv/fgv-dissertacao.sty"
```

Também confirme que o arquivo `.org` contém a classe correta, por exemplo:

```org
#+LATEX_CLASS: fgv-paper
```

ou:

```org
#+LATEX_CLASS: fgv-dissertacao
```

### 11.3. PDF não compila, mas ORG foi gerado

O `.org` e o `.bib` normalmente continuam válidos. Compile manualmente depois de corrigir o ambiente LaTeX/Emacs.

Sugestão:

```bash
cd output/documento/meu_documento
emacs --batch -Q -l ../../../../misc/academic-writing.el meu_documento.org --funcall org-latex-export-to-pdf
```

Ou desative temporariamente:

```toml
[documento]
exportar_pdf = false
```

### 11.4. Scopus/Web of Science não retorna resultados

Verifique:

```env
SCOPUS_API_KEY=
SCOPUS_INSTTOKEN=
WOS_API_KEY=
```

Nem todas as bases funcionam sem credenciais institucionais.

### 11.5. Muitos textos genéricos entram na triagem

Ajuste:

```toml
[triagem]
rigor = "estrito"
orientacao_inline = "Exclua textos genericamente relacionados ao tema que não tratem diretamente do recorte e do objetivo."
```

E reduza ambiguidades em `[pesquisa]`.

### 11.6. Poucos textos são encontrados

Tente:

```toml
[triagem]
rigor = "exploratorio"

[busca]
quantidade_triagem = 150
quantidade_selecionados = 20

[queries]
query_geral = ""
```

Também considere ampliar `periodo`, `idiomas` ou `bases`.

### 11.7. Documento ficou curto

Ajuste metas:

```toml
[documento]
min_palavras_total = 22000
alvo_palavras_total = 28000
geracao_em_etapas = true

[geracao]
modo = "etapas"
gerar_por_subsecoes = true
reparar_secao_curta = true
```

### 11.8. Citações não aparecem

Verifique:

```toml
[documento]
usar_bib_da_pesquisa = true
priorizar_citacoes_dos_selecionados = true
incluir_artigos_extras_no_bib = true
```

Confirme se o `.bib` foi copiado para a mesma pasta do `.org` final.

### 11.9. Documentos locais não foram encontrados

Confirme se usou `input_zip` ou `input_dir`:

```toml
[documentos_locais]
ativos = true
input_zip = "./inputs/textos.zip"
# ou
input_dir = "./inputs/textos"
```

Verifique também as extensões:

```toml
tipos = ["pdf", "docx", "txt", "md", "org"]
```

---

## 12. Checklist antes de distribuir o programa

Antes de entregar o bundle para outra pessoa:

- [ ] Remover chaves reais do `.env`.
- [ ] Substituir caminhos absolutos por caminhos relativos.
- [ ] Manter `requirements.txt` atualizado.
- [ ] Incluir templates em `templates/`.
- [ ] Incluir `misc/academic-writing.el` e estilos em `misc/fgv/`.
- [ ] Incluir prompts em `prompts/`.
- [ ] Incluir este `README.md`.
- [ ] Testar `mock_run = true`.
- [ ] Testar uma execução com `dry_run = true`.
- [ ] Testar pelo menos uma execução real pequena.

### `.env.example` recomendado

Distribua um `.env.example`, não o `.env` real:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5.4
SEMANTIC_SCHOLAR_API_KEY=
SCOPUS_API_KEY=
SCOPUS_INSTTOKEN=
WOS_API_KEY=
NCBI_API_KEY=
NCBI_EMAIL=
OPENALEX_API_KEY=
OPENALEX_EMAIL=
CROSSREF_EMAIL=
EUROPEPMC_EMAIL=
CORE_API_KEY=
UNPAYWALL_EMAIL=
ORG_LATEX_CLASS_INIT=./misc/academic-writing.el
LATEX_EXTRA_PATH=./misc/fgv/fgv-paper.sty
```

---

## 13. Apêndice — comandos CLI úteis

### Pesquisa

```bash
python scripts/research/gerador_pesquisa_rc_2.py --config config/research/template_toml_unificado_rc_2.toml
```

Com overrides:

```bash
python scripts/research/gerador_pesquisa_rc_2.py \
  --config config/research/template_toml_unificado_rc_2.toml \
  --tema "Capacidades estatais" \
  --recorte "Centralização e desempenho de políticas públicas" \
  --quantidade-triagem 60 \
  --quantidade-selecionados 8 \
  --triagem-rigor moderado
```

### Pipeline

```bash
python scripts/pipeline/gerador_pesquisa_documento_rc_6_refatorado_v5_2_10.py \
  --config config/pipeline/toml_pipeline_completo_v5_2_0.toml
```

Com override de modelo:

```bash
python scripts/pipeline/gerador_pesquisa_documento_rc_6_refatorado_v5_2_10.py \
  --config config/pipeline/toml_pipeline_completo_v5_2_0.toml \
  --model gpt-5.4
```

### Documento standalone

```bash
python scripts/document/gerador_documento_academico_rc_3.py \
  --config config/document/template_toml_documento_academico_rc_3.toml
```

Com preflight:

```bash
python scripts/document/gerador_documento_academico_rc_3.py \
  --config config/document/template_toml_documento_academico_rc_3.toml \
  --preflight-only
```

Com parâmetros diretos:

```bash
python scripts/document/gerador_documento_academico_rc_3.py \
  --tipo-documento paper \
  --bundle-dir ./output/bundle \
  --template ./templates/template_paper.org \
  --output-dir ./output/documento \
  --basename meu_paper \
  --citation-style apa \
  --exportar-pdf
```

---

## 14. Apêndice — modelo mínimo de TOML para cada tipo de documento

### 14.1. Paper

```toml
[documento]
tipo_documento = "paper"
prefixo = "meu_paper"
template_org = "./templates/template_paper.org"
output_dir = "./output/documento"
exportar_pdf = true
modo_escrita = "novo"
```

### 14.2. Dissertação

```toml
[documento]
tipo_documento = "dissertacao"
prefixo = "minha_dissertacao"
template_org = "./templates/template_dissertacao_fgv_apa_v2.org"
output_dir = "./output/documento"
exportar_pdf = true
institution_name = "Fundação Getúlio Vargas"
program_name = "Mestrado Profissional em Políticas Públicas e Governo"
area_de_concentracao = "Políticas Públicas e Governo"
ano = "2026"
linha_pesquisa = "Governança, Estado e Políticas Públicas"
geracao_em_etapas = true
min_palavras_total = 22000
alvo_palavras_total = 28000
```

### 14.3. Atividade

```toml
[documento]
tipo_documento = "atividade"
prefixo = "atividade_aula_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
min_palavras_atividade = 1200
alvo_palavras_atividade = 2500
```

### 14.4. Resposta discursiva

```toml
[documento]
tipo_documento = "resposta_discursiva"
prefixo = "resposta_discursiva_aula_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
orientacao_inline = "Responda em formato discursivo, com tese, desenvolvimento e conclusão."
```

### 14.5. Fichamento

```toml
[documento]
tipo_documento = "fichamento"
prefixo = "fichamento_texto_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
min_palavras_fichamento = 1200
alvo_palavras_fichamento = 2500
```

### 14.6. Resumo

```toml
[documento]
tipo_documento = "resumo"
prefixo = "resumo_texto_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
min_palavras_resumo = 600
alvo_palavras_resumo = 1200
```

### 14.7. Resumo expandido

```toml
[documento]
tipo_documento = "resumo_expandido"
prefixo = "resumo_expandido_texto_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
min_palavras_resumo = 800
alvo_palavras_resumo = 1800
```

### 14.8. Ensaio

```toml
[documento]
tipo_documento = "ensaio"
prefixo = "ensaio_aula_1"
template_org = "./templates/template_atividade_fgv_v5_2_7.org"
output_dir = "./output/documento"
exportar_pdf = true
orientacao_inline = "Produza ensaio acadêmico com tese clara, argumentação progressiva e conclusão crítica."
```

---

## 15. Convenções recomendadas de versionamento

Ao criar novas versões do pipeline, mantenha nomes explícitos:

```text
gerador_pesquisa_documento_rc_6_refatorado_v5_2_11.py
gerador_pesquisa_documento_rc_6_refatorado_v5_2_12.py
```

Para TOMLs:

```text
toml_pipeline_completo_v5_2_1.toml
toml_pipeline_dissertacao_fgv_v5_2_1.toml
toml_pipeline_atividade_fgv_v5_2_1.toml
```

Para saídas:

```toml
[saida]
prefixo = "tema_recorte_versao"

[documento]
prefixo = "tema_recorte_tipo_documento"
```

---

## 16. Resumo operacional

1. Configure `.env`.
2. Escolha o TOML correto.
3. Preencha `[pesquisa]`, `[atividade]` e `[documento]`.
4. Use caminhos relativos sempre que possível.
5. Rode primeiro com `dry_run = true` ou `mock_run = true`.
6. Rode a execução real.
7. Confira `output/pesquisa`, `output/documento` e `entrega_final`.
8. Se o PDF falhar, aproveite o `.org` e ajuste `[latex]`.

