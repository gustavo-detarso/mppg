---
document_id: MPPG-ACADEMIC-PIPELINE-MANUAL-001
canonical_filename: MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE.md
project: Software MPPG / Academic Pipeline
title: Manual técnico e operacional completo do Academic Pipeline
status: vigente
effective_from: 2026-08-07
last_revised: 2026-08-07
language: pt-BR
audience:
  - usuários do Academic Pipeline
  - estudantes e pesquisadores
  - mantenedores do Software MPPG
  - revisores de código
  - assistentes de IA
scope: funcionalidades operacionais, usabilidade, configuração, interfaces, fluxos, saídas, diagnóstico, PRISMA, renderização, módulos canônicos e governança documental
revision_control: git
canonical_source_root: software/academic_pipeline_mppg
baseline_branch: master
baseline_commit: 9e24ad6db001d56d7334bf3ab97c97a05cce579a
baseline_commit_subject: "feat(mppg): publish canonical Atestmed project artifacts"
refactor_program_state: closed
refactor_program_progress_percent: 100
legacy_runtime_state: retired
manual_synchronization_policy: mandatory_after_functional_change
mandatory_next_step: none
---

# Manual técnico e operacional completo do Academic Pipeline — Software MPPG

## 1. Função deste manual

Este documento é a referência técnica e de usabilidade do **Academic Pipeline MPPG** no baseline canônico indicado no cabeçalho.

Seu objetivo é responder, em um único lugar, a cinco perguntas:

1. **o que o programa faz;**
2. **como o usuário executa cada função;**
3. **quais entradas, configurações e arquivos participam de cada fluxo;**
4. **quais artefatos são produzidos;**
5. **como o software está organizado internamente para manutenção e evolução.**

A estrutura documental foi inspirada no modelo `UAC_DIRETRIZ_PERMANENTE_NOMENCLATURA_CANONICA_IA.md`, especialmente em sua ideia de fonte canônica estável, baseline explícito, seções operacionais, governança e sincronização documental. O conteúdo funcional deste manual, contudo, deriva do **Academic Pipeline real**, não do software UAC.

Este manual adota um nome canônico estável:

```text
MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE.md
```

A evolução futura deve ocorrer no Git. Não devem ser criadas autoridades paralelas como `..._v2.md`, `..._final.md` ou arquivos datados para representar a versão vigente.

---

## 2. Baseline canônico documentado

```yaml
repository:
  branch: master
  commit: 9e24ad6db001d56d7334bf3ab97c97a05cce579a
  canonical_software_root: software/academic_pipeline_mppg

package:
  name: academic-pipeline-mppg
  package_version: 0.1.0
  python: ">=3.11"
  console_entrypoint: academic-pipeline
  console_target: academic_pipeline.cli:main
  module_entrypoint: python -m academic_pipeline

refactor:
  state: closed
  progress_percent: 100
  legacy_productive_runtime: absent
```

O commit-base deste manual já contém a publicação dos artefatos canônicos do projeto Atestmed e é o HEAD de `master` utilizado para a presente documentação.

---

## 3. Visão geral: o que o Academic Pipeline faz

O Academic Pipeline é um pacote Python modular para **geração, validação, rastreabilidade e renderização de documentos acadêmicos**. Ele combina:

- configuração declarativa em TOML;
- documentos locais como corpus;
- orientações e prompts;
- bibliografia e DOI;
- geração estruturada por IA quando habilitada;
- modelo documental canônico em JSON;
- validações antes e depois da geração;
- renderização ORG/LaTeX/PDF;
- renderização DOCX;
- perfis e layouts institucionais;
- relatórios de qualidade e conformidade;
- mapas mentais;
- traduções;
- resumos e palavras-chave de papers;
- fluxos de revisão estruturada PRISMA;
- busca externa, triagem e curadoria de referências;
- interfaces CLI, TUI e GUI;
- utilitários de projeto, diagnóstico e recompilação sem IA.

O programa pode ser usado tanto para **gerar um documento novo** quanto para **reutilizar um `document.json` existente**, **recompilar um ORG**, **validar configurações/artefatos**, **gerenciar pesquisa PRISMA** ou **preparar um artigo final a partir de referências congeladas**.

---

## 4. Arquitetura operacional

O caminho público principal é:

```text
academic-pipeline
        │
        └── academic_pipeline.cli:main
                │
                └── academic_pipeline.runtime:run
                        ├── rotas informativas
                        ├── doctor
                        ├── check-config
                        ├── list-profiles
                        ├── institution-compliance
                        ├── DOI manifest
                        └── default_runtime
                                ├── configuração
                                ├── corpus
                                ├── bibliografia
                                ├── PRISMA
                                ├── document_model
                                ├── validação
                                ├── mapa mental
                                ├── ORG/PDF
                                ├── DOCX
                                ├── traduções
                                ├── prompt lock
                                ├── conformidade
                                └── qualidade/manifests
```

A mesma autoridade é alcançada por:

```bash
academic-pipeline ...
```

ou:

```bash
python -m academic_pipeline ...
```

O runtime canônico seleciona deliberadamente a rota apropriada. Não há fallback produtivo para o antigo monólito RC.

---

## 5. Requisitos e dependências Python

O pacote exige **Python 3.11 ou superior**.

Dependências declaradas:

```text
openai >= 1.0.0
pydantic >= 2.0
python-dotenv >= 1.0
pypdf >= 4.0
python-docx >= 1.1
openpyxl >= 3.1
```

Alguns fluxos também dependem de ferramentas externas disponíveis no ambiente, especialmente quando há compilação LaTeX/PDF, renderização de mapa mental ou interfaces opcionais.

### 5.1. Instalação do pacote em ambiente de desenvolvimento

A partir de `software/academic_pipeline_mppg`:

```bash
python -m pip install -e .
```

Após a instalação:

```bash
academic-pipeline --help
```

e:

```bash
python -m academic_pipeline --help
```

devem apontar para a mesma superfície pública.

### 5.2. Recursos incluídos na distribuição

O `pyproject.toml` inclui explicitamente como dados de pacote:

- exemplos TOML;
- assets FGV;
- templates DOCX e ORG;
- estilos LaTeX;
- prompts institucionais;
- validators TOML;
- prompts globais, de documento, pesquisa e PRISMA;
- templates CSL;
- recursos misc necessários ao runtime.

---

## 6. Quick start

### 6.1. Ver a ajuda

```bash
academic-pipeline --help
```

### 6.2. Diagnosticar o ambiente

```bash
academic-pipeline --doctor
```

### 6.3. Listar presets TOML

```bash
academic-pipeline --list-toml-profiles
```

### 6.4. Listar perfis institucionais

```bash
academic-pipeline --list-institutions
```

### 6.5. Explicar o perfil FGV

```bash
academic-pipeline --explain-profile fgv
```

### 6.6. Validar um TOML antes da execução

```bash
academic-pipeline --check-config --config caminho/projeto.toml
```

### 6.7. Executar um projeto configurado

```bash
academic-pipeline --config caminho/projeto.toml
```

### 6.8. Abrir a Central Operacional FGV

```bash
academic-pipeline --tui
```

### 6.9. Abrir o wizard de TOML

```bash
academic-pipeline --init-toml
```

ou com preset:

```bash
academic-pipeline --init-toml --toml-profile atividade_local_fgv
```

---

## 7. Formas de interação

### 7.1. CLI

É a superfície pública mais completa e previsível. Use quando:

- automatizar fluxos;
- executar por shell;
- validar em scripts;
- trabalhar com TOML já existente;
- depurar uma etapa isolada.

### 7.2. TUI — Central Operacional FGV

A TUI usa `prompt_toolkit` e oferece navegação guiada, seleção de projetos, confirmação antes de operações custosas, campos de caminho com conclusão e logs operacionais.

Fluxo visual declarado:

```text
escolher categoria
→ configurar
→ conferir
→ validar
→ gerar
→ revisar
```

A TUI mantém estado em:

```text
app_bundle/.academic_pipeline_tui_state.json
```

e logs em:

```text
app_bundle/output/tui_logs/
```

Ela separa explicitamente:

- **TOML global da pipeline**, usado nos fluxos gerais;
- **estado/TOML do artigo PRISMA**, usado no fluxo robusto de artigo final.

### 7.3. Perfis guiados da TUI

A Central Operacional expõe estes perfis:

1. `atividade_local_fgv` — atividade acadêmica local;
2. `resumo_artigos_local_fgv` — resumo analítico de artigos;
3. `paper_local_fgv` — paper a partir de corpus local;
4. `paper_prisma_fgv` — paper + relatório PRISMA;
5. `dissertacao_local_fgv` — dissertação local;
6. `dissertacao_prisma_fgv` — dissertação + PRISMA;
7. `relatorio_prisma_fgv` — relatório PRISMA autônomo;
8. `relatorio_prisma_busca_orientada_fgv` — busca externa, deduplicação e triagem humana;
9. `somente_renderizar_fgv` — renderização de `document.json` existente, sem nova geração textual.

### 7.4. GUI

A opção:

```bash
academic-pipeline --gui
```

abre a interface gráfica FGV disponível no bundle. Ela é especialmente orientada ao fluxo de atividades acadêmicas e projetos com configuração TOML.

### 7.5. Wizard TOML

O wizard:

```bash
academic-pipeline --init-toml
```

gera a configuração do projeto por perguntas e presets. A opção `--toml-profile` permite iniciar diretamente em um perfil.

---

## 8. Catálogo completo da superfície CLI

A tabela a seguir consolida a superfície pública encontrada no runtime/parser canônico, incluindo a opção dinâmica `--list-profiles` e os wrappers PRISMA tratados pelo runtime.

| Opção | Área | Finalidade |
| --- | --- | --- |
| --config <arquivo.toml> | Fluxo principal | Seleciona o TOML do projeto/documento a executar. |
| --tui | Interface | Abre a Central Operacional FGV em terminal, baseada em prompt_toolkit. |
| --gui | Interface | Abre a interface gráfica FGV para atividades acadêmicas. |
| --init-toml | Configuração | Abre o gerador interativo completo de TOML. |
| --toml-profile <perfil> | Configuração | Define o preset inicial do gerador TOML. |
| --no-clear | Interface | Evita limpeza convencional da tela durante o wizard/TUI. |
| --list-toml-profiles | Descoberta | Lista presets disponíveis do gerador TOML. |
| --list-profiles | Descoberta | Lista os perfis TOML disponibilizados pelo runtime canônico. |
| --list-institutions | Institucional | Lista perfis institucionais instalados. |
| --list-layouts | Institucional | Lista layouts disponíveis para o perfil institucional selecionado. |
| --explain-profile [perfil] | Institucional | Explica perfil institucional; sem valor explícito usa FGV. |
| --show-prompts | Prompts | Mostra prompts e diretivas ativos para o TOML informado. |
| --write-prompt-lock | Prompts | Gera prompt_lock.json e prompt_lock.md e encerra. |
| --check-institution-compliance | Validação | Valida artefatos existentes contra o perfil institucional. |
| --doctor | Diagnóstico | Diagnostica ambiente, ferramentas, recursos institucionais e estilo bibliográfico. |
| --check-config | Validação | Valida preventivamente o TOML e encerra. |
| --recompile | Renderização | Recompila um arquivo ORG existente sem nova chamada à IA. |
| --org <arquivo.org> | Renderização | Arquivo ORG usado por recompile, quality-report ou conformidade. |
| --academic-writing <arquivo.el> | Renderização | Override do academic-writing.el em recompilação. |
| --latex-extra-path <caminho> | Renderização | Override de caminho extra de LaTeX em recompilação. |
| --pdf-engine <motor> | Renderização | Override do motor PDF em recompilação. |
| --no-clean | Renderização | Preserva arquivos auxiliares de compilação no modo recompile. |
| --somente-renderizar | Reuso | Usa document.json existente e somente renderiza saídas. |
| --somente-mapa-mental | Mapa mental | Usa document.json existente e executa apenas a etapa de mapa mental. |
| --reusar-mapa-mental | Mapa mental | Reaproveita mapa já existente quando disponível, sem nova geração. |
| --forcar-regeneracao-mapa-mental | Mapa mental | Remove artefatos conhecidos do mapa mental e força recriação. |
| --document-json <arquivo.json> | Reuso | Caminho do document.json existente. |
| --prisma-importar-triagem <csv> | PRISMA | Importa triagem humana e consolida matriz/relatório PRISMA. |
| --init-project <nome> | Projeto | Cria a estrutura de projeto em app_bundle/projetos/<nome>. |
| --project-type <tipo> | Projeto | Tipo: paper, atividade, prisma, atividade_prisma ou paper_prisma. |
| --institution <perfil> | Projeto | Perfil institucional utilizado na criação do projeto; padrão fgv. |
| --base-dir <caminho> | Projeto | Raiz alternativa do academic_pipeline/app_bundle para init-project. |
| --overwrite-project | Projeto | Permite sobrescrever somente arquivos considerados seguros pelo inicializador. |
| --make-doi-manifest | DOI | Gera doi_manifest.csv de uma pasta ou ZIP de documentos. |
| --input-zip <arquivo.zip> | DOI | ZIP de documentos usado para criação do manifesto DOI. |
| --input-dir <diretório> | DOI | Diretório de documentos usado para criação do manifesto DOI. |
| --output <arquivo> | Saída | Arquivo de saída em comandos que aceitam destino explícito. |
| --output-dir <diretório> | Caminhos | Override de document_output_dir. |
| --work-dir <diretório> | Caminhos | Override de work_dir para extrações temporárias. |
| --cache-dir <diretório> | Caminhos | Override de cache_dir para fulltext_cache. |
| --research-output-dir <diretório> | Caminhos | Override do diretório de saída de pesquisa/PRISMA. |
| --output-prefix <prefixo> | Caminhos | Override do prefixo canônico dos artefatos do documento. |
| --layout <layout> | Documento | Override de documento.layout. |
| --tipo-conteudo <tipo> | Documento | Override de documento.tipo_conteudo. |
| --genero-academico <gênero> | Documento | Override de documento.genero_academico. |
| --no-output-subdir | Caminhos | Evita criação do subdiretório document_prefix sob o output. |
| --inspect-bib <arquivo.bib> | Bibliografia | Inspeciona BibTeX e gera relatório Markdown/JSON. |
| --quality-report | Qualidade | Gera quality_report.md a partir de document.json, opcionalmente ORG. |
| --bib <arquivo.bib> | Bibliografia | BibTeX opcional para qualidade ou conformidade. |
| --docx <arquivo.docx> | Conformidade | DOCX opcional para validação institucional. |
| --pdf <arquivo.pdf> | Conformidade | PDF opcional para validação institucional. |
| --prisma-curadoria-menu | PRISMA | Abre submenu de curadoria de referências. |
| --prisma-curadoria-ia | PRISMA | Executa curadoria assistida por IA e gera XLSX/CSV. |
| --prisma-curadoria-sem-ia | PRISMA | Executa curadoria por heurística local, sem chamada à IA. |
| --prisma-curadoria-reexportar-xlsx | PRISMA | Reexporta XLSX revisado para triagem_humana.csv. |
| --prisma-curadoria-importar | PRISMA | Importa triagem_humana.csv e produz PRISMA final. |
| --prisma-curadoria-fluxo-completo | PRISMA | Executa curadoria e depois a importação final. |
| --prisma-curadoria-prompt <yaml> | PRISMA | Prompt estruturado específico da curadoria. |
| --prisma-curadoria-input <arquivo> | PRISMA | Entrada XLSX/CSV específica da curadoria. |
| --prisma-curadoria-out-dir <diretório> | PRISMA | Diretório explícito de saída da curadoria/relatório. |
| --prisma-curadoria-max-incluir <n> | PRISMA | Limite máximo de referências incluídas. |
| --prisma-curadoria-top-n-candidatos <n> | PRISMA | Número de candidatos avaliados/enviados à curadoria IA. |
| --prisma-curadoria-limiar-minimo <n> | PRISMA | Limiar mínimo de inclusão da curadoria. |
| --prisma-exportar-bib | PRISMA/artigo | Wrapper canônico para exportação BibTeX do conjunto PRISMA. |
| --prisma-congelar-artigo | PRISMA/artigo | Wrapper para congelar insumos selecionados do artigo PRISMA. |
| --prisma-gerar-toml-artigo | PRISMA/artigo | Wrapper para gerar o TOML final do artigo a partir do estado PRISMA. |
| --prisma-gerar-artigo-final | PRISMA/artigo | Wrapper para gerar o artigo final com os insumos PRISMA congelados. |
| -h / --help | Ajuda | Exibe ajuda do entrypoint oficial academic-pipeline. |

### 8.1. Regra de precedência

Alguns comandos são rotas especializadas e deliberadamente não podem ser misturados livremente com outros comandos:

- `--doctor`;
- `--check-config`;
- `--list-profiles`;
- `--check-institution-compliance`;
- `--make-doi-manifest`.

O runtime contém verificações explícitas de combinação e retorna erro de uso quando uma rota especializada é misturada com opções incompatíveis.

---

## 9. Configuração TOML

O TOML é o contrato declarativo central do fluxo principal.

O pipeline carrega o arquivo, registra internamente seu caminho e diretório, aplica o perfil institucional e depois aceita overrides de CLI.

Seções observadas no runtime e módulos:

```text
[projeto]
[documento]
[paths]
[openai]
[documentos_locais]
[orientacoes]
[bibliografia]
[prompts]
[pipeline]
[latex]
[docx]
[mapa_mental]
[idiomas_saida]
[resumo_paper] / configurações equivalentes de resumos
[pesquisa]
[busca_prisma]
[atividade]
```

A presença exata de algumas seções depende do perfil/preset.

### 9.1. Paths principais

O runtime trabalha com:

```text
document_output_dir
document_prefix
work_dir
cache_dir
research_output_dir
create_document_subdir
```

A CLI pode sobrescrever os caminhos principais com:

```text
--output-dir
--work-dir
--cache-dir
--research-output-dir
--output-prefix
--no-output-subdir
```

### 9.2. Overrides do documento

```text
--layout
--tipo-conteudo
--genero-academico
```

permitem alterar aspectos do documento sem editar permanentemente o TOML.

### 9.3. Modelo OpenAI

O fluxo que exige geração textual cria o cliente OpenAI a partir da configuração/ambiente e seleciona o modelo definido no projeto. Modos de renderização/recompilação podem evitar nova chamada à IA.

---

## 10. Inicialização de projetos

Use:

```bash
academic-pipeline --init-project meu_projeto --project-type paper
```

Tipos aceitos:

```text
paper
atividade
prisma
atividade_prisma
paper_prisma
```

O inicializador utiliza templates de configuração do bundle e cria uma árvore de projeto em `app_bundle/projetos/<nome>`, com arquivos auxiliares e placeholders adequados ao tipo.

Exemplo institucional:

```bash
academic-pipeline   --init-project trabalho_final   --project-type atividade   --institution fgv
```

A opção `--overwrite-project` não é uma autorização para apagar arbitrariamente conteúdo; ela libera apenas sobrescritas consideradas seguras pelo inicializador.

---

## 11. Fluxo principal de geração de documento

O fluxo padrão por TOML pode ser resumido assim:

```text
1. carregar .env e TOML
2. aplicar perfil institucional
3. aplicar overrides da CLI
4. resolver outputs/work/cache
5. validar preventivamente a configuração
6. descobrir documentos locais
7. copiar documentos para fulltext_cache
8. carregar orientações
9. gerar/validar bibliografia
10. executar PRISMA, quando habilitado
11. criar AcademicDocument/document.json com IA
12. sanear vazamentos técnicos e bibkeys crus
13. validar document_model
14. gerar resumos/palavras-chave de paper, quando habilitados
15. gerar/reutilizar mapa mental, quando habilitado
16. salvar document.json
17. renderizar ORG
18. compilar PDF, conforme configuração
19. renderizar DOCX, conforme configuração
20. gerar traduções adicionais, quando configuradas
21. registrar prompt_lock
22. validar conformidade institucional
23. gerar relatório de qualidade
24. gerar run report e manifesto de outputs
```

### 11.1. Modo somente pesquisa

Quando a configuração desativa a produção do documento acadêmico e mantém a pesquisa, o programa pode gerar bibliografia/PRISMA e relatórios sem criar `document.json`, ORG, PDF ou DOCX final.

---

## 12. Corpus de documentos locais

O módulo `corpus_manager.py` é responsável por:

- localizar documentos locais;
- tratar entrada por diretório/ZIP conforme a configuração;
- extrair texto de formatos suportados;
- capturar avisos de PDFs estruturalmente irregulares;
- limitar tamanho textual quando necessário;
- produzir metadados de origem;
- copiar documentos para `fulltext_cache`;
- carregar documentos de orientação.

O fluxo principal usa:

```text
discover_local_documents
copy_documents_to_fulltext_cache
collect_orientation_docs
```

O cache é um artefato operacional, não a autoridade bibliográfica original.

---

## 13. Bibliografia e DOI

### 13.1. Construção de bibliografia

`bibliography_manager.py`:

- identifica DOI em texto/metadados;
- consulta provedores quando configurados;
- normaliza DOI;
- produz metadados bibliográficos;
- renderiza entradas BibTeX;
- lê e separa entradas BibTeX;
- identifica entradas por DOI ou metadados;
- calcula qualidade das entradas;
- deduplica versões concorrentes;
- tenta associar documentos do corpus às respectivas chaves BibTeX.

Funções técnicas importantes incluem:

```text
metadata_provider_statuses
extract_doi_from_text
bibtex_escape
render_bib_entry
split_bib_entries
bib_entry_key
extract_field
entry_identity
entry_quality
deduplicate_entries
build_bibliography
```

### 13.2. Manifesto DOI

Para gerar um CSV de apoio:

```bash
academic-pipeline   --make-doi-manifest   --input-dir ./fontes   --output ./doi_manifest.csv
```

ou:

```bash
academic-pipeline   --make-doi-manifest   --input-zip ./fontes.zip   --output ./doi_manifest.csv
```

A rota valida sua própria superfície de opções para evitar combinações ambíguas.

### 13.3. Inspeção de BibTeX

```bash
academic-pipeline --inspect-bib referencias.bib
```

gera relatórios de inspeção em Markdown/JSON.

---

## 14. Modelo documental canônico

O documento interno é modelado por Pydantic em `document_model.py`.

A autoridade intermediária principal é:

```text
<prefixo>.document.json
```

Ela representa a estrutura semântica usada pelos renderizadores.

O modelo contempla, entre outros elementos:

- metadados;
- seções;
- blocos;
- texto e conteúdo inline;
- citações;
- listas;
- tabelas;
- figuras;
- bibliografia;
- diagnósticos.

Essa separação permite gerar o conteúdo uma vez e depois renderizá-lo em diferentes formatos.

---

## 15. Geração estruturada por IA

`document_builder.py` transforma corpus, orientações, configuração, bibliografia e prompts em `AcademicDocument`.

O fluxo:

- seleciona diretivas/prompt bundle;
- monta contexto controlado;
- chama o modelo configurado;
- exige saída estruturada;
- pode trabalhar em etapas/checkpoints em perfis que habilitam geração seccional;
- valida a estrutura antes de seguir;
- não considera o texto bruto da IA como produto final: o produto passa por modelo, saneamento e validação.

O modo completo requer credencial OpenAI disponível no ambiente quando houver etapa de IA.

---

## 16. Validação do documento

`document_validator.py` valida:

- coerência estrutural do `AcademicDocument`;
- citações contra chaves bibliográficas disponíveis;
- conteúdo ORG;
- vazamentos ou estruturas técnicas indevidas;
- contratos necessários para renderização.

O runtime chama validação antes da publicação dos artefatos finais e interrompe o fluxo em erros relevantes.

---

## 17. Modo `--somente-renderizar`

Quando o conteúdo já foi gerado:

```bash
academic-pipeline   --config projeto.toml   --somente-renderizar   --document-json output/meu_documento.document.json
```

O programa:

1. carrega o `document.json`;
2. resolve a bibliografia associada;
3. saneia eventuais marcas técnicas/bibkeys crus;
4. reutiliza resumos existentes quando disponíveis;
5. opcionalmente reutiliza ou regenera mapa mental;
6. renderiza novamente as saídas sem regenerar o texto acadêmico principal.

É o modo recomendado para ajustes de layout/compilação quando o conteúdo semântico já está aprovado.

---

## 18. Recompilação ORG sem IA

Use:

```bash
academic-pipeline   --recompile   --config projeto.toml   --org output/meu_documento.org
```

Overrides disponíveis:

```text
--academic-writing
--latex-extra-path
--pdf-engine
--no-clean
```

Esse modo recompila um ORG já existente, sem solicitar nova geração textual à IA.

---

## 19. ORG, LaTeX e PDF

`render_org_latex.py` é um dos principais renderizadores.

Capacidades identificadas:

- títulos/seções normalizados;
- renderização de blocos textuais;
- citações em LaTeX;
- listas;
- quotes;
- figuras;
- quebras de página;
- tabelas ORG pequenas;
- tabelas largas convertidas para LaTeX responsivo `longtblr`;
- landscape para tabelas largas;
- configuração de bibliografia/biblatex;
- estilos ABNT, APA, author-year, IEEE/numeric conforme configuração;
- elementos e layouts institucionais;
- suporte a resumos multilíngues.

Funções relevantes incluem:

```text
clean_heading_title
org_heading
render_table_block_org
normalize_biblatex_style
enforce_abnt_biblatex_options
bibliography_style_from_cfg
biblatex_options_for_style
strip_org_cite_export_lines
render_bibliography_preamble
render_block_org
render_org_latex
```

A compilação final é coordenada por `latex_compile.py`.

---

## 20. DOCX

`render_docx.py` converte o `AcademicDocument` para DOCX.

O fluxo pode usar templates e recursos institucionais distribuídos no pacote. A validação posterior do DOCX é registrada nos relatórios do pipeline.

Quando o perfil institucional exige um modelo específico, a configuração do perfil/layout é incorporada antes da renderização.

---

## 21. Perfis institucionais e FGV

O sistema possui três camadas:

```text
institution_profiles.py
institution_layouts.py
institution_compliance.py
```

### 21.1. Perfis

O perfil institucional pode fornecer:

- defaults;
- paths de assets;
- templates;
- estilos;
- prompts;
- validators;
- layouts disponíveis.

Comandos:

```bash
academic-pipeline --list-institutions
academic-pipeline --explain-profile fgv
```

### 21.2. Layouts

```bash
academic-pipeline --list-layouts --config projeto.toml
```

O layout é resolvido a partir da configuração/perfil, podendo ser sobrescrito por:

```bash
--layout <nome>
```

### 21.3. Conformidade

Exemplo:

```bash
academic-pipeline   --check-institution-compliance   --config projeto.toml   --org output/documento.org   --bib output/documento.bib   --docx output/documento.docx   --pdf output/documento.pdf
```

A validação gera estrutura de relatório e pode produzir versões Markdown/JSON.

---

## 22. Prompts e diretivas

`prompt_manager.py` resolve prompts a partir de:

- configuração do projeto;
- diretório do TOML;
- perfil institucional;
- recursos `profile://`;
- categorias globais;
- documento;
- paper/atividade/dissertação;
- pesquisa/PRISMA.

O gerenciador:

- resolve paths;
- carrega texto;
- remove diretivas inadequadas de exposição de raciocínio interno em prompt geral;
- compõe bundles;
- calcula SHA-256 do material carregado;
- permite relatório dos prompts ativos.

### 22.1. Ver prompts ativos

```bash
academic-pipeline --show-prompts --config projeto.toml
```

### 22.2. Prompt lock

```bash
academic-pipeline --write-prompt-lock --config projeto.toml
```

produz:

```text
<prefixo>.prompt_lock.json
<prefixo>.prompt_lock.md
```

O prompt lock registra rastreabilidade, versão do pipeline, configuração, perfil e prompts utilizados.

---

## 23. Relatório de qualidade

Use:

```bash
academic-pipeline   --quality-report   --document-json output/documento.document.json   --org output/documento.org   --bib output/documento.bib
```

O módulo `quality_report.py`:

- conta palavras;
- mede distribuição por seção;
- coleta citações;
- examina ORG;
- procura termos técnicos indevidos;
- produz alertas;
- registra status de qualidade.

A geração completa também pode produzir o relatório automaticamente.

---

## 24. Mapas mentais

A configuração `[mapa_mental]` controla essa capacidade.

O módulo permite:

- verificar se a geração está ativa;
- construir prompt específico;
- sanear PlantUML;
- determinar paths de `.puml` e imagem;
- renderizar mapa existente;
- gerar novo mapa por IA;
- recolorir/normalizar;
- anexar figura ao `document_model`;
- reutilizar mapa;
- remover artefatos conhecidos antes de regeneração forçada.

### 24.1. Somente mapa mental

```bash
academic-pipeline   --config projeto.toml   --somente-mapa-mental   --document-json output/documento.document.json
```

### 24.2. Reutilizar

```bash
academic-pipeline   --config projeto.toml   --somente-renderizar   --document-json output/documento.document.json   --reusar-mapa-mental
```

### 24.3. Forçar regeneração

```bash
academic-pipeline   --config projeto.toml   --somente-renderizar   --document-json output/documento.document.json   --forcar-regeneracao-mapa-mental
```

A regeneração forçada pode exigir novamente o cliente OpenAI e o renderizador PlantUML.

---

## 25. Traduções

`document_translation.py` traduz apenas conteúdo substantivo.

O módulo:

- identifica strings translatáveis;
- protege paths/campos técnicos;
- evita traduzir URLs, IDs, chaves e metadados estruturais;
- divide conteúdo em lotes com limite configurável;
- chama o modelo de tradução;
- reinsere o texto traduzido na mesma estrutura;
- permite renderizações adicionais por idioma.

Configuração relevante:

```text
[idiomas_saida]
```

O limite de lote é normalizado pelo runtime para evitar requisições excessivamente pequenas ou grandes.

---

## 26. Resumos e palavras-chave multilíngues

`paper_abstracts.py` é responsável por:

- determinar idiomas de resumo;
- diferenciar idiomas do paper principal de cópias traduzidas;
- extrair conteúdo substantivo do documento;
- proteger bibliografia e metadados técnicos;
- gerar resumo e palavras-chave;
- salvar sidecar:

```text
<prefixo>.resumos_paper.json
```

O renderizador ORG/PDF e o DOCX podem incorporar os resumos conforme a configuração.

---

## 27. PRISMA — visão geral

O subsistema PRISMA cobre várias etapas distintas:

```text
busca
→ deduplicação
→ triagem/curadoria
→ revisão humana
→ consolidação
→ relatório
→ bibliografia selecionada
→ full text
→ congelamento de insumos
→ artigo final
```

Módulos centrais:

```text
prisma_busca_externa.py
prisma_curadoria_ia_referencias.py
prisma_pipeline.py
prisma_builder.py
prisma_model.py
prisma_validator.py
render_prisma_org.py
render_prisma_docx.py
render_prisma_xlsx.py
render_prisma_flow.py
prisma_exportar_bib.py
prisma_baixar_fulltext_artigos.py
prisma_fulltext_garantido.py
prisma_congelar_artigo.py
```

---

## 28. PRISMA — busca externa

`prisma_busca_externa.py` oferece infraestrutura para:

- catálogos/provedores de busca;
- estratégias configuráveis;
- normalização de registros;
- retries;
- rate limiting;
- deduplicação;
- geração de material para triagem;
- integração com pesquisa configurada no TOML.

A busca externa é executada apenas quando habilitada/configurada.

---

## 29. PRISMA — triagem humana

A opção:

```bash
academic-pipeline   --config projeto_prisma.toml   --prisma-importar-triagem caminho/triagem_humana.csv
```

importa a decisão humana e consolida os artefatos PRISMA.

Esse fluxo é especialmente associado ao perfil de busca orientada, em que a revisão humana é uma etapa explícita.

---

## 30. PRISMA — curadoria assistida por IA ou heurística

### 30.1. Curadoria IA

```bash
academic-pipeline   --config projeto.toml   --prisma-curadoria-ia
```

### 30.2. Sem IA

```bash
academic-pipeline   --config projeto.toml   --prisma-curadoria-sem-ia
```

### 30.3. Parâmetros

```text
--prisma-curadoria-input
--prisma-curadoria-out-dir
--prisma-curadoria-prompt
--prisma-curadoria-max-incluir
--prisma-curadoria-top-n-candidatos
--prisma-curadoria-limiar-minimo
```

### 30.4. Revisão XLSX

Após revisão humana:

```bash
academic-pipeline   --config projeto.toml   --prisma-curadoria-reexportar-xlsx   --prisma-curadoria-input caminho/revisado.xlsx
```

### 30.5. Importação final

```bash
academic-pipeline   --config projeto.toml   --prisma-curadoria-importar
```

### 30.6. Fluxo completo

```bash
academic-pipeline   --config projeto.toml   --prisma-curadoria-fluxo-completo
```

Esse comando encadeia curadoria e importação final conforme a lógica do runtime.

---

## 31. PRISMA — artigo final

O runtime reconhece quatro wrappers de alto nível:

```text
--prisma-exportar-bib
--prisma-congelar-artigo
--prisma-gerar-toml-artigo
--prisma-gerar-artigo-final
```

Eles formam a ponte entre um relatório PRISMA aprovado e o produto acadêmico final.

### 31.1. Exportar bibliografia incluída

Produz BibTeX das referências incluídas.

### 31.2. Congelar artigo

`prisma_congelar_artigo.py` reúne os insumos aprovados, registra SHA-256/metadados e prepara uma área estável para geração do artigo.

### 31.3. Gerar TOML do artigo

O gerador do artigo extrai tema, recorte, objetivo, pergunta, hipótese, tese e estrutura a partir dos dados da pesquisa/configuração e cria o TOML final baseado em template.

Ele exige os artefatos PRISMA necessários, incluindo a bibliografia e arquivos de pesquisa definidos pelo fluxo.

### 31.4. Gerar artigo final

Executa a geração final sobre os insumos congelados, reduzindo o risco de a pesquisa mudar entre a aprovação da triagem e a redação final.

---

## 32. Fluxo de estado do Artigo PRISMA

Existe uma CLI auxiliar:

```text
artigo_prisma_workflow.py
```

Ações:

```text
status
validate
mark-reviewed
```

Exemplos:

```bash
python -m app_bundle.scripts.pipeline.artigo_prisma_workflow   status   --art-dir caminho/artigo
```

```bash
python -m app_bundle.scripts.pipeline.artigo_prisma_workflow   validate   --art-dir caminho/artigo
```

A confirmação de revisão humana exige `--xlsx`.

---

## 33. Full text PRISMA

Os módulos:

```text
prisma_baixar_fulltext_artigos.py
prisma_fulltext_garantido.py
```

tratam obtenção/organização e garantia de full text no fluxo robusto. A finalidade é assegurar que a redação final opere sobre as evidências efetivamente selecionadas, com rastreabilidade dos arquivos disponíveis e ausentes.

---

## 34. Doctor

```bash
academic-pipeline --doctor
```

Executa diagnóstico do ambiente e pode verificar, conforme o contexto:

- configuração;
- paths;
- ferramentas;
- recursos institucionais;
- bibliografia;
- capacidades de pesquisa;
- diretórios de saída;
- condições necessárias às etapas posteriores.

`--doctor` é uma rota própria; o runtime rejeita combinações inadequadas com outros comandos de alto nível.

---

## 35. Check-config

```bash
academic-pipeline --check-config --config projeto.toml
```

É a validação preventiva do TOML. O fluxo carrega o perfil institucional, aplica overrides válidos e verifica erros/warnings antes da geração custosa.

A geração principal também executa uma validação preventiva e bloqueia quando existem erros de configuração.

---

## 36. Listagens e descoberta

### 36.1. Presets do wizard

```bash
academic-pipeline --list-toml-profiles
```

### 36.2. Perfis TOML do runtime

```bash
academic-pipeline --list-profiles
```

### 36.3. Instituições

```bash
academic-pipeline --list-institutions
```

### 36.4. Layouts

```bash
academic-pipeline --list-layouts --config projeto.toml
```

### 36.5. Explicação do perfil

```bash
academic-pipeline --explain-profile fgv
```

---

## 37. Artefatos produzidos

Dependendo do projeto/configuração, podem ser produzidos:

```text
<prefixo>.document.json
<prefixo>.org
<prefixo>.pdf
<prefixo>.docx
<prefixo>.bib
<prefixo>.prompt_lock.json
<prefixo>.prompt_lock.md
<prefixo>.quality_report.json
<prefixo>.quality_report.md
<prefixo>.compliance_report.json
<prefixo>.compliance_report.md
<prefixo>.run_report.json
<prefixo>.outputs.txt
<prefixo>.resumos_paper.json
```

Além destes, fluxos PRISMA podem gerar:

- matrizes CSV/XLSX;
- triagem humana;
- relatório PRISMA ORG/PDF/DOCX;
- diagrama de fluxo;
- bibliografia de referências incluídas;
- arquivos de full text;
- dados congelados do artigo;
- TOML específico do artigo final.

Nem todo fluxo gera todos os artefatos.

---

## 38. Rastreabilidade

O programa registra múltiplas camadas de rastreabilidade:

### 38.1. `document.json`

Autoridade intermediária estruturada do conteúdo.

### 38.2. `prompt_lock`

Registra quais prompts/diretivas foram usados.

### 38.3. `run_report`

Registra execução, configuração, modelo, outputs e warnings.

### 38.4. `outputs.txt`

Manifesto legível dos caminhos produzidos.

### 38.5. Relatórios de qualidade e conformidade

Separam avaliação textual de avaliação institucional.

### 38.6. SHA-256 em fluxos de congelamento

O artigo PRISMA utiliza hashes para estabilizar e identificar os insumos aprovados.

---

## 39. Tratamento de bibliografia no renderizador

O renderizador normaliza estilos BibLaTeX e possui suporte explícito a aliases de:

```text
ABNT
APA
authoryear
IEEE/numeric
Vancouver → numeric
```

Para ABNT, o renderizador impõe opções coerentes de `biblatex`, incluindo backend Biber e ordenação apropriada, salvo override explícito.

O sistema também remove diretivas `#+CITE_EXPORT` herdadas quando conflitam com a estratégia atual de citações LaTeX.

---

## 40. Tabelas e elementos gráficos

Tabelas pequenas permanecem preferencialmente em sintaxe ORG.

Tabelas largas ou com células extensas podem ser convertidas para LaTeX `longtblr` e, em casos mais largos, `landscape`, para reduzir cortes laterais no PDF.

Figuras, inclusive mapa mental, são registradas no `document_model` e renderizadas como parte do documento.

---

## 41. Caminhos e raiz do projeto

`academic_pipeline.repository_paths` fornece resolução segura da raiz.

Variável reconhecida:

```text
ACADEMIC_PIPELINE_PROJECT_ROOT
```

Quando definida, deve apontar para uma raiz válida contendo:

```text
pyproject.toml
academic_pipeline/
app_bundle/
```

`repository_resource()` impede que um recurso resolvido escape da raiz do projeto.

---

## 42. Módulos canônicos do core

| Módulo | Responsabilidade |
| --- | --- |
| academic_pipeline/__init__.py | API pública do pacote. |
| academic_pipeline/__main__.py | Entrada para python -m academic_pipeline. |
| academic_pipeline/cli.py | main() público; delega ao runtime canônico. |
| academic_pipeline/cli_parser.py | Superfície argparse pública. |
| academic_pipeline/runtime.py | Roteamento canônico e precedência das rotas. |
| academic_pipeline/default_runtime.py | Fluxo padrão completo de geração/renderização. |
| academic_pipeline/command_dispatch.py | Dispatchers especializados das opções de linha de comando. |
| academic_pipeline/doctor_runtime.py | Rota nativa --doctor. |
| academic_pipeline/check_config_runtime.py | Rota nativa --check-config. |
| academic_pipeline/list_profiles_runtime.py | Rota nativa --list-profiles. |
| academic_pipeline/institution_compliance_runtime.py | Rota nativa de conformidade institucional. |
| academic_pipeline/doi_manifest_runtime.py | Rota nativa de manifesto DOI. |
| academic_pipeline/document_orchestration.py | Estágios canônicos do documento. |
| academic_pipeline/prisma_generic_orchestration.py | Estágios/wrappers do PRISMA e artigo genérico. |
| academic_pipeline/repository_paths.py | Descoberta segura da raiz do projeto e recursos. |

---

## 43. Catálogo técnico de funções do core

Esta seção lista as funções e métodos identificados nos módulos canônicos que implementam entrypoint, roteamento e orquestração. Nomes iniciados por `_` são internos e não devem ser tratados como API pública estável.

### `academic_pipeline.runtime`

- `RuntimeContext.as_dispatch_mapping`
- `_load_config`
- `default_runtime_context`
- `_normalize_argv`
- `_matches_option`
- `_namespace_has_preceding_trigger`
- `_namespace_has_check_config_preceding_trigger`
- `_is_exact_list_profiles_invocation`
- `_is_exact_institution_compliance_invocation`
- `_is_exact_doi_manifest_invocation`
- `_has_prisma_generic_wrapper_trigger`
- `select_runtime_route`
- `_build_parser`
- `_normalize_dispatch_code`
- `_run_native_first_wave`
- `_run_native_doctor`
- `_run_native_check_config`
- `_run_check_config_combination_error`
- `_run_list_profiles_combination_error`
- `_run_native_list_profiles`
- `_run_native_institution_compliance`
- `_run_native_doi_manifest`
- `_run_native_default`
- `_run_doctor_combination_error`
- `_run_institution_compliance_error`
- `run`

### `academic_pipeline.cli_parser`

- `build_parser`
- `parse_args`

### `academic_pipeline.doctor_runtime`

- `_apply_cli_path_overrides`
- `_output_paths`
- `_research_output_paths`
- `default_doctor_runtime_context`
- `_normalize_argv`
- `_build_parser`
- `run_doctor_command`

### `academic_pipeline.check_config_runtime`

- `default_check_config_runtime_context`
- `_normalize_argv`
- `_build_parser`
- `_prepare_config`
- `run_check_config_command`

### `academic_pipeline.list_profiles_runtime`

- `print_profiles`
- `run_list_profiles_command`

### `academic_pipeline.institution_compliance_runtime`

- `default_institution_compliance_runtime_context`
- `_normalize_argv`
- `_build_parser`
- `_prepare_config`
- `_dispatch_runtime`
- `run_institution_compliance_command`

### `academic_pipeline.doi_manifest_runtime`

- `_normalize_argv`
- `_build_parser`
- `_validate_option_surface`
- `run_make_doi_manifest_command`

### `academic_pipeline.repository_paths`

- `_is_project_root`
- `_candidate_chain`
- `repository_project_root`
- `repository_resource`

### `academic_pipeline.document_orchestration`

- `apply_cli_path_overrides_impl`
- `load_existing_document_json_impl`
- `run_document_stage_001`
- `run_document_stage_002`
- `run_document_stage_003`
- `run_document_stage_004`
- `run_document_stage_005`
- `run_document_stage_006`
- `run_document_stage_007`
- `run_document_stage_008`
- `run_document_stage_009`
- `run_document_stage_010`
- `run_document_stage_011`
- `run_document_stage_012`

### `academic_pipeline.default_runtime`

- `stage`
- `_json_or_none`
- `load_config`
- `make_client`
- `_section`
- `output_paths`
- `research_output_paths`
- `work_cache_paths`
- `apply_cli_path_overrides`
- `load_existing_document_json`
- `resolve_bib_for_existing_document`
- `_openai_model_from_cfg`
- `_load_optional_config`
- `_resolve_latex_paths_for_recompile`
- `run_recompile`
- `render_external_prisma_outputs`
- `render_additional_language_versions`
- `_prisma_curadoria_default_config`
- `_prisma_curadoria_default_out_dir`
- `_prisma_curadoria_default_prompt`
- `_prisma_curadoria_script_path`
- `_prisma_curadoria_arg`
- `_prisma_curadoria_config_from_args`
- `_prisma_curadoria_out_from_args`
- `_prisma_curadoria_prompt_from_args`
- `_prisma_curadoria_input_from_args`
- `_prisma_curadoria_run_command`
- `_prisma_curadoria_build_cmd`
- `_prisma_curadoria_run_ia`
- `_prisma_curadoria_reexportar_xlsx`
- `_prisma_curadoria_pipeline_supports_flag`
- `_prisma_curadoria_importar_no_pipeline`
- `_prisma_curadoria_fluxo_completo`
- `_prisma_curadoria_mostrar_caminhos`
- `_prisma_curadoria_menu`
- `_prisma_curadoria_dispatch`
- `_ap003f_pipeline_core`
- `_prisma_artigo_generico_strip`
- `_prisma_artigo_generico_out_dir`
- `run_default`

### 43.1. Observação sobre helpers internos

O software possui também helpers locais, funções aninhadas, métodos de classes de UI e rotinas específicas nos módulos de domínio. O manual os agrupa pelo módulo na seção seguinte para manter a documentação de uso legível. A autoridade de API pública continua sendo o entrypoint e as opções documentadas na seção 8.

---

## 44. Catálogo técnico dos módulos de domínio

| Módulo | Responsabilidade operacional |
| --- | --- |
| academic_pipeline_gui.py | GUI FGV para criação/execução de atividades acadêmicas e seleção de projetos. |
| academic_pipeline_toml_generator.py | Gerador TOML não interativo/canônico. |
| academic_pipeline_toml_generator_interativo.py | Wizard extenso de TOML com presets e modo TUI FGV. |
| academic_pipeline_tui.py | Central Operacional FGV em terminal; orquestra configurar → conferir → validar → gerar → revisar. |
| academic_pipeline_tui_widgets.py | Widgets prompt_toolkit utilizados pela TUI. |
| artigo_prisma_workflow.py | CLI de status, validação e confirmação de revisão humana do fluxo Artigo PRISMA. |
| bibliography_manager.py | Metadados bibliográficos, DOI, BibTeX, deduplicação, correspondência documento↔referência. |
| citation_renderer.py | Renderização de citações em saídas textuais/LaTeX. |
| clean_bundle.py | Limpeza/organização de artefatos do bundle. |
| corpus_manager.py | Descoberta, extração, leitura e cache de documentos locais. |
| diagnostics.py | Doctor, check-config, relatórios de execução, manifests e validações auxiliares. |
| document_builder.py | Construção do AcademicDocument/document.json, incluindo geração estruturada por IA. |
| document_model.py | Modelos Pydantic canônicos do documento, blocos, citações, figuras e bibliografia. |
| document_translation.py | Tradução seletiva de conteúdo substantivo preservando campos técnicos. |
| document_validator.py | Validação estrutural/semântica do document_model e do ORG. |
| gerar_artigo_final_unificado.py | Orquestra geração final de artigo no fluxo PRISMA. |
| gerar_artigo_longo_fulltext_secional.py | Geração de artigo longo baseada em full text por seções. |
| institution_compliance.py | Verificação de conformidade institucional e geração de relatórios. |
| institution_explainer.py | Explicação textual dos perfis institucionais. |
| institution_layouts.py | Descoberta e resolução de layouts por perfil. |
| institution_profiles.py | Carregamento, descrição e aplicação de perfis institucionais. |
| latex_compile.py | Sequência de compilação LaTeX/PDF. |
| mindmap_manager.py | Geração, renderização, reutilização, anexação e regeneração de mapas mentais. |
| paper_abstracts.py | Resumo e palavras-chave de papers em um ou mais idiomas. |
| pipeline_orchestrator.py | Orquestrador executável tradicional do bundle. |
| preparar_artigo_longo_fulltext.py | Preparação de insumos para artigo longo/full text. |
| prisma_baixar_fulltext_artigos.py | Aquisição/organização de full text para artigos selecionados. |
| prisma_builder.py | Construção de estruturas/artefatos PRISMA. |
| prisma_busca_externa.py | Busca externa, provedores, deduplicação, retry/rate limit e triagem. |
| prisma_congelar_artigo.py | Congelamento de insumos PRISMA e geração de TOML para artigo final. |
| prisma_curadoria_ia_referencias.py | Curadoria IA ou heurística de referências, com XLSX/CSV e revisão humana. |
| prisma_diagrama_fluxo.py | Geração de diagrama/fluxo PRISMA. |
| prisma_exportar_bib.py | Exportação BibTeX das referências incluídas. |
| prisma_fulltext_garantido.py | Fluxo robusto para assegurar full text e rastreabilidade. |
| prisma_model.py | Modelos de dados PRISMA. |
| prisma_pipeline.py | Orquestração dos outputs de relatório PRISMA. |
| prisma_validator.py | Validação de estruturas e saídas PRISMA. |
| project_tools.py | Inicialização de projetos, manifesto DOI e inspeção de bibliografia. |
| prompt_lock.py | Registro de rastreabilidade de prompts em JSON e Markdown. |
| prompt_manager.py | Resolução, carregamento, saneamento e composição do banco de prompts. |
| quality_report.py | Métricas e alertas de qualidade textual/documental. |
| render_docx.py | Renderização DOCX a partir do modelo canônico. |
| render_docx_canonico.py | Apoio à aplicação/normalização canônica de DOCX. |
| render_org_latex.py | Renderização ORG/LaTeX, tabelas, citações, bibliografia e elementos institucionais. |
| render_prisma_docx.py | Renderização DOCX específica do relatório PRISMA. |
| render_prisma_flow.py | Renderização do fluxo/diagrama PRISMA. |
| render_prisma_org.py | Renderização ORG específica do relatório PRISMA. |
| render_prisma_xlsx.py | Renderização XLSX das matrizes/triagens PRISMA. |
| utils.py | Utilidades de caminhos, texto, escrita e normalização. |
| validar_artigo_longo_fulltext.py | Valida artigo longo/full text. |

### 44.1. Artefatos históricos na árvore

A árvore do bundle ainda contém alguns arquivos com `.orig` ou nomes antigos/versionados. Eles **não constituem a interface operacional canônica** e não devem ser usados para escolher entrypoints ou APIs atuais.

Exemplos encontrados na árvore de fontes:

```text
academic_pipeline_toml_generator_interativo.py.orig
academic_pipeline_tui_widgets.py.orig
academic_pipeline_toml_generator_v0_3_1.py
```

A presença desses arquivos é histórica; a autoridade atual são os módulos canônicos sem marcador de versão.

---

## 45. Funções técnicas representativas dos módulos de domínio

### 45.1. Bibliografia

```text
metadata_provider_statuses
extract_doi_from_text
render_bib_entry
split_bib_entries
bib_entry_key
extract_field
entry_identity
entry_quality
deduplicate_entries
build_bibliography
```

### 45.2. Corpus

```text
read_text_file
discover_local_documents
copy_documents_to_fulltext_cache
collect_orientation_docs
```

### 45.3. Mapa mental

```text
mindmap_config
should_generate_mindmap
sanitize_plantuml
build_mindmap_prompt
mindmap_artifact_paths
attach_mindmap_figure
render_existing_mindmap
render_or_generate_mindmap
delete_existing_mindmap_outputs
attach_existing_mindmap_if_available
generate_and_attach_mindmap
```

### 45.4. Tradução

```text
requested_translation_languages
translation_batch_size
collect_translatable_strings
translate_document_model
```

### 45.5. Resumos de paper

```text
abstract_sidecar_path
requested_abstract_languages
main_document_abstract_languages
paper_abstracts_enabled
generate_paper_abstract_bundle
write_paper_abstract_bundle
read_paper_abstract_bundle
inject_paper_abstracts_into_org
inject_paper_abstracts_into_docx
```

### 45.6. Prompts

```text
resolve_prompt_path
sanitize_general_execution_prompt
load_prompt_bundle
prompt_report_for_cfg
build_prompt_lock
write_prompt_lock
render_prompt_lock_markdown
write_prompt_lock_markdown
```

### 45.7. Qualidade

```text
build_quality_report
write_quality_report
```

### 45.8. Institucional

```text
apply_institution_profile
describe_institution_profiles
available_layouts
resolve_layout_spec
explain_profile
run_institution_compliance
render_compliance_markdown
write_compliance_reports
```

### 45.9. Renderização ORG/LaTeX

```text
clean_heading_title
org_heading
render_table_block_org
normalize_biblatex_style
enforce_abnt_biblatex_options
bibliography_style_from_cfg
biblatex_options_for_style
strip_org_cite_export_lines
render_bibliography_preamble
render_block_org
render_org_latex
```

### 45.10. Projetos e DOI

```text
init_project
make_doi_manifest
inspect_bib
render_bib_inspection_markdown
```

### 45.11. PRISMA

O subsistema possui muitas funções internas de busca, normalização, deduplicação, curadoria, exportação e renderização. A superfície de uso suportada é a CLI descrita nas seções 27–33, e os módulos responsáveis estão inventariados na seção 44.

---

## 46. Usabilidade recomendada por objetivo

### Quero apenas validar meu projeto

```bash
academic-pipeline --doctor
academic-pipeline --check-config --config projeto.toml
```

### Quero criar um projeto pela primeira vez

```bash
academic-pipeline --tui
```

ou:

```bash
academic-pipeline --init-toml
```

### Quero executar um TOML pronto

```bash
academic-pipeline --config projeto.toml
```

### Quero mudar apenas layout/saída, sem nova redação

```bash
academic-pipeline   --config projeto.toml   --somente-renderizar   --document-json output/documento.document.json
```

### Quero apenas recompilar o PDF a partir do ORG

```bash
academic-pipeline   --recompile   --config projeto.toml   --org output/documento.org
```

### Quero conferir a bibliografia

```bash
academic-pipeline --inspect-bib output/documento.bib
```

### Quero validar a aderência institucional

```bash
academic-pipeline   --check-institution-compliance   --config projeto.toml   --org output/documento.org   --pdf output/documento.pdf
```

### Quero somente relatório de qualidade

```bash
academic-pipeline   --quality-report   --document-json output/documento.document.json   --org output/documento.org
```

---

## 47. Erros comuns e diagnóstico

### 47.1. “Informe --config”

O fluxo padrão chegou à etapa que exige configuração e nenhum TOML foi fornecido.

Solução:

```bash
academic-pipeline --config caminho/projeto.toml
```

ou use uma rota autônoma como `--doctor`.

### 47.2. Combinação inválida com `--doctor`

`--doctor` é uma rota especializada. Execute-a separadamente.

### 47.3. Combinação inválida com `--check-config`

Execute a validação de configuração isoladamente com as opções de configuração/path aplicáveis.

### 47.4. `document.json` ausente em `--somente-renderizar`

Esse modo exige documento existente. Verifique `--document-json` ou o output/prefixo calculado pelo TOML.

### 47.5. Mapa mental não ativo

`--forcar-regeneracao-mapa-mental` exige que `[mapa_mental]` esteja ativado.

### 47.6. Bibliografia inconsistente

Use:

```bash
academic-pipeline --inspect-bib referencias.bib
```

e revise DOI, duplicatas e metadados.

### 47.7. Falha de PDF

Verifique:

```bash
academic-pipeline --doctor
```

e, quando já houver ORG, use `--recompile` para isolar a camada de compilação.

### 47.8. Falha de IA

Confirme credencial/configuração do provedor e se o modo utilizado realmente precisa de IA. `--recompile` e parte dos fluxos de reuso não exigem nova redação por IA.

### 47.9. Pesquisa PRISMA com estado incompleto

Valide a configuração, a triagem e, no fluxo de artigo, use a CLI de `artigo_prisma_workflow.py` para `status`/`validate`.

---

## 48. O que não deve ser usado como entrypoint

Não use como entrada oficial:

- arquivos históricos `.orig`;
- geradores com marcador de versão;
- scripts aposentados da árvore `academic_pipeline_rc10_7_conformidade`;
- módulos históricos citados em logs antigos.

Use:

```text
academic-pipeline
python -m academic_pipeline
```

e, somente quando documentado neste manual, CLIs auxiliares do bundle para fluxos especializados.

---

## 49. Política de manutenção deste manual

Toda alteração funcional futura deve classificar impacto documental.

### 49.1. Atualização obrigatória

Atualizar este manual quando houver:

- nova opção CLI;
- remoção/renomeação de opção;
- novo perfil ou layout;
- novo tipo de projeto;
- alteração de requisitos/dependências;
- mudança de comportamento do fluxo principal;
- alteração dos artefatos gerados;
- novo módulo operacional;
- mudança em PRISMA, DOI, bibliografia, renderização, TUI ou GUI;
- alteração de caminhos/configuração;
- mudança de requisitos de IA ou ferramentas externas.

### 49.2. Fonte canônica

```text
MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE.md
```

A história deve permanecer no Git.

### 49.3. Sincronização no mesmo ciclo

Uma mudança funcional não deve ser considerada documentalmente completa enquanto o manual não refletir o comportamento já validado.

Gate recomendado:

```yaml
mppg_manual_synchronization_gate:
  functional_change_identified: true
  canonical_manual_identified: true
  affected_sections_identified: true
  manual_updated: true
  obsolete_instructions_remaining: 0
  undocumented_new_public_functions: 0
  cli_catalog_matches_runtime: true
  examples_match_current_interface: true
  baseline_commit_recorded: true
```

Decisão:

```text
MPPG_MANUAL_SYNCHRONIZATION_GATE_APPROVED
```

---

## 50. Política de nomenclatura documental

Não criar versões paralelas vigentes como:

```text
MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE_v2.md
MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE_final.md
MPPG_MANUAL_TECNICO_ACADEMIC_PIPELINE_20260807.md
```

Esses nomes podem existir apenas como evidência histórica arquivada, nunca como nova autoridade vigente.

---

## 51. Estado global atual

```yaml
academic_pipeline:
  refactor_program: closed
  refactor_progress_percent: 100
  productive_legacy_runtime: retired
  canonical_branch: master
  documented_baseline_commit: 9e24ad6db001d56d7334bf3ab97c97a05cce579a

post_promotion_publication:
  state: closed
  pending_steps: 0

active_mandatory_front:
  none
```

O programa foi canonicalizado e publicado. Este manual não reabre AP-009 nem qualquer fase encerrada.

---

## 52. Referências técnicas usadas para esta edição

Baseline:

```text
https://github.com/gustavo-detarso/mppg
commit 9e24ad6db001d56d7334bf3ab97c97a05cce579a
```

Fontes canônicas principais consultadas:

```text
software/academic_pipeline_mppg/pyproject.toml
software/academic_pipeline_mppg/README.md

software/academic_pipeline_mppg/academic_pipeline/
software/academic_pipeline_mppg/app_bundle/scripts/pipeline/
```

Módulos especialmente relevantes:

```text
academic_pipeline/runtime.py
academic_pipeline/cli_parser.py
academic_pipeline/default_runtime.py
academic_pipeline/document_orchestration.py
academic_pipeline/prisma_generic_orchestration.py
academic_pipeline/doctor_runtime.py
academic_pipeline/check_config_runtime.py
academic_pipeline/institution_compliance_runtime.py
academic_pipeline/doi_manifest_runtime.py

app_bundle/scripts/pipeline/academic_pipeline_tui.py
app_bundle/scripts/pipeline/academic_pipeline_gui.py
app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py
app_bundle/scripts/pipeline/project_tools.py
app_bundle/scripts/pipeline/corpus_manager.py
app_bundle/scripts/pipeline/bibliography_manager.py
app_bundle/scripts/pipeline/document_builder.py
app_bundle/scripts/pipeline/document_model.py
app_bundle/scripts/pipeline/document_validator.py
app_bundle/scripts/pipeline/render_org_latex.py
app_bundle/scripts/pipeline/render_docx.py
app_bundle/scripts/pipeline/mindmap_manager.py
app_bundle/scripts/pipeline/document_translation.py
app_bundle/scripts/pipeline/paper_abstracts.py
app_bundle/scripts/pipeline/prompt_manager.py
app_bundle/scripts/pipeline/prompt_lock.py
app_bundle/scripts/pipeline/quality_report.py
app_bundle/scripts/pipeline/institution_profiles.py
app_bundle/scripts/pipeline/institution_layouts.py
app_bundle/scripts/pipeline/institution_compliance.py
app_bundle/scripts/pipeline/prisma_pipeline.py
app_bundle/scripts/pipeline/prisma_busca_externa.py
app_bundle/scripts/pipeline/prisma_curadoria_ia_referencias.py
app_bundle/scripts/pipeline/prisma_exportar_bib.py
app_bundle/scripts/pipeline/prisma_congelar_artigo.py
app_bundle/scripts/pipeline/prisma_fulltext_garantido.py
```

Modelo estrutural fornecido pelo usuário:

```text
UAC_DIRETRIZ_PERMANENTE_NOMENCLATURA_CANONICA_IA.md
```

---

## 53. Regra resumida de uso

> Para um usuário comum, o ponto de entrada é `academic-pipeline`. Use `--tui` ou `--init-toml` para criar/configurar projetos; `--check-config` e `--doctor` para validar; `--config projeto.toml` para executar; `--somente-renderizar` ou `--recompile` para reaproveitar conteúdo sem nova redação; os comandos `--prisma-*` para pesquisa estruturada e artigo PRISMA; e os relatórios `prompt_lock`, qualidade e conformidade para rastreabilidade. A autoridade semântica intermediária do documento é `document.json`; a configuração é o TOML; a história e evolução do software pertencem ao Git.
