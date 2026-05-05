# Manual unificado do projeto

## Visão geral

Este manual consolida o funcionamento do ecossistema atual do projeto, composto por dois níveis principais:

1. **Pesquisa**  
   Script: `gerador_pesquisa_rc_2.py`

2. **Pipeline integrado pesquisa + paper**  
   Script: `gerador_pesquisa_documento_rc_2.py`

A lógica geral é:

- o **gerador de pesquisa** conduz a etapa metodológica de busca, seleção, triagem, síntese e geração dos artefatos da pesquisa;
- o **pipeline integrado** pode:
  - rodar só a pesquisa;
  - rodar só o paper a partir de uma pesquisa já pronta;
  - rodar pesquisa + bundle + paper;
  - reescrever ou expandir um paper já existente;
  - incorporar artigos extras ao paper.

---

## Nomenclaturas padrão do projeto

A partir do estado atual do projeto, use sempre estas nomenclaturas:

- `template_research.org` → template da pesquisa
- `template_paper.org` → template do paper
- `template_dissertacao.org` → template base da dissertação

Esses nomes substituem o uso antigo de `template.org` como referência principal da pesquisa.

---

## Scripts principais

### 1. `gerador_pesquisa_rc_2.py`

Responsável por:

- ler o TOML da pesquisa;
- interpretar tema, recorte e objetivo;
- gerar palavras-chave e queries;
- executar busca, triagem e seleção;
- produzir artefatos da pesquisa;
- gerar `.org`, `.bib`, `.json`, fluxograma PRISMA e, se configurado, PDF.

### 2. `gerador_pesquisa_documento_rc_2.py`

Responsável por orquestrar:

- só pesquisa;
- só bundle;
- só paper;
- pesquisa + bundle + paper;
- reescrita ou expansão do paper;
- uso de artigos selecionados da pesquisa;
- uso de artigos extras;
- pacote final de entrega.

O `rc_12` já aponta, por padrão, para:

```toml
[pipeline]
script_pesquisa = "./gerador_pesquisa_rc_2.py"
```



### 3. `gerador_documento_academico_rc_1.py`

É o motor **standalone de redação acadêmica**.

Serve para:
- gerar um **paper**;
- gerar uma **dissertação**;
- futuramente acomodar outros tipos documentais sem mudar a lógica central do motor.

Ele utiliza:
- template externo `.org`;
- orientações externas unificadas;
- artigos selecionados da pesquisa;
- artigos extras;
- bundle ou pesquisa pronta;
- reescrita e expansão de documentos existentes.

---

## Script standalone do paper

Além do pipeline integrado, o projeto agora conta com um script standalone alinhado ao padrão novo:

- **Script:** `gerador_documento_academico_rc_1.py`
- **TOML-base:** `template_toml_documento_academico_rc_1.toml`

Ele foi criado para substituir o uso direto do script legado `gerar_paper_org_ai_interativo_v3_6_9.py` quando você quiser um fluxo mais próximo ao programa unificado.

### O que ele traz de novo

- suporte a `--config` em TOML;
- suporte a `bundle_dir` ou `pesquisa_dir_existente`;
- uso de `template_paper.org` com fallback organizacional para `template_research.org`;
- nomenclatura unificada de orientações:
  - `orientacoes_paths`
  - `orientacao_inline`
- uso de artigos selecionados da pesquisa como base principal do paper;
- suporte a artigos extras;
- suporte a `modo_escrita = "novo" | "reescrever" | "expandir"`;
- `dry_run`;
- geração de pacote `entrega_final/`.

### Exemplo de uso

```bash
python gerador_documento_academico_rc_1.py --config /caminho/para/template_toml_documento_academico_rc_1.toml
```

### Quando usar

Use o `gerador_documento_academico_rc_1.py` quando você quiser:
- gerar só o paper a partir de uma pesquisa já pronta;
- reescrever ou expandir um paper sem rodar o pipeline completo;
- usar bundle/pesquisa existente como fonte principal;
- manter a convenção nova de templates e orientações.

---

## Templates usados

### Pesquisa
Use:

```text
template_research.org
```

Esse template deve ser apontado no TOML da pesquisa, normalmente em:

```toml
[saida]
org_modelo = "/caminho/para/template_research.org"
```

### Paper
Use:

```text
template_paper.org
```

No pipeline integrado, ele pode ser apontado em:

```toml
[documento]
template_org = "/caminho/para/template_paper.org"
```

Se não for apontado, o pipeline tenta usar `template_paper.org` no diretório de execução.

### Dissertação
Use:

```text
template_dissertacao.org
```

No motor standalone de documento acadêmico, ele pode ser apontado em:

```toml
[documento]
tipo_documento = "dissertacao"
template_org = "/caminho/para/template_dissertacao.org"
```

---

## Nomenclatura unificada de orientações

O projeto foi ajustado para usar uma convenção única para orientações externas.

As chaves padronizadas são:

- `orientacoes_paths = []`
- `orientacao_inline = ""`

### Como funcionam

#### `orientacoes_paths`
Lista de arquivos de orientação.

Aceita múltiplos arquivos, por exemplo:

```toml
orientacoes_paths = [
  "/caminho/para/orientacao_geral_execucao.txt",
  "/caminho/para/triagem_especifica.txt"
]
```

#### `orientacao_inline`
Texto escrito diretamente no TOML.

Exemplo:

```toml
orientacao_inline = """
Priorize aderência substantiva ao recorte.
Evite ampliar para textos genéricos do campo.
"""
```

---

## Onde as orientações podem ser usadas

### 1. Em `[saida]`
Uso: orientação geral da etapa de pesquisa.

```toml
[saida]
orientacoes_paths = ["/caminho/para/orientacao_geral_execucao.txt"]
orientacao_inline = ""
```

Esse é o melhor lugar para diretrizes gerais de comportamento e execução da pesquisa.

### 2. Em `[triagem]`
Uso: orientação específica da triagem e do ranking.

```toml
[triagem]
orientacoes_paths = [
  "/caminho/para/triagem_prompt.txt",
  "/caminho/para/diretivas_extras.txt"
]
orientacao_inline = ""
```

Aqui entram orientações mais diretamente ligadas a:
- rigor;
- aderência temática;
- ranking;
- decisão entre amplitude e precisão.

### 3. Em `[documento]`
Uso: orientação da redação, reescrita ou expansão do paper.

```toml
[documento]
orientacoes_paths = [
  "/caminho/para/orientacao_geral_execucao.txt",
  "/caminho/para/orientacao_especifica_paper.txt"
]
orientacao_inline = ""
```

Esse é o melhor lugar para:
- perfil de escrita;
- profundidade argumentativa;
- tom analítico;
- diretrizes de redação do paper.

---

## TOML da pesquisa

Arquivo-base recomendado:

```text
template_toml_unificado_rc_2.toml
```

### Blocos principais

- `[atividade]`
- `[pesquisa]`
- `[bibliografia]`
- `[busca]`
- `[triagem]`
- `[queries]`
- `[saida]`
- `[latex]`
- `[openai]`
- `[controle]`

### Exemplo mínimo

```toml
[atividade]
modo = "revisao_sistematica"
disciplina = "Teorias de Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado de Políticas Públicas e Governo"
turma = "T-01"
polo = "Brasília"
aluno = "Seu nome"

[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização do poder e resultados de políticas públicas"
objetivo = "Investigar como a literatura explica a relação entre capacidades estatais, centralização e desempenho de políticas públicas"
trabalho = ""
tipo_estudo = "Revisão de literatura"
periodo = "2019-2026"
idiomas = ["inglês", "português"]
bases = ["semantic_scholar", "scopus", "pubmed", "openalex", "crossref", "europe_pmc", "core"]
palavras_chave = []

[triagem]
rigor = "moderado"
usar_score_hibrido = true
orientacoes_paths = ["./prompts/triagem_prompt.txt", "./prompts/diretivas_extras.txt"]
orientacao_inline = ""
permitir_textos_nao_publicos = false

[saida]
prefixo = "atividade_modelo"
output_dir = "/caminho/para/saida"
criar_subdiretorio = true
org_modelo = "/caminho/para/template_research.org"
orientacoes_paths = ["/caminho/para/orientacao_geral_execucao.txt"]
orientacao_inline = ""
exportar_pdf = true
```

---

## TOML do pipeline integrado

Arquivo-base recomendado:

```text
template_toml_pipeline_pesquisa_documento_rc_2.toml
```

### Blocos principais

- `[pipeline]`
- `[atividade]`
- `[pesquisa]`
- `[bibliografia]`
- `[busca]`
- `[triagem]`
- `[queries]`
- `[saida]`
- `[latex]`
- `[openai]`
- `[controle]`
- `[documento]`

---

## Modos de uso do pipeline

### 1. Só pesquisa

```toml
[pipeline]
executar_pesquisa = true
executar_bundle = false
executar_documento = false
script_pesquisa = "./gerador_pesquisa_rc_2.py"
```

### 2. Pesquisa + bundle + paper

```toml
[pipeline]
executar_pesquisa = true
executar_bundle = true
executar_documento = true
script_pesquisa = "./gerador_pesquisa_rc_2.py"
```

### 3. Só paper a partir de pesquisa pronta

```toml
[pipeline]
executar_pesquisa = false
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = "/caminho/para/saida_da_pesquisa"
script_pesquisa = "./gerador_pesquisa_rc_2.py"
```

---

## Configurações do paper

No bloco `[documento]`, você controla a etapa redacional.

### Exemplo estruturado

```toml
[documento]
prefixo = "atividade_modelo_paper"
criar_subdiretorio = true
template_org = "/caminho/para/template_paper.org"
exportar_pdf = true
orientacoes_paths = [
  "/caminho/para/orientacao_geral_execucao.txt",
  "/caminho/para/orientacao_especifica_paper.txt"
]
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
reescrever_a_partir_do_org_atual = false
paper_org_existente = ""
preservar_estrutura_do_org_anterior = false
```

---

## Modos de escrita do paper

### `modo_escrita = "novo"`
Cria um paper novo a partir da pesquisa.

### `modo_escrita = "reescrever"`
Reescreve um paper existente usando a pesquisa e, opcionalmente, o `.org` anterior.

### `modo_escrita = "expandir"`
Mantém a base anterior e amplia o paper com novos desenvolvimentos.

---

## Reescrita do paper sem nova pesquisa

Se você quiser reescrever o paper sem fazer nova pesquisa:

```toml
[pipeline]
executar_pesquisa = false
executar_bundle = true
executar_documento = true
pesquisa_dir_existente = "/caminho/para/saida_da_pesquisa"

[documento]
modo_escrita = "reescrever"
reescrever_a_partir_do_org_atual = true
paper_org_existente = "/caminho/para/paper_anterior.org"
usar_artigos_selecionados_pesquisa = true
artigos_extras_paths = ["/caminho/para/pasta_com_extras"]
```

---

## Uso de artigos extras

Os artigos extras não devem ser jogados dentro da pasta de cache da pesquisa.

O jeito correto é:

```toml
[documento]
artigos_extras_paths = [
  "/caminho/para/pasta_com_extras",
  "/caminho/para/artigo_extra.pdf"
]
```

Eles entram como **base complementar** do paper.

---

## Regras bibliográficas do pipeline

O pipeline diferencia:

- artigos selecionados formalmente na pesquisa;
- artigos extras adicionados depois.

### Controles relevantes

```toml
[documento]
usar_bib_da_pesquisa = true
incluir_artigos_extras_no_bib = true
priorizar_citacoes_dos_selecionados = true
extras_so_complementam = true
minimo_citacoes_dos_selecionados = 3
```

A recomendação é manter:
- selecionados da pesquisa como base principal;
- extras como complemento.

---

## Dry run

Antes de rodar uma execução pesada, use:

```toml
[controle]
dry_run = true
```

Isso serve para validar:
- caminhos;
- templates;
- existência da pesquisa anterior;
- presença dos arquivos de orientação;
- consistência do TOML.

Depois, para a execução real, mude para:

```toml
[controle]
dry_run = false
```

---

## Artefatos gerados

### Pela pesquisa
Em geral:
- `.org`
- `.bib`
- `.json`
- fluxograma PRISMA (`.svg` / `.pdf`)
- PDF final, se habilitado
- cache de full texts dos selecionados

### Pelo paper
Em geral:
- `.org`
- `.bib`
- `.json`
- auditoria de prompts
- PDF do paper, se habilitado

### Pelo pipeline
Em geral:
- bundle de handoff
- manifest
- proveniência
- mapas de uso das referências
- pacote final de entrega

---

## Pacote final de entrega

O pipeline mais completo pode gerar um diretório `entrega_final/` com:
- `pesquisa.org`
- `paper.org`
- PDFs
- `.bib`
- TOML da execução
- manifest
- proveniência
- mapa de uso das referências
- arquivos extras usados

---

## Comandos recomendados

### Pesquisa apenas

```bash
python gerador_pesquisa_rc_2.py --config /caminho/para/template_toml_unificado_rc_2.toml
```

### Pipeline integrado

```bash
python gerador_pesquisa_documento_rc_2.py --config /caminho/para/template_toml_pipeline_pesquisa_documento_rc_2.toml
```

---

## Recomendações finais

### Melhor prática 1
Para primeiro teste, comece com:
- `executar_pesquisa = false`
- `executar_documento = true`
- `pesquisa_dir_existente` apontando para uma pesquisa já pronta

### Melhor prática 2
Use sempre:
- `template_research.org` para pesquisa
- `template_paper.org` para paper

### Melhor prática 3
Coloque orientações gerais em:
- `[saida]` para pesquisa
- `[documento]` para redação do paper

### Melhor prática 4
Use `[triagem]` só para instruções específicas de seleção e ranking.

### Melhor prática 5
Não misture artigos extras com o cache de full text da pesquisa.

---

## Resumo curto

- **Pesquisa** → `gerador_pesquisa_rc_2.py`
- **Pipeline integrado** → `gerador_pesquisa_documento_rc_2.py`
- **Template da pesquisa** → `template_research.org`
- **Template do paper** → `template_paper.org`
- **Orientações externas** → `orientacoes_paths` + `orientacao_inline`
- **Script de pesquisa padrão do pipeline** → `./gerador_pesquisa_rc_2.py`

---

## Mapa de uso recomendado

Este mapa responde à pergunta prática: **qual arquivo abrir primeiro em cada cenário de trabalho**.

### Cenário 1 — Quero fazer só a pesquisa
Use:
- script: `gerador_pesquisa_rc_2.py`
- TOML: `template_toml_unificado_rc_2.toml`
- template: `template_research.org`

Fluxo recomendado:
1. preencha `template_toml_unificado_rc_2.toml`;
2. aponte `org_modelo` para `template_research.org`;
3. configure tema, recorte, objetivo, bases e triagem;
4. rode `gerador_pesquisa_rc_2.py`.

Quando usar:
- revisão sistemática/PRISMA;
- pesquisa empírica;
- quando você ainda não quer escrever o paper.

### Cenário 2 — Quero fazer pesquisa + paper no mesmo fluxo
Use:
- script: `gerador_pesquisa_documento_rc_2.py`
- TOML: `template_toml_pipeline_pesquisa_documento_rc_2.toml`
- templates:
  - `template_research.org`
  - `template_paper.org`

Fluxo recomendado:
1. preencha `template_toml_pipeline_pesquisa_documento_rc_2.toml`;
2. deixe `executar_pesquisa = true`;
3. deixe `executar_documento = true`;
4. aponte o template da pesquisa para `template_research.org`;
5. aponte o template do paper para `template_paper.org`;
6. rode o pipeline integrado.

Quando usar:
- quando você quer ir da busca até o paper em uma única execução;
- quando quer gerar bundle e pacote final de entrega.

### Cenário 3 — Já tenho a pesquisa pronta e quero gerar só o paper
Use:
- script: `gerador_pesquisa_documento_rc_2.py` **ou** `gerador_documento_academico_rc_1.py`
- TOML:
  - `template_toml_pipeline_pesquisa_documento_rc_2.toml` para o pipeline integrado
  - `template_toml_documento_academico_rc_1.toml` para o documento acadêmico standalone
- template: `template_paper.org`

Fluxo recomendado com o pipeline:
1. defina `executar_pesquisa = false`;
2. defina `executar_documento = true`;
3. informe `pesquisa_dir_existente`;
4. rode o pipeline.

Fluxo recomendado com o standalone do paper:
1. preencha `template_toml_documento_academico_rc_1.toml`;
2. informe `pesquisa_dir_existente` ou `bundle_dir`;
3. aponte `template_org` para `template_paper.org`;
4. rode `gerador_documento_academico_rc_1.py`.

Quando usar:
- quando a pesquisa já foi concluída;
- quando você quer reaproveitar os selecionados e o `.bib`.

### Cenário 4 — Quero reescrever um paper já existente
Use:
- script: `gerador_pesquisa_documento_rc_2.py` **ou** `gerador_documento_academico_rc_1.py`
- template: `template_paper.org`

Configuração recomendada:
- `modo_escrita = "reescrever"`
- `reescrever_a_partir_do_org_atual = true`
- `paper_org_existente = "/caminho/para/paper_anterior.org"`

Quando usar:
- quando o texto já existe, mas precisa ser refeito com base melhor;
- quando você quer manter a pesquisa consolidada e mudar a redação.

### Cenário 5 — Quero expandir um paper com artigos extras
Use:
- script: `gerador_pesquisa_documento_rc_2.py` **ou** `gerador_documento_academico_rc_1.py`
- template: `template_paper.org`

Configuração recomendada:
- `modo_escrita = "expandir"`
- `usar_artigos_selecionados_pesquisa = true`
- `artigos_extras_paths = ["/caminho/para/pasta_com_extras"]`

Quando usar:
- quando o paper já está bom, mas precisa ganhar densidade;
- quando você quer agregar novas referências sem refazer toda a pesquisa.

### Cenário 6 — Quero só montar o bundle e revisar depois
Use:
- script: `gerador_pesquisa_documento_rc_2.py`
- TOML: `template_toml_pipeline_pesquisa_documento_rc_2.toml`

Configuração recomendada:
- `executar_pesquisa = true`
- `executar_bundle = true`
- `executar_documento = false`

Quando usar:
- quando você quer primeiro revisar os selecionados;
- quando ainda não quer disparar a redação do paper.

### Cenário 7 — Quero validar tudo antes de rodar de verdade
Use:
- o mesmo script do cenário desejado;
- o respectivo TOML;
- `dry_run = true`.

Quando usar:
- para validar caminhos;
- para checar templates;
- para confirmar presença dos artefatos esperados;
- para evitar rodadas desnecessárias da IA.

### Regra prática final

Se estiver em dúvida, siga esta ordem:

1. **Só pesquisa** → `gerador_pesquisa_rc_2.py`
2. **Pesquisa + paper no mesmo fluxo** → `gerador_pesquisa_documento_rc_2.py`
3. **Só paper com pesquisa pronta** → `gerador_documento_academico_rc_1.py`
4. **Reescrever ou expandir paper** → `gerador_documento_academico_rc_1.py` ou pipeline integrado, conforme o caso

### Mapa mínimo de abertura de arquivos

- Quer pesquisar?  
  Abra primeiro: `template_toml_unificado_rc_2.toml`

- Quer pipeline completo?  
  Abra primeiro: `template_toml_pipeline_pesquisa_documento_rc_2.toml`

- Quer só o paper?  
  Abra primeiro: `template_toml_documento_academico_rc_1.toml`

- Quer entender tudo antes?  
  Abra primeiro: `manual_unificado_rc_14.md`


---

## Motor de redação acadêmica

O motor standalone deixa de ser centrado apenas em "paper" e passa a ser organizado por **tipo documental**.

Arquivo principal:
- `gerador_documento_academico_rc_1.py`

Arquivo TOML correspondente:
- `template_toml_documento_academico_rc_1.toml`

### Lógica geral

Em vez de um script nomeado pelo produto final, a nova lógica organiza o motor por um conceito mais amplo:

- **documento acadêmico**

Isso permite adicionar, de forma incremental, novos tipos no futuro, como:
- paper
- dissertacao
- relatorio
- qualificacao
- memorial

Hoje, os tipos previstos diretamente são:
- `paper`
- `dissertacao`

### Campo principal

No TOML standalone, o campo central passa a ser:

```toml
[documento]
tipo_documento = "paper"
```

ou

```toml
[documento]
tipo_documento = "dissertacao"
```

### Templates por tipo

- `paper` → `template_paper.org`
- `dissertacao` → `template_dissertacao.org`
- fallback organizacional → `template_research.org`

### Exemplo: paper

```toml
[documento]
tipo_documento = "paper"
template_org = "/caminho/para/template_paper.org"
pesquisa_dir_existente = "/caminho/para/saida_da_pesquisa"
usar_artigos_selecionados_pesquisa = true
modo_escrita = "novo"
```

### Exemplo: dissertação

```toml
[documento]
tipo_documento = "dissertacao"
template_org = "/caminho/para/template_dissertacao.org"
pesquisa_dir_existente = "/caminho/para/saida_da_pesquisa"
usar_artigos_selecionados_pesquisa = true
artigos_extras_paths = ["/caminho/para/pasta_com_extras"]
modo_escrita = "expandir"
```

### Quando usar o motor standalone

Use `gerador_documento_academico_rc_1.py` quando:
- a pesquisa já está pronta;
- você quer só a etapa redacional;
- quer escolher entre paper e dissertação;
- quer reescrever, expandir ou regenerar o documento final.


---

## Estrutura profissional do bundle

O bundle distribuído foi reorganizado com uma estrutura interna de software mais profissional.

### Árvore recomendada

```text
bundle_projeto_pesquisa_documento_rc_17/
├── README.md
├── .env.example
├── requirements.txt
├── docs/
│   └── manual_unificado_rc_18.md
├── scripts/
│   ├── research/
│   │   └── gerador_pesquisa_rc_2.py
│   ├── pipeline/
│   │   └── gerador_pesquisa_documento_rc_2.py
│   └── document/
│       └── gerador_documento_academico_rc_3.py
├── config/
│   ├── research/
│   │   └── template_toml_unificado_rc_2.toml
│   ├── pipeline/
│   │   └── template_toml_pipeline_pesquisa_documento_rc_2.toml
│   └── document/
│       └── template_toml_documento_academico_rc_3.toml
└── templates/
    ├── template_research.org
    ├── template_paper.org
    └── template_dissertacao.org
```

### Regra de execução

Para que os caminhos relativos dos TOMLs funcionem como distribuídos no bundle, execute os comandos **a partir da raiz do bundle**.

### Quickstart

**Só pesquisa**
```bash
python ./scripts/research/gerador_pesquisa_rc_2.py --config ./config/research/template_toml_unificado_rc_2.toml
```

**Pesquisa + bundle + paper**
```bash
python ./scripts/pipeline/gerador_pesquisa_documento_rc_2.py --config ./config/pipeline/template_toml_pipeline_pesquisa_documento_rc_2.toml
```

**Só documento acadêmico (paper ou dissertação)**
```bash
python ./scripts/document/gerador_documento_academico_rc_3.py --config ./config/document/template_toml_documento_academico_rc_3.toml
```

---

## Confirmação sobre geração via IA e edição em Org

Sim. Foi mantida a lógica de geração **via IA** para:

- pesquisa;
- paper;
- dissertação.

E também foi mantida a geração do arquivo **`.org` editável** para o usuário revisar e ajustar manualmente depois, caso queira.

Em termos práticos:

- a **pesquisa** continua podendo gerar `.org` como artefato principal de saída;
- o **paper** continua sendo gerado em `.org`;
- a **dissertação** também passa a seguir a mesma lógica, usando `template_dissertacao.org` e gerando `.org` para edição posterior.

O PDF permanece como saída opcional, derivada da compilação do `.org`.


---

## Confirmação sobre geração via IA e `.org` editável

Sim. Foi mantida a geração **via IA** de:

- pesquisa;
- paper;
- dissertação.

E também foi mantida a geração do arquivo **`.org` editável** em todos esses fluxos.

Ou seja:
- a pesquisa continua gerando `.org`;
- o documento acadêmico standalone continua gerando `.org`;
- o pipeline integrado continua gerando `.org` da pesquisa e `.org` do documento final;
- o PDF continua sendo opcional e derivado da compilação do `.org`.


---

## Limpeza de nomenclatura legada

Nesta revisão, os nomes legados ligados a "paper" foram mantidos **apenas** quando se referem ao tipo documental legítimo `paper` ou ao arquivo `template_paper.org`.

Foram corrigidos, inclusive dentro dos códigos:
- `executar_paper` → `executar_documento`
- `paper_org_existente` → `documento_org_existente`
- `PaperContext` → `DocumentContext`
- `build_paper_context` → `build_document_context`
- `maybe_rewrite_paper_context` → `maybe_rewrite_document_context`
- `incluir_paper_pdf` → `incluir_documento_pdf`

Também foi revista a integração do motor standalone, que agora usa:
- `gerador_documento_academico_rc_3.py`
- `gerar_documento_org_ai_interativo_rc_1.py`

com a nomenclatura coerente com:
- `paper`
- `dissertacao`
- `documento acadêmico`


---

## Diagrama da arquitetura

O bundle inclui dois arquivos do diagrama arquitetural:

- `docs/diagrama_arquitetura.md`
- `docs/diagrama_arquitetura.svg`

Eles mostram a relação entre:
- pesquisa;
- pipeline integrado;
- motor standalone de documento acadêmico;
- motor textual IA;
- templates;
- artefatos e saídas finais.
