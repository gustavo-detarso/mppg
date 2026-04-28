# Manual unificado do projeto — gerador_pesquisa_rc_1.py

Este documento reúne, em um único arquivo, os dois guias do projeto:

1. **Guia do programa e uso do arquivo TOML**
2. **Guia completo de formulação de Tema, Recorte e Objetivo para otimizar as buscas**

---

## Sumário

- [Parte I — Guia do programa e uso do arquivo TOML](#parte-i--guia-do-programa-e-uso-do-arquivo-toml)
- [Parte II — Guia completo de formulação de Tema, Recorte e Objetivo](#parte-ii--guia-completo-de-formulação-de-tema-recorte-e-objetivo)

---

# Parte I — Guia do programa e uso do arquivo TOML

# Guia do programa e uso do arquivo TOML

Este documento explica **o que o programa faz**, **quais são os modos de uso**, **como preencher o arquivo TOML** e **como executar o script**.

Arquivo analisado: `gerar_atividade_prisma_api_multibase_interativo_v3_8_67.py`  
Template analisado: `template_toml_unificado.toml`

---

## 1. O que o programa faz

O programa gera atividades acadêmicas em **Org-mode** e pode exportar o resultado para **PDF**. Ele trabalha em **dois modos**:

### a) `revisao_sistematica`
Nesse modo, o script executa um fluxo de busca e seleção inspirado no **PRISMA**:

1. lê parâmetros pelo terminal, por flags ou por um arquivo TOML;
2. consulta bases acadêmicas;
3. deduplica registros e aplica filtros iniciais;
4. usa a OpenAI para apoiar:
   - sugestão de palavras-chave;
   - construção/ajuste de queries;
   - triagem dos candidatos;
   - análise dos textos selecionados;
   - síntese final;
5. gera:
   - documento `.org`;
   - fluxograma PRISMA em `.svg`;
   - versão PDF do fluxograma;
   - arquivo `.bib` com referências;
   - arquivo de debug em `.json`;
   - opcionalmente o PDF final do trabalho.

### b) `pesquisa_empirica`
Nesse modo, o script **não executa fluxo PRISMA**. Em vez disso, ele gera uma **proposta de pesquisa empírica**, com elementos como:

- contextualização;
- problema de pesquisa;
- justificativa;
- objetivo geral;
- objetivos específicos;
- proposta de condução;
- possibilidades de coleta e análise de dados;
- hipótese ou resposta teórica;
- modelo teórico;
- texto corrido para entrega.

Nesse modo, o foco é gerar o **roteiro/projeto empírico**, e não buscar e selecionar artigos por PRISMA.

---

## 2. O que o programa precisa para funcionar

### Dependências Python
Instale, no mínimo:

```bash
pip install openai requests python-dotenv pydantic prompt_toolkit
```

### Arquivo `.env`
As chaves de API **não ficam no TOML**. Elas devem ficar em um arquivo `.env` ao lado do script.

Exemplos de variáveis aceitas pelo programa:

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
CORE_EMAIL=
HTTP_PROXY=
HTTPS_PROXY=
```

Nem todas são obrigatórias. Isso depende das bases que você ativar.

---

## 3. Bases suportadas

No modo PRISMA, o script reconhece estas bases:

- `semantic_scholar`
- `scopus`
- `web_of_science`
- `pubmed`
- `openalex`
- `crossref`
- `europe_pmc`
- `core`

Use esses nomes exatamente assim no TOML.

---

## 4. Como o TOML funciona

O arquivo `template_toml_unificado.toml` é um **arquivo de configuração única** para os dois modos.

A lógica principal é esta:

- você preenche o TOML;
- o script lê esse arquivo;
- os valores do TOML são aplicados como se fossem argumentos de linha de comando;
- se `nao_interativo = true`, o script tenta rodar sem ficar perguntando coisas no terminal.

### Regra mais importante
No bloco `[atividade]`, escolha **apenas um** dos modos:

```toml
modo = "revisao_sistematica"
```

ou

```toml
modo = "pesquisa_empirica"
```

---

## 5. Estrutura do TOML, bloco por bloco

## `[atividade]`
Define os metadados gerais da atividade.

Exemplo:

```toml
[atividade]
modo = "revisao_sistematica"
disciplina = "Teorias de Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado de Políticas Públicas e Governo"
turma = "T-01"
polo = "Brasília"
aluno = "Seu nome"
```

### Campos
- `modo`: define o fluxo do programa.
- `disciplina`, `professor`, `curso`, `turma`, `polo`, `aluno`: entram como metadados do trabalho.

---

## `[pesquisa]`
É o núcleo intelectual da execução. Esse bloco orienta tanto a busca PRISMA quanto a geração da proposta empírica.

```toml
[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização do poder e resultados de políticas públicas"
objetivo = "Identificar como a literatura recente explica a relação entre capacidades estatais, centralização e desempenho de políticas públicas"
trabalho = ""
tipo_estudo = "Revisão de literatura"
periodo = "2019-2026"
idiomas = ["inglês", "português"]
bases = ["semantic_scholar", "scopus", "pubmed"]
palavras_chave = []
```

### Campos
- `tema`: assunto central.
- `recorte`: delimitação específica do tema.
- `objetivo`: objetivo do trabalho.
- `trabalho`: título do trabalho. Se ficar vazio, a IA pode sugerir o título.
- `tipo_estudo`: tipo de estudo priorizado.
- `periodo`: usado no PRISMA para recorte temporal.
- `idiomas`: idiomas aceitos na busca/análise.
- `bases`: bases a consultar no modo PRISMA.
- `palavras_chave`: palavras-chave iniciais. Se deixar vazio, a IA pode sugerir.

### Observação
- `periodo` e `bases` são essencialmente campos de **PRISMA**.
- No modo empírico, o mais importante aqui é a tríade **tema + recorte + objetivo** e o `tipo_estudo`.

---

## `[bibliografia]`
Usado principalmente no modo PRISMA.

```toml
[bibliografia]
estilo_citacao = "APA"
```

Valores comuns:

- `ABNT`
- `APA`
- `Chicago`
- `MLA`
- `Vancouver`

No modo empírico, esse bloco pode continuar preenchido, mas não é o centro do fluxo.

---

## `[busca]`
Controla como a busca e a geração de termos serão conduzidas.

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

### Campos
- `sugerir_palavras_chave_ia`: deixa a IA sugerir palavras-chave/conceitos.
- `query_bilingue`: tenta montar/sugerir buscas em português + inglês.
- `quantidade_triagem`: número máximo de candidatos que entram na triagem detalhada.
- `quantidade_selecionados`: quantos textos você deseja selecionar ao final.
- `salvar_busca_bruta_json`: salva respostas brutas das bases em JSON.
- `incluir_analise_detalhada_ia`: inclui análises individuais dos textos selecionados.
- `incluir_sintese_integradora_ia`: inclui a síntese final integrada.

### Importante
Os campos `quantidade_triagem`, `quantidade_selecionados`, `salvar_busca_bruta_json`, `incluir_analise_detalhada_ia` e `incluir_sintese_integradora_ia` fazem sentido principalmente no **modo PRISMA**.

---

## `[triagem]`
Bloco específico do modo PRISMA.

```toml
[triagem]
rigor = "moderado"
usar_score_hibrido = true
triagem_prompt_path = ""
diretivas_extras = ""
permitir_textos_nao_publicos = false
```

### Campos
- `rigor`: pode ser `estrito`, `moderado` ou `exploratorio`.
- `usar_score_hibrido`: combina heurística local + IA para calibrar ranking/triagem.
- `triagem_prompt_path`: caminho para um arquivo com instruções extras de triagem.
- `diretivas_extras`: texto inline ou caminho para outro arquivo com diretrizes adicionais.
- `permitir_textos_nao_publicos`: se `true`, o script pode manter textos sem download local do PDF quando houver ao menos um link verificável do registro.

### Quando usar `permitir_textos_nao_publicos = true`
Use apenas quando você aceita um resultado metodologicamente mais flexível. O padrão mais rigoroso é manter `false`.

---

## `[queries]`
Permite informar manualmente as queries por base.

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

### Como usar
#### Opção 1 — automática
Deixe tudo vazio. O Python monta as queries com base em:

- tema;
- recorte;
- objetivo;
- tipo de estudo;
- palavras-chave;
- idiomas;
- modo bilíngue.

#### Opção 2 — manual
Preencha as queries que você quiser controlar diretamente.

Exemplo:

```toml
query_scopus = "TITLE-ABS-KEY(\"state capacity\" AND \"policy implementation\" AND \"literature review\")"
```

### Recomendação prática
Se você quer reprodutibilidade com menos trabalho manual, deixe as `query_*` vazias e use `palavras_chave = []` com `sugerir_palavras_chave_ia = true`.

---

## `[saida]`
Controla os arquivos gerados.

```toml
[saida]
prefixo = "atividade_capacidades_estatais"
output_dir = "/caminho/para/saida"
criar_subdiretorio = true
org_modelo = "/caminho/para/template.org"
arquivo_orientacao = ""
exportar_pdf = true
gerar_env_example = false
remover_auxiliares = true
```

### Campos
- `prefixo`: prefixo dos nomes dos arquivos gerados.
- `output_dir`: diretório-base de saída.
- `criar_subdiretorio`: se `true`, cria uma pasta com o nome do prefixo.
- `org_modelo`: caminho do template `.org` usado como base.
- `arquivo_orientacao`: arquivo opcional com diretrizes complementares do trabalho.
- `exportar_pdf`: tenta converter o `.org` final para PDF.
- `gerar_env_example`: gera um `.env.example`.
- `remover_auxiliares`: remove arquivos auxiliares ao final.

### Sobre `arquivo_orientacao`
Esse arquivo pode ser `.txt`, `.md`, `.org`, `.docx` ou até `.pdf` em alguns casos de extração. Ele complementa o contexto do trabalho e influencia a geração final.

---

## `[latex]`
Controla a exportação do Org para PDF.

```toml
[latex]
org_latex_class_init = "/caminho/para/academic-writing.el"
latex_extra_path = "/caminho/para/fgv-paper.sty"
comando_exportacao_pdf = ""
fgv_logo_path = "/caminho/para/logo.png"
```

### Campos
- `org_latex_class_init`: arquivo `.el` que registra a classe LaTeX no Emacs batch.
- `latex_extra_path`: caminho de `.sty`, pasta ou recurso LaTeX adicional.
- `comando_exportacao_pdf`: comando externo opcional para exportar PDF.
- `fgv_logo_path`: logo usado no cabeçalho do relatório.

### Dica importante
Se seu template Org usa uma classe como `fgv-paper`, esse bloco é decisivo. Sem isso, a exportação em lote pode falhar porque o Emacs batch ou o LaTeX não encontram a classe/pacotes necessários.

---

## `[openai]`
Define o modelo.

```toml
[openai]
model = "gpt-5.4"
```

---

## `[controle]`
Controla a execução automática e o salvamento da configuração final.

```toml
[controle]
nao_interativo = true
salvar_config = true
config_output = ""
```

### Campos
- `nao_interativo`: se `true`, evita perguntas no terminal e força uso do TOML/CLI.
- `salvar_config`: se `true`, o script salva a configuração final em TOML.
- `config_output`: caminho do TOML final salvo pelo script.

### Comportamento prático
Se `salvar_config = true` e `config_output` estiver vazio, o script tende a salvar automaticamente um TOML final no diretório de saída, com nome derivado do prefixo.

---

## 6. Como preencher o TOML para cada modo

## Exemplo mínimo — modo PRISMA

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
quantidade_triagem = 25
quantidade_selecionados = 3
salvar_busca_bruta_json = true
incluir_analise_detalhada_ia = true
incluir_sintese_integradora_ia = true

[triagem]
rigor = "moderado"
usar_score_hibrido = true
triagem_prompt_path = ""
diretivas_extras = ""
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
prefixo = "atividade_capacidades_estatais_prisma"
output_dir = "/caminho/para/saida"
criar_subdiretorio = true
org_modelo = "/caminho/para/template.org"
arquivo_orientacao = ""
exportar_pdf = true
gerar_env_example = false
remover_auxiliares = true

[latex]
org_latex_class_init = "/caminho/para/academic-writing.el"
latex_extra_path = "/caminho/para/fgv-paper.sty"
comando_exportacao_pdf = ""
fgv_logo_path = "/caminho/para/logo.png"

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
salvar_config = true
config_output = ""
```

---

## Exemplo mínimo — modo empírico

```toml
[atividade]
modo = "pesquisa_empirica"
disciplina = "Teorias de Administração Pública"
professor = "Bernardo Buta"
curso = "Mestrado de Políticas Públicas e Governo"
turma = "T-01"
polo = "Brasília"
aluno = "Seu nome"

[pesquisa]
tema = "Capacidades estatais e desempenho estatal"
recorte = "Relação entre coerção, centralização do poder e resultados de políticas públicas"
objetivo = "Elaborar uma proposta de pesquisa empírica sobre capacidades estatais, capacidade coercitiva, centralização político-administrativa e desempenho de políticas públicas"
trabalho = ""
tipo_estudo = "Métodos mistos"
idiomas = ["inglês", "português"]
palavras_chave = []

[busca]
sugerir_palavras_chave_ia = true
query_bilingue = true

[saida]
prefixo = "atividade_capacidades_estatais_empirica"
output_dir = "/caminho/para/saida"
criar_subdiretorio = true
org_modelo = "/caminho/para/template.org"
arquivo_orientacao = ""
exportar_pdf = true
gerar_env_example = false
remover_auxiliares = true

[latex]
org_latex_class_init = "/caminho/para/academic-writing.el"
latex_extra_path = "/caminho/para/fgv-paper.sty"
comando_exportacao_pdf = ""
fgv_logo_path = "/caminho/para/logo.png"

[openai]
model = "gpt-5.4"

[controle]
nao_interativo = true
salvar_config = true
config_output = ""
```

### Observação
No modo empírico, os blocos `[bibliografia]`, `[triagem]` e `[queries]` podem até continuar presentes, mas deixam de ser o centro do processamento.

---

## 7. Como executar com o TOML

Supondo que o TOML esteja salvo como `template_toml_unificado.toml`:

```bash
python gerar_atividade_prisma_api_multibase_interativo_v3_8_67.py --config /caminho/para/template_toml_unificado.toml
```

Se quiser garantir execução sem perguntas no terminal, mantenha no TOML:

```toml
[controle]
nao_interativo = true
```

---

## 8. Como executar sem TOML

O script também pode rodar de forma interativa ou por flags. Exemplo:

```bash
python gerar_atividade_prisma_api_multibase_interativo_v3_8_67.py \
  --disciplina "Teorias de Administração Pública" \
  --professor "Bernardo Buta" \
  --curso "Mestrado de Políticas Públicas e Governo" \
  --tema "stakeholders" \
  --recorte "artigo de revisão recente" \
  --objetivo "localizar um artigo de revisão recente e analisá-lo" \
  --palavras-chave "stakeholder,stakeholders,stakeholder engagement" \
  --bases "semantic_scholar,scopus,web_of_science" \
  --aluno "Gustavo M. Mendes de Tarso"
```

Mas, para repetibilidade, o TOML é a melhor opção.

---

## 9. Arquivos que o programa gera

## No modo PRISMA
Tipicamente o script gera:

- `PREFIXO.org`
- `PREFIXO_prisma.svg`
- `PREFIXO_prisma.pdf` (PDF do fluxograma)
- `PREFIXO.bib`
- `PREFIXO_debug.json`
- `PREFIXO_config.toml` ou outro TOML final, se `salvar_config = true`
- `PREFIXO.pdf` se a exportação final do Org funcionar
- logs por fonte
- JSON bruto por fonte, se habilitado

Além disso, ele pode criar cache temporário de texto completo e outros auxiliares, que depois podem ser removidos se `remover_auxiliares = true`.

## No modo empírico
Tipicamente o script gera:

- `PREFIXO.org`
- `PREFIXO_debug.json`
- `PREFIXO_config.toml` ou equivalente, se habilitado
- `PREFIXO.pdf` se a exportação final funcionar

Nesse modo não há fluxograma PRISMA nem `.bib` obrigatório como peça central do fluxo.

---

## 10. Estratégias de uso recomendadas

## Cenário A — quero praticidade
Use:

- `palavras_chave = []`
- `query_* = ""`
- `sugerir_palavras_chave_ia = true`
- `query_bilingue = true`

Assim o script deixa a IA sugerir termos e o Python monta as queries.

## Cenário B — quero controle metodológico fino
Preencha manualmente:

- `palavras_chave`
- uma ou mais `query_*`
- `triagem_prompt_path`
- `diretivas_extras`

Isso te dá mais controle sobre a estratégia de busca e os critérios de triagem.

## Cenário C — quero repetir o mesmo experimento depois
Use:

- `nao_interativo = true`
- `salvar_config = true`

Assim você roda com o mesmo TOML depois e compara saídas.

---

## 11. Erros e cuidados comuns

### 1. Chaves de API no lugar errado
As chaves **não** devem ir no TOML. Devem ir no `.env`.

### 2. Nome errado da base
No TOML, use exatamente os nomes aceitos, como `semantic_scholar` e `europe_pmc`.

### 3. Exportação PDF falhando
Se o template Org usa classe LaTeX customizada, você provavelmente precisará preencher corretamente:

- `org_latex_class_init`
- `latex_extra_path`
- possivelmente `comando_exportacao_pdf`

### 4. Modo errado
Se você colocar `modo = "pesquisa_empirica"`, o programa **não fará PRISMA**, mesmo que os blocos de triagem estejam preenchidos.

### 5. TOML muito manual e contraditório
Evite misturar:

- palavras-chave manuais muito rígidas;
- queries manuais muito restritivas;
- triagem extremamente estrita;
- poucas bases.

Essa combinação pode zerar a seleção final.

---

## 12. Recomendação final

Se a intenção é usar o script como fluxo principal, o melhor caminho é:

1. começar pelo template TOML unificado;
2. preencher bem `[atividade]` e `[pesquisa]`;
3. decidir o `modo` corretamente;
4. deixar a IA sugerir palavras-chave e queries na primeira rodada;
5. só depois travar queries manuais, se necessário;
6. salvar a configuração final para reaproveitar a execução.

---

## 13. Resumo rápido

- O programa serve para gerar **atividade PRISMA** ou **proposta de pesquisa empírica**.
- O arquivo TOML é a forma mais estável de controlar a execução.
- O campo mais importante do TOML é `modo`.
- No PRISMA, o script busca em bases, tria, analisa e gera `.org`, `.svg`, `.bib`, `.json` e opcionalmente `.pdf`.
- No empírico, ele gera um projeto/roteiro de pesquisa em `.org` e opcionalmente `.pdf`.
- As chaves de API ficam no `.env`, não no TOML.



---

# Parte II — Guia completo de formulação de Tema, Recorte e Objetivo

# Guia completo de formulação de Tema, Recorte e Objetivo para otimizar as buscas

## 1. Finalidade deste guia

No **gerador_pesquisa_rc_1.py**, a qualidade da busca depende menos de “ter muitas palavras” e mais de **formular corretamente a tríade Tema–Recorte–Objetivo**. Essa tríade não serve apenas para descrever o assunto do trabalho. Ela orienta, de forma indireta, quase todo o fluxo:

- geração automática de palavras-chave;
- expansão controlada de termos relacionados;
- montagem de blocos conceituais;
- construção das queries por base;
- triagem temática;
- ranking dos textos;
- justificativas de inclusão e exclusão;
- síntese final.

Por isso, preencher mal essa tríade costuma produzir dois problemas clássicos:

- **abertura excessiva da busca**, com muitos textos “do campo”, mas poucos textos realmente úteis;
- **estreitamento excessivo**, com queries literais demais e baixo retorno em certas bases.

O objetivo deste guia é te ajudar a escrever Tema, Recorte e Objetivo de modo que o script encontre o melhor equilíbrio entre **precisão temática**, **recall suficiente** e **aderência analítica**.

---

## 2. Regra central: o papel de cada elemento

A forma mais segura de pensar a tríade é esta:

- **Tema** = o núcleo obrigatório
- **Recorte** = a fronteira substantiva
- **Objetivo** = a relação analítica que a busca deve privilegiar

Em linguagem simples:

- o **Tema** diz **sobre o que a pesquisa é, sem dúvida**;
- o **Recorte** diz **qual parte específica desse tema realmente importa**;
- o **Objetivo** diz **o que você quer que a literatura explique, compare, mostre ou esclareça**.

Se os três estiverem bem escritos, o script tende a funcionar melhor.  
Se um deles estiver fraco, os demais precisam “compensar”, e isso quase sempre piora a qualidade da busca.

---

## 3. Como o script “lê” essa tríade

Embora o usuário escreva só três campos, o sistema tenta inferir deles várias camadas.

### 3.1 O Tema tende a ser interpretado como eixo principal
O Tema costuma gerar:
- conceitos-base;
- termos centrais do campo;
- âncoras para as queries;
- requisito mínimo da triagem.

Se o Tema for mal formulado, todo o resto nasce torto.

### 3.2 O Recorte tende a funcionar como calibrador de aderência
O Recorte ajuda a distinguir:
- textos que são apenas “do assunto”;
- textos que realmente abordam a **relação específica** que interessa.

Em rigor moderado, ele não deve funcionar como faca cega que elimina tudo, mas como filtro forte de prioridade.

### 3.3 O Objetivo orienta o tipo de relação analítica
O Objetivo ajuda o script a entender:
- se a busca quer mapear um campo;
- explicar mecanismos;
- comparar arranjos;
- identificar efeitos;
- localizar tensões;
- construir um projeto empírico;
- encontrar revisão de literatura, scoping review, estudo de caso etc.

Em termos práticos, o Objetivo diz ao sistema **que tipo de texto deve subir no ranking**.

---

## 4. O que faz uma tríade ser boa

Uma tríade boa normalmente tem estas características:

### 4.1 Claridade
Cada campo deve ter função distinta.  
Se Tema, Recorte e Objetivo disserem quase a mesma coisa, o script perde resolução.

### 4.2 Densidade analítica
O texto deve usar conceitos reais, não rótulos vagos.

Melhor:
- capacidade coercitiva
- coordenação intergovernamental
- enforcement regulatório
- desempenho estatal
- arranjos institucionais
- implementação de políticas públicas

Pior:
- fatores relevantes
- elementos importantes
- vários aspectos do tema
- questões relacionadas

### 4.3 Delimitação sem literalismo
Você quer delimitar o problema, mas não obrigar a busca a depender de uma única frase rara.

### 4.4 Separação entre núcleo e detalhe
O Tema deve dar o núcleo.  
O Recorte deve introduzir a especificidade.  
O Objetivo deve introduzir a operação analítica.

### 4.5 Reutilização
A formulação deve permitir que o script use a lógica em outros temas, sem ficar “viciado” num caso específico.

---

## 5. Como escrever o Tema

## 5.1 O que o Tema deve fazer
O Tema deve responder à pergunta:

**“Qual é o campo central desta busca?”**

Ele deve:
- nomear o objeto principal;
- dizer qual discussão central está em jogo;
- ser amplo o suficiente para permitir busca;
- ser específico o suficiente para impedir dispersão total.

## 5.2 O que o Tema não deve fazer
O Tema não deve:
- tentar incluir todos os detalhes do problema;
- virar um mini-parágrafo com várias relações analíticas;
- acumular sinônimos demais;
- misturar resultados, métodos, contexto e hipótese num único campo.

## 5.3 Estrutura recomendada do Tema
A forma mais segura é:

**[conceito principal 1] + [conceito principal 2]**

ou

**[objeto] + [fenômeno]**

ou

**[campo substantivo] + [dimensão principal]**

## 5.4 Exemplos bons de Tema
- `Capacidades estatais e desempenho estatal`
- `Coordenação federativa e implementação de políticas públicas`
- `Governança regulatória e enforcement estatal`
- `Descentralização e efetividade governamental`
- `Burocracia pública e capacidade de implementação`

## 5.5 Exemplos ruins de Tema
- `Estudo sobre o funcionamento do Estado`
- `Alguns aspectos da governança pública`
- `Análise ampla sobre vários fatores do desempenho estatal`
- `Governança, implementação, coordenação, burocracia, regulação, controle e capacidade`

## 5.6 Erro típico no Tema
O erro mais comum é querer resolver o problema inteiro nele.  
Quando isso acontece, o Tema vira uma sequência de rótulos e a busca passa a capturar textos do campo em vez de textos do problema.

---

## 6. Como escrever o Recorte

## 6.1 O que o Recorte deve fazer
O Recorte deve responder:

**“Qual parte específica do tema me interessa e o que diferencia um texto aderente de um texto apenas vizinho?”**

Ele é a peça que reduz a abertura excessiva.  
Na prática, o Recorte serve para impedir que o sistema trate como suficiente um texto que só menciona o campo de forma genérica.

## 6.2 O que o Recorte deve conter
Ele costuma funcionar melhor quando inclui pelo menos dois destes elementos:

- mecanismo;
- dimensão institucional;
- relação entre variáveis;
- condição contextual;
- nível de governo;
- tipo de efeito ou resultado.

## 6.3 Estrutura recomendada do Recorte
Uma fórmula muito boa é:

**“Relação entre [dimensão A], [dimensão B] e [resultado], considerando [contexto/condição].”**

Outra fórmula possível:

**“Análise de como [mecanismo] afeta [resultado], em [contexto institucional ou territorial].”**

## 6.4 Exemplos bons de Recorte
- `Relação entre capacidades coercitivas, centralização político-administrativa e resultados de políticas públicas`
- `Como arranjos institucionais interferem na implementação e na efetividade de políticas sociais`
- `Impactos da descentralização sobre coordenação intergovernamental e desempenho estatal`
- `Efeitos da capacidade burocrática e do enforcement regulatório sobre a implementação de políticas ambientais`
- `Como mecanismos de coordenação federativa afetam entrega de serviços públicos`

## 6.5 Exemplos ruins de Recorte
- `Análise do tema`
- `Com foco no assunto`
- `Estudos relacionados`
- `Aspectos importantes da governança`
- `Questões que envolvem o tema principal`

## 6.6 O que torna um Recorte forte
Um Recorte forte geralmente:
- introduz tensão ou mecanismo;
- elimina, sem dizer “excluir”, boa parte dos textos genéricos;
- ajuda a IA a priorizar textos mais focalizados;
- não exige uma formulação tão rara que mate o recall.

## 6.7 Erro típico no Recorte
O erro mais frequente é fazer um Recorte tão abstrato que ele não recorta nada.

---

## 7. Como escrever o Objetivo

## 7.1 O que o Objetivo deve fazer
O Objetivo deve responder:

**“O que eu quero que a busca me ajude a descobrir, explicar, comparar ou mapear?”**

Ele não é só uma frase bonita.  
Ele orienta o tipo de literatura que sobe no ranking.

## 7.2 Verbos que funcionam melhor
Use verbos analíticos claros, como:
- investigar
- identificar
- examinar
- comparar
- mapear
- avaliar
- analisar
- compreender
- verificar

## 7.3 O que o Objetivo deve explicitar
Ele deve mostrar:
- a relação entre os conceitos;
- o efeito, mecanismo ou padrão procurado;
- a utilidade analítica da busca.

## 7.4 Estrutura recomendada do Objetivo
Uma estrutura muito boa é:

**“Investigar como [A] influencia [B], identificando [mecanismos, limites, tensões, efeitos ou padrões].”**

Outra boa forma:

**“Mapear como a literatura recente explica [relação X], com atenção a [mecanismos ou resultados].”**

## 7.5 Exemplos bons de Objetivo
- `Investigar como capacidades estatais e centralização do poder influenciam a implementação e os resultados de políticas públicas`
- `Identificar mecanismos explicativos que conectem arranjos institucionais, coordenação e desempenho estatal`
- `Mapear como a literatura explica a relação entre descentralização, enforcement e efetividade governamental`
- `Examinar de que forma a capacidade coercitiva se articula com coordenação institucional e resultados de políticas públicas`

## 7.6 Exemplos ruins de Objetivo
- `Estudar o tema`
- `Entender melhor o assunto`
- `Pesquisar sobre capacidades estatais`
- `Discutir algumas questões do tema`

## 7.7 Erro típico no Objetivo
O problema mais comum é escrever um objetivo apenas temático, quando ele deveria ser relacional.

---

## 8. Relação correta entre os três campos

Uma tríade ruim geralmente tem superposição.  
Uma tríade boa tem **camadas**.

### 8.1 Forma ideal
- o **Tema** introduz o campo;
- o **Recorte** introduz a especificidade;
- o **Objetivo** introduz a operação analítica.

### 8.2 Exemplo bem construído
**Tema:**  
`Capacidades estatais e desempenho estatal`

**Recorte:**  
`Relação entre capacidades coercitivas, centralização político-administrativa e resultados de políticas públicas`

**Objetivo:**  
`Investigar como capacidades estatais, centralização do poder e coordenação institucional influenciam a implementação e a efetividade de políticas públicas, identificando mecanismos explicativos e tensões entre autoridade e desempenho.`

### 8.3 Exemplo mal construído
**Tema:**  
`Capacidades estatais, coordenação, coerção, centralização, desempenho, implementação e efetividade`

**Recorte:**  
`Análise do tema`

**Objetivo:**  
`Entender melhor o assunto`

Aqui o Tema faz tudo, o Recorte não recorta e o Objetivo não orienta nada.

---

## 9. Como otimizar a tríade para o modo PRISMA

No modo PRISMA, o script precisa:
- gerar termos de busca;
- construir queries por base;
- recuperar candidatos;
- fazer triagem temática;
- ranquear aderência;
- justificar seleção final.

Por isso, no modo PRISMA a tríade deve ser mais cuidadosa em três pontos.

## 9.1 O Tema deve ser conceitualmente forte
Ele será o eixo obrigatório.  
Se estiver frouxo, a triagem começa frouxa.

## 9.2 O Recorte deve impedir inflation
Ele deve impedir que “governança”, “implementação” ou “administração pública” bastem sozinhas como justificativa de inclusão.

## 9.3 O Objetivo deve privilegiar explicação
Revisões muito boas não apenas “mapeiam o tema”; elas mapeiam **como a literatura trata uma relação específica**.

## 9.4 No PRISMA, perguntas úteis são:
- Que mecanismo relaciona A e B?
- Que efeito institucional aparece entre X e Y?
- Em que condições o arranjo Z melhora ou piora o desempenho?
- Como a literatura trata a tensão entre autoridade e efetividade?

---

## 10. Como otimizar a tríade para o modo empírico

No modo empírico, a tríade precisa funcionar menos como “query de busca” e mais como base para:
- problema de pesquisa;
- justificativa;
- hipótese ou resposta teórica;
- proposta metodológica;
- desenho empírico.

Por isso:

## 10.1 O Tema continua sendo o núcleo
Sem ele, o projeto vira um agregado disperso.

## 10.2 O Recorte ganha ainda mais importância
Ele ajuda a transformar um campo amplo em um problema empiricamente estudável.

## 10.3 O Objetivo deve ser operacional
No empírico, ele precisa sugerir:
- variável;
- relação;
- mecanismo;
- caso;
- comparação;
- evidência possível.

### Exemplo
**Tema:**  
`Capacidades estatais e desempenho estatal`

**Recorte:**  
`Relação entre centralização político-administrativa e coordenação intergovernamental na implementação de políticas públicas`

**Objetivo:**  
`Investigar como diferentes graus de centralização político-administrativa afetam a coordenação intergovernamental e a implementação de políticas públicas em contextos federativos.`

Aqui já nasce uma pergunta empírica plausível.

---

## 11. Como escolher o nível certo de detalhe

Esse é um dos pontos mais importantes.

## 11.1 Quando está amplo demais
Sinais de excesso de amplitude:
- tudo parece caber;
- o script começa a trazer textos de campo muito genéricos;
- muitas palavras amplas entram na query;
- a triagem precisa “salvar” a busca sozinha.

## 11.2 Quando está estreito demais
Sinais de excesso de estreitamento:
- a query vira quase uma frase rara;
- certas bases retornam 0 com frequência;
- você depende de formulações literais;
- o tema parece mais um resumo de conclusão do que um campo de busca.

## 11.3 O ponto ideal
O ponto ideal é:
- **Tema claro**
- **Recorte forte**
- **Objetivo analítico**
- mas sem transformar os três em uma frase única hiperliteral.

---

## 12. O uso de sinônimos e termos relacionados

O script já tenta lidar com:
- sinônimos;
- equivalentes em inglês;
- blocos conceituais;
- expansão controlada de termos.

Por isso, o usuário **não precisa tentar colocar todos os sinônimos manualmente** no Tema, Recorte e Objetivo.

## 12.1 O que fazer
Use os conceitos principais.

## 12.2 O que evitar
Não escreva assim:
`capacidade estatal, state capacity, capacidade governamental, government capacity, governança, governance, administração pública, public administration...`

Isso tende a gerar poluição em vez de precisão.

## 12.3 Regra prática
Escreva o conceito principal de forma natural.  
Deixe o script expandir com parcimônia.

---

## 13. Como lidar com termos amplos

Alguns termos são úteis, mas perigosos:
- governança
- administração pública
- implementação
- desempenho
- efetividade
- coordenação

Eles **não são ruins**.  
O problema é quando aparecem sem amarração com:
- um mecanismo;
- um arranjo;
- uma tensão;
- um resultado;
- um eixo substantivo.

## 13.1 Como usar bem
- `coordenação intergovernamental`
- `efetividade de políticas públicas`
- `desempenho estatal`
- `arranjos institucionais de implementação`
- `governança regulatória`

## 13.2 Como usar mal
- `governança`
- `implementação`
- `coordenação`
- `efetividade`
- `desempenho`

isolados e sem contexto.

---

## 14. Como escrever para melhorar a geração automática de queries

Como o script monta queries a partir da tríade, você pode ajudar a IA e o parser do script com algumas escolhas.

## 14.1 Prefira conceitos recuperáveis
Melhor:
- `capacidade coercitiva`
- `centralização político-administrativa`
- `coordenação intergovernamental`
- `arranjos institucionais`
- `implementação de políticas públicas`
- `desempenho estatal`

Pior:
- `estrutura de mando`
- `funcionamento do sistema`
- `capacidade de agir`
- `problemas do campo`

## 14.2 Prefira relações explícitas
Melhor:
- `relação entre A e B`
- `efeitos de A sobre B`
- `como A influencia B`
- `mecanismos que conectam A e B`

Pior:
- `alguns aspectos de A e B`
- `discussão sobre A e B`

## 14.3 Evite frases excessivamente longas
Uma formulação clara e densa é melhor do que uma frase enorme e barroca.

---

## 15. Como transformar uma ideia vaga em uma tríade boa

## 15.1 Ideia vaga
`Quero pesquisar como o Estado funciona melhor ou pior`

## 15.2 Transformação
Perguntas para refinar:
- Qual dimensão do Estado?
- Melhor ou pior em quê?
- Em qual processo?
- Por qual mecanismo?
- Em qual arranjo?

## 15.3 Resultado
**Tema:**  
`Capacidades estatais e desempenho estatal`

**Recorte:**  
`Relação entre coordenação institucional, centralização do poder e implementação de políticas públicas`

**Objetivo:**  
`Investigar como diferentes arranjos de coordenação e centralização influenciam a implementação e os resultados de políticas públicas.`

---

## 16. Como saber se a tríade ficou boa antes de rodar

Faça este teste mental.

### 16.1 Teste do Tema
Se eu lesse só o Tema, eu saberia claramente o campo?

### 16.2 Teste do Recorte
Se eu lesse só o Recorte, eu saberia o que diferencia um texto aderente de um texto genérico do campo?

### 16.3 Teste do Objetivo
Se eu lesse só o Objetivo, eu saberia que tipo de relação, mecanismo ou resultado a busca quer privilegiar?

Se a resposta for “não” em algum deles, o campo precisa de revisão.

---

## 17. Checklist de qualidade antes da execução

Use esta lista toda vez que for preencher o TOML.

### Tema
- O núcleo do campo está claro?
- Há no máximo 1 ou 2 conceitos centrais?
- O Tema evita virar um parágrafo?

### Recorte
- O que realmente interessa está explícito?
- Há mecanismo, relação ou condição?
- Ele ajuda a excluir textos apenas genéricos do campo?

### Objetivo
- O verbo é analítico?
- A relação entre conceitos está clara?
- O objetivo ajuda a identificar o que conta como aderência?

### Tríade como conjunto
- Os três campos fazem funções diferentes?
- Há precisão sem literalismo excessivo?
- O problema está claro sem depender de sinônimos múltiplos?

---

## 18. Modelos prontos para preenchimento

## 18.1 Modelo geral
**Tema:**  
`[conceito principal] e [fenômeno central]`

**Recorte:**  
`Relação entre [mecanismo/dimensão A], [mecanismo/dimensão B] e [resultado de interesse], considerando [contexto/condição].`

**Objetivo:**  
`Investigar como [A] e [B] influenciam [resultado], identificando [mecanismos, limites, tensões, efeitos ou padrões].`

## 18.2 Modelo mais explicativo
**Tema:**  
`[campo substantivo] e [resultado ou processo]`

**Recorte:**  
`Análise de como [arranjo institucional/mecanismo] afeta [implementação, desempenho, efetividade, coordenação etc.]`

**Objetivo:**  
`Examinar de que forma [conceito 1] e [conceito 2] se articulam na produção de [resultado], com atenção a [mecanismos, tensões ou limites].`

## 18.3 Modelo empírico
**Tema:**  
`[campo] e [fenômeno]`

**Recorte:**  
`Relação entre [variável institucional] e [resultado observável], em [contexto/caso/nível de governo].`

**Objetivo:**  
`Investigar como [variável] afeta [resultado], identificando [mecanismos causais, padrões comparativos ou condicionantes].`

---

## 19. Exemplo completo aplicado ao seu caso

### Tema
`Capacidades estatais e desempenho estatal`

### Recorte
`Relação entre capacidades coercitivas, centralização político-administrativa e resultados de políticas públicas, considerando o papel dos arranjos institucionais na implementação e na efetividade estatal.`

### Objetivo
`Investigar de que forma capacidades estatais, especialmente capacidade coercitiva e centralização do poder, influenciam o desempenho estatal e os resultados de políticas públicas, identificando mecanismos explicativos, limites analíticos e tensões entre autoridade estatal, coordenação institucional e efetividade governamental.`

## Por que essa formulação funciona
- o **Tema** fixa o núcleo;
- o **Recorte** impede abertura excessiva;
- o **Objetivo** força a busca a privilegiar explicações e relações, não só mapeamento do campo.

---

## 20. Erros estratégicos que mais prejudicam o gerador_pesquisa_rc_1.py

### 20.1 Tema muito amplo
Abre a busca demais.

### 20.2 Recorte vazio
Faz a triagem carregar sozinha o peso da precisão.

### 20.3 Objetivo descritivo
Favorece textos panorâmicos demais.

### 20.4 Excesso de sinônimos
Pode bagunçar a expansão de termos.

### 20.5 Linguagem vaga
Reduz a capacidade do script de gerar blocos conceituais úteis.

### 20.6 Frase rara demais
Derruba recall em bases mais sensíveis.

---

## 21. Regra final de ouro

Se eu tivesse que resumir o guia inteiro em uma fórmula única, seria esta:

**Tema define o campo.  
Recorte define a fronteira.  
Objetivo define a relação analítica que a busca precisa capturar.**

Quando essa tríade está bem construída:
- as keywords saem melhores;
- as queries ficam menos artificiais;
- a triagem precisa “consertar” menos a busca;
- os textos finais tendem a ficar mais aderentes.

---

## 22. Versão resumida para lembrar sempre

Antes de rodar, pergunte a si mesmo:

- **Tema:** sobre o que esta busca é?
- **Recorte:** qual parte exata disso me interessa?
- **Objetivo:** o que eu quero que a literatura explique?

Se essas três respostas estiverem claras, o restante do fluxo tende a melhorar bastante.

