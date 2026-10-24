# Bundle canônico — MPPG / Orçamento Público / 4B

Revisão do bundle: **1.4 — 25/09/2026**.

Este diretório é a fonte canônica da revisão de literatura do subtema **4B — Controle, avaliação e participação social**, com ênfase recorrente em orçamento participativo.

## Regra de nomenclatura e sobrescrita

Os arquivos canônicos internos **não recebem sufixos de versão**. Alterações futuras sobrescrevem o arquivo correspondente. O histórico deve ser mantido por Git, backup externo ou pelo ZIP de entrega, não por nomes `_v10`, `_v11` etc.

## Diretório de compilação — opção A

`03_REVISAO/` contém somente os elementos necessários à recompilação e o PDF gerado:

- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org`
- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.bib`
- `assets/`
- `compilar.sh`
- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf` — criado/sobrescrito pela compilação bem-sucedida.

O conjunto de entrega é materializado automaticamente em `05_ENTREGAS/`: PDF e DOCX da revisão, além de PDF e DOCX do glossário. O PDF da revisão em `05_ENTREGAS/` é uma cópia byte-idêntica do PDF canônico validado em `03_REVISAO/`; os DOCX preservam a mesma composição visual página a página.

## Compilação do PDF

No diretório `03_REVISAO`:

```bash
chmod +x compilar.sh
./compilar.sh
```

O script detecta o Academic Pipeline MPPG, exporta ORG para TEX, aplica `biblatex-abnt`, executa LuaLaTeX/Biber/LuaLaTeX/LuaLaTeX e roda os gates. **O PDF canônico só é substituído depois de todos os gates fatais passarem.**

### Limpeza automática

Quando a compilação termina com sucesso, são excluídos automaticamente:

- `.build/` e todo o seu conteúdo;
- TEX exportado e arquivos auxiliares;
- `.aux`, `.bbl`, `.bcf`, `.blg`, `.log`, `.out`, `.run.xml` e equivalentes;
- arquivo temporário de validação;
- `compilacao.log`.

Assim, nenhuma sobra de compilação permanece em `03_REVISAO/`. Se houver **falha**, `.build/` e `compilacao.log` permanecem para diagnóstico.

Para preservar temporários mesmo quando houver sucesso:

```bash
MPPG_KEEP_BUILD=1 ./compilar.sh
```

Para apontar explicitamente o pipeline:

```bash
MPPG_ACADEMIC_PIPELINE_ROOT="$HOME/Documentos/mppg/software/academic_pipeline_mppg" ./compilar.sh
```

## DOCX

O comando canônico agora gera o DOCX automaticamente **depois** que o PDF correspondente foi compilado e validado. A regra é **PDF → DOCX de fidelidade visual**: cada página do PDF é rasterizada e inserida integralmente na página equivalente do Word, mantendo paginação, quebras, figuras, quadros, tipografia e posicionamento visual.

Saídas materializadas em `05_ENTREGAS/`:

- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf`;
- `REVISAO_LITERATURA_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.docx`;
- `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf`;
- `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.docx`.

Essa solução prioriza **paridade visual com o PDF**. Consequência técnica: o texto das páginas fica incorporado como imagem e não é editável/selecionável como texto Word. Se futuramente for necessária uma versão DOCX editável, ela deve ser tratada como uma saída distinta, cuja aparência pode aproximar-se, mas não ser garantida como pixel-idêntica ao PDF.

A resolução padrão é 200 dpi e pode ser alterada, por exemplo:

```bash
MPPG_DOCX_DPI=240 ./gerar_tudo.sh
```

Para gerar somente os DOCX a partir de PDFs já existentes:

```bash
./gerar_docx.sh
```

## Hierarquia do bundle

1. `01_GOVERNANCA/` — instruções, Prompt Master, Executor e política canônica;
2. `02_FONTES_INSTITUCIONAIS/` — fontes oficiais e modelos da disciplina;
3. `03_REVISAO/` — ORG, BIB, assets, compilador e PDF canônicos;
4. `04_CHECKPOINT/` — checkpoint e inventário de fontes acadêmicas;
5. `05_ENTREGAS/` — conjunto autocontido de entrega: PDF/DOCX da revisão e PDF/DOCX do glossário.

Dentro de `03_REVISAO/glossario/` ficam a fonte canônica dos verbetes, o wrapper ORG do glossário e seu compilador.

## Proveniência

A fonte canônica deriva do ORG corrigido em 19/09/2026, com adequação do fichamento ao modelo da disciplina e uso das citações selecionadas na Seção 3. A revisão 1.2 acrescentou o glossário obrigatório em fonte canônica única, sua materialização autônoma e sua inclusão como Apêndice A. A revisão 1.3 tornou `gerar_tudo.sh` responsável por gerar também os DOCX a partir dos PDFs finais validados, em modo de fidelidade visual. A revisão 1.4 passou a materializar também uma cópia byte-idêntica do PDF principal em `05_ENTREGAS/`, de modo que a pasta reúna as quatro saídas finais.


## Glossário obrigatório
O bundle gera o glossário do 4B em duas materializações sincronizadas a partir de **uma única fonte canônica**:

`03_REVISAO/glossario/GLOSSARIO_CONTEUDO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.org`

Essa fonte alimenta:
- o **arquivo autônomo** `GLOSSARIO_MPPG_ORCAMENTO_4B_CONTROLE_AVALIACAO_PARTICIPACAO.pdf`;
- o **APÊNDICE A — GLOSSÁRIO ORÇAMENTÁRIO DO SUBTEMA 4B** da revisão.

Os dez verbetes obrigatórios são: controle interno; controle externo (TCU); auditoria; controle social; conselho gestor; orçamento participativo; indicadores de desempenho; avaliação de políticas públicas; ciclo de políticas públicas; risco fiscal.

Para compilar somente o glossário:

```bash
cd 03_REVISAO/glossario
./compilar_glossario.sh
```

Para gerar revisão + glossário em **PDF e DOCX** de uma só vez:

```bash
cd 03_REVISAO
./gerar_tudo.sh
```

O `compilar.sh` principal contém gates que impedem uma compilação bem-sucedida se faltar algum dos dez verbetes, o campo “Não confundir com”, a inclusão canônica ou o Apêndice A no PDF. Depois disso, `gerar_docx.sh` cria as duas saídas DOCX, verifica paridade de número de páginas PDF/DOCX e materializa uma cópia byte-idêntica do PDF principal em `05_ENTREGAS/`. O `gerar_tudo.sh` só conclui com sucesso se as quatro saídas finais existirem.
