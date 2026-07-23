# AP-007B.3 — Encerramento formal da AP-007B

## Situação final

A AP-007B está encerrada tecnicamente em estado pré-commit. O entrypoint público `academic_pipeline.cli:main(argv)` delega a `academic_pipeline.runtime:run(argv)`. A primeira onda executa cinco superfícies sem chamar o monólito; os demais comandos seguem por fallback legado explícito e injetado.

## Primeira onda nativa

- `--help`;
- `--list-toml-profiles`;
- `--list-institutions`;
- `--list-layouts`;
- `--explain-profile`.

O `RuntimeContext` possui seis dependências explícitas e não usa `globals()` ou `locals()`.

## Validações consolidadas

- contrato AP-007B: 20 aprovados;
- regressões source-tree: 24 aprovadas e 4 desmarcadas;
- comparação tríplice: cinco superfícies equivalentes;
- fallback preservado como `list[str]`;
- `sys.argv`, `sys.path` e diretório de trabalho preservados;
- 63 opções longas registradas e 66 tokens no texto de ajuda.

Os quatro testes desmarcados pertencem ao contrato distributivo e de instalação isolada da AP-007E.

## Evidência formal

`ap007b2_validacao_integrada_comparativa_v1_20260722_215803.log`, SHA-256 `6e0c24d61054bcb9f111f354e9ce8d13fcf48f49091a17b18032bac85b127c20`.

## Aprendizados

As falhas iniciais vieram de inferências não verificadas sobre parser, assinaturas, opções, dependências e `DispatchResult`. A solução foi substituir inferências por contratos canônicos e inspeção comportamental exata. O reparo v10 corrigiu o porcelain, preservou `list[str]` e separou source tree de distribuição isolada.

## Escopo candidato ao commit

Exatamente dez caminhos: dois produtivos e oito artefatos formais. O staging permanece vazio.

## Decisão

A AP-007B está pronta para commit isolado e publicação somente após autorização explícita. Nenhuma tag está autorizada.
