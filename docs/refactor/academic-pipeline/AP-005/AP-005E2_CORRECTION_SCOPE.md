# AP-005E.2 — Escopo vinculante da correção AP-005E.3

## Princípio

Corrigir apenas defeitos demonstrados na instalação isolada, preservando nome, versão, entrypoints públicos e bridge legado.

## Correções obrigatórias

- PEP 621 runtime dependencies.
- operational package-data allowlist.
- article_workflow import correction.
- PRISMA helper resolution correction.
- hardcoded prompt path correction.
- fresh-wheel regression gates.

## Correções condicionais

- self-invocation by __file__ correction.
- requirements/Pipfile normalization beyond runtime essentials.

## Proibições

- rename distribution.
- change version.
- remove legacy bridge.
- change public entrypoints.
- package app_bundle/projetos.
- package app_bundle/output.
- broad module reorganization.

## Package data

A correção deve usar uma allowlist operacional mínima. Não é permitido incluir os 184 arquivos não-Python em bloco. `app_bundle/projetos`, `app_bundle/output`, históricos e documentos de desenvolvimento permanecem fora do artefato.

O gate mínimo deve comprovar em wheel novo:

- `--list-institutions` encontra `fgv`;
- `--explain-profile fgv` retorna com sucesso;
- `--init-project` cria projeto em diretório externo;
- templates, prompts, perfis e assets necessários vêm do venv instalado.

## Dependências

O wheel deve declarar as dependências necessárias para que seu entrypoint público execute após `pip install` normal. `pip check` sozinho não é suficiente: os entrypoints e módulos operacionais devem ser exercitados em venv novo.

## Gate de encerramento da AP-005E.3

- wheel e sdist construídos em clone limpo;
- instalação normal do wheel, sem requirements externo;
- `pip check` aprovado;
- ambos os entrypoints aprovados fora do checkout;
- 65 módulos passivos importáveis ou exclusão justificada;
- comandos institucionais e `--init-project` aprovados;
- helpers PRISMA resolvidos no layout instalado;
- ausência de caminho pessoal em código produtivo;
- suíte canônica integral aprovada.
