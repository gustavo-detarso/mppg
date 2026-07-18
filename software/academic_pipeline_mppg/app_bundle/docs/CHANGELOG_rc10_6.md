# CHANGELOG rc10.7 — prompt bank

## Adicionado

- `app_bundle/scripts/pipeline/prompt_manager.py`.
- Pasta `app_bundle/prompts/` com prompts globais, de pesquisa, documento e PRISMA.
- Pasta `app_bundle/institutions/fgv/prompts/` para diretivas institucionais da FGV.
- Seção `[prompts]` nos TOMLs de exemplo.
- Comando `--show-prompts`.
- Registro dos prompts ativos no `run_report.json`.
- Validação de caminhos de prompts no `--check-config`.

## Corrigido

- A orientação geral de execução foi saneada para remover a exigência explícita de cadeia de pensamento/CoT, preservando planejamento interno rigoroso e justificativas sintéticas.

## Uso

```toml
[prompts]
ativos = true
global_paths = ["../../prompts/global/orientacao_geral_execucao.txt"]
research_paths = ["../../prompts/research/triagem_prompt.txt", "../../prompts/research/diretivas_extras.txt"]
paper_paths = ["../../prompts/document/paper.txt"]
```
