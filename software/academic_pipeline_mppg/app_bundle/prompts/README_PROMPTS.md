# Prompt bank do academic_pipeline

Use esta pasta para guardar diretivas reutilizáveis.

Estrutura recomendada:

- `global/`: diretrizes gerais de comportamento e operação.
- `research/`: geração de queries, triagem, ranking e seleção bibliográfica.
- `document/`: diretrizes por tipo de documento.
- `prisma/`: diretrizes do relatório metodológico PRISMA.

No TOML, ative com:

```toml
[prompts]
ativos = true
global_paths = ["app://prompts/global/orientacao_geral_execucao.txt"]
research_paths = [
  "app://prompts/research/triagem_prompt.txt",
  "app://prompts/research/diretivas_extras.txt"
]
document_paths = []
paper_paths = ["app://prompts/document/paper.txt"]
atividade_paths = ["app://prompts/document/atividade.txt"]
dissertacao_paths = ["app://prompts/document/dissertacao.txt"]
prisma_paths = ["app://prompts/prisma/relatorio_prisma.txt"]
```
