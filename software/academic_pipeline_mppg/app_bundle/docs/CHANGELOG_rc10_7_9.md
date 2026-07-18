# rc10.7.9 — ORG validator visual fix

- Corrige falso positivo de vazamento técnico em validação do ORG renderizado.
- A validação de menções técnicas agora ignora diretivas Org, comentários LaTeX, headers e caminhos de arquivos, mantendo a validação de citações sobre o ORG integral.
- Remove a palavra interna "pipeline" de comentários não visíveis no `academic-writing.el`.
- Mantém o visual de atividade FGV introduzido na rc10.7.8.
