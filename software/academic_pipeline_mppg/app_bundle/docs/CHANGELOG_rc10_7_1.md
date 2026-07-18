# CHANGELOG rc10.7.1 — schema estrito e etapas visíveis

## Correções

- Corrigido erro `Invalid schema for response_format AcademicDocument` no endpoint OpenAI Responses.
- Removido `dict[str, Any]` livre do modelo `AcademicDocument.diagnostics`.
- Adicionado `extra=forbid` nos modelos Pydantic usados em Structured Outputs, garantindo `additionalProperties=false`.
- `diagnostics` agora usa campos JSON serializados (`prompts_json`, `mindmap_json`, `source_info_json`, `relatorio_pesquisa_json`).

## Usabilidade

- Adicionada impressão de etapas em tempo real com prefixo `[ETAPA]` durante execução completa e renderização.
- Etapas exibidas incluem: validação de configuração, extração de documentos, carregamento de orientações, bibliografia, chamada à IA, validações, renderização ORG, PDF, DOCX, conformidade, qualidade e manifestos.
