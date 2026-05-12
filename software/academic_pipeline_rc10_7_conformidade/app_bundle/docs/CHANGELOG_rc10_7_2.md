# CHANGELOG rc10.7.2 — reparo de menções técnicas no document_model

## Correções

- Corrige falso positivo na validação do `document_model` causado por `diagnostics.prompts_json`, que pode conter caminhos internos do projeto.
- A validação de menções técnicas agora considera apenas conteúdo visível do documento acadêmico, ignorando campos de auditoria, hashes e paths.
- Adiciona saneamento automático antes da validação do `document_model`, reescrevendo termos operacionais como `OCR` e `pipeline` quando escaparem para o texto gerado.
- Mantém a validação do ORG renderizado, mas ignora regiões não visíveis, como blocos de comentário.

## Impacto

A execução de atividade/paper/dissertação não deve mais falhar por menções técnicas que apareçam apenas em campos internos de rastreabilidade. Quando a IA deixar termos operacionais no texto visível, o pipeline tentará reescrevê-los antes de validar e registrará o reparo em `run_report.json`.
