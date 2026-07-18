# academic_pipeline rc10.7.5 — validação de vazamentos técnicos

## Correção

- Corrige falso positivo na validação de menções técnicas: a sigla `OCR` agora é detectada apenas como termo isolado, e não dentro de palavras acadêmicas como `democracia`.
- A validação de termos técnicos agora usa fronteiras de palavra/frase após normalização textual.
- Mantém saneamento de termos técnicos visíveis quando eles realmente aparecem no conteúdo.

