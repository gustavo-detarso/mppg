# Perfis institucionais

A pasta `app_bundle/institutions/` contém perfis institucionais reutilizáveis.
Cada perfil pode definir templates, estilos LaTeX, modelos DOCX, assets, prompts e
regras de validação institucional.

## Como ativar no TOML

```toml
[instituicao]
perfil = "fgv"
```

## Comandos úteis

```bash
pipenv run python -m academic_pipeline --list-institutions
pipenv run python -m academic_pipeline --explain-profile fgv
pipenv run python -m academic_pipeline --config caminho.toml --check-institution-compliance
```
