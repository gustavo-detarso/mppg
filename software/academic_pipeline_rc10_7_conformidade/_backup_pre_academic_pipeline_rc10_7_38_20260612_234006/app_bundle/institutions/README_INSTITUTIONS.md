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
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --list-institutions
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --explain-profile fgv
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --config caminho.toml --check-institution-compliance
```
