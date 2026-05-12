# rc10.5 — Perfis institucionais

## Novidades

- Nova camada `app_bundle/institutions/<perfil>/`.
- Perfil inicial `fgv` com defaults de templates, LaTeX, DOCX, bibliografia e regras de validação.
- Novo bloco TOML:

```toml
[instituicao]
perfil = "fgv"
```

- Novo comando:

```bash
pipenv run python app_bundle/scripts/pipeline/academic_pipeline_rc10.py --list-institutions
```

- `--init-project` agora aceita `--institution fgv`.
- O `doctor`, `check-config` e `run_report.json` registram o perfil institucional carregado.

## Observação

Os arquivos locais `app_bundle/misc/academic-writing.el` e `app_bundle/misc/fgv.png` continuam externos ao bundle e devem ser copiados pelo usuário.
