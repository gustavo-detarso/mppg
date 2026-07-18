# Changelog rc10.7 — Conformidade institucional

Esta versão acrescenta uma camada de conformidade institucional auditável sobre a
arquitetura rc10.6.

## Novos comandos

```bash
--explain-profile fgv
--check-institution-compliance
--write-prompt-lock
```

## Novos arquivos

```text
app_bundle/scripts/pipeline/institution_compliance.py
app_bundle/scripts/pipeline/institution_explainer.py
app_bundle/scripts/pipeline/prompt_lock.py
app_bundle/institutions/fgv/validators/paper_rules.toml
app_bundle/institutions/fgv/validators/atividade_rules.toml
app_bundle/institutions/fgv/validators/dissertacao_rules.toml
```

## Novas saídas por execução completa

```text
<prefixo>.prompt_lock.json
<prefixo>.prompt_lock.md
<prefixo>.compliance_report.json
<prefixo>.compliance_report.md
```

## Objetivo

A versão passa a registrar quais prompts/diretivas foram usados e a verificar
artefatos gerados contra o perfil institucional escolhido, especialmente o perfil
FGV 2025.
