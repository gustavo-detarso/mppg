# State and Evidence Layout

## Versioned normative layer

`/home/gustavodetarso/Documentos/mppg/governance/`

Contains:

- Prompt Master;
- machine-readable derivative policy;
- schemas;
- contracts;
- templates;
- cryptographic manifests.

## Operational state layer — outside Git

`~/.local/state/mppg-orchestrator/`

Suggested:

- `current/front_authority.json`
- `current/state.json`
- `current/approvals.jsonl`
- `current/blockers.json`
- `evidence/<front-id>/`
- `logs/`

## Runtime/share layer — outside Git

`~/.local/share/mppg-orchestrator/`

Contains future deterministic controller/runtime assets that are not themselves repository authorities.

The repository Prompt Master and `FRONT_AUTHORITY_LEDGER` remain higher authorities than ephemeral runtime state.
