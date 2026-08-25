# MPPG Governance

This directory is the versionable normative layer for Software MPPG / Academic Pipeline governance.

## Primary authority

`MPPG_PROMPT_MASTER_CANONICO.md`

SHA-256 at bootstrap:

`3f3997c771c0b579089598787b9cedd85ba6749a1dab1c3724e212f50052896d`

The Prompt Master is the first normative authority for every front.

## Directory map

- `MPPG_PROMPT_MASTER_CANONICO.md` — normative authority.
- `policy/MPPG_POLICY_COMPILED.json` — machine-readable derivative; never overrides the Master.
- `schemas/` — schemas for the future orchestrator's ledgers, approvals and AI adjudications.
- `contracts/` — operational architecture and gate contracts.
- `templates/` — initial state templates.
- `MANIFEST.sha256` — byte-level integrity of this governance package.

## External operational state

The bootstrap also prepares, without putting them in Git:

- `~/.local/state/mppg-orchestrator/current/`
- `~/.local/state/mppg-orchestrator/evidence/`
- `~/.local/state/mppg-orchestrator/logs/`
- `~/.local/share/mppg-orchestrator/`

## API key

The future orchestrator must read `OPENAI_API_KEY` only from the environment.

No governance artifact may contain the API token.

## Git governance

Materializing this directory does not authorize staging, commit or publication. Those remain separate gates.

## Acceptance guard

`validators/acceptance_guard.py` is a deterministic policy guard for the future
orchestrator. It validates `FRONT_CLASS`, derives product/user acceptance
requirements fail-closed, blocks staging while required user acceptance is
pending, and computes closure from gate state rather than hardcoded success.

See `contracts/USER_ACCEPTANCE_DERIVATION.md`.

## Permanent AI orchestrator

Routine command after closure: `mppg-orchestrator run`.
