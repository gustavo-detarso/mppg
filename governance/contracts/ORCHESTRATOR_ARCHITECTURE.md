# MPPG Governance Orchestrator — Architecture Contract

## Authority

The normative authority is:

`/home/gustavodetarso/Documentos/mppg/governance/MPPG_PROMPT_MASTER_CANONICO.md`

Expected SHA-256 at bootstrap:

`3f3997c771c0b579089598787b9cedd85ba6749a1dab1c3724e212f50052896d`

`policy/MPPG_POLICY_COMPILED.json` is derivative and never overrides the Prompt Master.

## Separation of responsibilities

### Deterministic local controller

The controller owns:

- repository discovery;
- current `master` / tracking / `origin/master` resolution;
- `FRONT_BASELINE_HEAD` freeze;
- gate-state transitions;
- cryptographic tokens;
- exact staging;
- commit execution;
- fast-forward/non-force publication;
- protected-state guards;
- evidence and state persistence;
- authorization enforcement.

### AI semantic agent

The AI may perform:

- semantic adjudication;
- blocker-domain classification;
- root-cause analysis;
- read-only investigation planning;
- repair design;
- contract drafting;
- structured recommendations.

The AI must not receive unrestricted shell control and must not directly execute Git mutations.

## API credential

Use `OPENAI_API_KEY` from the process environment only.

Never persist or print the token in:

- Git;
- governance files;
- state JSON;
- logs;
- evidence bundles;
- prompts;
- stdout/stderr.

A guard may emit only `OPENAI_API_KEY_PRESENT=true|false`.

## Persistent runtime locations

State:

`~/.local/state/mppg-orchestrator/`

Recommended subdirectories:

- `current/`
- `evidence/`
- `logs/`

Runtime/share:

`~/.local/share/mppg-orchestrator/`

The state directories are operational and must not become repository authority.

## Canonical state machine

`INCEPTION_READONLY`
→ `AWAITING_MATERIALIZATION_AUTHORIZATION`
→ `MATERIALIZED_NON_STAGED`
→ `TECHNICAL_VALIDATION`
→ `MACHINE_PRODUCT_ACCEPTANCE` when applicable
→ `AWAITING_USER_ACCEPTANCE` when applicable
→ `STAGING_READY`
→ `AWAITING_STAGING_AUTHORIZATION`
→ `STAGED_EXACT`
→ `AWAITING_COMMIT_AUTHORIZATION`
→ `COMMITTED_ISOLATED`
→ `AWAITING_PUBLICATION_AUTHORIZATION`
→ `PUBLISHED_FAST_FORWARD`
→ `POST_PUBLICATION_CLOSURE`
→ `CLOSED`

Any blocker returns to read-only adjudication. A mutation-required remedy stops at a new explicit authorization gate.

## Harness failure rule

After two consecutive failures attributable exclusively to the auditor/harness, rebuild the harness from the last substantive PASS authorities instead of incremental patching.

## Acceptance-policy guard — mandatory

Before the first mutation, the deterministic controller must:

1. validate `FRONT_CLASS` against the Prompt Master enum;
2. inspect `ARTIFACT_ROLE_MAP`;
3. derive `PRODUCT_ARTIFACT_REQUIRED`;
4. derive `USER_ACCEPTANCE_REQUIRED` fail-closed;
5. freeze the rationale and authority;
6. create `PRODUCT_ACCEPTANCE_CONTRACT` and `USER_ACCEPTANCE_CONTRACT` when applicable.

The controller must not rely on an LLM to decide final gate state. AI may propose
classification, but deterministic policy validation adjudicates whether the
result is structurally admissible.

When `USER_ACCEPTANCE_REQUIRED=true`, the controller must place the user
acceptance gate before staging unless a frozen contract explicitly marks
`USER_ACCEPTANCE_DEFERRED_TO_PUBLISHED_RUNTIME=true`.

`FRONT_CLOSED=true` is always computed from gate state. It must never be a
hardcoded success line in an auditor or shell script.


## Permanent AI orchestrator

Canonical source: `governance/orchestrator/`

Runtime copy: `~/.local/share/mppg-orchestrator/runtime/`

Launcher: `~/.local/bin/mppg-orchestrator`

Routine command: `mppg-orchestrator run`

The deterministic controller owns all mutation gates. AI receives only strict
allowlisted read-only tools and never receives unrestricted shell access.

Harness safety requires subprocess/subshell containment for strict shell options
and forbids sourcing user profiles in the parent interactive shell.
