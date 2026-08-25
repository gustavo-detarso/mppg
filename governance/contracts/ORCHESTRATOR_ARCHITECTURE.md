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

## Byte-safe Git I/O and binary academic artifacts

All Git subprocess output that may contain repository paths or academic content
is captured as raw bytes first. String views use UTF-8 with `surrogateescape`;
patch/path identity hashes use the raw Git bytes. JSON that may carry path data
uses ASCII escaping.

Candidate staging isolates both `GIT_INDEX_FILE` and `GIT_OBJECT_DIRECTORY`.
The canonical object database is available only through
`GIT_ALTERNATE_OBJECT_DIRECTORIES`.

The repository-root `.gitattributes` is the canonical Git semantic authority
for binary academic/artifact formats currently present in the ingestion
universe. Those formats are marked `binary`; therefore the mandatory
`git diff --cached --check` remains active without applying textual whitespace
or EOL semantics to PDF, DOCX, ODT, PNG, ZIP, SQLite, BIN and QDA containers.

This is semantic file treatment, not a scanner exclusion. Text files remain
subject to the normal cached diff check.

## External untracked semantic boundary

External untracked content outside `software/academic_pipeline_mppg/` is protected out-of-scope state by default. Discovery is evidence, not ingestion intent. `mppg-orchestrator run` may inventory and fingerprint that content intra-run, but it must not create a repository-content-ingestion front or reach staging merely because such files exist.

When external untracked content is the only repository difference, the controller classifies the state as `external_untracked_preserved`, recomputes its byte-safe fingerprint before closure, reports `EXTERNAL_UNTRACKED_POLICY=PRESERVE_OUT_OF_SCOPE`, and exits without staging, commit or publication.

The historical automatic ingestion path is fail-closed behind `EXPLICIT_INGESTION_FRONT_REQUIRED`. Any future intentional repository-content ingestion requires a separate governed front with explicit semantic scope authority before staging.

## Closed-loop AI supervisor

`mppg-orchestrator run` executes a bounded supervisor: detect blocker → AI adjudication → deterministic allowlisted action → evidence recomputation → AI re-adjudication. Automatic action labels are executable capabilities, not advisory text.

Ephemeral auditor rebuilds are declarative JSON probe plans only. Shadow patches are applied only in temporary clones outside the canonical repository. If a validated shadow patch requires canonical materialization, the controller freezes its patch/scope/target hashes and requests an explicit MATERIALIZATION authorization inline.

No-progress detection terminates fail-closed. Two consecutive harness-only failures force a clean declarative rebuild from the latest substantive PASS. Consumed mutation authorizations are never silently reused.

The normal operator workflow is one terminal session. Routine diagnostic log shuttling to an external chat is not part of the runtime contract.

## Resilient supervisor boundary

The closed loop has two fault-containment layers. Core context/probe/action exceptions are normalized to blockers and fed back to AI. The launcher starts a separate `mppg_recovery_kernel.py`; if the core exits with a recoverable internal/harness event, the kernel performs read-only AI diagnosis, validates any candidate patch in a temporary clone, and requests fresh `KERNEL RECOVERY MATERIALIZATION` authorization before canonical writes. The kernel cannot stage, commit or publish.

Machine acceptance uses `mppg-orchestrator acceptance-test`, a real OpenAI multi-round canary requiring at least two executed automatic probe cycles. Machine-acceptance failure after materialization rolls back before AI re-entry. Gates prefer `/dev/tty`.

## Real-index opacity during read-only phases

Before explicit STAGING authorization, `.git/index` is protected state.
Worktree/status/diff inspection MUST use a disposable copy via `GIT_INDEX_FILE`.
This prevents Git's worktree stat-cache refresh from changing physical index
bytes and from being misclassified as staging.

The real index is touched only by the explicit exact-staging transition.
After staging, cached patch/path/blob identity is the semantic authority.

## Persistent Git proof continuity

The live exact-staging, commit, publication, and closure transitions use one
persistent checkpoint chain rooted at:

`~/.local/share/mppg-ai-supervisor/fronts/<front>/`

Checkpoint records are canonical JSON accompanied by `.sha256` sidecars. The
records bind the front, gate, exact scope, gate token, transition evidence, and
explicit predecessor SHA-256. `STAGING` has no predecessor; `COMMIT` requires
and names `STAGING`; `PUBLICATION` requires and names `COMMIT`; `CLOSURE`
requires and names `PUBLICATION`.

The controller validates the JSON bytes against the sidecar and recursively
validates predecessor continuity before performing the next Git mutation.
Scope, token, ref, cached-patch, or cached-path mismatch fails closed. A valid
staged or committed checkpoint may be recovered after process restart through
the same mutation functions and authorization gates used by the uninterrupted
flow. Checkpoint recovery does not make authorization transitive:
MATERIALIZATION, USER_ACCEPTANCE, STAGING, COMMIT, and PUBLICATION continue to
require their own human authorization.

## Portable host-derived candidate freeze

A candidate cached-patch SHA MUST NOT be transported as a cross-host authority
when the authoritative baseline is a Git commit in the canonical repository.

The portable contract freezes:
- baseline commit/ref identity;
- exact target paths;
- target file SHA-256;
- target file size;
- target Git mode;
- expected presence/absence and mode of baseline paths.

Immediately before MATERIALIZATION, the canonical host constructs a temporary
index/object database, proves every candidate `(mode, blob, path)` against the
portable target authority, requires `git diff --cached --check`, and only then
freezes the candidate patch SHA for the materialization token.

Filesystem permissions produced by ZIP extraction are never an authority.
