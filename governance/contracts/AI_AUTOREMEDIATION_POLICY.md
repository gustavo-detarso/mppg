# AI Auto-Remediation Policy

Prompt Master SHA-256: `3f3997c771c0b579089598787b9cedd85ba6749a1dab1c3724e212f50052896d`

AI semantic blocker adjudication is mandatory when deterministic evidence alone
cannot close a blocker.

Automatic actions are limited to:

```text
NOOP
RERUN_READONLY_PROBE
RECLASSIFY_FROM_EVIDENCE
REBUILD_EPHEMERAL_AUDITOR
RECOMPUTE_EPHEMERAL_EVIDENCE
RETRY_EXTERNAL_READONLY
SIMULATE_PATCH_IN_SHADOW
```

All automatic actions are read-only or ephemeral/shadow only.

Explicit authorization remains mandatory for canonical materialization,
persistent environment/profile mutation, dependency installation/upgrades,
runtime-service mutation, staging, commit and publication.

USER_ACCEPTANCE can never be auto-passed.

Harness safety:

```text
HOST_INTERACTIVE_SHELL_OPTION_MUTATION=PROHIBITED
SHELL_STRICT_MODE=SUBPROCESS_OR_SUBSHELL_ONLY
SOURCE_USER_PROFILE_IN_PARENT_SHELL=PROHIBITED
PROFILE_VALIDATION=CHILD_SHELL_ONLY
HARNESS_REBUILD_AFTER_TWO_HARNESS_ONLY_FAILURES=true
```

OpenAI:
- endpoint: `https://api.openai.com/v1/responses`
- default model: `gpt-5.6-luna`
- credential: `OPENAI_API_KEY` from the process environment only
- strict function schemas
- strict final JSON schema
- `parallel_tool_calls=false`
- `store=false`
- no unrestricted shell tool
- no direct Git mutation by AI
- API key never printed, persisted or included in request body

## Executable closed-loop policy

The controller independently executes and validates allowlisted read-only/ephemeral actions. Model assertions cannot suppress a physically live repository-state blocker, waive a failed diff-check, authorize canonical mutation, approve USER_ACCEPTANCE, expand mutation scope, or reuse a consumed gate.

A model-proposed shadow patch is non-authoritative until the controller validates exact scope, protected-path denial, patch application in a temporary clone, configured validation probes, and byte hashes. Canonical application requires an explicit MATERIALIZATION token.

## Crash-resilient recovery

Core read-only exceptions are promoted to blockers. Core process failures are supervised by the separate recovery kernel. AI recovery patches remain shadow-only until a fresh explicit materialization authorization. The kernel has no staging, commit or publication authority.
