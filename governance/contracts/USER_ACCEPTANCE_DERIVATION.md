# User Acceptance Derivation Contract

## Normative purpose

The orchestrator must never treat `USER_ACCEPTANCE_REQUIRED` as a convenient
developer constant. It is a derived front property and must be decided before
the first mutation.

The derivation is fail-closed.

## FRONT_CLASS validation

Only these values are valid:

- `functional`
- `structural`
- `documentation`
- `dependency`
- `runtime`
- `persistent_data`
- `security_access`
- `mixed`
- `other`

Any other value is `BLOCKER_DOMAIN=authority_model`.

A governance front that combines normative documentation, machine-readable
policy, schemas and control contracts should normally use `FRONT_CLASS=mixed`.

## Perceptible artifact rule

If the `ARTIFACT_ROLE_MAP` includes a user-perceptible artifact or a role such as:

- normative authority;
- current documentation;
- operational contract documentation;
- document/PDF/DOCX/HTML;
- dashboard/chart/report/interface;
- template/state template;
- generation prompt;
- AI-produced content;
- user-perceptible output;

then, by default:

```text
PRODUCT_ARTIFACT_REQUIRED=true
USER_ACCEPTANCE_REQUIRED=true
```

A false result is allowed only when a contractual exception is explicit,
justified, frozen before mutation, and supported by a non-AI authority.

## Machine product acceptance

Objective validation may be automated. Examples for governance artifacts:

- Prompt Master SHA-256 exact;
- JSON parse/schema validity;
- manifest integrity;
- required contracts and schemas present;
- no placeholder content;
- internal references resolvable;
- policy evaluator self-tests pass.

`MACHINE_PRODUCT_ACCEPTANCE=PASS` does not imply human acceptance.

## Human acceptance

The AI must never emit `USER_ACCEPTANCE=PASS`.

When required, the user must review a representative real result and explicitly
approve it. The preferred point is before staging.

If pending:

```text
TECHNICAL_GATES=PASS
MACHINE_PRODUCT_ACCEPTANCE=PASS
USER_ACCEPTANCE=PENDING
FRONT_PROGRESS=AWAITING_USER_ACCEPTANCE
FRONT_CLOSED=false
```

## Staging guard

Unless a previously frozen contract explicitly defers acceptance to published
runtime, exact staging is blocked until:

```text
TECHNICAL_POST_MATERIALIZATION=PASS
MACHINE_PRODUCT_ACCEPTANCE=PASS
USER_ACCEPTANCE=PASS
```

## Closure guard

The orchestrator computes, rather than hardcodes, closure:

```text
FRONT_CLOSED =
TECHNICAL_CLOSURE_PASS
AND MACHINE_PRODUCT_ACCEPTANCE_PASS
AND USER_ACCEPTANCE_PASS   # when required
AND ZERO_APPLICABLE_BLOCKERS
```

A closure auditor may never convert `USER_ACCEPTANCE=PENDING` to `PASS`.
