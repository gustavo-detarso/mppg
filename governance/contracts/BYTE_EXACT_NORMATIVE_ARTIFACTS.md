# Byte-Exact Normative Artifacts

## Scope

`MPPG_PROMPT_MASTER_CANONICO.md` is a byte-exact normative authority.

Expected SHA-256:

`3f3997c771c0b579089598787b9cedd85ba6749a1dab1c3724e212f50052896d`

## Git whitespace adjudication

The Prompt Master intentionally contains Markdown hard line breaks represented
by trailing spaces. Those bytes are part of the frozen normative authority and
must not be rewritten merely to satisfy a generic whitespace checker.

Therefore `governance/.gitattributes` applies the narrowly scoped rule:

`MPPG_PROMPT_MASTER_CANONICO.md -diff`

This is not a generic scanner exclusion. It applies to one named immutable
authority so that Git treats it as opaque for textual diff/whitespace analysis.

`git diff --cached --check` remains mandatory for the governance staging gate
and continues to apply to every other staged governance artifact.

## Prohibited interpretations

This rule does not authorize:

- broad `*.md -diff`;
- broad `governance/** -diff`;
- disabling `git diff --check`;
- rewriting the Prompt Master to remove intentional bytes;
- suppressing findings in other files.

Any future change to this exception requires a separate semantic adjudication.
