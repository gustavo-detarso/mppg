# Authorization Gates

The following authorities are independent and non-transitive:

1. MATERIALIZATION
2. USER_ACCEPTANCE, when required
3. STAGING
4. COMMIT
5. PUBLICATION

Additional explicit mutation gates may be created for:

- `.env`;
- persistent databases;
- runtime/service state;
- security/access control;
- branches/worktrees;
- other protected external state.

A prior authorization never authorizes the next gate.

`USER_ACCEPTANCE=PASS` is not Git authorization.

`STAGING_AUTHORIZED=true` does not imply `COMMIT_AUTHORIZED=true`.

`COMMIT_AUTHORIZED=true` does not imply `PUBLICATION_AUTHORIZED=true`.

Git proof continuity is additionally fail-closed. Successful exact staging creates a
persistent `STAGING` checkpoint; `COMMIT` requires that checkpoint. Successful
commit creates a `COMMIT` checkpoint whose explicit predecessor is the validated
`STAGING` checkpoint; `PUBLICATION` requires the `COMMIT` checkpoint. Successful
publication creates a `PUBLICATION` checkpoint, and closure requires it.

Each checkpoint is stored under
`~/.local/share/mppg-ai-supervisor/fronts/<front>/` as canonical JSON with a
SHA-256 sidecar. Every checkpoint after `STAGING` records the SHA-256 of its
immediate predecessor. A missing, malformed, tampered, wrong-scope, wrong-token,
or discontinuous checkpoint blocks the next transition. Persistence supports
restart recovery but never reuses or implies a human authorization for another
gate. MATERIALIZATION, USER_ACCEPTANCE, STAGING, COMMIT, and PUBLICATION remain
separate human decisions.

Publication must be fast-forward, non-force, and limited to the authorized ref.
