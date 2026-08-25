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

Publication must be fast-forward, non-force, and limited to the authorized ref.
