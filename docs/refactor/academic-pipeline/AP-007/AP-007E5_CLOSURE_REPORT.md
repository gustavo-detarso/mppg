# AP-007E.5 — Compatibility, regression and formal closure

Status: **ready_for_isolated_commit_decision**.

## Baseline

- Commit: `766956710435f1c338d2e0332d24e55106b981b7`
- Tree: `1d673e7c324b74f1fef033578aa995e836da1014`
- Branch: `ap-refactor/04-consumer-canonicalization`
- Runtime SHA-256: `b3ef27fb472985634e6b678158a607bcbda3d01d942b767a139048d25369388c`

## Regression result

- Stable post-phase contracts: **16 passed**.
- Integrated census: **748 passed**, **70 exact classified historical/phase-local failures**, and **0 blocking unknown failures**.
- Current productive suite after nominal deselection of the exact classified debt: return code **0**.
- Current productive passed: **748**.
- Frozen xfails: **3**, with the exact expected node IDs.
- Historical deferred packaging tests: **4** executed individually.
- Historical blocking failures: **0**.

## Source/distribution evidence carried forward

- Runtime executions: **30**.
- Runtime comparisons: **24**.
- Divergences: **0**.
- Module hash parity: **True**.
- Critical resource hash parity: **True**.

## Phase-local separation

The validator-execution contract from AP-007E.0 was not replayed because it asserts the original four-path precommit scope. Its stable semantic contracts were executed by exact node ID, while immutable artifact hashes and schemas were revalidated.

## Historical compatibility decision

The integrated census was allowed to return code 1 only because its failed-node set matched the frozen 70-node catalog exactly. Those failures were classified into seven named historical/phase-local categories and assigned to AP-007F. The same suite was then rerun with only those exact node IDs deselected and passed cleanly.

The four packaging/legacy tests were also executed individually. Passing tests are recorded as current contracts; the exact `ModuleNotFoundError` caused by the absent direct-source bridge is classified as non-blocking debt for AP-007F. No unknown node ID, changed category count, changed xfail set, timeout, error, or xpass was accepted.

## Consolidated corrections

- `POST_COMMIT_PHASE_LOCAL_VALIDATORS_NOT_REPLAYED` — revalidate immutable hashes, schemas and stable exact node IDs instead of replaying precommit scope gates
- `VENV_LAUNCHER_PRESERVED_WITH_REAL_EXECUTABLE_SEPARATE` — execute the canonical virtualenv launcher while recording its resolved base interpreter independently
- `HISTORICAL_TESTS_EXECUTED_INDIVIDUALLY` — classify each exact node ID without maxfail or batch masking
- `DIRECT_PEP517_BACKEND_HOOKS_NO_FRONTEND_INSTALL` — use the declared setuptools.build_meta hooks when the optional build frontend is absent
- `TRACKED_RESIDUAL_ARCHIVE_MEMBERS_FILTERED_BEFORE_EXTRACTION` — classify git-archive member names before destination path construction or filesystem extraction
- `METADATA_HEADER_NORMALIZED_TO_JSON_NATIVE_TYPES` — parse email metadata with the default policy and recursively reject non-native JSON values
- `STABLE_POST_PHASE_CONTRACTS_SELECTED_BY_EXACT_NODEID` — exclude only nominal phase-local validator execution tests from later expanded scopes
- `MINIMAL_DEPENDENCY_SYMMETRY_FOR_SOURCE_WHEEL_SDIST` — compare source, wheel and sdist in equivalent dependency-minimal disposable environments
- `NOMINAL_VOLATILE_NORMALIZATION_ONLY` — normalize only documented paths and traceback framing while preserving semantic stdout and stderr
- `JSON_NATIVE_PREFLIGHT_BEFORE_MANIFEST_WRITE` — validate the complete manifest recursively before atomic serialization
- `INTEGRATED_CENSUS_SPLIT_FROM_CURRENT_PRODUCTIVE_REGRESSION` — freeze and classify the exact historical/phase-local failure set before rerunning only the current operational suite

## Commit decision scope

Exactly **24 nonproductive artifacts** compose the AP-007E candidate set. No productive module was edited, and no staging, commit, tag or push was performed. An isolated commit requires explicit user authorization.

## Next phase

AP-007F.0 must audit real `run_legacy` consumers, fallback commands and published legacy wrappers before any removal decision.
