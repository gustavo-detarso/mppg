# Blocker and AI Adjudication Policy

Every blocker must be classified into a `BLOCKER_DOMAIN`.

The AI is permitted to continue automatically when the next actions are read-only.

Examples:

- auditor/harness bug → diagnose, rebuild auditor, self-test, rerun read-only;
- evidence packaging issue → repair evidence packaging read-only;
- scanner-model defect → redesign scanner read-only;
- authority-model ambiguity → reconstruct authority ledger read-only.

If resolution requires real mutation, the AI may design and validate the proposed remedy in a safe/disposable environment, freeze the remediation contract, and must then stop for explicit authorization.

The AI must never modify software merely to make a scanner green.

`USER_ACCEPTANCE` cannot be self-issued by AI.
