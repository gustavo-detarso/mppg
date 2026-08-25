# MPPG Orchestrator — resilient AI closed loop

Normal operation:

```bash
cd /home/gustavodetarso/Documentos/mppg
mppg-orchestrator run
```

The launcher starts a recovery kernel supervising the main orchestrator. Read-only/action exceptions are fed back inside the core. If the core itself exits with a recoverable harness/internal failure, the kernel captures the structured event, obtains an AI diagnosis with read-only tools, validates a candidate patch in a temporary clone and asks an inline materialization authorization.

The AI cannot self-authorize materialization, USER_ACCEPTANCE, staging, commit or publication. Machine acceptance is `mppg-orchestrator acceptance-test`, a real OpenAI multi-round closed-loop canary with no canonical mutation.
