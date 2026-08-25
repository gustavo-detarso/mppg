#!/usr/bin/env python3
from acceptance_guard import PolicyError, compute_closure, compute_stage_ready, derive_acceptance

CURRENT_GOVERNANCE_FRONT = {
    "front_class": "mixed",
    "artifact_role_map": {
        "governance/MPPG_PROMPT_MASTER_CANONICO.md": "normative_authority",
        "governance/contracts/*": "operational_contract_documentation",
        "governance/templates/*": "state_template",
    },
    "user_acceptance_required": False,
    "product_artifact_required": False,
}

d = derive_acceptance(CURRENT_GOVERNANCE_FRONT)
assert d["product_artifact_required"] is True
assert d["user_acceptance_required"] is True
assert d["user_acceptance_derivation_rationale"] == "fail_closed_perceptible_artifact"

pending = {
    **CURRENT_GOVERNANCE_FRONT,
    "technical_closure_pass": True,
    "machine_product_acceptance": "PASS",
    "user_acceptance": "PENDING",
    "applicable_blockers": 0,
}
c = compute_closure(pending)
assert c["front_closed"] is False
assert c["front_progress"] == "AWAITING_USER_ACCEPTANCE"

approved = {**pending, "user_acceptance": "PASS"}
c = compute_closure(approved)
assert c["front_closed"] is True
assert c["front_progress"] == "100_PERCENT"

stage_pending = {
    **CURRENT_GOVERNANCE_FRONT,
    "technical_post_materialization": "PASS",
    "machine_product_acceptance": "PASS",
    "user_acceptance": "PENDING",
    "user_acceptance_deferred_to_published_runtime": False,
}
assert compute_stage_ready(stage_pending)["staging_ready"] is False

stage_approved = {**stage_pending, "user_acceptance": "PASS"}
assert compute_stage_ready(stage_approved)["staging_ready"] is True

invalid = dict(CURRENT_GOVERNANCE_FRONT)
invalid["front_class"] = "governance"
try:
    derive_acceptance(invalid)
except PolicyError:
    pass
else:
    raise AssertionError("invalid FRONT_CLASS must fail closed")

ai_exception = {
    **CURRENT_GOVERNANCE_FRONT,
    "user_acceptance_required": False,
    "user_acceptance_exception_justification": "claimed exception",
    "user_acceptance_exception_authority": "assistant",
    "user_acceptance_exception_source": "AI",
}
assert derive_acceptance(ai_exception)["user_acceptance_required"] is True

human_exception = {
    **CURRENT_GOVERNANCE_FRONT,
    "user_acceptance_required": False,
    "user_acceptance_exception_justification": "explicit frozen exception",
    "user_acceptance_exception_authority": "user approval event abc123",
    "user_acceptance_exception_source": "USER",
}
assert derive_acceptance(human_exception)["user_acceptance_required"] is False

print("ACCEPTANCE_GUARD_SELF_TEST=PASS")
