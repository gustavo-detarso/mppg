#!/usr/bin/env python3
"""Deterministic acceptance/closure policy guard for MPPG fronts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

ALLOWED_FRONT_CLASSES = {
    "functional",
    "structural",
    "documentation",
    "dependency",
    "runtime",
    "persistent_data",
    "security_access",
    "mixed",
    "other",
}

PERCEPTIBLE_ROLE_MARKERS = {
    "normative_authority",
    "documentation_current",
    "operational_contract_documentation",
    "document",
    "pdf",
    "docx",
    "html",
    "dashboard",
    "chart",
    "report",
    "interface",
    "template",
    "state_template",
    "generation_prompt",
    "prompt",
    "ai_content",
    "user_perceptible_output",
    "product_artifact",
}


class PolicyError(ValueError):
    pass


def _flatten_roles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            out.extend(_flatten_roles(k))
            out.extend(_flatten_roles(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_roles(item))
        return out
    return [str(value)]


def _contains_perceptible_role(artifact_role_map: Any) -> bool:
    roles = [x.lower() for x in _flatten_roles(artifact_role_map)]
    for item in roles:
        normalized = item.replace("-", "_").replace(" ", "_")
        if normalized in PERCEPTIBLE_ROLE_MARKERS:
            return True
        if any(marker in normalized for marker in PERCEPTIBLE_ROLE_MARKERS):
            return True
    return False


def derive_acceptance(profile: dict[str, Any]) -> dict[str, Any]:
    front_class = profile.get("front_class")
    if front_class not in ALLOWED_FRONT_CLASSES:
        raise PolicyError(
            f"invalid FRONT_CLASS={front_class!r}; "
            f"allowed={sorted(ALLOWED_FRONT_CLASSES)}"
        )

    perceptible = _contains_perceptible_role(profile.get("artifact_role_map"))
    explicit_user = profile.get("user_acceptance_required")
    explicit_product = profile.get("product_artifact_required")

    exception_justification = (
        profile.get("user_acceptance_exception_justification") or ""
    ).strip()
    exception_authority = (
        profile.get("user_acceptance_exception_authority") or ""
    ).strip()
    exception_source = (
        profile.get("user_acceptance_exception_source") or ""
    ).strip().upper()

    valid_exception = bool(
        exception_justification
        and exception_authority
        and exception_source not in {"AI", "LLM", "ASSISTANT"}
    )

    derived_product = bool(explicit_product) or perceptible

    if explicit_user is True:
        derived_user = True
        rationale = "explicit_true"
    elif perceptible or derived_product:
        if explicit_user is False and valid_exception:
            derived_user = False
            rationale = "explicit_false_with_valid_contractual_exception"
        else:
            derived_user = True
            rationale = "fail_closed_perceptible_artifact"
    else:
        derived_user = bool(explicit_user)
        rationale = "non_perceptible_default"

    return {
        "front_class": front_class,
        "perceptible_artifact_detected": perceptible,
        "product_artifact_required": derived_product,
        "user_acceptance_required": derived_user,
        "user_acceptance_derivation_rationale": rationale,
        "contractual_exception_valid": valid_exception,
    }


def compute_stage_ready(profile: dict[str, Any]) -> dict[str, Any]:
    derived = derive_acceptance(profile)
    technical = profile.get("technical_post_materialization")
    machine = profile.get("machine_product_acceptance")
    user = profile.get("user_acceptance")
    deferred = bool(profile.get("user_acceptance_deferred_to_published_runtime"))

    ready = technical == "PASS" and machine in {"PASS", "NOT_APPLICABLE"}

    if derived["user_acceptance_required"] and not deferred:
        ready = ready and user == "PASS"

    return {
        **derived,
        "staging_ready": bool(ready),
        "user_acceptance_deferred_to_published_runtime": deferred,
    }


def compute_closure(profile: dict[str, Any]) -> dict[str, Any]:
    derived = derive_acceptance(profile)
    technical = bool(profile.get("technical_closure_pass"))
    machine = profile.get("machine_product_acceptance")
    user = profile.get("user_acceptance")
    blockers = int(profile.get("applicable_blockers", 0))

    machine_pass = machine in {"PASS", "NOT_APPLICABLE"}
    user_pass = (
        user == "PASS"
        if derived["user_acceptance_required"]
        else user in {"PASS", "NOT_APPLICABLE"}
    )

    closed = technical and machine_pass and user_pass and blockers == 0

    if derived["user_acceptance_required"] and user != "PASS":
        progress = "AWAITING_USER_ACCEPTANCE"
    elif blockers:
        progress = "BLOCKED"
    elif closed:
        progress = "100_PERCENT"
    else:
        progress = "TECHNICALLY_INCOMPLETE"

    return {
        **derived,
        "technical_closure_pass": technical,
        "machine_product_acceptance": machine,
        "user_acceptance": user,
        "applicable_blockers": blockers,
        "front_progress": progress,
        "front_closed": bool(closed),
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"derive", "stage", "closure"}:
        print(
            "usage: acceptance_guard.py derive|stage|closure <profile.json>",
            file=sys.stderr,
        )
        return 2

    profile = json.load(open(sys.argv[2], "r", encoding="utf-8"))
    try:
        if sys.argv[1] == "derive":
            result = derive_acceptance(profile)
        elif sys.argv[1] == "stage":
            result = compute_stage_ready(profile)
        else:
            result = compute_closure(profile)
    except (PolicyError, ValueError, TypeError) as exc:
        print(json.dumps({
            "policy_guard": "BLOCKED",
            "blocker_domain": "authority_model",
            "reason": str(exc),
        }, ensure_ascii=False, sort_keys=True))
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
