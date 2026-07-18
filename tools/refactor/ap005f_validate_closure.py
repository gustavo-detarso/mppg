#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "ap005f.closure-manifest.v1"
PHASE = "AP-005F"
EXPECTED_BRANCH = "ap-refactor/04-consumer-canonicalization"
BASELINE_HEAD = "e5e0d85178d8498c303ad2e8ccc9102f2c8222c8"
UPSTREAM = "origin/ap-refactor/04-consumer-canonicalization"
REMOTE_URL = "git@github.com:gustavo-detarso/mppg.git"

REPORT_REL = (
    "docs/refactor/academic-pipeline/AP-005/"
    "AP-005F_CLOSURE_REPORT.md"
)
MANIFEST_REL = (
    "docs/refactor/academic-pipeline/AP-005/"
    "ap005f_closure_manifest.json"
)
TEST_REL = (
    "software/academic_pipeline_rc10_7_conformidade/"
    "tests/characterization/test_ap005f_closure_contract.py"
)
VALIDATOR_REL = "tools/refactor/ap005f_validate_closure.py"

CLOSURE_ARTIFACTS = [
    REPORT_REL,
    MANIFEST_REL,
    TEST_REL,
    VALIDATOR_REL,
]

PHASE_COMMITS = [
    {
        "hash": "6ef568b250390e12dc2e86b86a8c530188604a28",
        "subject": "refactor(academic-pipeline): inventariar consumidores da AP-005A",
    },
    {
        "hash": "9372de8f621c9012a28d4c4a9a64e252a398bdf3",
        "subject": "refactor(ap005): canonicalize PRISMA consumers",
    },
    {
        "hash": "b8cb7ba3a3175ac79799b78a5d0678224076ef80",
        "subject": "refactor(ap005): canonicalize TOML capture aliases",
    },
    {
        "hash": "78f3be0fce0dd8f79e55729a7111a9359c9edb8d",
        "subject": "fix(ap005): support post-commit validation of AP-005C",
    },
    {
        "hash": "ba28822c826c37022581bf88c6a1b488e2c618de",
        "subject": "docs(ap005): formalize AP-005D facade preservation",
    },
    {
        "hash": "162df76eea94b3a5889ca217a907690f4d62c649",
        "subject": "fix(academic-pipeline): congelar universo histórico da AP-005D",
    },
    {
        "hash": "0d553c975ad7948762f74aa4fcff3903578712de",
        "subject": "chore(academic-pipeline): materializar inventário da AP-005E.1",
    },
    {
        "hash": "b16d1389486f220f829235e87adf88a191cefa87",
        "subject": "test(academic-pipeline): caracterizar instalação isolada AP-005E.2",
    },
    {
        "hash": "71b0c490463edfeb24d6c733ce0a6c698b970510",
        "subject": "fix(academic-pipeline): corrigir instalação distribuída AP-005E.3",
    },
    {
        "hash": BASELINE_HEAD,
        "subject": "chore(academic-pipeline): estabilizar metadados de distribuição AP-005E.4",
    },
]

TRACKED_DOCUMENTATION_BEFORE_CLOSURE = [
    "docs/refactor/academic-pipeline/AP-005/AP-005A_CONSUMER_DEPENDENCY_INVENTORY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005A_CONSUMER_MIGRATION_STRATEGY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005B2_PRISMA_ADAPTER_BATCHES.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005B_CONSUMER_CANONICALIZATION_PLAN.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005C2_STABILIZATION_VALIDATION.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005C_CLOSURE_REPORT.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005C_TOML_CAPTURE_ALIAS_STRATEGY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005D_FACADE_STRATEGY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005E1_INSTALLATION_ENTRYPOINT_INVENTORY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005E1_INSTALLATION_ENTRYPOINT_STRATEGY.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005E2_CORRECTION_SCOPE.md",
    "docs/refactor/academic-pipeline/AP-005/AP-005E2_ISOLATED_BUILD_INSTALLATION_CHARACTERIZATION.md",
    "docs/refactor/academic-pipeline/AP-005/ap005a_consumer_dependency_inventory.json",
    "docs/refactor/academic-pipeline/AP-005/ap005b2_prisma_adapter_batches.json",
    "docs/refactor/academic-pipeline/AP-005/ap005b_consumer_canonicalization_plan.json",
    "docs/refactor/academic-pipeline/AP-005/ap005c2_stabilization_manifest.json",
    "docs/refactor/academic-pipeline/AP-005/ap005c3_closure_manifest.json",
    "docs/refactor/academic-pipeline/AP-005/ap005c_toml_capture_alias_inventory.json",
    "docs/refactor/academic-pipeline/AP-005/ap005d_facade_inventory.json",
    "docs/refactor/academic-pipeline/AP-005/ap005e1_installation_entrypoint_inventory.json",
    "docs/refactor/academic-pipeline/AP-005/ap005e2_isolated_build_installation_characterization.json",
]

CHARACTERIZATION_CONTRACTS_BEFORE_CLOSURE = [
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005a_consumer_dependency_inventory_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005b2_prisma_adapter_batches_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005b2_prisma_adapter_equivalence_characterization.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005b_consumer_canonicalization_plan_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005b_consumer_contract_reclassification.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c1_toml_capture_alias_application_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c2_stabilization_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c3_closure_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c_toml_capture_alias_inventory_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005c_toml_capture_alias_semantics_characterization.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005d_facade_inventory_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005e1_installation_entrypoint_inventory_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005e2_isolated_build_installation_contract.py",
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/test_ap005e3_distribution_corrections_contract.py",
]

KNOWN_XFAILS = [
    {
        "test": (
            "app_bundle/tests/test_article_workflow_characterization.py::"
            "test_refresh_from_files_should_keep_downstream_stages_blocked_after_first_failure"
        ),
        "status": "legacy_defect_catalogued",
    },
    {
        "test": (
            "app_bundle/tests/test_canonical_docx_characterization.py::"
            "test_extract_resumos_should_separate_inline_keywords_from_heading_abstract"
        ),
        "status": "legacy_defect_catalogued",
    },
    {
        "test": (
            "app_bundle/tests/test_rc10_configuration_characterization.py::"
            "test_reference_strip_should_remove_parenthetical_citations"
        ),
        "status": "legacy_defect_catalogued",
    },
]


def run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def canonical_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": PHASE,
        "baseline": {
            "branch": EXPECTED_BRANCH,
            "head": BASELINE_HEAD,
            "upstream": UPSTREAM,
            "origin": REMOTE_URL,
        },
        "scope": {
            "purpose": "stabilization_and_closure",
            "production_code_changes": 0,
            "closure_artifact_count": len(CLOSURE_ARTIFACTS),
        },
        "phase_commits": PHASE_COMMITS,
        "tracked_documentation_before_closure": (
            TRACKED_DOCUMENTATION_BEFORE_CLOSURE
        ),
        "characterization_contracts_before_closure": (
            CHARACTERIZATION_CONTRACTS_BEFORE_CLOSURE
        ),
        "closure_artifacts": CLOSURE_ARTIFACTS,
        "validation": {
            "canonical_suite": {
                "passed": 573,
                "xfailed": 3,
                "returncode": 0,
            },
            "distribution": {
                "package_data_resources": 38,
                "passive_modules": 65,
                "description_content_type": "text/markdown",
                "console_entrypoint": (
                    "academic-pipeline = academic_pipeline.cli:main"
                ),
                "build_warnings_classified": 0,
            },
            "isolated_installation": {
                "pip_check": "passed",
                "console_help": "passed",
                "module_help": "passed",
                "list_institutions": "passed",
                "explain_profile_fgv": "passed",
                "prisma_exportar_bib_help": "passed",
                "prisma_congelar_artigo_help": "passed",
                "passive_import_failures": 0,
            },
        },
        "build_evidence": {
            "wheel_sha256": (
                "2d97c03fa36475d3813497219867d51d58adbd89023d3fafbe82d1623af283c1"
            ),
            "sdist_sha256": (
                "85035062e279334c748cd38c85df4af887de3b2b4e25d594a935721c2052d1db"
            ),
        },
        "known_xfails": KNOWN_XFAILS,
        "closure_decision": (
            "ready_for_explicit_commit_and_publication_approval"
        ),
        "next_phase": {
            "id": "AP-006",
            "reserved_scope": (
                "physical package/directory naming and post-AP-005 evolution"
            ),
        },
    }
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def render_report(payload: dict[str, Any]) -> str:
    commit_lines = "\n".join(
        f"- `{item['hash']}` — {item['subject']}"
        for item in payload["phase_commits"]
    )
    xfail_lines = "\n".join(
        f"- `{item['test']}` — {item['status']}"
        for item in payload["known_xfails"]
    )
    artifact_lines = "\n".join(
        f"- `{path}`" for path in payload["closure_artifacts"]
    )

    validation = payload["validation"]
    suite = validation["canonical_suite"]
    distribution = validation["distribution"]

    return f"""# AP-005F — Relatório de encerramento da AP-005

## 1. Decisão

A AP-005 está tecnicamente estabilizada e pronta para consolidação
explícita. A decisão registrada é
`{payload['closure_decision']}`.

A AP-005F não altera código produtivo. Seu escopo é congelar as
evidências de encerramento, registrar os contratos finais e preparar a
baseline da AP-006.

## 2. Baseline

- Branch: `{payload['baseline']['branch']}`
- HEAD publicado: `{payload['baseline']['head']}`
- Upstream: `{payload['baseline']['upstream']}`
- Origin: `{payload['baseline']['origin']}`
- Fingerprint do manifesto: `{payload['fingerprint']}`

## 3. Resultados consolidados

- Suíte canônica: **{suite['passed']} passed e {suite['xfailed']} xfailed**.
- Recursos não Python no wheel: **{distribution['package_data_resources']}**.
- Módulos passivos instalados: **{distribution['passive_modules']}**.
- Falhas de importação passiva: **0**.
- `pip check`: aprovado.
- Console `academic-pipeline`: aprovado.
- Módulo `python -P -m academic_pipeline`: aprovado.
- Perfil institucional FGV: aprovado.
- Avisos residuais classificados de build: **0**.
- Documentos AP-005 anteriores ao encerramento: **{len(TRACKED_DOCUMENTATION_BEFORE_CLOSURE)}**.
- Contratos AP-005 anteriores ao encerramento: **{len(CHARACTERIZATION_CONTRACTS_BEFORE_CLOSURE)}**.

## 4. Trajetória registrada

{commit_lines}

## 5. Síntese por subfase

- **AP-005A:** inventário de consumidores e estratégia de migração.
- **AP-005B:** canonicalização dos consumidores PRISMA.
- **AP-005C:** migração dos aliases de captura TOML e estabilização.
- **AP-005D:** consolidação das fachadas e preservação explícita da API.
- **AP-005E:** metadados, build, instalação isolada, recursos e entrypoints.
- **AP-005F:** auditoria integrada e encerramento documental/contratual.

## 6. Defeitos legados mantidos como xfail

{xfail_lines}

Esses três defeitos permanecem fora do escopo da AP-005 e não impedem
o encerramento porque estão catalogados e preservados como `xfail`.

## 7. Evidência distributiva

- Wheel SHA-256 observado:
  `{payload['build_evidence']['wheel_sha256']}`
- sdist SHA-256 observado:
  `{payload['build_evidence']['sdist_sha256']}`
- `Description-Content-Type`: `text/markdown`
- Entrypoint:
  `{distribution['console_entrypoint']}`

Os hashes registram a execução de auditoria de encerramento de
18/07/2026. Não constituem requisito de reprodutibilidade byte a byte,
pois os formatos de distribuição podem incorporar metadados temporais.

## 8. Artefatos de encerramento

{artifact_lines}

## 9. Limites e transição

A AP-005 não renomeia fisicamente
`software/academic_pipeline_rc10_7_conformidade`. Essa eventual mudança
permanece reservada para a AP-006.

A baseline para a próxima fase é o estado publicado em
`{payload['baseline']['head']}`, acrescido somente dos quatro artefatos
de encerramento desta AP-005F após aprovação explícita de commit e
publicação.
"""


def validate_repository(root: Path) -> None:
    branch = run_git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"Branch inesperada: {branch}")

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit(
            f"O HEAD atual não descende do baseline {BASELINE_HEAD}."
        )

    for item in PHASE_COMMITS:
        subject = run_git(root, "show", "-s", "--format=%s", item["hash"])
        if subject != item["subject"]:
            raise SystemExit(
                f"Assunto divergente para {item['hash']}: {subject!r}"
            )

    required = (
        TRACKED_DOCUMENTATION_BEFORE_CLOSURE
        + CHARACTERIZATION_CONTRACTS_BEFORE_CLOSURE
        + CLOSURE_ARTIFACTS
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise SystemExit(f"Arquivos obrigatórios ausentes: {missing}")

    status = run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    changed = []
    for line in status.splitlines():
        if not line:
            continue
        changed.append(line[3:])

    unexpected = sorted(set(changed) - set(CLOSURE_ARTIFACTS))
    if unexpected:
        raise SystemExit(
            f"Alterações fora do fechamento AP-005F: {unexpected}"
        )


def write_artifacts(root: Path) -> dict[str, Any]:
    payload = build_payload()
    manifest = root / MANIFEST_REL
    report = root / REPORT_REL

    manifest.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_report(payload), encoding="utf-8")
    return payload


def check_artifacts(root: Path) -> dict[str, Any]:
    payload = build_payload()
    manifest = root / MANIFEST_REL
    report = root / REPORT_REL

    observed_manifest = manifest.read_text(encoding="utf-8")
    expected_manifest = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    if observed_manifest != expected_manifest:
        raise SystemExit("Manifesto AP-005F diverge da forma canônica.")

    observed_report = report.read_text(encoding="utf-8")
    expected_report = render_report(payload)
    if observed_report != expected_report:
        raise SystemExit("Relatório AP-005F diverge da forma canônica.")

    validate_repository(root)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera ou valida o encerramento da AP-005F."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if args.write:
        payload = write_artifacts(root)
        action = "gravados"
    else:
        payload = check_artifacts(root)
        action = "verificados"

    print(f"schema={payload['schema']}")
    print(f"fingerprint={payload['fingerprint']}")
    print(f"commits AP-005={len(payload['phase_commits'])}")
    print(
        "documentos pré-encerramento="
        f"{len(payload['tracked_documentation_before_closure'])}"
    )
    print(
        "contratos pré-encerramento="
        f"{len(payload['characterization_contracts_before_closure'])}"
    )
    print(f"artefatos de encerramento={len(payload['closure_artifacts'])}")
    print(f"decisão={payload['closure_decision']}")
    print(f"arquivos {action}=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
