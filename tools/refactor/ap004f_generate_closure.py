#!/usr/bin/env python3
"""Gerador de encerramento da AP-004F do Academic Pipeline.

Este utilitário valida a cadeia AP-004A–E, executa os gates funcionais finais,
avalia a prontidão de integração sem criar merge/rebase/cherry-pick e gera os
artefatos documentais e contratuais da AP-004F.

Não altera código produtivo, não cria commit, não realiza push e não integra
branches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "ap004f.closure-manifest.v1"
EXPECTED_REPOSITORY = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline"
)
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
EXPECTED_HEAD = "b5f924ae2b55c961f251a8d65f3405eb3cea35b8"
EXPECTED_HEAD_SUBJECT = (
    "refactor(academic-pipeline): consolidar compatibilidades da AP-004E"
)
REMOTE_REF = "origin/ap-refactor/03-orchestrator-decomposition"
INTEGRATION_REF = "origin/refactor/academic-pipeline"

SOFTWARE_REL = Path("software/academic_pipeline_mppg")
TOP_AP004_DOCS_REL = Path("docs/refactor/academic-pipeline/AP-004")
SOFTWARE_AP004_DOCS_REL = SOFTWARE_REL / "docs/refactor/academic-pipeline/AP-004"
TOOL_REL = Path("tools/refactor/ap004f_generate_closure.py")

FINAL_VALIDATION_REL = TOP_AP004_DOCS_REL / "AP-004F_FINAL_VALIDATION.md"
CLOSURE_REPORT_REL = TOP_AP004_DOCS_REL / "AP-004F_CLOSURE_REPORT.md"
INTEGRATION_DECISION_REL = TOP_AP004_DOCS_REL / "AP-004F_INTEGRATION_DECISION.md"
MANIFEST_REL = TOP_AP004_DOCS_REL / "ap004f_closure_manifest.json"
TEST_REL = SOFTWARE_REL / "tests/characterization/test_ap004f_closure_contract.py"
OUTPUT_RELS = (
    FINAL_VALIDATION_REL,
    CLOSURE_REPORT_REL,
    INTEGRATION_DECISION_REL,
    MANIFEST_REL,
    TEST_REL,
)

NAMING_CONVENTION_REL = (
    SOFTWARE_AP004_DOCS_REL / "AP-004_NAMING_CONVENTION.md"
)
AP004D_INVENTORY_REL = TOP_AP004_DOCS_REL / "ap004d_version_marker_inventory.json"
AP004E_INVENTORY_REL = TOP_AP004_DOCS_REL / "ap004e_compatibility_inventory.json"
AP004E_TEST_REL = (
    SOFTWARE_REL
    / "tests/characterization/test_ap004e_compatibility_inventory_contract.py"
)
AP004D_TEST_REL = (
    SOFTWARE_REL
    / "tests/characterization/test_ap004d_version_marker_inventory_contract.py"
)

AP004_COMMITS = (
    (
        "AP-004A",
        "6de61fc9741035187836460d97da6d672708998a",
        "chore(academic-pipeline): consolidar inventário de nomes da AP-004A",
    ),
    (
        "AP-004B",
        "aa9829f09a5c1b9e69c634637c311b03f360b07e",
        "refactor(academic-pipeline): consolidar módulos e arquivos da AP-004B",
    ),
    (
        "AP-004C",
        "81293d79e86da8b4d0407b483fc3dedaf27768cb",
        "refactor(academic-pipeline): consolidar símbolos internos da AP-004C",
    ),
    (
        "AP-004D",
        "389f0ae526d12327a58ce23937225cf05b032566",
        "refactor(academic-pipeline): consolidar marcadores de versão da AP-004D",
    ),
    (
        "AP-004E",
        "b5f924ae2b55c961f251a8d65f3405eb3cea35b8",
        "refactor(academic-pipeline): consolidar compatibilidades da AP-004E",
    ),
)
AP004_BASE_PARENT = AP004_COMMITS[0][1] + "^"

EXPECTED_AP004E_FILES = {
    "docs/refactor/academic-pipeline/AP-004/AP-004E_COMPATIBILITY_INVENTORY.md",
    "docs/refactor/academic-pipeline/AP-004/AP-004E_COMPATIBILITY_STRATEGY.md",
    "docs/refactor/academic-pipeline/AP-004/ap004e_compatibility_inventory.json",
    "software/academic_pipeline_mppg/tests/characterization/"
    "test_ap004e_compatibility_inventory_contract.py",
    "tools/refactor/ap004e_inventory_compatibility.py",
}

EXPECTED_AP004E_SCHEMA = "ap004e.compatibility-inventory.v6"
EXPECTED_AP004E_FINGERPRINT = (
    "cee4120c2602bb12e78fe7d41cf22fc261b8a64647c2c2b9d6e256903d5574e3"
)
EXPECTED_AP004E_DECISIONS = {
    "preservar": 22,
    "preservar congelado": 2,
    "preservar ou migrar consumidores antes": 38,
    "preservar sem alteração": 2,
}
EXPECTED_XFAIL_NODEIDS = (
    "app_bundle/tests/test_article_workflow_characterization.py::"
    "test_refresh_from_files_should_keep_downstream_stages_blocked_after_first_failure",
    "app_bundle/tests/test_canonical_docx_characterization.py::"
    "test_extract_resumos_should_separate_inline_keywords_from_heading_abstract",
    "app_bundle/tests/test_rc10_configuration_characterization.py::"
    "test_reference_strip_should_remove_parenthetical_citations",
)

COMPILE_RELS = (
    SOFTWARE_REL / "academic_pipeline/cli_parser.py",
    SOFTWARE_REL / "academic_pipeline/command_dispatch.py",
    SOFTWARE_REL / "academic_pipeline/document_orchestration.py",
    SOFTWARE_REL / "academic_pipeline/prisma_generic_orchestration.py",
    SOFTWARE_REL / "academic_pipeline/__main__.py",
    SOFTWARE_REL / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    SOFTWARE_REL
    / "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py",
)

ALLOWED_DIRTY_RELS = frozenset(
    {
        NAMING_CONVENTION_REL.as_posix(),
        TOOL_REL.as_posix(),
        *(path.as_posix() for path in OUTPUT_RELS),
    }
)


class ClosureError(RuntimeError):
    """Falha segura de validação da AP-004F."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PytestSummary:
    passed: int
    xfailed: int
    xpassed: int
    failed: int
    errors: int
    skipped: int
    duration_seconds: float | None
    short_summary_lines: tuple[str, ...]


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    echo: bool = False,
) -> CommandResult:
    if echo:
        print("$", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    result = CommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if echo:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(
                result.stderr,
                end="" if result.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise ClosureError(
            f"comando falhou ({result.returncode}): {' '.join(command)}"
            + (f"\n{details}" if details else "")
        )
    return result


def git(
    repo_root: Path,
    *args: str,
    check: bool = True,
    echo: bool = False,
) -> CommandResult:
    return run_command(
        ("git", *args),
        cwd=repo_root,
        check=check,
        echo=echo,
    )


def ensure_file(path: Path) -> None:
    if not path.is_file():
        raise ClosureError(f"arquivo obrigatório ausente: {path}")
    if path.stat().st_size == 0:
        raise ClosureError(f"arquivo obrigatório vazio: {path}")


def read_json(path: Path) -> dict[str, Any]:
    ensure_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"JSON inválido em {path}: {exc}") from exc


def git_status_paths(repo_root: Path) -> dict[str, str]:
    result = git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    statuses: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path_part = raw_line[3:]
        if " -> " in path_part:
            _, path_part = path_part.split(" -> ", 1)
        statuses[path_part] = status
    return statuses


def validate_allowed_dirty_tree(repo_root: Path) -> dict[str, str]:
    statuses = git_status_paths(repo_root)
    unexpected = sorted(set(statuses) - ALLOWED_DIRTY_RELS)
    if unexpected:
        details = "\n".join(f"- {statuses[path]} {path}" for path in unexpected)
        raise ClosureError(
            "a árvore contém alterações fora do conjunto permitido da AP-004F:\n"
            + details
        )
    naming_status = statuses.get(NAMING_CONVENTION_REL.as_posix())
    if naming_status not in {" M", "M ", "MM"}:
        raise ClosureError(
            "o saneamento de EOF esperado não está presente em "
            f"{NAMING_CONVENTION_REL}; status observado: {naming_status!r}"
        )
    return statuses


def validate_naming_eof_fix(repo_root: Path) -> dict[str, Any]:
    rel = NAMING_CONVENTION_REL.as_posix()
    path = repo_root / NAMING_CONVENTION_REL
    ensure_file(path)
    current = path.read_bytes()
    head = git(repo_root, "show", f"HEAD:{rel}").stdout.encode("utf-8")
    expected = head.rstrip(b" \t\r\n") + b"\n"
    if current != expected:
        raise ClosureError(
            "a alteração de AP-004_NAMING_CONVENTION.md não corresponde "
            "exclusivamente à normalização do EOF"
        )
    if current == head:
        raise ClosureError("o saneamento documental não produziu diferença contra HEAD")
    return {
        "path": rel,
        "head_bytes": len(head),
        "working_tree_bytes": len(current),
        "bytes_removed": len(head) - len(current),
        "change": "remoção de linha vazia excedente no EOF",
    }


def validate_git_worktree(repo_root: Path) -> dict[str, str]:
    """Valida a estrutura Git sem assumir que ``.git`` seja diretório.

    Em um linked worktree, ``.git`` é um arquivo que aponta para o diretório
    administrativo real. Os comandos ``git rev-parse`` são a fonte canônica
    para ambos os formatos.
    """
    inside = git(repo_root, "rev-parse", "--is-inside-work-tree").stdout.strip()
    if inside != "true":
        raise ClosureError(f"o caminho não é um worktree Git válido: {repo_root}")

    git_dir_text = git(repo_root, "rev-parse", "--git-dir").stdout.strip()
    common_dir_text = git(repo_root, "rev-parse", "--git-common-dir").stdout.strip()

    def resolve_git_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    git_dir = resolve_git_path(git_dir_text)
    common_dir = resolve_git_path(common_dir_text)

    if not git_dir.is_dir():
        raise ClosureError(f"diretório administrativo Git ausente: {git_dir}")
    if not common_dir.is_dir():
        raise ClosureError(f"diretório Git comum ausente: {common_dir}")

    # Funciona tanto com HEAD físico no git-dir quanto com referências simbólicas.
    git(repo_root, "rev-parse", "--verify", "HEAD")

    return {
        "git_dir": str(git_dir),
        "git_common_dir": str(common_dir),
        "worktree_format": (
            "linked_worktree"
            if (repo_root / ".git").is_file()
            else "standard_repository"
        ),
    }


def validate_repository(repo_root: Path, *, fetch: bool) -> dict[str, Any]:
    if repo_root != EXPECTED_REPOSITORY.resolve():
        raise ClosureError(
            f"repositório inesperado: {repo_root}; esperado: {EXPECTED_REPOSITORY}"
        )
    git_layout = validate_git_worktree(repo_root)
    if fetch:
        print("=== FETCH ORIGIN ===", flush=True)
        git(repo_root, "fetch", "origin", echo=True)

    branch = git(repo_root, "branch", "--show-current").stdout.strip()
    head = git(repo_root, "rev-parse", "HEAD").stdout.strip()
    subject = git(repo_root, "show", "-s", "--format=%s", "HEAD").stdout.strip()
    remote_head = git(repo_root, "rev-parse", REMOTE_REF).stdout.strip()
    divergence_text = git(
        repo_root,
        "rev-list",
        "--left-right",
        "--count",
        f"HEAD...{REMOTE_REF}",
    ).stdout.strip()
    divergence = [int(part) for part in divergence_text.split()]

    if branch != EXPECTED_BRANCH:
        raise ClosureError(f"branch incorreta: {branch}; esperada: {EXPECTED_BRANCH}")
    if head != EXPECTED_HEAD:
        raise ClosureError(f"HEAD incorreto: {head}; esperado: {EXPECTED_HEAD}")
    if subject != EXPECTED_HEAD_SUBJECT:
        raise ClosureError(
            f"mensagem do HEAD incorreta: {subject!r}; esperada: {EXPECTED_HEAD_SUBJECT!r}"
        )
    if remote_head != EXPECTED_HEAD:
        raise ClosureError(
            f"HEAD remoto incorreto: {remote_head}; esperado: {EXPECTED_HEAD}"
        )
    if divergence != [0, 0]:
        raise ClosureError(f"divergência local/remoto inesperada: {divergence}")

    statuses = validate_allowed_dirty_tree(repo_root)
    eof_fix = validate_naming_eof_fix(repo_root)

    return {
        "git_layout": git_layout,
        "branch": branch,
        "head": head,
        "head_subject": subject,
        "remote_ref": REMOTE_REF,
        "remote_head": remote_head,
        "divergence": divergence,
        "dirty_paths": [
            {"path": path, "status": status}
            for path, status in sorted(statuses.items())
        ],
        "expected_documental_fix": eof_fix,
    }


def validate_commits(repo_root: Path) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for phase, commit, expected_subject in AP004_COMMITS:
        exists = git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
        if exists.returncode != 0:
            raise ClosureError(f"commit ausente para {phase}: {commit}")
        actual_subject = git(
            repo_root, "show", "-s", "--format=%s", commit
        ).stdout.strip()
        committed_at = git(
            repo_root, "show", "-s", "--format=%cI", commit
        ).stdout.strip()
        if actual_subject != expected_subject:
            raise ClosureError(
                f"mensagem divergente em {phase}: {actual_subject!r}; "
                f"esperada: {expected_subject!r}"
            )
        ancestor = git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
            check=False,
        ).returncode == 0
        if not ancestor:
            raise ClosureError(f"{phase} ({commit}) não é ancestral de HEAD")
        if previous_hash is not None:
            ordered = git(
                repo_root,
                "merge-base",
                "--is-ancestor",
                previous_hash,
                commit,
                check=False,
            ).returncode == 0
            if not ordered:
                raise ClosureError(
                    f"ordem histórica inválida: {previous_hash} não é ancestral de {commit}"
                )
        phases.append(
            {
                "phase": phase,
                "commit": commit,
                "subject": actual_subject,
                "committed_at": committed_at,
                "ancestor_of_head": True,
            }
        )
        previous_hash = commit
    return phases


def validate_ap004e_commit(repo_root: Path) -> dict[str, Any]:
    previous = AP004_COMMITS[-2][1]
    current = AP004_COMMITS[-1][1]
    paths = {
        line.strip()
        for line in git(
            repo_root, "diff", "--name-only", f"{previous}..{current}"
        ).stdout.splitlines()
        if line.strip()
    }
    unexpected = sorted(paths - EXPECTED_AP004E_FILES)
    missing = sorted(EXPECTED_AP004E_FILES - paths)
    if unexpected or missing:
        raise ClosureError(
            "o commit AP-004E não contém exatamente os cinco artefatos autorizados; "
            f"inesperados={unexpected}; ausentes={missing}"
        )
    return {
        "previous_commit": previous,
        "commit": current,
        "files": sorted(paths),
        "productive_code_changed": False,
    }


def validate_ap004e_inventory(repo_root: Path) -> dict[str, Any]:
    data = read_json(repo_root / AP004E_INVENTORY_REL)
    summary = data.get("summary", {})
    gate = data.get("gate", {})
    if data.get("schema_version") != EXPECTED_AP004E_SCHEMA:
        raise ClosureError(
            f"schema AP-004E divergente: {data.get('schema_version')!r}"
        )
    if data.get("contract_fingerprint") != EXPECTED_AP004E_FINGERPRINT:
        raise ClosureError("fingerprint AP-004E divergente")
    expected_values = {
        "item_count": 64,
        "manual_decision_items": 0,
        "removal_candidates": 0,
        "blocked_items": 0,
        "syntax_errors": 0,
    }
    for key, expected in expected_values.items():
        if summary.get(key) != expected:
            raise ClosureError(
                f"resumo AP-004E divergente em {key}: {summary.get(key)!r}; "
                f"esperado: {expected!r}"
            )
    if summary.get("decision_counts") != EXPECTED_AP004E_DECISIONS:
        raise ClosureError(
            "decisões AP-004E divergentes: "
            f"{summary.get('decision_counts')!r}"
        )
    for key in (
        "productive_applicator_allowed",
        "productive_changes_allowed",
        "commit_allowed",
        "push_allowed",
        "integration_allowed",
    ):
        if gate.get(key) is not False:
            raise ClosureError(f"gate AP-004E não está bloqueado em {key}")
    return {
        "schema_version": data["schema_version"],
        "contract_fingerprint": data["contract_fingerprint"],
        "summary": summary,
        "conclusion": (
            "64 superfícies preservadas ou dependentes de migração prévia; "
            "nenhum candidato seguro à remoção; aplicador produtivo não necessário"
        ),
    }


def parse_numstat(text: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    inserted = 0
    deleted = 0
    binary = 0
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        add_text, delete_text, path = parts
        if add_text == "-" or delete_text == "-":
            binary += 1
            files.append({"path": path, "insertions": None, "deletions": None})
            continue
        add = int(add_text)
        delete = int(delete_text)
        inserted += add
        deleted += delete
        files.append({"path": path, "insertions": add, "deletions": delete})
    return {
        "file_count": len(files),
        "insertions": inserted,
        "deletions": deleted,
        "binary_files": binary,
        "files": files,
    }


def validate_consolidated_diff(repo_root: Path) -> dict[str, Any]:
    check = git(repo_root, "diff", "--check", AP004_BASE_PARENT, check=False)
    if check.returncode != 0 or check.stdout.strip() or check.stderr.strip():
        raise ClosureError(
            "git diff --check consolidado falhou:\n"
            + (check.stdout + check.stderr).strip()
        )
    numstat = git(
        repo_root, "diff", "--numstat", AP004_BASE_PARENT
    ).stdout
    name_status = [
        line
        for line in git(
            repo_root, "diff", "--name-status", AP004_BASE_PARENT
        ).stdout.splitlines()
        if line.strip()
    ]
    summary = parse_numstat(numstat)
    summary["name_status"] = name_status
    summary["diff_check_clean"] = True
    return summary


def parse_pytest_summary(output: str) -> PytestSummary:
    patterns = {
        "passed": r"(\d+) passed",
        "xfailed": r"(\d+) xfailed",
        "xpassed": r"(\d+) xpassed",
        "failed": r"(\d+) failed",
        "errors": r"(\d+) errors?",
        "skipped": r"(\d+) skipped",
    }
    values: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        values[key] = int(match.group(1)) if match else 0
    duration_match = re.search(r"in\s+([0-9.]+)s", output)
    short_lines = tuple(
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(("XFAIL ", "XPASS ", "FAILED ", "ERROR "))
    )
    return PytestSummary(
        passed=values["passed"],
        xfailed=values["xfailed"],
        xpassed=values["xpassed"],
        failed=values["failed"],
        errors=values["errors"],
        skipped=values["skipped"],
        duration_seconds=(
            float(duration_match.group(1)) if duration_match else None
        ),
        short_summary_lines=short_lines,
    )


def pytest_summary_dict(summary: PytestSummary) -> dict[str, Any]:
    return {
        "passed": summary.passed,
        "xfailed": summary.xfailed,
        "xpassed": summary.xpassed,
        "failed": summary.failed,
        "errors": summary.errors,
        "skipped": summary.skipped,
        "duration_seconds": summary.duration_seconds,
        "short_summary_lines": list(summary.short_summary_lines),
    }


def run_final_validations(repo_root: Path) -> dict[str, Any]:
    software_root = repo_root / SOFTWARE_REL
    print("=== PY_COMPILE AP-004 ===", flush=True)
    compile_command = [
        sys.executable,
        "-m",
        "py_compile",
        *[
            str(path.relative_to(SOFTWARE_REL))
            for path in COMPILE_RELS
        ],
    ]
    run_command(compile_command, cwd=software_root, echo=True)

    print("=== CONTRATOS AP-004D/AP-004E ===", flush=True)
    contracts = run_command(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-ra",
            str(AP004D_TEST_REL.relative_to(SOFTWARE_REL)),
            str(AP004E_TEST_REL.relative_to(SOFTWARE_REL)),
        ),
        cwd=software_root,
        echo=True,
    )
    contract_summary = parse_pytest_summary(contracts.stdout + contracts.stderr)
    if (
        contract_summary.passed != 7
        or contract_summary.failed
        or contract_summary.errors
        or contract_summary.xpassed
    ):
        raise ClosureError(
            "contratos AP-004D/AP-004E divergentes do esperado: "
            f"{pytest_summary_dict(contract_summary)}"
        )

    print("=== SUÍTE CANÔNICA PRÉ-AP-004F ===", flush=True)
    suite = run_command(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-ra",
            "app_bundle/tests",
            "tests",
        ),
        cwd=software_root,
        echo=True,
    )
    suite_output = suite.stdout + suite.stderr
    suite_summary = parse_pytest_summary(suite_output)
    if (
        suite_summary.passed != 489
        or suite_summary.xfailed != 3
        or suite_summary.xpassed != 0
        or suite_summary.failed != 0
        or suite_summary.errors != 0
    ):
        raise ClosureError(
            "suíte canônica divergente do esperado antes da AP-004F: "
            f"{pytest_summary_dict(suite_summary)}"
        )
    found_xfails = {
        line.split(" - ", 1)[0].removeprefix("XFAIL ")
        for line in suite_summary.short_summary_lines
        if line.startswith("XFAIL ")
    }
    if found_xfails != set(EXPECTED_XFAIL_NODEIDS):
        raise ClosureError(
            "conjunto de xfail divergente; "
            f"encontrado={sorted(found_xfails)}; "
            f"esperado={sorted(EXPECTED_XFAIL_NODEIDS)}"
        )
    return {
        "py_compile": {
            "status": "passed",
            "module_count": len(COMPILE_RELS),
            "modules": [path.as_posix() for path in COMPILE_RELS],
        },
        "contract_tests": pytest_summary_dict(contract_summary),
        "canonical_suite_before_ap004f_contract": pytest_summary_dict(
            suite_summary
        ),
        "expected_xfails": list(EXPECTED_XFAIL_NODEIDS),
    }


def assess_integration(repo_root: Path) -> dict[str, Any]:
    exists = git(
        repo_root,
        "show-ref",
        "--verify",
        f"refs/remotes/{INTEGRATION_REF}",
        check=False,
    )
    if exists.returncode != 0:
        raise ClosureError(f"branch de integração remota ausente: {INTEGRATION_REF}")
    integration_head = git(repo_root, "rev-parse", INTEGRATION_REF).stdout.strip()
    divergence_text = git(
        repo_root,
        "rev-list",
        "--left-right",
        "--count",
        f"{INTEGRATION_REF}...HEAD",
    ).stdout.strip()
    target_only, source_only = [int(value) for value in divergence_text.split()]
    merge_base = git(
        repo_root, "merge-base", INTEGRATION_REF, "HEAD"
    ).stdout.strip()
    target_is_ancestor = git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        INTEGRATION_REF,
        "HEAD",
        check=False,
    ).returncode == 0
    source_is_ancestor = git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        "HEAD",
        INTEGRATION_REF,
        check=False,
    ).returncode == 0

    if target_is_ancestor:
        integration_mode = "fast-forward"
        conflict_assessment = "not_applicable_fast_forward"
        merge_tree_clean = True
    elif source_is_ancestor:
        integration_mode = "already_integrated_or_target_ahead"
        conflict_assessment = "source_is_ancestor_of_target"
        merge_tree_clean = True
    else:
        merge_tree = git(
            repo_root,
            "merge-tree",
            "--write-tree",
            INTEGRATION_REF,
            "HEAD",
            check=False,
        )
        merge_tree_clean = merge_tree.returncode == 0
        conflict_assessment = (
            "clean_non_fast_forward_merge"
            if merge_tree_clean
            else "conflicts_detected"
        )
        integration_mode = "merge_commit_required"

    technically_ready = bool(
        source_only > 0
        and not source_is_ancestor
        and merge_tree_clean
    )
    if source_is_ancestor:
        decision = "no_integration_needed_source_already_contained"
    elif technically_ready:
        decision = "ready_for_explicit_integration_approval"
    else:
        decision = "blocked_pending_conflict_resolution"

    return {
        "target_ref": INTEGRATION_REF,
        "target_head": integration_head,
        "source_head": EXPECTED_HEAD,
        "merge_base": merge_base,
        "divergence": {
            "target_only": target_only,
            "source_only": source_only,
        },
        "target_is_ancestor_of_source": target_is_ancestor,
        "source_is_ancestor_of_target": source_is_ancestor,
        "integration_mode": integration_mode,
        "merge_tree_clean": merge_tree_clean,
        "conflict_assessment": conflict_assessment,
        "technically_ready": technically_ready,
        "decision": decision,
        "integration_executed": False,
    }


def fingerprint_payload(payload: dict[str, Any]) -> str:
    basis = {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "contract_fingerprint"}
    }
    encoded = json.dumps(
        basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    baseline: dict[str, Any],
    phases: list[dict[str, Any]],
    ap004e_commit: dict[str, Any],
    ap004e_inventory: dict[str, Any],
    consolidated_diff: dict[str, Any],
    validations: dict[str, Any],
    integration: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": baseline,
        "phases": phases,
        "ap004e_commit_contract": ap004e_commit,
        "ap004e_inventory_contract": ap004e_inventory,
        "consolidated_diff": consolidated_diff,
        "validation": validations,
        "integration_assessment": integration,
        "closure_decision": {
            "ap004_status": "technically_closed",
            "productive_applicator_required": False,
            "productive_changes_in_ap004f": False,
            "remaining_manual_inventory_decisions": 0,
            "safe_removal_candidates": 0,
            "residual_known_defects": 3,
            "residual_known_defects_policy": "preserved_as_xfail",
            "integration_recommended": integration["technically_ready"],
            "integration_authorized": False,
            "next_required_action": (
                "review_and_commit_ap004f_artifacts"
                if integration["technically_ready"]
                else "review_integration_blocker"
            ),
        },
        "artifacts": {
            "generated": [path.as_posix() for path in OUTPUT_RELS],
            "generator": TOOL_REL.as_posix(),
            "documental_sanitation": NAMING_CONVENTION_REL.as_posix(),
        },
        "gate": {
            "productive_changes_allowed": False,
            "merge_allowed": False,
            "rebase_allowed": False,
            "cherry_pick_allowed": False,
            "commit_allowed_before_review": False,
            "push_allowed_before_review": False,
            "integration_allowed_before_explicit_approval": False,
            "integration_executed": False,
            "message": (
                "[BLOQUEIO] Integração em refactor/academic-pipeline depende "
                "de revisão dos artefatos AP-004F e aprovação expressa."
            ),
        },
    }
    payload["contract_fingerprint"] = fingerprint_payload(payload)
    return payload


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def render_final_validation(payload: dict[str, Any]) -> str:
    validation = payload["validation"]
    suite = validation["canonical_suite_before_ap004f_contract"]
    contracts = validation["contract_tests"]
    diff = payload["consolidated_diff"]
    integration = payload["integration_assessment"]
    phase_rows = [
        (item["phase"], f"`{item['commit']}`", item["subject"], "OK")
        for item in payload["phases"]
    ]
    lines = [
        "# AP-004F — Validação final da AP-004",
        "",
        "## Resultado executivo",
        "",
        "A cadeia AP-004A–E foi validada na branch canônica, com histórico linear, "
        "HEAD local e remoto sincronizados, saneamento documental do EOF e ausência "
        "de alterações produtivas na AP-004E.",
        "",
        f"Fingerprint do encerramento: `{payload['contract_fingerprint']}`.",
        "",
        "## Baseline",
        "",
        markdown_table(
            ("Critério", "Resultado"),
            (
                ("Branch", f"`{payload['baseline']['branch']}`"),
                ("HEAD", f"`{payload['baseline']['head']}`"),
                ("Remoto", f"`{payload['baseline']['remote_head']}`"),
                ("Divergência", "0 0"),
                ("Diff check consolidado", "aprovado"),
            ),
        ),
        "",
        "## Marcos AP-004A–E",
        "",
        markdown_table(("Fase", "Commit", "Mensagem", "Ancestralidade"), phase_rows),
        "",
        "## Validação funcional",
        "",
        markdown_table(
            ("Gate", "Resultado"),
            (
                (
                    "Py_compile",
                    f"{validation['py_compile']['module_count']} módulos aprovados",
                ),
                (
                    "Contratos AP-004D/AP-004E",
                    f"{contracts['passed']} passed",
                ),
                (
                    "Suíte canônica pré-contrato AP-004F",
                    f"{suite['passed']} passed, {suite['xfailed']} xfailed",
                ),
                ("Xpass", suite["xpassed"]),
                ("Falhas", suite["failed"] + suite["errors"]),
            ),
        ),
        "",
        "### Defeitos históricos preservados",
        "",
    ]
    for nodeid in validation["expected_xfails"]:
        lines.append(f"- `{nodeid}`")
    lines.extend(
        [
            "",
            "## Integridade do conjunto de mudanças",
            "",
            f"- Arquivos no diff consolidado: **{diff['file_count']}**.",
            f"- Inserções: **{diff['insertions']}**.",
            f"- Exclusões: **{diff['deletions']}**.",
            "- Binários: **0**.",
            "- `git diff --check`: aprovado.",
            "- AP-004E: exatamente cinco artefatos não produtivos.",
            "",
            "## Avaliação da integração",
            "",
            markdown_table(
                ("Critério", "Resultado"),
                (
                    ("Branch alvo", f"`{integration['target_ref']}`"),
                    ("HEAD alvo", f"`{integration['target_head']}`"),
                    ("Modo previsto", integration["integration_mode"]),
                    ("Merge-tree limpo", integration["merge_tree_clean"]),
                    ("Prontidão técnica", integration["technically_ready"]),
                    ("Integração executada", "não"),
                ),
            ),
            "",
            "## Conclusão",
            "",
            "A validação final da AP-004 está aprovada. A execução da integração "
            "permanece bloqueada até aprovação expressa posterior ao commit e ao push "
            "dos artefatos da AP-004F.",
            "",
        ]
    )
    return "\n".join(lines)


def render_closure_report(payload: dict[str, Any]) -> str:
    ap004e = payload["ap004e_inventory_contract"]
    integration = payload["integration_assessment"]
    lines = [
        "# AP-004F — Relatório de encerramento da AP-004",
        "",
        "## Objetivo encerrado",
        "",
        "A AP-004 consolidou nomenclatura, módulos, símbolos internos, marcadores "
        "de versão e superfícies de compatibilidade do Academic Pipeline sem "
        "reabrir a decomposição arquitetural concluída na AP-003.",
        "",
        "## Síntese por subfase",
        "",
        markdown_table(
            ("Subfase", "Resultado consolidado"),
            (
                (
                    "AP-004A",
                    "inventário de nomes e convenção canônica publicados",
                ),
                (
                    "AP-004B",
                    "módulos e arquivos consolidados com wrappers históricos preservados",
                ),
                (
                    "AP-004C",
                    "símbolos internos normalizados, protegidos ou adiados conforme contrato",
                ),
                (
                    "AP-004D",
                    "marcadores internos de versão substituídos por nomes semânticos duráveis",
                ),
                (
                    "AP-004E",
                    "64 superfícies de compatibilidade classificadas; nenhuma remoção segura",
                ),
            ),
        ),
        "",
        "## Decisões arquiteturais finais",
        "",
        "- O arquivo histórico `academic_pipeline_rc10.py` permanece suportado.",
        "- `pipeline_orchestrator.py` permanece como alias canônico.",
        "- Os entrypoints `python -m academic_pipeline` e `academic-pipeline` "
        "permanecem públicos e duráveis.",
        "- Os arquivos fulltext `v1_13` e `v1_14` permanecem congelados.",
        "- Os cinco símbolos protegidos permanecem fora de remoção.",
        "- Os três defeitos históricos permanecem congelados por `xfail`.",
        "- A AP-004E não exige aplicador produtivo.",
        "",
        "## Compatibilidades",
        "",
        f"- Itens inventariados: **{ap004e['summary']['item_count']}**.",
        f"- Decisões manuais: **{ap004e['summary']['manual_decision_items']}**.",
        f"- Candidatos seguros à remoção: **{ap004e['summary']['removal_candidates']}**.",
        f"- Colisões: **{ap004e['summary']['blocked_items']}**.",
        "",
        "## Riscos residuais",
        "",
        "1. Consumidores externos de wrappers históricos não são integralmente "
        "observáveis pelo repositório.",
        "2. A remoção futura das 38 superfícies internas classificadas para migração "
        "prévia exigirá uma fase própria, com contratos adicionais.",
        "3. A branch alvo de integração pode evoluir após esta fotografia; o gate "
        "de integração deverá ser repetido imediatamente antes da operação.",
        "4. Os três `xfail` permanecem dívida técnica deliberadamente fora do escopo.",
        "",
        "## Estado de encerramento",
        "",
        markdown_table(
            ("Dimensão", "Estado"),
            (
                ("AP-004", "tecnicamente encerrada"),
                ("Código produtivo na AP-004F", "inalterado"),
                ("Aplicador produtivo AP-004E", "não necessário"),
                ("Integração", "não executada"),
                ("Prontidão técnica", integration["technically_ready"]),
            ),
        ),
        "",
        f"Fingerprint contratual: `{payload['contract_fingerprint']}`.",
        "",
    ]
    return "\n".join(lines)


def render_integration_decision(payload: dict[str, Any]) -> str:
    integration = payload["integration_assessment"]
    closure = payload["closure_decision"]
    recommendation = (
        "RECOMENDADA, SOB APROVAÇÃO EXPRESSA"
        if closure["integration_recommended"]
        else "NÃO RECOMENDADA NO ESTADO ATUAL"
    )
    lines = [
        "# AP-004F — Decisão de integração da AP-004",
        "",
        f"## Decisão: {recommendation}",
        "",
        "A AP-004 está tecnicamente encerrada e validada. Esta decisão não executa "
        "integração e não concede autorização automática para merge, rebase ou "
        "cherry-pick.",
        "",
        "## Evidências",
        "",
        markdown_table(
            ("Critério", "Resultado"),
            (
                ("Branch de origem", f"`{EXPECTED_BRANCH}`"),
                ("HEAD de origem", f"`{integration['source_head']}`"),
                ("Branch alvo", f"`{integration['target_ref']}`"),
                ("HEAD alvo", f"`{integration['target_head']}`"),
                ("Merge-base", f"`{integration['merge_base']}`"),
                (
                    "Divergência",
                    f"alvo={integration['divergence']['target_only']}; "
                    f"origem={integration['divergence']['source_only']}",
                ),
                ("Modo previsto", integration["integration_mode"]),
                ("Conflitos detectados", not integration["merge_tree_clean"]),
                ("Prontidão técnica", integration["technically_ready"]),
            ),
        ),
        "",
        "## Condições obrigatórias antes da integração",
        "",
        "1. Revisar e aprovar os artefatos AP-004F.",
        "2. Consolidar o saneamento documental e os seis artefatos AP-004F em "
        "commit isolado.",
        "3. Publicar o commit na branch de origem e confirmar divergência `0 0`.",
        "4. Repetir `git fetch origin`, suíte canônica e avaliação de conflitos.",
        "5. Obter autorização expressa para a operação de integração.",
        "6. Integrar sem reescrever o histórico dos commits AP-004A–F.",
        "",
        "## Operações ainda bloqueadas",
        "",
        "```text",
        "[BLOQUEIO] Não executar merge.",
        "[BLOQUEIO] Não executar rebase.",
        "[BLOQUEIO] Não executar cherry-pick.",
        "[BLOQUEIO] Não publicar alteração na branch de integração.",
        "```",
        "",
        f"Fingerprint contratual: `{payload['contract_fingerprint']}`.",
        "",
    ]
    return "\n".join(lines)


def render_characterization_test(payload: dict[str, Any]) -> str:
    phases_literal = repr(
        [(item["phase"], item["commit"]) for item in payload["phases"]]
    )
    xfails_literal = repr(list(EXPECTED_XFAIL_NODEIDS))
    integration = payload["integration_assessment"]
    return f'''"""Contrato de encerramento da AP-004F.

Gerado por tools/refactor/ap004f_generate_closure.py.
Não editar manualmente: regenere após repetir o gate da AP-004F.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_SCHEMA = {SCHEMA_VERSION!r}
EXPECTED_HEAD = {EXPECTED_HEAD!r}
EXPECTED_FINGERPRINT = {payload['contract_fingerprint']!r}
EXPECTED_PHASES = {phases_literal}
EXPECTED_XFAILS = {xfails_literal}
EXPECTED_INTEGRATION_TARGET = {integration['target_ref']!r}
EXPECTED_INTEGRATION_MODE = {integration['integration_mode']!r}
EXPECTED_INTEGRATION_READY = {integration['technically_ready']!r}


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        marker = parent / {MANIFEST_REL.as_posix()!r}
        if marker.is_file():
            return parent
    raise AssertionError("não foi possível localizar a raiz do repositório")


def _load_manifest() -> dict:
    path = _repo_root() / {MANIFEST_REL.as_posix()!r}
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint_basis(data: dict) -> str:
    basis = {{
        key: value
        for key, value in data.items()
        if key not in {{"generated_at", "contract_fingerprint"}}
    }}
    encoded = json.dumps(
        basis,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_ap004f_manifest_contract_is_frozen() -> None:
    data = _load_manifest()
    assert data["schema_version"] == EXPECTED_SCHEMA
    assert data["baseline"]["head"] == EXPECTED_HEAD
    assert data["baseline"]["remote_head"] == EXPECTED_HEAD
    assert data["baseline"]["divergence"] == [0, 0]
    assert [(item["phase"], item["commit"]) for item in data["phases"]] == EXPECTED_PHASES
    assert data["contract_fingerprint"] == EXPECTED_FINGERPRINT
    assert _fingerprint_basis(data) == EXPECTED_FINGERPRINT


def test_ap004f_final_validation_contract() -> None:
    data = _load_manifest()
    contracts = data["validation"]["contract_tests"]
    suite = data["validation"]["canonical_suite_before_ap004f_contract"]
    assert contracts["passed"] == 7
    assert contracts["failed"] == 0
    assert contracts["errors"] == 0
    assert suite["passed"] == 489
    assert suite["xfailed"] == 3
    assert suite["xpassed"] == 0
    assert suite["failed"] == 0
    assert suite["errors"] == 0
    assert data["validation"]["expected_xfails"] == EXPECTED_XFAILS


def test_ap004f_closes_ap004_without_productive_applicator() -> None:
    data = _load_manifest()
    closure = data["closure_decision"]
    ap004e = data["ap004e_inventory_contract"]
    assert closure["ap004_status"] == "technically_closed"
    assert closure["productive_applicator_required"] is False
    assert closure["productive_changes_in_ap004f"] is False
    assert closure["remaining_manual_inventory_decisions"] == 0
    assert closure["safe_removal_candidates"] == 0
    assert closure["residual_known_defects"] == 3
    assert ap004e["summary"]["item_count"] == 64
    assert ap004e["summary"]["removal_candidates"] == 0
    assert ap004e["summary"]["blocked_items"] == 0


def test_ap004f_integration_remains_explicitly_blocked() -> None:
    data = _load_manifest()
    integration = data["integration_assessment"]
    gate = data["gate"]
    assert integration["target_ref"] == EXPECTED_INTEGRATION_TARGET
    assert integration["integration_mode"] == EXPECTED_INTEGRATION_MODE
    assert integration["technically_ready"] is EXPECTED_INTEGRATION_READY
    assert integration["integration_executed"] is False
    assert gate["productive_changes_allowed"] is False
    assert gate["merge_allowed"] is False
    assert gate["rebase_allowed"] is False
    assert gate["cherry_pick_allowed"] is False
    assert gate["commit_allowed_before_review"] is False
    assert gate["push_allowed_before_review"] is False
    assert gate["integration_allowed_before_explicit_approval"] is False
    assert gate["integration_executed"] is False
'''


def prepare_outputs(payload: dict[str, Any]) -> dict[Path, str]:
    return {
        FINAL_VALIDATION_REL: render_final_validation(payload),
        CLOSURE_REPORT_REL: render_closure_report(payload),
        INTEGRATION_DECISION_REL: render_integration_decision(payload),
        MANIFEST_REL: json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        TEST_REL: render_characterization_test(payload),
    }


def normalize_text(text: str) -> bytes:
    return (text.rstrip(" \t\r\n") + "\n").encode("utf-8")


def transaction_write(
    repo_root: Path,
    outputs: dict[Path, str],
) -> tuple[Path, list[str]]:
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix="ap004f_closure_backup_",
            dir=os.environ.get("TMPDIR") or None,
        )
    )
    previous: dict[Path, bytes | None] = {}
    changed: list[str] = []
    try:
        for rel, text in outputs.items():
            path = repo_root / rel
            new_bytes = normalize_text(text)
            old_bytes = path.read_bytes() if path.exists() else None
            previous[path] = old_bytes
            if old_bytes == new_bytes:
                continue
            if old_bytes is not None:
                backup_path = backup_dir / rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_bytes(old_bytes)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(new_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
            changed.append(rel.as_posix())
        return backup_dir, changed
    except Exception:
        for path, old_bytes in previous.items():
            if old_bytes is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(old_bytes)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida e gera o encerramento AP-004F sem alterar código produtivo "
            "ou integrar branches."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=EXPECTED_REPOSITORY,
        help=f"raiz Git (padrão: {EXPECTED_REPOSITORY})",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="não executar git fetch origin antes dos gates",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="somente para inspeção local; não permite escrita dos artefatos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="executar gates e renderização sem escrever artefatos",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        baseline = validate_repository(repo_root, fetch=not args.no_fetch)
        phases = validate_commits(repo_root)
        ap004e_commit = validate_ap004e_commit(repo_root)
        ap004e_inventory = validate_ap004e_inventory(repo_root)
        ensure_file(repo_root / AP004D_INVENTORY_REL)
        consolidated_diff = validate_consolidated_diff(repo_root)
        integration = assess_integration(repo_root)

        if args.skip_tests:
            if not args.dry_run:
                raise ClosureError("--skip-tests exige --dry-run")
            validations = {
                "py_compile": {"status": "skipped"},
                "contract_tests": {"status": "skipped"},
                "canonical_suite_before_ap004f_contract": {"status": "skipped"},
                "expected_xfails": list(EXPECTED_XFAIL_NODEIDS),
            }
        else:
            validations = run_final_validations(repo_root)

        payload = build_manifest(
            baseline=baseline,
            phases=phases,
            ap004e_commit=ap004e_commit,
            ap004e_inventory=ap004e_inventory,
            consolidated_diff=consolidated_diff,
            validations=validations,
            integration=integration,
        )
        outputs = prepare_outputs(payload)

        backup_dir: Path | None = None
        changed: list[str] = []
        if not args.dry_run:
            backup_dir, changed = transaction_write(repo_root, outputs)

        print("=== AP-004F — ENCERRAMENTO GERADO ===")
        print(f"Repositório: {repo_root}")
        print(f"Branch: {payload['baseline']['branch']}")
        print(f"HEAD: {payload['baseline']['head']}")
        print(f"Fases validadas: {len(payload['phases'])}")
        print(
            "Suíte canônica pré-AP-004F: "
            f"{payload['validation']['canonical_suite_before_ap004f_contract'].get('passed', 'skipped')} passed, "
            f"{payload['validation']['canonical_suite_before_ap004f_contract'].get('xfailed', 'skipped')} xfailed"
        )
        print(
            "Integração: "
            f"modo={integration['integration_mode']}; "
            f"tecnicamente_pronta={integration['technically_ready']}"
        )
        print(f"Fingerprint: {payload['contract_fingerprint']}")
        if args.dry_run:
            print("Modo: DRY-RUN; nenhum arquivo escrito.")
        else:
            print(f"Backup externo: {backup_dir}")
            if changed:
                print("Arquivos criados/atualizados:")
                for rel in changed:
                    print(f"- {rel}")
            else:
                print("Artefatos já idênticos; nenhuma reescrita necessária.")
        print("[BLOQUEIO] Não alterar código produtivo.")
        print("[BLOQUEIO] Não criar commit ou push antes da revisão.")
        print("[BLOQUEIO] Não executar merge, rebase ou cherry-pick.")
        print("[BLOQUEIO] Não integrar em refactor/academic-pipeline.")
        return 0
    except ClosureError as exc:
        print(f"ERRO SEGURO: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERRO INESPERADO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
