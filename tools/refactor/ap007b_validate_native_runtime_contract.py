\
#!/usr/bin/env python3
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_BASE_HEAD = "17725a5505eb2f9c0b1a6cfd5899e38d70031f80"
EXPECTED_COMMIT_SUBJECT = (
    "refactor(academic-pipeline): materialize AP-007B native runtime"
)
EXPECTED_PATHS = {
    "software/academic_pipeline_mppg/academic_pipeline/cli.py",
    "software/academic_pipeline_mppg/academic_pipeline/runtime.py",
    "docs/refactor/academic-pipeline/AP-007/AP-007B_NATIVE_RUNTIME_CONTRACT.md",
    "docs/refactor/academic-pipeline/AP-007/ap007b_native_runtime_contract.json",
    "software/academic_pipeline_mppg/tests/characterization/test_ap007b_native_runtime_contract.py",
    "tools/refactor/ap007b_validate_native_runtime_contract.py",
}
PRESERVED_HASHES = {
    "software/academic_pipeline_mppg/academic_pipeline/legacy.py": (
        "f11ddffc30f60ac0c5e0856e8bf00ffaae866a8df806fd3c2b99f1afaa09e6b9"
    ),
    "software/academic_pipeline_mppg/academic_pipeline/__main__.py": (
        "31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4"
    ),
    "software/academic_pipeline_mppg/academic_pipeline/command_dispatch.py": (
        "9255c4b924fd61b7120b8c5e02684d338f6788de42ae7c352b049a488a308afe"
    ),
    "software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_rc10.py": (
        "f385b32fed0445dde90a596440903a7c174e42eac2e1675251ddbd0ce516288f"
    ),
    "software/academic_pipeline_mppg/pyproject.toml": (
        "0c5225e5e9bc8f94ae0964e84b180444908af5d1de5fd7929574686951384d80"
    ),
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dirty_paths(repo: Path) -> set[str]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    paths: set[str] = set()
    for line in output.splitlines():
        if not line:
            continue
        if len(line) < 4 or line[2] != " ":
            raise AssertionError(
                f"registro porcelain inesperado: {line!r}"
            )
        paths.add(line[3:])
    return paths


def _commit_paths(repo: Path, revision: str) -> set[str]:
    output = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        revision,
    )
    return {line for line in output.splitlines() if line}


def validate(repo: Path, mode: str = "auto") -> dict[str, Any]:
    repo = repo.resolve()
    payload_path = (
        repo
        / "docs/refactor/academic-pipeline/AP-007/"
        "ap007b_native_runtime_contract.json"
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    assert payload["schema"] == "ap007b-native-runtime-contract-v1"
    assert payload["phase"] == "AP-007B.1"
    assert payload["baseline"]["head"] == EXPECTED_BASE_HEAD
    assert set(payload["materialized_paths"]) == EXPECTED_PATHS
    assert payload["first_wave"] == [
        "--help",
        "--list-toml-profiles",
        "--list-institutions",
        "--list-layouts",
        "--explain-profile",
    ]
    assert payload["topology"]["fallback"] == "explicit_legacy_fallback"
    assert payload["topology"]["normal_first_wave_legacy_calls"] == 0
    assert payload["parser"]["registered_long_option_count"] == 63
    assert payload["parser"]["help_text_option_token_count"] == 66
    assert payload["parser"]["registered_options_are_in_help"] is True
    assert payload["dispatch_runtime_key_count"] == 6
    assert payload["dispatch_runtime_keys"] == [
        "Path",
        "available_layouts",
        "describe_institution_profiles",
        "explain_profile",
        "load_config",
        "resolve_layout_spec",
    ]
    assert payload["dispatch_result_fields"] == [
        "handled",
        "value",
    ]
    assert payload["dispatch_result_semantics"] == {
        "handled_field": "handled",
        "value_field": "value",
        "handled_zero": {
            "handled": True,
            "value": 0,
        },
        "not_handled": {
            "handled": False,
            "value": None,
        },
    }

    for relative, expected in payload["artifact_sha256"].items():
        assert _sha(repo / relative) == expected

    for relative, expected in PRESERVED_HASHES.items():
        assert _sha(repo / relative) == expected

    for relative, expected in payload["preserved_sha256"].items():
        assert _sha(repo / relative) == expected

    cli_path = (
        repo
        / "software/academic_pipeline_mppg/academic_pipeline/cli.py"
    )
    runtime_path = (
        repo
        / "software/academic_pipeline_mppg/academic_pipeline/runtime.py"
    )
    cli_source = cli_path.read_text(encoding="utf-8")
    runtime_source = runtime_path.read_text(encoding="utf-8")
    cli_tree = ast.parse(cli_source)
    runtime_tree = ast.parse(runtime_source)

    assert "from .runtime import run" in cli_source
    assert "from .legacy import run_legacy" in cli_source
    assert "return run(argv, legacy_runner=run_legacy)" in cli_source

    parser_tree = ast.parse(
        (repo / "software/academic_pipeline_mppg/academic_pipeline/cli_parser.py")
        .read_text(encoding="utf-8")
    )
    parser_functions = {
        node.name: node
        for node in parser_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"build_parser", "parse_args"} <= set(parser_functions)

    assert "pipeline_version=PIPELINE_VERSION" in runtime_source
    assert "globals(" not in runtime_source
    assert "locals(" not in runtime_source
    assert "sys.path" not in runtime_source
    assert "importlib" not in runtime_source
    assert "academic_pipeline_rc10" not in runtime_source

    classes = {
        node.name: node
        for node in runtime_tree.body
        if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name: node
        for node in runtime_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "RuntimeContext" in classes
    assert {"run", "select_runtime_route", "_run_native_first_wave"} <= set(
        functions
    )

    if mode == "auto":
        head = _git(repo, "rev-parse", "HEAD")
        mode = "precommit" if head == EXPECTED_BASE_HEAD else "postcommit"

    if mode == "precommit":
        assert _git(repo, "rev-parse", "HEAD") == EXPECTED_BASE_HEAD
        assert _dirty_paths(repo) == EXPECTED_PATHS
        assert not _git(repo, "diff", "--cached", "--name-only")
    elif mode == "postcommit":
        subject = _git(repo, "show", "-s", "--format=%s", "HEAD")
        parent = _git(repo, "rev-parse", "HEAD^")
        assert subject == EXPECTED_COMMIT_SUBJECT
        assert parent == EXPECTED_BASE_HEAD
        assert _commit_paths(repo, "HEAD") == EXPECTED_PATHS
    else:
        raise ValueError(f"modo inválido: {mode}")

    return {
        "ok": True,
        "mode": mode,
        "path_count": len(EXPECTED_PATHS),
        "first_wave_count": len(payload["first_wave"]),
        "runtime_context_fields": payload["runtime_context_fields"],
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo = Path(args[0]).resolve() if args else Path.cwd().resolve()
    mode = args[1] if len(args) > 1 else "auto"
    result = validate(repo, mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print("AP-007B native runtime contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
