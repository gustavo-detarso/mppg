#!/usr/bin/env python3
"""Inventário reproduzível de instalação, metadata e entrypoints da AP-005E.1."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from setuptools import find_namespace_packages, find_packages

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_REL = Path("software/academic_pipeline_rc10_7_conformidade")
SOFTWARE_ROOT = ROOT / SOFTWARE_REL
DOC_DIR = ROOT / "docs/refactor/academic-pipeline/AP-005"

EXPECTED_BRANCH = "ap-refactor/04-consumer-canonicalization"
EXPECTED_BASELINE_COMMIT = "ba28822c826c37022581bf88c6a1b488e2c618de"
EXPECTED_UPSTREAM = "origin/ap-refactor/04-consumer-canonicalization"
SCHEMA_VERSION = "ap005e1.installation-entrypoint-inventory.v1"

INVENTORY_JSON = DOC_DIR / "ap005e1_installation_entrypoint_inventory.json"
INVENTORY_MD = DOC_DIR / "AP-005E1_INSTALLATION_ENTRYPOINT_INVENTORY.md"
STRATEGY_MD = DOC_DIR / "AP-005E1_INSTALLATION_ENTRYPOINT_STRATEGY.md"

PYPROJECT_REL = SOFTWARE_REL / "pyproject.toml"
ENTRYPOINT_FILES = (
    SOFTWARE_REL / "academic_pipeline/__init__.py",
    SOFTWARE_REL / "academic_pipeline/__main__.py",
    SOFTWARE_REL / "academic_pipeline/cli.py",
)

EXPECTED_DISCOVERED_PACKAGES = [
    "academic_pipeline",
    "app_bundle",
    "app_bundle.scripts",
    "app_bundle.scripts.pipeline",
    "app_bundle.scripts.pipeline.article_workflow",
]

EXPECTED_SOURCE_ROOT_CENSUS = {
    "tracked_total": 274,
    "python_files": 90,
    "non_python_files": 184,
    "init_files": 5,
    "selected_package_python_files": 65,
    "excluded_test_python_files": 23,
    "other_python_files": 2,
    "selected_package_non_python_files": 3,
}

EXPECTED_ENTRYPOINT_HASHES = {
    "software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__init__.py":
        "4d08f2e264da8da506351ceaf4a5efd17fa4211bae7b44da458086618137413c",
    "software/academic_pipeline_rc10_7_conformidade/academic_pipeline/__main__.py":
        "31840fb9a79716886a21e2026f9255e4df5bdf897531cecf63399692ada047f4",
    "software/academic_pipeline_rc10_7_conformidade/academic_pipeline/cli.py":
        "bbda5a88d1234f649bb6d171a14b75e51d03bf3d721d91656ad624912f26c2db",
}

RELEVANT_TESTS = [
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_entrypoints_orchestration_characterization.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_official_package_entrypoint.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_package_imports_document_core.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_package_imports_entrypoints.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_package_imports_prisma_core.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_package_imports_rendering.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_package_imports_support_services.py",
    "software/academic_pipeline_rc10_7_conformidade/app_bundle/tests/"
    "test_packaging_metadata.py",
]

SNAPSHOT_FILES = [
    "software/academic_pipeline_rc10_7_conformidade/tests/characterization/"
    "snapshots/ap003a/package_module_help.txt",
]

LAYOUT_RISK_RECORDS = [
    {
        "id": "legacy-relative-path-bridge",
        "path": "software/academic_pipeline_rc10_7_conformidade/"
        "academic_pipeline/legacy.py",
        "lines": [17, 29, 61, 63, 66],
        "classification": "intentional_compatibility_bridge",
        "decision": "preserve",
        "reason": (
            "O entrypoint oficial localiza app_bundle/scripts/pipeline "
            "relativamente ao pacote instalado e normaliza sys.path."
        ),
    },
    {
        "id": "hardcoded-user-prompt-default",
        "path": "software/academic_pipeline_rc10_7_conformidade/"
        "academic_pipeline/prisma_generic_orchestration.py",
        "lines": [163],
        "classification": "portability_risk",
        "decision": "characterize_in_ap005e2",
        "reason": (
            "O valor padrão contém caminho absoluto sob /home/gustavodetarso "
            "e não é portável para instalação isolada."
        ),
    },
    {
        "id": "module-self-invocation-by-file",
        "path": "software/academic_pipeline_rc10_7_conformidade/"
        "academic_pipeline/prisma_generic_orchestration.py",
        "lines": [310, 332, 336, 550],
        "classification": "installed_layout_risk",
        "decision": "characterize_in_ap005e2",
        "reason": (
            "Chamadas subprocess usam __file__ como script; o comportamento "
            "deve ser validado após instalação do wheel."
        ),
    },
    {
        "id": "helper-sibling-assumption",
        "path": "software/academic_pipeline_rc10_7_conformidade/"
        "academic_pipeline/prisma_generic_orchestration.py",
        "lines": [505, 537],
        "classification": "installed_layout_risk",
        "decision": "characterize_in_ap005e2",
        "reason": (
            "Helpers são procurados como irmãos do módulo academic_pipeline; "
            "a presença e o destino no artefato instalado precisam ser provados."
        ),
    },
]


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def verify_git_baseline() -> None:
    branch = run_git("branch", "--show-current").stdout.strip()
    head = run_git("rev-parse", "HEAD").stdout.strip()
    upstream = run_git(
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    ).stdout.strip()
    baseline_exists = run_git(
        "cat-file",
        "-e",
        f"{EXPECTED_BASELINE_COMMIT}^{{commit}}",
        check=False,
    )
    baseline_is_ancestor = run_git(
        "merge-base",
        "--is-ancestor",
        EXPECTED_BASELINE_COMMIT,
        "HEAD",
        check=False,
    )

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Branch divergente: {branch}")
    if upstream != EXPECTED_UPSTREAM:
        raise RuntimeError(f"Upstream divergente: {upstream}")
    if baseline_exists.returncode != 0:
        raise RuntimeError(
            "Commit baseline não existe no repositório: "
            f"{EXPECTED_BASELINE_COMMIT}"
        )
    if baseline_is_ancestor.returncode != 0:
        raise RuntimeError(
            "HEAD não descende do baseline AP-005D: "
            f"baseline={EXPECTED_BASELINE_COMMIT}; head={head}"
        )


def tracked_files_under_package_roots() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            str(SOFTWARE_REL / "academic_pipeline"),
            str(SOFTWARE_REL / "app_bundle"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return sorted(
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )


def source_root_census(files: list[str]) -> dict[str, int]:
    prefix = f"{SOFTWARE_REL.as_posix()}/"
    python_files = [path for path in files if path.endswith(".py")]
    non_python_files = [path for path in files if not path.endswith(".py")]
    init_files = [path for path in files if path.endswith("/__init__.py")]

    selected_python = []
    excluded_test_python = []
    other_python = []
    selected_non_python = []

    for path in files:
        relative = path.removeprefix(prefix)
        selected = (
            relative.startswith("academic_pipeline/")
            or relative.startswith("app_bundle/scripts/")
        )
        if path.endswith(".py"):
            if selected:
                selected_python.append(path)
            elif relative.startswith("app_bundle/tests/"):
                excluded_test_python.append(path)
            else:
                other_python.append(path)
        elif selected:
            selected_non_python.append(path)

    return {
        "tracked_total": len(files),
        "python_files": len(python_files),
        "non_python_files": len(non_python_files),
        "init_files": len(init_files),
        "selected_package_python_files": len(selected_python),
        "excluded_test_python_files": len(excluded_test_python),
        "other_python_files": len(other_python),
        "selected_package_non_python_files": len(selected_non_python),
    }


def load_pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / PYPROJECT_REL).read_text(encoding="utf-8"))


def discover_packages(pyproject: dict[str, Any]) -> dict[str, Any]:
    find_cfg = (
        pyproject.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
    )
    where_values = find_cfg.get("where") or ["."]
    include = find_cfg.get("include")
    exclude = find_cfg.get("exclude")
    namespaces = find_cfg.get("namespaces", True)

    discovery: dict[str, Any] = {}
    for where in where_values:
        kwargs: dict[str, Any] = {"where": str(SOFTWARE_ROOT / where)}
        if include:
            kwargs["include"] = include
        if exclude:
            kwargs["exclude"] = exclude

        classic = sorted(find_packages(**kwargs))
        namespace = sorted(find_namespace_packages(**kwargs))
        selected = namespace if namespaces is not False else classic
        discovery[where] = {
            "find_packages": classic,
            "find_namespace_packages": namespace,
            "selected": selected,
        }
    return {
        "configuration": find_cfg,
        "discovery": discovery,
        "selected_packages": discovery["."]["selected"],
    }


def ast_summary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []
    main_guards: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(
                {
                    "kind": "import",
                    "names": [
                        {"name": alias.name, "asname": alias.asname}
                        for alias in node.names
                    ],
                }
            )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "kind": "from",
                    "module": node.module,
                    "level": node.level,
                    "names": [
                        {"name": alias.name, "asname": alias.asname}
                        for alias in node.names
                    ],
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(
                {
                    "kind": type(node).__name__,
                    "name": node.name,
                    "lineno": node.lineno,
                }
            )
        elif isinstance(node, ast.If):
            condition = ast.get_source_segment(source, node.test) or ""
            if "__name__" in condition and "__main__" in condition:
                main_guards.append(
                    {"lineno": node.lineno, "condition": condition}
                )

    relative = path.relative_to(ROOT).as_posix()
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "imports": imports,
        "definitions": definitions,
        "main_guards": main_guards,
    }


def build_inventory() -> dict[str, Any]:
    pyproject = load_pyproject()
    project = pyproject["project"]
    setuptools_cfg = pyproject.get("tool", {}).get("setuptools", {})
    package_discovery = discover_packages(pyproject)
    tracked_files = tracked_files_under_package_roots()
    census = source_root_census(tracked_files)

    entrypoint_summaries = [
        ast_summary(ROOT / relative)
        for relative in ENTRYPOINT_FILES
    ]
    entrypoint_hashes = {
        item["path"]: item["sha256"]
        for item in entrypoint_summaries
    }

    if package_discovery["selected_packages"] != EXPECTED_DISCOVERED_PACKAGES:
        raise RuntimeError(
            "Pacotes descobertos divergentes: "
            f"{package_discovery['selected_packages']}"
        )
    if census != EXPECTED_SOURCE_ROOT_CENSUS:
        raise RuntimeError(f"Censo de fontes divergente: {census}")
    if entrypoint_hashes != EXPECTED_ENTRYPOINT_HASHES:
        raise RuntimeError(f"Hashes de entrypoint divergentes: {entrypoint_hashes}")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "AP-005E.1",
        "baseline": {
            "branch": EXPECTED_BRANCH,
            "commit": EXPECTED_BASELINE_COMMIT,
            "upstream": EXPECTED_UPSTREAM,
        },
        "metadata": {
            "build_system": pyproject["build-system"],
            "project": {
                "name": project.get("name"),
                "version": project.get("version"),
                "description": project.get("description"),
                "requires_python": project.get("requires-python"),
                "dependencies": project.get("dependencies"),
                "scripts": project.get("scripts"),
            },
            "setuptools": {
                "include_package_data": setuptools_cfg.get(
                    "include-package-data"
                ),
                "packages_find": setuptools_cfg.get("packages", {}).get(
                    "find", {}
                ),
                "package_data": setuptools_cfg.get("package-data"),
                "exclude_package_data": setuptools_cfg.get(
                    "exclude-package-data"
                ),
                "data_files": setuptools_cfg.get("data-files"),
            },
        },
        "package_discovery": package_discovery,
        "source_root_census": census,
        "entrypoints": {
            "console_script": {
                "name": "academic-pipeline",
                "target": "academic_pipeline.cli:main",
            },
            "module_entrypoint": {
                "command": "python -m academic_pipeline",
                "target": "academic_pipeline.cli:main",
            },
            "public_package_function": {
                "symbol": "academic_pipeline.main",
                "target": "academic_pipeline.cli:main",
                "all": ["main"],
            },
            "compatibility_chain": [
                "academic-pipeline",
                "academic_pipeline.cli:main",
                "academic_pipeline.legacy:run_legacy",
                "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
            ],
            "files": entrypoint_summaries,
        },
        "existing_contracts": {
            "tests": RELEVANT_TESTS,
            "snapshots": SNAPSHOT_FILES,
        },
        "audit_observations": {
            "read_only_audit_log": (
                "ap005e1_inventario_metadata_entrypoints_20260718_063956.log"
            ),
            "current_pipenv_distribution": {
                "classification": "environment_observation_not_gate",
                "distribution": "academic-pipeline-mppg",
                "version": "0.1.0",
                "entrypoint": "academic_pipeline.cli:main",
                "file_count": 141,
                "import_origin": "site-packages/academic_pipeline/__init__.py",
                "cwd": "/tmp",
                "pythonpath_removed": True,
            },
            "raw_layout_reference_counts": {
                "total": 194,
                "tests": 154,
                "app_bundle_scripts": 28,
                "academic_pipeline": 12,
                "classification": "algorithmic_evidence_not_gate",
            },
            "historical_python_universe": {
                "tracked_total": 265,
                "top_level_backups_relative_to_software": 119,
                "auditable": 146,
            },
            "source_audit_fingerprint": (
                "22827bef51dfc701824caf9bf60d37477"
                "af8dfdd939bfc7ef3110acee2a2c495"
            ),
        },
        "risks": [
            {
                "id": "package-data-coverage",
                "classification": "unresolved_build_artifact_scope",
                "decision": "characterize_in_ap005e2",
                "evidence": (
                    "include-package-data=true sem package-data, "
                    "MANIFEST.in ou data-files declarados; o censo de fontes "
                    "não é isomórfico ao conteúdo instalado."
                ),
            },
            {
                "id": "runtime-dependency-authority",
                "classification": "distribution_metadata_gap_by_design",
                "decision": "preserve_and_characterize_in_ap005e2",
                "evidence": (
                    "project.dependencies é vazio e Pipfile/Pipfile.lock "
                    "permanecem como autoridade de dependências."
                ),
            },
            *LAYOUT_RISK_RECORDS,
        ],
        "decisions": {
            "productive_change_required_in_ap005e1": False,
            "preserve_distribution_name": True,
            "preserve_distribution_version": True,
            "preserve_console_script": True,
            "preserve_module_entrypoint": True,
            "preserve_public_main": True,
            "preserve_legacy_entrypoint": True,
            "preserve_legacy_path_bridge": True,
            "wheel_contents_are_proven": False,
            "isolated_installation_is_proven": False,
            "defer_corrections_until_ap005e2": True,
            "broad_package_reorganization_allowed": False,
        },
        "ap005e2_gates": [
            "build wheel and sdist from a clean temporary descendant",
            "inspect exact archive manifests and reject accidental residues",
            "install wheel into a fresh temporary virtual environment",
            "remove PYTHONPATH and run from outside the checkout",
            "prepend the temporary environment bin directory to PATH",
            "prove academic_pipeline.__file__ belongs to the temporary environment",
            "prove academic-pipeline resolves to the temporary environment",
            "compare academic-pipeline and python -m academic_pipeline help",
            "exercise the legacy bridge without importing from the checkout",
            "characterize package data required by operational commands",
            "characterize hardcoded and sibling-helper layout risks",
        ],
    }

    payload["fingerprint"] = sha256_bytes(canonical_bytes(payload))
    return payload


def render_inventory(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]["project"]
    census = payload["source_root_census"]
    selected = payload["package_discovery"]["selected_packages"]
    risks = payload["risks"]

    lines = [
        "# AP-005E.1 — Inventário de instalação, metadata e entrypoints",
        "",
        "## Baseline",
        "",
        f"- Branch: `{payload['baseline']['branch']}`",
        f"- Commit: `{payload['baseline']['commit']}`",
        f"- Upstream: `{payload['baseline']['upstream']}`",
        f"- Fingerprint: `{payload['fingerprint']}`",
        "",
        "## Metadata pública",
        "",
        f"- Distribuição: `{metadata['name']}`",
        f"- Versão: `{metadata['version']}`",
        f"- Python: `{metadata['requires_python']}`",
        "- Backend: `setuptools.build_meta`",
        "- Dependências PEP 621 declaradas: nenhuma",
        "- Autoridade operacional de dependências: `Pipfile` e `Pipfile.lock`",
        "- Console script: `academic-pipeline = academic_pipeline.cli:main`",
        "",
        "## Pacotes descobertos",
        "",
    ]
    lines.extend(f"- `{package}`" for package in selected)
    lines.extend(
        [
            "",
            "## Censo sob as raízes de pacote",
            "",
            f"- Arquivos rastreados: **{census['tracked_total']}**",
            f"- Python: **{census['python_files']}**",
            f"- Não Python: **{census['non_python_files']}**",
            f"- `__init__.py`: **{census['init_files']}**",
            (
                "- Python sob pacotes selecionados: "
                f"**{census['selected_package_python_files']}**"
            ),
            (
                "- Testes Python excluídos da descoberta: "
                f"**{census['excluded_test_python_files']}**"
            ),
            "",
            "O censo das raízes não representa o manifesto do wheel. Ele inclui "
            "testes excluídos, documentos, projetos, outputs e outros arquivos "
            "rastreados. A cobertura real somente poderá ser decidida após "
            "construção e inspeção do artefato na AP-005E.2.",
            "",
            "## Cadeia de entrypoints",
            "",
            "1. `academic-pipeline` → `academic_pipeline.cli:main`;",
            "2. `python -m academic_pipeline` → `academic_pipeline.cli:main`;",
            "3. `academic_pipeline.main` → `academic_pipeline.cli:main`;",
            "4. `academic_pipeline.cli:main` → `academic_pipeline.legacy:run_legacy`;",
            "5. o bridge legado carrega `app_bundle/scripts/pipeline/"
            "academic_pipeline_rc10.py`.",
            "",
            "A cadeia é coerente no código-fonte e está coberta por contratos "
            "existentes. A AP-005E.1 não autoriza alteração desses entrypoints.",
            "",
            "## Riscos inventariados",
            "",
        ]
    )
    for risk in risks:
        lines.append(
            f"- `{risk['id']}` — **{risk['classification']}**; "
            f"decisão: `{risk['decision']}`."
        )
    lines.extend(
        [
            "",
            "## Decisão",
            "",
            "A AP-005E.1 é documental e de caracterização. Nenhuma alteração "
            "produtiva é necessária ou autorizada.",
            "",
            "Permanecem não demonstrados:",
            "",
            "- o manifesto exato do wheel e do sdist;",
            "- a instalação em ambiente virtual realmente novo;",
            "- a suficiência dos arquivos de dados distribuídos;",
            "- a ausência de importação acidental do checkout;",
            "- a portabilidade dos caminhos registrados como risco.",
            "",
            "Esses pontos constituem os gates obrigatórios da AP-005E.2.",
            "",
        ]
    )
    return "\n".join(lines)


def render_strategy(payload: dict[str, Any]) -> str:
    lines = [
        "# AP-005E.1 — Estratégia de instalação e entrypoints",
        "",
        "## Objetivo",
        "",
        "Congelar a superfície instalável atualmente declarada antes de construir "
        "artefatos e testar uma instalação isolada.",
        "",
        "## Superfícies preservadas",
        "",
        "- distribuição `academic-pipeline-mppg`, versão `0.1.0`;",
        "- console script `academic-pipeline`;",
        "- execução `python -m academic_pipeline`;",
        "- função pública `academic_pipeline.main` e `__all__ = [\"main\"]`;",
        "- bridge `academic_pipeline.legacy` para o runtime histórico;",
        "- script histórico `academic_pipeline_rc10.py` como compatibilidade.",
        "",
        "## Exclusões",
        "",
        "- não renomear pacotes ou módulos;",
        "- não remover wrappers, facades ou aliases;",
        "- não alterar o runtime acadêmico;",
        "- não preencher `project.dependencies` nesta subfase;",
        "- não decidir package data por contagem bruta;",
        "- não antecipar reorganizações da AP-006.",
        "",
        "## Interpretação das evidências",
        "",
        "A instalação observada no Pipenv atual prova somente que existe uma "
        "distribuição `0.1.0` importável a partir de `site-packages` e um console "
        "script no `bin` desse ambiente. Ela não é gate de encerramento porque "
        "pode refletir instalação anterior e não foi construída em descendente "
        "temporário limpo.",
        "",
        "Os 274 arquivos rastreados sob `academic_pipeline` e `app_bundle` não "
        "podem ser comparados diretamente aos 141 registros da distribuição "
        "instalada. Os universos têm semânticas diferentes.",
        "",
        "## Gates da AP-005E.2",
        "",
    ]
    lines.extend(f"{index}. {gate}." for index, gate in enumerate(
        payload["ap005e2_gates"], start=1
    ))
    lines.extend(
        [
            "",
            "## Critério de aplicação",
            "",
            "A AP-005E.3 somente poderá alterar metadata, package data ou "
            "entrypoints quando a AP-005E.2 reproduzir um defeito concreto no "
            "artefato instalado. Caso todos os gates passem, a aplicação será "
            "formalmente `no-op`.",
            "",
        ]
    )
    return "\n".join(lines)


def generated_files(payload: dict[str, Any]) -> dict[Path, bytes]:
    return {
        INVENTORY_JSON: (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        INVENTORY_MD: (render_inventory(payload).rstrip("\n") + "\n").encode("utf-8"),
        STRATEGY_MD: (render_strategy(payload).rstrip("\n") + "\n").encode("utf-8"),
    }


def write_files(files: dict[Path, bytes]) -> None:
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"[WRITE] {path.relative_to(ROOT)}")


def check_files(files: dict[Path, bytes]) -> None:
    failures: list[str] = []
    for path, expected in files.items():
        if not path.is_file():
            failures.append(f"ausente: {path.relative_to(ROOT)}")
            continue
        actual = path.read_bytes()
        if actual != expected:
            failures.append(f"divergente: {path.relative_to(ROOT)}")
    if failures:
        raise RuntimeError("; ".join(failures))
    for path in files:
        print(f"[OK] {path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_git_baseline()
    payload = build_inventory()
    files = generated_files(payload)
    if args.write:
        write_files(files)
    else:
        check_files(files)
    print(f"fingerprint={payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
