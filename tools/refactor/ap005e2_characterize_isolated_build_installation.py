#!/usr/bin/env python3
"""Caracterização reproduzível do build e da instalação isolada da AP-005E.2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_REL = Path("software/academic_pipeline_rc10_7_conformidade")
SOFTWARE_ROOT = ROOT / SOFTWARE_REL
DOC_DIR = ROOT / "docs/refactor/academic-pipeline/AP-005"

EXPECTED_BRANCH = "ap-refactor/04-consumer-canonicalization"
EXPECTED_BASELINE_COMMIT = "0d553c975ad7948762f74aa4fcff3903578712de"
EXPECTED_UPSTREAM = "origin/ap-refactor/04-consumer-canonicalization"

SCHEMA_VERSION = "ap005e2.isolated-build-installation-characterization.v1"

CHARACTERIZATION_JSON = (
    DOC_DIR / "ap005e2_isolated_build_installation_characterization.json"
)
REPORT_MD = (
    DOC_DIR / "AP-005E2_ISOLATED_BUILD_INSTALLATION_CHARACTERIZATION.md"
)
SCOPE_MD = DOC_DIR / "AP-005E2_CORRECTION_SCOPE.md"

PYPROJECT_REL = SOFTWARE_REL / "pyproject.toml"
REQUIREMENTS_REL = SOFTWARE_REL / "requirements.txt"
PIPFILE_REL = SOFTWARE_REL / "Pipfile"

SELECTED_PACKAGE_PREFIXES = (
    "academic_pipeline/",
    "app_bundle/scripts/",
)
SELECTED_ROOT_PACKAGE_FILES = {
    "app_bundle/__init__.py",
}

KEY_SOURCE_FILES = {
    "article_workflow_wrapper": (
        SOFTWARE_REL / "app_bundle/scripts/pipeline/artigo_prisma_workflow.py"
    ),
    "prisma_orchestration": (
        SOFTWARE_REL / "academic_pipeline/prisma_generic_orchestration.py"
    ),
    "project_tools": (
        SOFTWARE_REL / "app_bundle/scripts/pipeline/project_tools.py"
    ),
}

DYNAMIC_EVIDENCE = {
    "logs": [
        "ap005e2_build_instalacao_isolada_20260718_075814.log",
        "ap005e2_dependencias_package_data_v3_20260718_080928.log",
    ],
    "build": {
        "wheel": {
            "filename": "academic_pipeline_mppg-0.1.0-py3-none-any.whl",
            "sha256": (
                "cea15ade083c2a0a530693dc04cdabe192de049bc7a4078e2f769cb456ade85c"
            ),
            "size_kib_approx": 358,
            "python_file_count": 66,
            "package_non_python_file_count": 0,
            "entry_count": 71,
            "strong_residues": [],
        },
        "sdist": {
            "filename": "academic_pipeline_mppg-0.1.0.tar.gz",
            "sha256": (
                "527ce87fb2702ef6906e605e4bcbb93d33a06c56c53059cb5b1f2d1b8c316318"
            ),
            "entry_count": 74,
            "strong_residues": [],
            "contains_pyproject": True,
            "contains_requirements": False,
            "contains_pipfile": False,
            "contains_pipfile_lock": False,
        },
        "metadata": {
            "name": "academic-pipeline-mppg",
            "version": "0.1.0",
            "requires_python": ">=3.11",
            "requires_dist": [],
            "console_script": "academic-pipeline = academic_pipeline.cli:main",
        },
        "reproducible_wheel_hash_observations": 3,
    },
    "wheel_only_environment": {
        "installation_succeeded": True,
        "pip_check_returncode": 0,
        "academic_pipeline_imported_from_temporary_venv": True,
        "legacy_runtime_imported": False,
        "legacy_runtime_error": "ModuleNotFoundError: No module named 'dotenv'",
        "console_help_returncode": 1,
        "module_help_returncode": 1,
    },
    "requirements_assisted_environment": {
        "requirements_install_returncode": 0,
        "academic_pipeline_imported_from_temporary_venv": True,
        "legacy_runtime_imported": True,
        "console_help_returncode": 0,
        "module_help_returncode": 0,
        "help_stdout_equal": True,
        "help_stderr_equal": True,
    },
    "passive_module_imports": {
        "requirements_only": {
            "total": 65,
            "passed": 64,
            "failed": 1,
        },
        "pipfile_direct_dependencies": {
            "total": 65,
            "passed": 64,
            "failed": 1,
        },
        "failure": {
            "module": "app_bundle.scripts.pipeline.artigo_prisma_workflow",
            "error": "ModuleNotFoundError: No module named 'article_workflow'",
        },
    },
    "installed_operational_commands": {
        "list_institutions": {
            "returncode": 0,
            "stdout": "Nenhum perfil institucional encontrado.",
        },
        "explain_profile_fgv": {
            "returncode": 1,
            "error": (
                "FileNotFoundError: Perfil institucional não encontrado: "
                "fgv. Disponíveis: nenhum"
            ),
        },
        "init_project_fgv": {
            "returncode": 1,
            "error": (
                "FileNotFoundError: Template TOML não encontrado: "
                "app_bundle/config/examples/atividade_rc10_exemplo.toml"
            ),
        },
        "doctor": {
            "returncode": 2,
            "classification": "environmental_diagnostic_not_packaging_gate",
            "python_dependencies_detected": True,
            "external_tool_failures": ["lualatex", "biber"],
        },
        "list_layouts_attempt": {
            "returncode": 1,
            "classification": "invalid_probe_not_used_as_defect_evidence",
            "reason": "--list-layouts exige --config caminho.toml",
        },
    },
}

EXPECTED_DATA_COUNTS = {
    "tracked_non_python_total": 184,
    "institutions": 18,
    "templates": 9,
    "prompts": 9,
    "misc": 6,
    "projetos": 111,
}

EXPECTED_DIRECT_REQUIREMENTS = [
    "openai>=1.0.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
    "pypdf>=4.0",
    "python-docx>=1.1",
    "openpyxl>=3.1",
]

EXPECTED_PIPFILE_DIRECT_NAMES = [
    "matplotlib",
    "openai",
    "openpyxl",
    "prompt-toolkit",
    "pydantic",
    "pypdf",
    "python-docx",
    "python-dotenv",
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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_git_baseline() -> None:
    branch = run_git("branch", "--show-current").stdout.strip()
    head = run_git("rev-parse", "HEAD").stdout.strip()

    if branch:
        upstream_proc = run_git(
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            check=False,
        )
        if upstream_proc.returncode != 0:
            raise RuntimeError(
                "Branch ativa sem upstream resolvível: "
                f"{branch}: {upstream_proc.stderr.strip()}"
            )
        upstream = upstream_proc.stdout.strip()

        if branch != EXPECTED_BRANCH:
            raise RuntimeError(f"Branch divergente: {branch}")
        if upstream != EXPECTED_UPSTREAM:
            raise RuntimeError(f"Upstream divergente: {upstream}")
    else:
        # Descendentes temporários são validados em HEAD destacado.
        upstream = None

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_BASELINE_COMMIT,
            head,
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "O baseline AP-005E.1 não é ancestral do HEAD atual: "
            f"{EXPECTED_BASELINE_COMMIT} -> {head}"
        )


def tracked_files() -> list[str]:
    proc = subprocess.run(
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
        raw.decode("utf-8")
        for raw in proc.stdout.split(b"\0")
        if raw
    )


def source_snapshot() -> dict[str, Any]:
    pyproject = tomllib.loads(
        (ROOT / PYPROJECT_REL).read_text(encoding="utf-8")
    )
    requirements = [
        line.strip()
        for line in (ROOT / REQUIREMENTS_REL)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pipfile = tomllib.loads(
        (ROOT / PIPFILE_REL).read_text(encoding="utf-8")
    )

    files = tracked_files()
    project_prefix = f"{SOFTWARE_REL.as_posix()}/"
    relative_files = [
        path.removeprefix(project_prefix)
        for path in files
    ]

    non_test_python = sorted(
        path
        for path in relative_files
        if path.endswith(".py")
        and not path.startswith("app_bundle/tests/")
    )
    packaged_python_source = sorted(
        path
        for path in non_test_python
        if (
            path.startswith(SELECTED_PACKAGE_PREFIXES)
            or path in SELECTED_ROOT_PACKAGE_FILES
        )
    )
    tracked_python_outside_wheel = sorted(
        set(non_test_python) - set(packaged_python_source)
    )
    tracked_non_python = sorted(
        path
        for path in relative_files
        if not path.endswith(".py")
    )

    data_counts = {}
    for name in ("institutions", "templates", "prompts", "misc", "projetos"):
        prefix = f"app_bundle/{name}/"
        data_counts[name] = sum(
            path.startswith(prefix)
            for path in tracked_non_python
        )
    data_counts["tracked_non_python_total"] = len(tracked_non_python)

    source_text = {
        name: (ROOT / path).read_text(
            encoding="utf-8",
            errors="replace",
        )
        for name, path in KEY_SOURCE_FILES.items()
    }

    article_tree = ast.parse(
        source_text["article_workflow_wrapper"],
        filename=str(KEY_SOURCE_FILES["article_workflow_wrapper"]),
    )
    absolute_article_imports = []
    for node in ast.walk(article_tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "article_workflow":
                absolute_article_imports.append(
                    {
                        "line": node.lineno,
                        "module": node.module,
                        "names": [alias.name for alias in node.names],
                    }
                )

    prisma_text = source_text["prisma_orchestration"]
    project_tools_text = source_text["project_tools"]

    snapshot = {
        "pyproject": {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "requires_python": pyproject["project"]["requires-python"],
            "dependencies": pyproject["project"].get("dependencies", []),
            "console_script": pyproject["project"]["scripts"][
                "academic-pipeline"
            ],
            "include_package_data": (
                pyproject.get("tool", {})
                .get("setuptools", {})
                .get("include-package-data")
            ),
            "package_data": (
                pyproject.get("tool", {})
                .get("setuptools", {})
                .get("package-data")
            ),
        },
        "requirements": requirements,
        "pipfile_direct_names": sorted(pipfile.get("packages", {})),
        "packaged_python_source_count": len(packaged_python_source),
        "tracked_python_outside_wheel": tracked_python_outside_wheel,
        "tracked_data_counts": data_counts,
        "source_defects": {
            "article_workflow_absolute_imports": absolute_article_imports,
            "hardcoded_user_prompt_path": (
                "/home/gustavodetarso/" in prisma_text
            ),
            "self_invocation_via_dunder_file": (
                "[sys.executable, __file__" in prisma_text
            ),
            "helper_export_sibling_assumption": (
                "Path(__file__).with_name('prisma_exportar_bib.py')"
                in prisma_text
            ),
            "helper_freeze_sibling_assumption": (
                "Path(__file__).with_name('prisma_congelar_artigo.py')"
                in prisma_text
            ),
            "init_project_config_example_path": (
                "app_bundle / 'config' / 'examples'"
                in project_tools_text
                or 'app_bundle / "config" / "examples"'
                in project_tools_text
                or "app_bundle / 'config' / \"examples\""
                in project_tools_text
            ),
        },
    }

    if snapshot["pyproject"]["dependencies"] != []:
        raise RuntimeError("project.dependencies deixou de estar vazio.")
    if requirements != EXPECTED_DIRECT_REQUIREMENTS:
        raise RuntimeError(f"requirements.txt divergente: {requirements}")
    if snapshot["pipfile_direct_names"] != EXPECTED_PIPFILE_DIRECT_NAMES:
        raise RuntimeError(
            "Dependências diretas do Pipfile divergentes: "
            f"{snapshot['pipfile_direct_names']}"
        )
    if snapshot["packaged_python_source_count"] != 66:
        raise RuntimeError(
            "Contagem do manifesto Python selecionado divergente: "
            f"{snapshot['packaged_python_source_count']}"
        )
    if len(snapshot["tracked_python_outside_wheel"]) != 1:
        raise RuntimeError(
            "Quantidade de Python rastreado fora do wheel divergente: "
            f"{snapshot['tracked_python_outside_wheel']}"
        )
    if data_counts != EXPECTED_DATA_COUNTS:
        raise RuntimeError(
            f"Contagens de dados rastreados divergentes: {data_counts}"
        )
    defects = snapshot["source_defects"]
    if len(defects["article_workflow_absolute_imports"]) != 1:
        raise RuntimeError(
            "Import absoluto article_workflow não foi localizado exatamente uma vez."
        )
    for key in (
        "hardcoded_user_prompt_path",
        "self_invocation_via_dunder_file",
        "helper_export_sibling_assumption",
        "helper_freeze_sibling_assumption",
    ):
        if defects[key] is not True:
            raise RuntimeError(f"Risco esperado ausente: {key}")

    return snapshot


def build_payload() -> dict[str, Any]:
    snapshot = source_snapshot()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "AP-005E.2",
        "baseline": {
            "branch": EXPECTED_BRANCH,
            "commit": EXPECTED_BASELINE_COMMIT,
            "upstream": EXPECTED_UPSTREAM,
        },
        "source_snapshot": snapshot,
        "dynamic_evidence": DYNAMIC_EVIDENCE,
        "findings": [
            {
                "id": "distribution-dependencies-empty",
                "severity": "blocking",
                "classification": "confirmed_distribution_metadata_defect",
                "evidence": (
                    "O wheel declara Requires-Dist vazio; instalação isolada "
                    "é aceita por pip check, mas o entrypoint falha por ausência "
                    "de dotenv."
                ),
                "ap005e3_action": (
                    "Declarar dependências operacionais mínimas em "
                    "project.dependencies e alinhar requirements/Pipfile."
                ),
            },
            {
                "id": "operational-package-data-absent",
                "severity": "blocking",
                "classification": "confirmed_package_data_defect",
                "evidence": (
                    "O wheel contém zero arquivos não-Python; perfis FGV não "
                    "são encontrados e --init-project não localiza seu template."
                ),
                "ap005e3_action": (
                    "Definir allowlist de dados operacionais; não empacotar "
                    "indiscriminadamente projetos, outputs ou documentos históricos."
                ),
            },
            {
                "id": "article-workflow-absolute-import",
                "severity": "blocking",
                "classification": "confirmed_installed_import_defect",
                "evidence": (
                    "64 de 65 módulos passivos importam; "
                    "artigo_prisma_workflow falha com import absoluto "
                    "de article_workflow."
                ),
                "ap005e3_action": (
                    "Converter para import relativo/canônico preservando "
                    "a API pública article_workflow."
                ),
            },
            {
                "id": "prisma-helper-sibling-resolution",
                "severity": "blocking",
                "classification": "confirmed_installed_layout_defect",
                "evidence": (
                    "Os helpers existem em app_bundle.scripts.pipeline, mas "
                    "o módulo instalado os procura como irmãos de academic_pipeline."
                ),
                "ap005e3_action": (
                    "Resolver helpers por módulo canônico ou caminho derivado "
                    "do pacote app_bundle."
                ),
            },
            {
                "id": "hardcoded-personal-prompt-path",
                "severity": "blocking",
                "classification": "confirmed_portability_defect",
                "evidence": (
                    "O código produtivo instalado contém caminho absoluto "
                    "sob /home/gustavodetarso."
                ),
                "ap005e3_action": (
                    "Substituir por configuração explícita ou recurso "
                    "relativo/ausência segura."
                ),
            },
            {
                "id": "module-self-invocation-by-file",
                "severity": "risk",
                "classification": "installed_layout_risk_not_fully_exercised",
                "evidence": (
                    "Subprocessos invocam __file__ do módulo "
                    "prisma_generic_orchestration."
                ),
                "ap005e3_action": (
                    "Caracterizar e, se necessário, redirecionar para "
                    "python -m academic_pipeline ou módulo canônico."
                ),
            },
            {
                "id": "list-layouts-probe-invalid",
                "severity": "excluded",
                "classification": "invalid_probe_not_defect",
                "evidence": (
                    "A tentativa sem --config falhou conforme o contrato atual."
                ),
                "ap005e3_action": "Não usar esse resultado como evidência de defeito.",
            },
            {
                "id": "doctor-external-tool-errors",
                "severity": "excluded",
                "classification": "environmental_not_packaging_defect",
                "evidence": (
                    "O doctor detectou dependências Python, mas retornou 2 por "
                    "lualatex e biber ausentes no ambiente temporário."
                ),
                "ap005e3_action": "Não alterar metadata Python por esse resultado.",
            },
        ],
        "decisions": {
            "productive_change_required_in_ap005e2": False,
            "ap005e2_is_characterization_only": True,
            "ap005e3_correction_required": True,
            "ap005e3_may_be_noop": False,
            "preserve_distribution_name": True,
            "preserve_distribution_version": True,
            "preserve_console_script": True,
            "preserve_module_entrypoint": True,
            "preserve_public_main": True,
            "broad_package_reorganization_allowed": False,
            "package_all_tracked_non_python_allowed": False,
            "package_projects_outputs_or_historical_docs_allowed": False,
        },
        "ap005e3_scope": {
            "required": [
                "PEP 621 runtime dependencies",
                "operational package-data allowlist",
                "article_workflow import correction",
                "PRISMA helper resolution correction",
                "hardcoded prompt path correction",
                "fresh-wheel regression gates",
            ],
            "conditional": [
                "self-invocation by __file__ correction",
                "requirements/Pipfile normalization beyond runtime essentials",
            ],
            "forbidden": [
                "rename distribution",
                "change version",
                "remove legacy bridge",
                "change public entrypoints",
                "package app_bundle/projetos",
                "package app_bundle/output",
                "broad module reorganization",
            ],
        },
    }
    payload["fingerprint"] = sha256_bytes(canonical_bytes(payload))
    return payload


def render_report(payload: dict[str, Any]) -> str:
    dynamic = payload["dynamic_evidence"]
    source = payload["source_snapshot"]
    findings = payload["findings"]

    lines = [
        "# AP-005E.2 — Caracterização de build e instalação isolada",
        "",
        "## Baseline",
        "",
        f"- Branch: `{payload['baseline']['branch']}`",
        f"- Commit: `{payload['baseline']['commit']}`",
        f"- Upstream: `{payload['baseline']['upstream']}`",
        f"- Fingerprint: `{payload['fingerprint']}`",
        "",
        "## Artefatos construídos",
        "",
        (
            "- Wheel: "
            f"`{dynamic['build']['wheel']['filename']}` — "
            f"SHA-256 `{dynamic['build']['wheel']['sha256']}`."
        ),
        (
            "- Sdist: "
            f"`{dynamic['build']['sdist']['filename']}` — "
            f"SHA-256 `{dynamic['build']['sdist']['sha256']}`."
        ),
        (
            "- Wheel reproduzido com o mesmo hash em "
            f"**{dynamic['build']['reproducible_wheel_hash_observations']}** "
            "execuções."
        ),
        "- Metadata, nome, versão e console script: corretos.",
        "- Resíduos fortes nos arquivos de distribuição: nenhum.",
        "",
        "## Conteúdo instalado",
        "",
        (
            "- Arquivos Python no wheel: "
            f"**{dynamic['build']['wheel']['python_file_count']}**."
        ),
        (
            "- Arquivos Python da fonte selecionados pelo layout de pacotes: "
            f"**{source['packaged_python_source_count']}**."
        ),
        (
            "- Python rastreado fora do wheel: "
            f"`{source['tracked_python_outside_wheel'][0]}`."
        ),
        "- Arquivos não-Python do pacote no wheel: **0**.",
        (
            "- Arquivos não-Python rastreados sob as raízes do pacote: "
            f"**{source['tracked_data_counts']['tracked_non_python_total']}**."
        ),
        "- A contagem rastreada não é uma lista de inclusão: contém projetos, "
        "outputs, documentação e resíduos que não devem integrar o wheel.",
        "",
        "## Entry points",
        "",
        "- `academic-pipeline --help`: aprovado com dependências externas.",
        "- `python -m academic_pipeline --help`: aprovado com dependências externas.",
        "- stdout e stderr dos dois entrypoints: idênticos.",
        "- Os imports foram resolvidos pelo venv temporário, sem checkout ou PYTHONPATH.",
        "",
        "## Defeitos confirmados",
        "",
    ]
    for finding in findings:
        if finding["severity"] == "blocking":
            lines.append(
                f"- `{finding['id']}` — **{finding['classification']}**: "
                f"{finding['evidence']}"
            )
    lines.extend(
        [
            "",
            "## Resultados excluídos como defeito",
            "",
        ]
    )
    for finding in findings:
        if finding["severity"] == "excluded":
            lines.append(
                f"- `{finding['id']}` — {finding['evidence']}"
            )
    lines.extend(
        [
            "",
            "## Decisão",
            "",
            "A AP-005E.2 é uma subfase de caracterização e não altera código "
            "produtivo. Os defeitos reproduzidos tornam a AP-005E.3 obrigatória; "
            "ela não poderá ser encerrada como `no-op`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_scope(payload: dict[str, Any]) -> str:
    scope = payload["ap005e3_scope"]
    lines = [
        "# AP-005E.2 — Escopo vinculante da correção AP-005E.3",
        "",
        "## Princípio",
        "",
        "Corrigir apenas defeitos demonstrados na instalação isolada, preservando "
        "nome, versão, entrypoints públicos e bridge legado.",
        "",
        "## Correções obrigatórias",
        "",
    ]
    lines.extend(f"- {item}." for item in scope["required"])
    lines.extend(
        [
            "",
            "## Correções condicionais",
            "",
        ]
    )
    lines.extend(f"- {item}." for item in scope["conditional"])
    lines.extend(
        [
            "",
            "## Proibições",
            "",
        ]
    )
    lines.extend(f"- {item}." for item in scope["forbidden"])
    lines.extend(
        [
            "",
            "## Package data",
            "",
            "A correção deve usar uma allowlist operacional mínima. Não é permitido "
            "incluir os 184 arquivos não-Python em bloco. `app_bundle/projetos`, "
            "`app_bundle/output`, históricos e documentos de desenvolvimento "
            "permanecem fora do artefato.",
            "",
            "O gate mínimo deve comprovar em wheel novo:",
            "",
            "- `--list-institutions` encontra `fgv`;",
            "- `--explain-profile fgv` retorna com sucesso;",
            "- `--init-project` cria projeto em diretório externo;",
            "- templates, prompts, perfis e assets necessários vêm do venv instalado.",
            "",
            "## Dependências",
            "",
            "O wheel deve declarar as dependências necessárias para que seu "
            "entrypoint público execute após `pip install` normal. `pip check` "
            "sozinho não é suficiente: os entrypoints e módulos operacionais "
            "devem ser exercitados em venv novo.",
            "",
            "## Gate de encerramento da AP-005E.3",
            "",
            "- wheel e sdist construídos em clone limpo;",
            "- instalação normal do wheel, sem requirements externo;",
            "- `pip check` aprovado;",
            "- ambos os entrypoints aprovados fora do checkout;",
            "- 65 módulos passivos importáveis ou exclusão justificada;",
            "- comandos institucionais e `--init-project` aprovados;",
            "- helpers PRISMA resolvidos no layout instalado;",
            "- ausência de caminho pessoal em código produtivo;",
            "- suíte canônica integral aprovada.",
            "",
        ]
    )
    return "\n".join(lines)


def generated_files(payload: dict[str, Any]) -> dict[Path, bytes]:
    return {
        CHARACTERIZATION_JSON: (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).rstrip("\n")
            + "\n"
        ).encode("utf-8"),
        REPORT_MD: (render_report(payload).rstrip("\n") + "\n").encode("utf-8"),
        SCOPE_MD: (render_scope(payload).rstrip("\n") + "\n").encode("utf-8"),
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
        if path.read_bytes() != expected:
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
    payload = build_payload()
    files = generated_files(payload)
    if args.write:
        write_files(files)
    else:
        check_files(files)
    print(f"fingerprint={payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
