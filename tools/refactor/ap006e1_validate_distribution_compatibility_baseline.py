#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

PHASE = "AP-006E.1"
SOURCE_COMMIT_DEFAULT = "aed79d72f6c26fabcdda00f25d058b32fdc3fd75"
CANONICAL_REL = "software/academic_pipeline_mppg"
BRIDGE_REL = "software/academic_pipeline_rc10_7_conformidade"
OLD_TOKEN = "academic_pipeline_rc10_7_conformidade"
NEW_TOKEN = "academic_pipeline_mppg"
EXPECTED_DISTRIBUTION = "academic-pipeline-mppg"
EXPECTED_CONSOLE = {"academic-pipeline": "academic_pipeline.cli:main"}
EXPECTED_PYTHON = ">=3.11"
EXPECTED_BRIDGE_TARGET = "academic_pipeline_mppg"

DOC_REL = "docs/refactor/academic-pipeline/AP-006/AP-006E1_DISTRIBUTION_COMPATIBILITY_BASELINE.md"
JSON_REL = "docs/refactor/academic-pipeline/AP-006/ap006e1_distribution_compatibility_baseline.json"
TEST_REL = "software/academic_pipeline_mppg/tests/characterization/test_ap006e1_distribution_compatibility_baseline.py"
VALIDATOR_REL = "tools/refactor/ap006e1_validate_distribution_compatibility_baseline.py"
CONTRACT_OWNED_PATHS = frozenset({DOC_REL, JSON_REL, TEST_REL, VALIDATOR_REL})

DISTRIBUTION_ROOT_FILES = frozenset({
    f"{CANONICAL_REL}/pyproject.toml",
    f"{CANONICAL_REL}/Pipfile",
    f"{CANONICAL_REL}/Pipfile.lock",
    f"{CANONICAL_REL}/requirements.txt",
    f"{CANONICAL_REL}/install_rc10.sh",
    f"{CANONICAL_REL}/setup_pipenv_env.sh",
    f"{CANONICAL_REL}/.env.template",
})
ACTIVE_SUFFIXES = frozenset({
    ".py", ".sh", ".bash", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".service", ".desktop"
})
HISTORICAL_SUFFIXES = frozenset({
    ".md", ".json", ".txt", ".log", ".csv", ".tsv", ".org", ".el", ".bib", ".tex"
})
DEFERRED_COMPONENTS = frozenset({
    "output", "outputs", "cache", ".academic_pipeline", "build", "dist", ".pytest_cache", "dados_prisma"
})
BACKUP_COMPONENT_PATTERN = re.compile(
    r"(?:^|[._-])(?:backup|backups|bak|archive|snapshot|patch[_-]?backups?)(?:$|[._-])",
    re.IGNORECASE,
)


def run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = run(["git", *args], cwd=repo, check=check)
    return result.stdout


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def source_blob(repo: Path, commit: str, path: str) -> bytes:
    return git_bytes(repo, "show", f"{commit}:{path}")


def source_text(repo: Path, commit: str, path: str) -> str:
    return source_blob(repo, commit, path).decode("utf-8", errors="replace")


def source_tree_paths(repo: Path, commit: str) -> set[str]:
    raw = git_bytes(repo, "ls-tree", "-r", "-z", "--name-only", commit)
    return {
        part.decode("utf-8", errors="surrogateescape")
        for part in raw.split(b"\0")
        if part
    }


def backup_component(part: str) -> bool:
    lowered = part.lower()
    return (
        lowered in {"backup", "backups", ".patch_backups", "archive", "archives", "snapshots"}
        or bool(BACKUP_COMPONENT_PATTERN.search(lowered))
    )


def classify_reference_path(path: str) -> str:
    pure = PurePosixPath(path)
    parts = pure.parts
    part_set = set(parts)
    suffix = pure.suffix.lower()
    backup_count = sum(1 for part in parts if backup_component(part))

    if path in CONTRACT_OWNED_PATHS:
        return "contract_owned_path"
    if backup_count >= 2 or len(path) > 300:
        return "pathological_recursive_backup_evidence"
    if backup_count == 1:
        return "explicit_backup_archive_component"
    if part_set & DEFERRED_COMPONENTS:
        return "scan_deferred_component"
    if path in DISTRIBUTION_ROOT_FILES:
        return "canonical_distribution_contract"
    if path.startswith(f"{CANONICAL_REL}/tests/") or "/tests/" in path or path.startswith("tools/refactor/"):
        return "canonical_test_or_validator"
    if path.startswith(f"{CANONICAL_REL}/docs/") or path.startswith("docs/"):
        return "historical_evidence"
    if path.startswith(f"{CANONICAL_REL}/") and suffix in ACTIVE_SUFFIXES:
        return "canonical_runtime_or_config"
    if not path.startswith(f"{CANONICAL_REL}/") and suffix in ACTIVE_SUFFIXES:
        return "external_operational_source"
    if not path.startswith(f"{CANONICAL_REL}/") and suffix == ".org":
        return "external_document_source"
    if suffix in HISTORICAL_SUFFIXES:
        return "historical_evidence"
    return "other_tracked_reference"


def scan_physical_name_references(repo: Path, commit: str) -> dict[str, Any]:
    result = run(
        ["git", "grep", "-n", "-I", "-e", OLD_TOKEN, "-e", NEW_TOKEN, commit, "--", "."],
        cwd=repo,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or "git grep falhou")

    classification_counts: Counter[str] = Counter()
    token_occurrence_counts: Counter[str] = Counter()
    active_records: list[dict[str, Any]] = []
    candidate_paths: set[str] = set()
    line_count = 0

    for raw_line in result.stdout.splitlines():
        prefix, path, line_text, text = raw_line.split(":", 3)
        if prefix != commit:
            raise AssertionError(f"Prefixo inesperado no git grep: {prefix}")
        line_number = int(line_text)
        tokens = {
            OLD_TOKEN: text.count(OLD_TOKEN),
            NEW_TOKEN: text.count(NEW_TOKEN),
        }
        tokens = {key: value for key, value in tokens.items() if value}
        if not tokens:
            continue
        classification = classify_reference_path(path)
        classification_counts[classification] += 1
        for key, value in tokens.items():
            token_occurrence_counts[key] += value
        line_count += 1

        if classification in {
            "canonical_distribution_contract",
            "canonical_runtime_or_config",
            "canonical_test_or_validator",
            "external_operational_source",
            "external_document_source",
        }:
            record = {
                "path": path,
                "line": line_number,
                "classification": classification,
                "tokens": tokens,
                "text_sha256": sha256_bytes(text.encode("utf-8", errors="surrogateescape")),
                "preview": text[:500],
            }
            active_records.append(record)
            if OLD_TOKEN in tokens:
                candidate_paths.add(path)

    active_records.sort(key=lambda row: (row["path"], row["line"], row["classification"]))
    if sum(classification_counts.values()) != line_count:
        raise AssertionError("Partição por classificação não cobre todas as linhas.")
    if CONTRACT_OWNED_PATHS & candidate_paths:
        raise AssertionError("Artefatos próprios entraram no conjunto candidato.")

    return {
        "matched_line_count": line_count,
        "matched_occurrence_count": sum(token_occurrence_counts.values()),
        "classification_counts": dict(sorted(classification_counts.items())),
        "token_occurrence_counts": dict(sorted(token_occurrence_counts.items())),
        "active_record_count": len(active_records),
        "active_records": active_records,
        "recorded_candidate_path_count": len(candidate_paths),
        "recorded_candidate_paths": sorted(candidate_paths),
        "coverage_assertion": "sum(classification_counts) == matched_line_count",
        "groups_are_disjoint": True,
        "contract_owned_paths_excluded": True,
    }


def parse_public_contract(repo: Path, commit: str) -> dict[str, Any]:
    pyproject_path = f"{CANONICAL_REL}/pyproject.toml"
    pyproject = tomllib.loads(source_text(repo, commit, pyproject_path))
    project = pyproject.get("project", {})
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    packages_find = setuptools.get("packages", {}).get("find", {})

    pipfile = tomllib.loads(source_text(repo, commit, f"{CANONICAL_REL}/Pipfile"))
    pipfile_lock = json.loads(source_text(repo, commit, f"{CANONICAL_REL}/Pipfile.lock"))

    scripts = dict(project.get("scripts", {}))
    if project.get("name") != EXPECTED_DISTRIBUTION:
        raise AssertionError(f"Distribuição inesperada: {project.get('name')}")
    if scripts != EXPECTED_CONSOLE:
        raise AssertionError(f"Console inesperado: {scripts}")
    if project.get("requires-python") != EXPECTED_PYTHON:
        raise AssertionError(f"requires-python inesperado: {project.get('requires-python')}")

    tree_paths = source_tree_paths(repo, commit)
    package_roots = {
        "academic_pipeline": f"{CANONICAL_REL}/academic_pipeline/__init__.py" in tree_paths,
        "app_bundle": f"{CANONICAL_REL}/app_bundle/__init__.py" in tree_paths,
    }
    if not all(package_roots.values()):
        raise AssertionError(f"Pacotes públicos ausentes: {package_roots}")

    return {
        "distribution_name": project.get("name"),
        "distribution_version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "console_scripts": scripts,
        "package_roots": package_roots,
        "setuptools_package_find": packages_find,
        "include_package_data": setuptools.get("include-package-data"),
        "package_data_keys": sorted(setuptools.get("package-data", {}).keys()),
        "pipfile_python_version": pipfile.get("requires", {}).get("python_version"),
        "pipfile_lock_python_version": pipfile_lock.get("_meta", {}).get("requires", {}).get("python_version"),
    }


def parse_entrypoint_chain(repo: Path, commit: str) -> dict[str, Any]:
    paths = {
        "module_entrypoint": f"{CANONICAL_REL}/academic_pipeline/__main__.py",
        "cli": f"{CANONICAL_REL}/academic_pipeline/cli.py",
        "legacy": f"{CANONICAL_REL}/academic_pipeline/legacy.py",
    }
    payload: dict[str, Any] = {"files": {}}
    for label, path in paths.items():
        text = source_text(repo, commit, path)
        tree = ast.parse(text, filename=path)
        functions = {
            node.name: {
                "positional": [arg.arg for arg in node.args.args],
                "keyword_only": [arg.arg for arg in node.args.kwonlyargs],
                "vararg": node.args.vararg.arg if node.args.vararg else None,
                "kwarg": node.args.kwarg.arg if node.args.kwarg else None,
            }
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        payload["files"][label] = {
            "path": path,
            "sha256": sha256_bytes(text.encode("utf-8")),
            "functions": functions,
            "contains_cli_main_reference": "academic_pipeline.cli" in text or "from .cli import main" in text,
            "contains_legacy_reference": "run_legacy" in text,
        }

    cli_main = payload["files"]["cli"]["functions"].get("main")
    legacy_run = payload["files"]["legacy"]["functions"].get("run_legacy")
    if cli_main is None:
        raise AssertionError("academic_pipeline.cli.main não encontrado via AST.")
    if legacy_run is None:
        raise AssertionError("academic_pipeline.legacy.run_legacy não encontrado via AST.")
    if not payload["files"]["module_entrypoint"]["contains_cli_main_reference"]:
        raise AssertionError("__main__.py não referencia o main do CLI.")

    payload["declared_chain"] = [
        "academic-pipeline -> academic_pipeline.cli:main",
        "python -m academic_pipeline -> academic_pipeline.cli:main",
        "academic_pipeline.cli:main -> academic_pipeline.legacy:run_legacy (fallback preservado)",
    ]
    payload["inspection_method"] = "AST e blobs Git; sem importar o orquestrador completo"
    return payload


def bridge_contract(repo: Path, commit: str) -> dict[str, Any]:
    raw = git(repo, "ls-tree", commit, BRIDGE_REL).strip()
    if not raw:
        raise AssertionError("Ponte não existe na tree-fonte.")
    metadata, path = raw.split("\t", 1)
    mode, object_type, oid = metadata.split()
    target = source_blob(repo, commit, BRIDGE_REL).decode("utf-8")
    if mode != "120000" or object_type != "blob":
        raise AssertionError(f"Ponte não é symlink Git: {raw}")
    if target != EXPECTED_BRIDGE_TARGET:
        raise AssertionError(f"Destino da ponte inesperado: {target}")
    return {
        "path": path,
        "mode": mode,
        "oid": oid,
        "target": target,
        "required_through_phase": "AP-006F",
        "removal_authorized": False,
    }


def build_inventory(repo: Path, source_commit: str) -> dict[str, Any]:
    source_tree = git(repo, "rev-parse", f"{source_commit}^{{tree}}").strip()
    source_date = git(repo, "show", "-s", "--format=%cI", source_commit).strip()
    deterministic = {
        "phase": PHASE,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "public_contract": parse_public_contract(repo, source_commit),
        "entrypoint_chain": parse_entrypoint_chain(repo, source_commit),
        "compatibility_bridge": bridge_contract(repo, source_commit),
        "reference_partition": scan_physical_name_references(repo, source_commit),
        "scope_decision": {
            "productive_files_changed_in_ap006e1": 0,
            "contract_artifact_count": len(CONTRACT_OWNED_PATHS),
            "neutral_uninstalled_import_failure": "expected_environment_observation_not_source_regression",
            "persistent_installation_allowed": False,
            "persistent_pth_allowed": False,
            "fallback_removal_phase": "AP-006F",
            "next_gate": "isolated_build_install_console_and_subprocess_validation",
        },
        "contract_owned_paths": sorted(CONTRACT_OWNED_PATHS),
    }
    fingerprint = sha256_bytes(canonical_json(deterministic))
    return {
        "schema_version": 1,
        "phase": PHASE,
        "status": "distribution_compatibility_baseline_materialized",
        "source_commit_date": source_date,
        "inventory_fingerprint_sha256": fingerprint,
        "deterministic": deterministic,
    }


def render_markdown(data: dict[str, Any]) -> str:
    d = data["deterministic"]
    public = d["public_contract"]
    refs = d["reference_partition"]
    lines = [
        "# AP-006E.1 — Baseline de distribuição e compatibilidade",
        "",
        "## Estado",
        "",
        f"- Commit-fonte: `{d['source_commit']}`",
        f"- Tree-fonte: `{d['source_tree']}`",
        f"- Fingerprint: `{data['inventory_fingerprint_sha256']}`",
        "- Natureza: contrato declarativo e teste de caracterização.",
        "- Código produtivo alterado nesta subfase: **zero**.",
        "",
        "## Contrato público preservado",
        "",
        f"- Distribuição: `{public['distribution_name']}` versão `{public['distribution_version']}`.",
        f"- Python: `{public['requires_python']}`.",
        "- Console: `academic-pipeline = academic_pipeline.cli:main`.",
        "- Módulos públicos: `academic_pipeline` e `app_bundle`.",
        "- `python -m academic_pipeline` converge para o mesmo `main`.",
        "- O fallback `academic_pipeline.legacy:run_legacy` permanece preservado.",
        "",
        "## Ponte de compatibilidade",
        "",
        f"- `{BRIDGE_REL} -> {EXPECTED_BRIDGE_TARGET}`.",
        "- Retenção obrigatória durante toda a AP-006E.",
        "- Remoção, substituição ou retenção definitiva pertence à AP-006F.",
        "",
        "## Partição das referências físicas",
        "",
        f"- Linhas classificadas: **{refs['matched_line_count']}**.",
        f"- Ocorrências dos dois nomes: **{refs['matched_occurrence_count']}**.",
        f"- Registros ativos para revisão: **{refs['active_record_count']}**.",
        f"- Caminhos candidatos com nome legado: **{refs['recorded_candidate_path_count']}**.",
        "",
    ]
    for name, count in refs["classification_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "As classes são disjuntas, cobrem integralmente as linhas encontradas e",
        "separam backups explícitos, caminhos adiados, evidência patológica e",
        "documentação histórica de consumidores operacionais.",
        "",
        "## Decisão ambiental",
        "",
        "A ausência da distribuição no virtualenv persistente e a falha de importação",
        "em diretório neutro sem instalação são observações ambientais. Elas não",
        "contradizem o contrato da árvore-fonte. A validação distributiva deverá ser",
        "feita em ambiente temporário isolado, sem instalação persistente e sem `.pth`",
        "residual.",
        "",
        "## Próximo gate",
        "",
        "Executar build, instalação, console, `python -m`, subprocessos e imports em",
        "clone/venv temporários. Somente evidência dessa validação poderá justificar",
        "alterações produtivas na AP-006E.2 ou AP-006E.3.",
        "",
    ])
    return "\n".join(lines)


def probe_import(cwd: Path) -> dict[str, Any]:
    code = r'''
import json
payload = {}
for name in ("academic_pipeline", "app_bundle"):
    try:
        module = __import__(name)
        payload[name] = {"status": "ok", "file": getattr(module, "__file__", None)}
    except Exception as exc:
        payload[name] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if all(item.get("status") == "ok" for item in payload.values()) else 1)
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = run([sys.executable, "-E", "-c", code], cwd=cwd, check=False, env=env)
    payload = json.loads(result.stdout.strip() or "{}")
    return {"returncode": result.returncode, "modules": payload, "stderr": result.stderr.strip()}


def current_environment_validation(repo: Path) -> dict[str, Any]:
    canonical = (repo / CANONICAL_REL).resolve()
    bridge = repo / BRIDGE_REL
    if not bridge.is_symlink():
        raise AssertionError("Ponte atual não é link simbólico.")
    if os.readlink(bridge) != EXPECTED_BRIDGE_TARGET:
        raise AssertionError(f"Destino textual atual da ponte: {os.readlink(bridge)}")
    if bridge.resolve() != canonical:
        raise AssertionError("Ponte atual não resolve para a árvore canônica.")

    current_pyproject = tomllib.loads((canonical / "pyproject.toml").read_text(encoding="utf-8"))
    project = current_pyproject.get("project", {})
    if project.get("name") != EXPECTED_DISTRIBUTION:
        raise AssertionError("Nome atual da distribuição divergiu.")
    if dict(project.get("scripts", {})) != EXPECTED_CONSOLE:
        raise AssertionError("Console atual divergiu.")

    source_probe = probe_import(canonical)
    bridge_probe = probe_import(bridge)
    for label, probe in (("canonical", source_probe), ("bridge", bridge_probe)):
        if probe["returncode"] != 0:
            raise AssertionError(f"Importação pela árvore {label} falhou: {probe}")
        for module_name, module in probe["modules"].items():
            if module["status"] != "ok":
                raise AssertionError(f"{label}/{module_name} falhou: {module}")
            resolved = Path(module["file"]).resolve()
            if canonical not in resolved.parents:
                raise AssertionError(f"{label}/{module_name} importado fora da árvore canônica: {resolved}")

    try:
        distribution = importlib.metadata.distribution(EXPECTED_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        distribution = None

    with tempfile.TemporaryDirectory(prefix="ap006e1_neutral_") as temp_dir:
        neutral_probe = probe_import(Path(temp_dir))

    purelib = Path(sysconfig.get_paths()["purelib"])
    relevant_pth = []
    for path in sorted(purelib.glob("*.pth"), key=lambda item: item.name):
        text = path.read_text(encoding="utf-8", errors="replace")
        if str(repo) in text or OLD_TOKEN in text or NEW_TOKEN in text:
            relevant_pth.append(str(path))
    if relevant_pth:
        raise AssertionError(f"Overlay .pth persistente detectado: {relevant_pth}")

    console_path = Path(sys.executable).parent / "academic-pipeline"
    if distribution is None:
        if neutral_probe["returncode"] == 0:
            raise AssertionError("Imports neutros funcionam sem distribuição instalada; possível overlay implícito.")
        mode = "source_tree_only_uninstalled_distribution"
        if console_path.exists():
            raise AssertionError("Console existe no venv sem metadado de distribuição.")
    else:
        if neutral_probe["returncode"] != 0:
            raise AssertionError("Distribuição instalada, mas imports neutros falharam.")
        if not console_path.is_file():
            raise AssertionError("Distribuição instalada sem console academic-pipeline.")
        mode = "installed_distribution"

    return {
        "mode": mode,
        "distribution_installed": distribution is not None,
        "source_probe": source_probe,
        "bridge_probe": bridge_probe,
        "neutral_probe": neutral_probe,
        "console_path": str(console_path) if console_path.exists() else None,
        "relevant_pth_files": relevant_pth,
        "purelib": str(purelib),
    }


def validate(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    data = json.loads((repo / JSON_REL).read_text(encoding="utf-8"))
    if data.get("phase") != PHASE:
        raise AssertionError("Fase incorreta no contrato.")
    deterministic = data["deterministic"]
    source_commit = deterministic["source_commit"]

    ancestor = run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=repo,
        check=False,
    )
    if ancestor.returncode != 0:
        raise AssertionError("Commit-fonte não é ancestral do HEAD atual.")

    recomputed = build_inventory(repo, source_commit)
    if recomputed["inventory_fingerprint_sha256"] != data["inventory_fingerprint_sha256"]:
        raise AssertionError("Fingerprint do inventário não é reproduzível.")
    if recomputed["deterministic"] != deterministic:
        raise AssertionError("Conteúdo determinístico divergiu da fonte Git.")

    candidates = set(deterministic["reference_partition"]["recorded_candidate_paths"])
    if not CONTRACT_OWNED_PATHS.isdisjoint(candidates):
        raise AssertionError("Artefatos do contrato autoclassificados como candidatos.")

    missing = [path for path in CONTRACT_OWNED_PATHS if not (repo / path).is_file()]
    if missing:
        raise AssertionError(f"Artefatos próprios ausentes: {missing}")

    head = git(repo, "rev-parse", "HEAD").strip()
    tracked = set(git(repo, "ls-files").splitlines())
    if head != source_commit and not CONTRACT_OWNED_PATHS <= tracked:
        raise AssertionError("Após mudança do HEAD, todos os artefatos próprios devem estar rastreados.")

    environment = current_environment_validation(repo)
    return {
        "phase": PHASE,
        "status": "ok",
        "source_commit": source_commit,
        "source_tree": deterministic["source_tree"],
        "inventory_fingerprint_sha256": data["inventory_fingerprint_sha256"],
        "contract_owned_path_count": len(CONTRACT_OWNED_PATHS),
        "candidate_path_count": deterministic["reference_partition"]["recorded_candidate_path_count"],
        "environment_mode": environment["mode"],
        "distribution_installed": environment["distribution_installed"],
        "relevant_pth_file_count": len(environment["relevant_pth_files"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", default=SOURCE_COMMIT_DEFAULT)
    parser.add_argument("--emit-json", type=Path)
    parser.add_argument("--emit-md", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()

    if bool(args.emit_json) != bool(args.emit_md):
        parser.error("--emit-json e --emit-md devem ser usados juntos")
    if args.emit_json and args.emit_md:
        inventory = build_inventory(repo, args.source_commit)
        args.emit_json.parent.mkdir(parents=True, exist_ok=True)
        args.emit_md.parent.mkdir(parents=True, exist_ok=True)
        args.emit_json.write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.emit_md.write_text(render_markdown(inventory), encoding="utf-8")

    summary = validate(repo)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
