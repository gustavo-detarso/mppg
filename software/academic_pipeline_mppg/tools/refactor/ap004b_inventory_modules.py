#!/usr/bin/env python3
"""AP-004B — inventário preparatório de módulos e arquivos.

Ferramenta somente leitura produtiva. Ela não renomeia, move, copia nem edita
módulos do Academic Pipeline. A execução:

- valida diretório, branch, árvore, HEAD local/remoto e o commit AP-004A;
- valida a matriz AP-004A v4.2 e a arquitetura congelada da AP-003;
- inventaria seis módulos/arquivos priorizados pela AP-004A;
- mapeia imports estáticos, imports dinâmicos, subprocessos, referências de
  caminho/string, entrypoints e documentação/testes consumidores;
- compara as versões v1_13 e v1_14 do executor full-text sem escolher uma;
- gera manifesto de hashes e contratos AST para um futuro aplicador AP-004B;
- grava os cinco artefatos AP-004B e mantém durável o contrato AP-004A;
- cria backup externo, usa escrita atômica e realiza rollback integral;
- executa py_compile, git diff --check, suíte específica e suíte consolidada.

Execute a partir da raiz do software e mantenha este arquivo fora do
repositório, por exemplo em ~/Downloads.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import difflib
import errno
import hashlib
import json
import os
import py_compile
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, NoReturn, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


EXPECTED_SOFTWARE_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline/"
    "software/academic_pipeline_rc10_7_conformidade"
)
EXPECTED_REPOSITORY_ROOT = Path(
    "/home/gustavodetarso/Documentos/mppg-refactor-academic-pipeline"
)
EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
EXPECTED_REMOTE_REF = f"origin/{EXPECTED_BRANCH}"
EXPECTED_REMOTE_BRANCH_REF = f"refs/heads/{EXPECTED_BRANCH}"
EXPECTED_HEAD = "6de61fc9741035187836460d97da6d672708998a"
EXPECTED_AP004A_SUBJECT = (
    "chore(academic-pipeline): consolidar inventário de nomes da AP-004A"
)
SOFTWARE_DIRNAME = "academic_pipeline_rc10_7_conformidade"

PHASE = "AP-004B"
MODE = "module-file-inventory-v1.6-read-only"
TOOL_REVISION = "1.6"
INVENTORY_REVISION = "1.6"
TOOL_VERSION = 1
INVENTORY_SCHEMA_VERSION = 2
BASELINE_PASSED = 418
BASELINE_XFAILED = 3
EXPECTED_CONTRACT_TESTS = 13

DOC_DIR = Path("docs/refactor/academic-pipeline/AP-004")
REPORT_REL = DOC_DIR / "AP-004B_MODULE_FILE_INVENTORY.md"
STRATEGY_REL = DOC_DIR / "AP-004B_MODULE_FILE_STRATEGY.md"
INVENTORY_REL = DOC_DIR / "ap004b_module_file_inventory.json"
TOOL_REL = Path("tools/refactor/ap004b_inventory_modules.py")
TEST_REL = Path(
    "tests/characterization/test_ap004b_module_file_inventory_contract.py"
)
OUTPUT_RELS = (
    REPORT_REL,
    STRATEGY_REL,
    INVENTORY_REL,
    TOOL_REL,
    TEST_REL,
)

AP004A_REPORT_REL = DOC_DIR / "AP-004A_NAMING_INVENTORY.md"
AP004A_CONVENTION_REL = DOC_DIR / "AP-004_NAMING_CONVENTION.md"
AP004A_INVENTORY_REL = DOC_DIR / "ap004a_naming_inventory.json"
AP004A_TEST_REL = Path(
    "tests/characterization/test_ap004a_naming_inventory_contract.py"
)
AP004A_TOOL_REL = Path("tools/refactor/ap004a_inventory_names.py")
AP004A_COMMIT_FILES = (
    AP004A_REPORT_REL,
    AP004A_CONVENTION_REL,
    AP004A_INVENTORY_REL,
    AP004A_TEST_REL,
    AP004A_TOOL_REL,
)

# Manutenção contratual necessária para que a AP-004A continue válida após
# o próprio commit e durante subfases posteriores. Não altera produção.
AP004A_CONTRACT_MAINTENANCE_RELS = (AP004A_TEST_REL,)
OUTPUT_RELS = OUTPUT_RELS + AP004A_CONTRACT_MAINTENANCE_RELS

ORCHESTRATOR_REL = Path(
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
PACKAGE_MAIN_REL = Path("academic_pipeline/__main__.py")
PACKAGE_CLI_REL = Path("academic_pipeline/cli.py")
PACKAGE_LEGACY_REL = Path("academic_pipeline/legacy.py")
PRISMA_ORCHESTRATION_REL = Path(
    "academic_pipeline/prisma_generic_orchestration.py"
)
PYPROJECT_REL = Path("pyproject.toml")

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "pipeline_orchestrator",
        "current_path": ORCHESTRATOR_REL.as_posix(),
        "current_module": "app_bundle.scripts.pipeline.academic_pipeline_rc10",
        "bare_module": "academic_pipeline_rc10",
        "proposed_path": "app_bundle/scripts/pipeline/pipeline_orchestrator.py",
        "proposed_module": "app_bundle.scripts.pipeline.pipeline_orchestrator",
        "classification": "renomeação com compatibilidade",
        "target_phase": "AP-004B/AP-004E",
        "compatibility_policy": "wrapper obrigatório no caminho histórico",
        "rationale": (
            "Orquestrador canônico da AP-003; possui execução direta, imports "
            "internos e contrato histórico. A migração não pode remover o "
            "arquivo antigo de imediato."
        ),
    },
    {
        "key": "toml_generator",
        "current_path": (
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_v0_3_1.py"
        ),
        "current_module": (
            "app_bundle.scripts.pipeline."
            "academic_pipeline_toml_generator_v0_3_1"
        ),
        "bare_module": "academic_pipeline_toml_generator_v0_3_1",
        "proposed_path": (
            "app_bundle/scripts/pipeline/academic_pipeline_toml_generator.py"
        ),
        "proposed_module": (
            "app_bundle.scripts.pipeline.academic_pipeline_toml_generator"
        ),
        "classification": "renomeação com compatibilidade",
        "target_phase": "AP-004B/AP-004E",
        "compatibility_policy": "wrapper obrigatório no caminho histórico",
        "rationale": (
            "Módulo executável com guarda main e nome versionado; chamadas "
            "diretas e referências por caminho precisam continuar válidas."
        ),
    },
    {
        "key": "prisma_ai_prescreen_configurator",
        "current_path": "configurar_pretriagem_ia_prisma_v16.py",
        "current_module": "configurar_pretriagem_ia_prisma_v16",
        "bare_module": "configurar_pretriagem_ia_prisma_v16",
        "proposed_path": "configurar_pretriagem_ia_prisma.py",
        "proposed_module": "configurar_pretriagem_ia_prisma",
        "classification": "renomeação com compatibilidade",
        "target_phase": "AP-004B/AP-004E",
        "compatibility_policy": "wrapper de script histórico até AP-004E",
        "rationale": (
            "Script de raiz executável e potencialmente chamado por nome de "
            "arquivo; requer compatibilidade transitória."
        ),
    },
    {
        "key": "article_diagnostic_log",
        "current_path": "gerar_log_diagnostico_artigo_v1_18.py",
        "current_module": "gerar_log_diagnostico_artigo_v1_18",
        "bare_module": "gerar_log_diagnostico_artigo_v1_18",
        "proposed_path": "gerar_log_diagnostico_artigo.py",
        "proposed_module": "gerar_log_diagnostico_artigo",
        "classification": "renomeação com compatibilidade",
        "target_phase": "AP-004B/AP-004E",
        "compatibility_policy": "wrapper de script histórico até AP-004E",
        "rationale": (
            "Script de diagnóstico com guarda main e nome versionado; o nome "
            "histórico deve continuar invocável durante a transição."
        ),
    },
    {
        "key": "fulltext_executor_v1_13",
        "current_path": "executar_artigo_longo_fulltext_v1_13.py",
        "current_module": "executar_artigo_longo_fulltext_v1_13",
        "bare_module": "executar_artigo_longo_fulltext_v1_13",
        "proposed_path": None,
        "proposed_module": None,
        "suspended_target": "executar_artigo_longo_fulltext.py",
        "classification": "renomeação de alto risco",
        "target_phase": "revisão manual antes da AP-004B produtiva",
        "compatibility_policy": "nenhuma decisão até comparar versões",
        "collision_group": "fulltext_executor",
        "rationale": (
            "Converge para o mesmo destino sugerido à versão v1_14; não é "
            "seguro escolher pelo número da versão."
        ),
    },
    {
        "key": "fulltext_executor_v1_14",
        "current_path": "executar_artigo_longo_fulltext_v1_14.py",
        "current_module": "executar_artigo_longo_fulltext_v1_14",
        "bare_module": "executar_artigo_longo_fulltext_v1_14",
        "proposed_path": None,
        "proposed_module": None,
        "suspended_target": "executar_artigo_longo_fulltext.py",
        "classification": "renomeação de alto risco",
        "target_phase": "revisão manual antes da AP-004B produtiva",
        "compatibility_policy": "nenhuma decisão até comparar versões",
        "collision_group": "fulltext_executor",
        "rationale": (
            "Converge para o mesmo destino sugerido à versão v1_13; não é "
            "seguro escolher pelo número da versão."
        ),
    },
)

EXPECTED_CANDIDATE_PATHS: tuple[str, ...] = (
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_v0_3_1.py",
    "configurar_pretriagem_ia_prisma_v16.py",
    "gerar_log_diagnostico_artigo_v1_18.py",
    "executar_artigo_longo_fulltext_v1_13.py",
    "executar_artigo_longo_fulltext_v1_14.py",
)

FORBIDDEN_PHANTOM_VERSION_TOKENS: tuple[str, ...] = (
    "gerar_log_diagnostico_artigo_v1_28",
    "executar_artigo_longo_fulltext_v1_23",
    "executar_artigo_longo_fulltext_v1_24",
)


REFERENCE_CATEGORIES: tuple[str, ...] = (
    "actionable_productive",
    "compatibility_contract",
    "historical_immutable",
    "physical_directory_reference",
    "protected_operational",
    "contextual_non_actionable",
)
EFFECTIVE_CONSUMER_CATEGORIES = {
    "actionable_productive",
    "compatibility_contract",
}
HISTORICAL_PREFIXES: tuple[str, ...] = (
    "docs/refactor/academic-pipeline/AP-003/",
    "tests/characterization/snapshots/ap003",
    "tests/characterization/test_ap003",
    "tools/refactor/ap003",
)
HISTORICAL_EXACT_PATHS: set[str] = {
    AP004A_REPORT_REL.as_posix(),
    AP004A_CONVENTION_REL.as_posix(),
    AP004A_INVENTORY_REL.as_posix(),
    AP004A_TOOL_REL.as_posix(),
}
PROTECTED_ROOT_PREFIXES: tuple[str, ...] = (
    "aplicar_",
    "atualizar_",
    "install_",
    "setup_",
)
KIND_PRIORITY = {
    "static_import": 100,
    "dynamic_import": 90,
    "subprocess_or_exec": 80,
    "shell_or_config_reference": 70,
    "python_path_assignment": 65,
    "python_string_reference": 60,
    "test_reference": 50,
    "documentation_reference": 40,
    "python_string_literal": 30,
}


RUNTIME_PATH_CALLS = {
    "open", "Path", "run_path", "runpy.run_path",
    "spec_from_file_location", "importlib.util.spec_from_file_location",
    "copy", "copy2", "copyfile", "move", "replace",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.move",
    "os.execv", "os.execve", "os.execl", "os.spawnv",
}


PATH_TARGET_PATTERN = re.compile(
    r"(?:^|_)(?:path|script|module|entrypoint|command|cmd|pipeline|"
    r"orchestrator|executable|file|filename)(?:$|_)",
    re.IGNORECASE,
)


def validate_static_candidate_configuration() -> None:
    actual = tuple(item["current_path"] for item in CANDIDATE_SPECS)
    if actual != EXPECTED_CANDIDATE_PATHS:
        fail(
            "Matriz estática AP-004B divergente.\n"
            f"Esperado: {json.dumps(EXPECTED_CANDIDATE_PATHS, indent=2, ensure_ascii=False)}\n"
            f"Atual: {json.dumps(actual, indent=2, ensure_ascii=False)}"
        )
    serialized = json.dumps(CANDIDATE_SPECS, ensure_ascii=False)
    phantom = [token for token in FORBIDDEN_PHANTOM_VERSION_TOKENS if token in serialized]
    if phantom:
        fail(
            "Matriz AP-004B contém versões inexistentes: " + ", ".join(phantom)
        )
    compatible = [
        item for item in CANDIDATE_SPECS
        if item["classification"] == "renomeação com compatibilidade"
    ]
    high_risk = [
        item for item in CANDIDATE_SPECS
        if item["classification"] == "renomeação de alto risco"
    ]
    if len(compatible) != 4 or len(high_risk) != 2:
        fail(
            "Matriz AP-004B deve conter exatamente quatro renomeações com "
            "compatibilidade e duas de alto risco."
        )


TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".org", ".toml", ".cfg",
    ".ini", ".json", ".yaml", ".yml", ".sh", ".bash", ".zsh",
    ".fish", ".service", ".desktop", ".el", ".mk",
}
TEXT_FILENAMES = {"Pipfile", "Dockerfile", "Makefile", "LICENSE"}
MAX_TEXT_BYTES = 3_000_000
IGNORED_PATH_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".academic_pipeline", ".patch_backups",
    "backups", "build", "dist", "site-packages",
}
EPHEMERAL_SUFFIXES = (".pyc", ".pyo")

DYNAMIC_IMPORT_CALLS = {
    "importlib.import_module",
    "__import__",
    "importlib.util.spec_from_file_location",
    "SourceFileLoader",
    "importlib.machinery.SourceFileLoader",
}
SUBPROCESS_CALLS = {
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_call", "subprocess.check_output", "os.system",
    "os.popen", "os.execv", "os.execve", "os.execl", "os.execlp",
}

SUMMARY_PATTERN = re.compile(
    r"(?P<passed>\d+) passed"
    r"(?:, (?P<xfailed>\d+) xfailed)?"
)


class InventoryError(RuntimeError):
    """Falha controlada da AP-004B."""

def validate_semantic_classifier_configuration() -> None:
    orchestrator = dict(CANDIDATE_SPECS[0])
    tokens = candidate_tokens(orchestrator)
    if text_mentions(SOFTWARE_DIRNAME, tokens):
        fail("Classificador confunde módulo academic_pipeline_rc10 com diretório físico.")
    expected = [("academic_pipeline_rc10", 1)]
    if text_mentions("academic_pipeline_rc10", tokens) != expected:
        fail("Classificador não reconhece o módulo bare academic_pipeline_rc10.")
    full_path = "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
    if text_mentions(full_path, tokens) != [(full_path, 1)]:
        fail("Classificador não prioriza o caminho completo do orquestrador.")
    duplicate = deduplicate_reference_records([
        {
            "candidate_key": "pipeline_orchestrator",
            "consumer_path": "module.py",
            "line": 1,
            "kind": "python_string_literal",
            "matched_tokens": ["academic_pipeline_rc10.py"],
            "excerpt": "academic_pipeline_rc10.py",
        },
        {
            "candidate_key": "pipeline_orchestrator",
            "consumer_path": "module.py",
            "line": 1,
            "kind": "subprocess_or_exec",
            "matched_tokens": ["academic_pipeline_rc10.py"],
            "excerpt": "academic_pipeline_rc10.py",
        },
    ])
    if len(duplicate) != 1 or duplicate[0]["kind"] != "subprocess_or_exec":
        fail("Deduplicação semântica não preserva a evidência mais forte.")
    if classify_reference({
        "consumer_path": "tests/characterization/test_ap003a_orchestrator_contract.py",
        "kind": "test_reference",
    })[0] != "historical_immutable":
        fail("Contrato AP-003 não foi classificado como histórico imutável.")
    if classify_reference({
        "consumer_path": "aplicar_docx_canonico_v11.py",
        "kind": "python_string_literal",
    })[0] != "protected_operational":
        fail("Aplicador histórico não foi classificado como operacional protegido.")



def fail(message: str) -> NoReturn:
    raise InventoryError(message)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class BackupRecord:
    path: Path
    existed: bool
    backup: Path | None


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> CommandResult:
    merged_env = os.environ.copy()
    merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            [str(item) for item in args],
            cwd=cwd,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError:
        fail(f"Executável obrigatório não encontrado: {args[0]}")
    except subprocess.TimeoutExpired:
        fail(
            f"Comando excedeu o limite de {timeout}s: "
            + " ".join(str(item) for item in args)
        )
    result = CommandResult(
        args=tuple(str(item) for item in args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        fail(
            "Comando falhou: " + " ".join(result.args)
            + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def git(root: Path, *args: str, check: bool = True, timeout: int = 180) -> CommandResult:
    return run(("git", *args), cwd=root, check=check, timeout=timeout)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_output(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.splitlines()) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temp, path.stat().st_mode)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def create_backups(
    paths: Iterable[Path], *, software_root: Path
) -> tuple[Path, list[BackupRecord]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        Path.home() / ".cache" / "academic-pipeline-refactor" / "backups"
        / PHASE / timestamp
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    records: list[BackupRecord] = []
    for path in paths:
        if path.exists():
            relative = path.relative_to(software_root)
            destination = backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            records.append(BackupRecord(path, True, destination))
        else:
            records.append(BackupRecord(path, False, None))
    return backup_root, records


def rollback(records: Iterable[BackupRecord]) -> None:
    for record in records:
        if record.existed and record.backup is not None:
            record.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.backup, record.path)
        elif record.path.exists():
            record.path.unlink()


def compile_python(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ap004b-pyc-") as temporary:
        py_compile.compile(
            str(path),
            cfile=str(Path(temporary) / f"{path.name}c"),
            doraise=True,
        )


def normalize_status_path(line: str) -> str:
    # Porcelain v1 begins with the two-column XY status. Some Git versions
    # render an additional separator space while others expose paths in forms
    # where slicing three characters would consume the first filename byte.
    # Remove exactly XY and then only whitespace.
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip().strip('"').replace("\\", "/")


def repository_relative(path: Path, *, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def software_relative_status_path(
    line: str, *, software_root: Path, repository_root: Path
) -> str:
    raw = normalize_status_path(line)
    prefix = repository_relative(
        software_root, repository_root=repository_root
    ).rstrip("/") + "/"
    if raw.startswith(prefix):
        return raw[len(prefix):]
    return raw


def parse_remote_head(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"Branch remota não resolvida univocamente:\n{output}")
    fields = lines[0].split()
    if len(fields) != 2:
        fail(f"Saída inválida de git ls-remote: {lines[0]}")
    return fields[0]


def is_ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        any(part in IGNORED_PATH_PARTS for part in parts)
        or path.endswith(EPHEMERAL_SUFFIXES)
    )


def validate_git_state(
    software_root: Path, *, skip_remote_check: bool
) -> tuple[Path, dict[str, Any]]:
    if software_root.resolve() != EXPECTED_SOFTWARE_ROOT.resolve():
        fail(
            "Diretório incorreto. Execute em:\n"
            f"  {EXPECTED_SOFTWARE_ROOT}\nAtual:\n  {software_root}"
        )
    repository_root = Path(
        git(software_root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    if repository_root != EXPECTED_REPOSITORY_ROOT.resolve():
        fail(
            f"Worktree inesperado: {repository_root}\n"
            f"Esperado: {EXPECTED_REPOSITORY_ROOT}"
        )
    branch = git(repository_root, "branch", "--show-current").stdout.strip()
    if branch != EXPECTED_BRANCH:
        fail(f"Branch incorreta: {branch!r}; esperada: {EXPECTED_BRANCH}")

    status_lines = [
        line for line in git(
            repository_root, "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout.splitlines() if line.strip()
    ]
    expected_outputs = {item.as_posix() for item in OUTPUT_RELS}
    new_outputs = {item.as_posix() for item in (
        REPORT_REL, STRATEGY_REL, INVENTORY_REL, TOOL_REL, TEST_REL
    )}
    actual_paths: set[str] = set()
    unexpected: list[str] = []
    for line in status_lines:
        relative = software_relative_status_path(
            line, software_root=software_root, repository_root=repository_root
        )
        if is_ephemeral(relative):
            continue
        actual_paths.add(relative)
        code = line[:2]
        allowed_existing = (
            (relative in new_outputs and code == "??")
            or (relative == AP004A_TEST_REL.as_posix() and code == " M")
        )
        if relative not in expected_outputs or not allowed_existing:
            unexpected.append(f"{code} {relative}")
    if unexpected:
        fail(
            "A árvore contém alterações não relacionadas. A AP-004B v1.6 aceita "
            "árvore limpa ou somente os artefatos preparatórios da execução "
            "anterior (cinco arquivos não rastreados e o contrato AP-004A "
            "modificado):\n"
            + "\n".join(f"  - {item}" for item in unexpected)
        )

    head = git(repository_root, "rev-parse", "HEAD").stdout.strip()
    if head != EXPECTED_HEAD:
        fail(
            "HEAD divergente do encerramento canônico da AP-004A.\n"
            f"Esperado: {EXPECTED_HEAD}\nAtual:    {head}"
        )
    tracking_head = git(
        repository_root, "rev-parse", EXPECTED_REMOTE_REF
    ).stdout.strip()
    if tracking_head != head:
        fail(
            f"{EXPECTED_REMOTE_REF} diverge do HEAD local.\n"
            f"Local: {head}\nTracking: {tracking_head}"
        )
    upstream_result = git(
        repository_root,
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}",
        check=False,
    )
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else ""
    if upstream and upstream != EXPECTED_REMOTE_REF:
        fail(f"Upstream inesperado: {upstream}; esperado: {EXPECTED_REMOTE_REF}")

    remote_head = tracking_head
    remote_check = "skipped"
    if not skip_remote_check:
        remote = git(
            repository_root,
            "ls-remote", "--heads", "origin", EXPECTED_REMOTE_BRANCH_REF,
            timeout=60,
        )
        remote_head = parse_remote_head(remote.stdout)
        remote_check = "passed"
        if remote_head != head:
            fail(
                "HEAD local não corresponde ao remoto publicado.\n"
                f"Local:  {head}\nRemoto: {remote_head}"
            )

    return repository_root, {
        "branch": branch,
        "head": head,
        "tracking_head": tracking_head,
        "remote_head": remote_head,
        "remote_check": remote_check,
        "upstream": upstream or EXPECTED_REMOTE_REF,
        "initial_tree_state": "clean" if not actual_paths else "ap004b-artifacts-only",
        "initial_paths": sorted(actual_paths),
    }


def validate_ap004a_closure(
    *, software_root: Path, repository_root: Path
) -> dict[str, Any]:
    subject = git(
        repository_root, "show", "-s", "--format=%s", EXPECTED_HEAD
    ).stdout.strip()
    if subject != EXPECTED_AP004A_SUBJECT:
        fail(
            "Assunto do commit AP-004A divergente.\n"
            f"Esperado: {EXPECTED_AP004A_SUBJECT}\nAtual: {subject}"
        )
    changed = [
        line.strip() for line in git(
            repository_root,
            "diff-tree", "--no-commit-id", "--name-only", "-r", EXPECTED_HEAD,
        ).stdout.splitlines() if line.strip()
    ]
    prefix = repository_relative(
        software_root, repository_root=repository_root
    ).rstrip("/") + "/"
    relative_changed = sorted(
        item[len(prefix):] if item.startswith(prefix) else item for item in changed
    )
    expected = sorted(item.as_posix() for item in AP004A_COMMIT_FILES)
    if relative_changed != expected:
        fail(
            "O commit AP-004A não contém exatamente os cinco artefatos esperados.\n"
            f"Esperado:\n{json.dumps(expected, indent=2, ensure_ascii=False)}\n"
            f"Atual:\n{json.dumps(relative_changed, indent=2, ensure_ascii=False)}"
        )
    for relative in AP004A_COMMIT_FILES:
        path = software_root / relative
        if not path.is_file():
            fail(f"Artefato AP-004A ausente: {relative}")

    try:
        data = json.loads((software_root / AP004A_INVENTORY_REL).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Inventário AP-004A inválido: {exc}")
    if data.get("phase") != "AP-004A":
        fail("Inventário AP-004A não identifica a fase AP-004A.")
    if data.get("inventory_schema_version") != 4:
        fail("Schema do inventário AP-004A não é 4.")
    if data.get("inventory_revision") != "4.2":
        fail("Revisão do inventário AP-004A não é 4.2.")

    actionable = [
        item
        for item in data.get("actionable_candidates", [])
        if isinstance(item, dict)
    ]
    expected_classes = {
        ORCHESTRATOR_REL.as_posix(): "renomeação com compatibilidade",
        (
            "app_bundle/scripts/pipeline/"
            "academic_pipeline_toml_generator_v0_3_1.py"
        ): "renomeação com compatibilidade",
        "configurar_pretriagem_ia_prisma_v16.py": "renomeação com compatibilidade",
        "gerar_log_diagnostico_artigo_v1_18.py": "renomeação com compatibilidade",
        "executar_artigo_longo_fulltext_v1_13.py": "renomeação de alto risco",
        "executar_artigo_longo_fulltext_v1_14.py": "renomeação de alto risco",
    }
    candidate_bindings: dict[str, dict[str, str]] = {}
    for path, classification in expected_classes.items():
        expected_name = Path(path).name
        matches = [
            item
            for item in actionable
            if item.get("path") == path
            and item.get("category") == "arquivo/módulo"
            and item.get("current_name") == expected_name
        ]
        if len(matches) != 1:
            summaries = [
                {
                    "category": item.get("category"),
                    "current_name": item.get("current_name"),
                    "classification": item.get("classification"),
                    "line": item.get("line"),
                }
                for item in actionable
                if item.get("path") == path
            ]
            fail(
                "Candidato de módulo AP-004A não é único para "
                f"{path} (encontrados: {len(matches)}).\n"
                "Registros no mesmo caminho:\n"
                f"{json.dumps(summaries, indent=2, ensure_ascii=False)}"
            )
        item = matches[0]
        if item.get("classification") != classification:
            fail(
                f"Classificação AP-004A divergente para o módulo {path}: "
                f"{item.get('classification')!r}"
            )
        candidate_bindings[path] = {
            "category": str(item.get("category")),
            "current_name": str(item.get("current_name")),
            "classification": str(item.get("classification")),
        }
    protected = data.get("protected_names", {})
    if protected.get("physical_directory") != SOFTWARE_DIRNAME:
        fail("Diretório físico não está protegido até AP-006 no inventário AP-004A.")

    return {
        "commit": EXPECTED_HEAD,
        "subject": subject,
        "changed_files": relative_changed,
        "inventory_path": AP004A_INVENTORY_REL.as_posix(),
        "inventory_sha256": sha256_path(software_root / AP004A_INVENTORY_REL),
        "schema_version": 4,
        "revision": "4.2",
        "candidate_classifications": expected_classes,
        "candidate_bindings": candidate_bindings,
    }


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def literal_strings(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
        elif isinstance(child, ast.JoinedStr):
            text = "".join(
                part.value
                for part in child.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if text:
                values.append(text)
    return values


def direct_main_guard_calls(tree: ast.Module) -> list[str]:
    calls: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test_dump = ast.dump(node.test, include_attributes=False)
        if "__name__" not in test_dump or "__main__" not in test_dump:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = call_name(child.func)
                if name:
                    calls.append(name)
    return calls


def ast_signature(tree: ast.Module) -> dict[str, Any]:
    top_functions: list[dict[str, Any]] = []
    top_classes: list[dict[str, Any]] = []
    assignments: list[str] = []
    imports: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_functions.append({
                "name": node.name,
                "async": isinstance(node, ast.AsyncFunctionDef),
                "line": node.lineno,
                "parameters": [arg.arg for arg in (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                )],
            })
        elif isinstance(node, ast.ClassDef):
            top_classes.append({"name": node.name, "line": node.lineno})
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.append(target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "form": "import", "module": alias.name,
                    "name": None, "asname": alias.asname, "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "form": "from", "module": node.module or "",
                    "name": alias.name, "asname": alias.asname, "line": node.lineno,
                    "level": node.level,
                })
    normalized_dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return {
        "ast_sha256": sha256_bytes(normalized_dump.encode("utf-8")),
        "top_level_functions": top_functions,
        "top_level_classes": top_classes,
        "top_level_assignments": sorted(set(assignments)),
        "imports": imports,
        "main_guard_calls": direct_main_guard_calls(tree),
    }


def validate_ap003_architecture(software_root: Path) -> dict[str, Any]:
    required = [
        ORCHESTRATOR_REL, PACKAGE_MAIN_REL, PACKAGE_CLI_REL,
        PACKAGE_LEGACY_REL, PRISMA_ORCHESTRATION_REL,
    ]
    for relative in required:
        if not (software_root / relative).is_file():
            fail(f"Arquivo estrutural ausente: {relative}")

    orchestrator_source = (software_root / ORCHESTRATOR_REL).read_text(encoding="utf-8")
    orchestrator_tree = ast.parse(orchestrator_source, filename=ORCHESTRATOR_REL.as_posix())
    top_names = [
        node.name for node in orchestrator_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if top_names.count("main") != 1:
        fail("Orquestrador não possui exatamente um main() público.")
    if top_names.count("_ap003f_pipeline_core") != 1:
        fail("Orquestrador não possui exatamente um _ap003f_pipeline_core.")
    if "_original_main_before_prisma_artigo_generico_wrapper" in orchestrator_source:
        fail("Alias histórico proibido reapareceu no orquestrador.")
    guard_calls = direct_main_guard_calls(orchestrator_tree)
    if "main" not in guard_calls:
        fail("Guarda __main__ do orquestrador não chama main().")

    package_main_source = (software_root / PACKAGE_MAIN_REL).read_text(encoding="utf-8")
    if "main" not in package_main_source:
        fail("academic_pipeline/__main__.py não referencia main().")
    prisma_source = (software_root / PRISMA_ORCHESTRATION_REL).read_text(encoding="utf-8")
    if "_ap003f_pipeline_core" not in prisma_source:
        fail("Módulo PRISMA não referencia _ap003f_pipeline_core.")

    return {
        "orchestrator": ORCHESTRATOR_REL.as_posix(),
        "orchestrator_sha256": sha256_path(software_root / ORCHESTRATOR_REL),
        "top_level_function_names": top_names,
        "main_guard_calls": guard_calls,
        "public_entrypoints": ["academic-pipeline", "python -m academic_pipeline"],
        "historical_alias_absent": True,
        "prisma_calls_internal_core": True,
    }


def ignored_tracked_path(relative: str) -> bool:
    """Decide exclusão sem consultar o filesystem.

    O repositório histórico pode conter caminhos rastreados sob árvores de
    backup recursivas. A decisão precisa ocorrer sobre a string retornada por
    ``git ls-files`` antes de construir/consultar o caminho físico; caso
    contrário, ``Path.is_file()`` pode levantar ``ENAMETOOLONG``.
    """
    parts = PurePosixPath(relative).parts
    return any(part in IGNORED_PATH_PARTS for part in parts)


def tracked_files(
    *, software_root: Path, repository_root: Path
) -> list[Path]:
    prefix = repository_relative(
        software_root, repository_root=repository_root
    ).rstrip("/") + "/"
    result = git(repository_root, "ls-files", "-z")
    files: list[Path] = []
    for raw in result.stdout.split("\0"):
        if not raw or not raw.startswith(prefix):
            continue
        relative = raw[len(prefix):]
        if ignored_tracked_path(relative):
            continue
        path = software_root / relative
        try:
            if path.is_file():
                files.append(path)
        except OSError as exc:
            if exc.errno in {
                errno.ENAMETOOLONG, errno.ELOOP, errno.ENOENT, errno.ENOTDIR
            }:
                continue
            raise
    return sorted(files)


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def historical_reference_path(relative: str) -> bool:
    return (
        relative in HISTORICAL_EXACT_PATHS
        or any(relative.startswith(prefix) for prefix in HISTORICAL_PREFIXES)
    )


def protected_consumer_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    parts = path.parts
    name = path.name.lower()
    if relative.startswith("app_bundle/projetos/") and any(
        part == "execucoes_anteriores" or part.startswith("output")
        for part in parts
    ):
        return True
    if "assets" in parts:
        return True
    if relative in {
        "app_bundle/.academic_pipeline_tui_state.json",
        "app_bundle/clean_institutional_tree_report.json",
    }:
        return True
    if any(path.name.startswith(prefix) for prefix in PROTECTED_ROOT_PREFIXES):
        return True
    if name.startswith("install") and path.suffix in {".sh", ".py"}:
        return True
    if "report" in name and path.suffix.lower() in {".json", ".log", ".txt"}:
        return True
    return False


def compatibility_contract_path(relative: str) -> bool:
    return (
        relative == PACKAGE_LEGACY_REL.as_posix()
        or relative.startswith("app_bundle/tests/")
        or relative.startswith("tests/")
    )


def documentation_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (
        relative.startswith("docs/")
        or relative.startswith("app_bundle/docs/")
        or path.name.lower().startswith("readme")
        or path.suffix.lower() in {".md", ".rst", ".adoc"}
    )


def candidate_tokens(spec: dict[str, Any]) -> set[str]:
    path = spec["current_path"]
    stem = Path(path).stem
    tokens = {
        path,
        Path(path).name,
        stem,
        spec["current_module"],
        spec["bare_module"],
    }
    return {token for token in tokens if token}


def _token_pattern(token: str) -> re.Pattern[str]:
    # Identificadores e nomes de arquivo não podem casar dentro de um token
    # maior. Isso impede academic_pipeline_rc10 de casar com o diretório
    # academic_pipeline_rc10_7_conformidade.
    left = r"(?<![A-Za-z0-9_])" if token and (token[0].isalnum() or token[0] == "_") else ""
    right = r"(?![A-Za-z0-9_])" if token and (token[-1].isalnum() or token[-1] == "_") else ""
    return re.compile(left + re.escape(token) + right)


def text_mentions(text: str, tokens: set[str]) -> list[tuple[str, int]]:
    candidates: list[tuple[int, int, str]] = []
    for token in tokens:
        for match in _token_pattern(token).finditer(text):
            candidates.append((match.start(), match.end(), token))
    # Resolve sobreposição pela evidência mais específica (token mais longo).
    selected: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, token in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])
    ):
        if any(not (end <= old_start or start >= old_end) for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        selected.append((start, end, token))
    counts = Counter(token for _, _, token in selected)
    return sorted(counts.items(), key=lambda item: (-len(item[0]), item[0]))


def module_matches(value: str, spec: dict[str, Any]) -> bool:
    value = value.strip()
    current = spec["current_module"]
    bare = spec["bare_module"]
    return (
        value == current or value == bare
        or value.startswith(current + ".")
        or value.startswith(bare + ".")
    )


def classify_reference(record: dict[str, Any]) -> tuple[str, str]:
    relative = record["consumer_path"]
    kind = record["kind"]
    if historical_reference_path(relative):
        return "historical_immutable", "preservar como evidência histórica; não editar"
    if protected_consumer_path(relative):
        return "protected_operational", "excluir do aplicador e do manifesto produtivo"
    if compatibility_contract_path(relative):
        return "compatibility_contract", "preservar ou ajustar apenas para provar o wrapper histórico"
    if documentation_path(relative):
        return "contextual_non_actionable", "documentação contextual; fora do primeiro aplicador"
    if kind in {
        "static_import", "dynamic_import", "subprocess_or_exec",
        "shell_or_config_reference", "python_path_assignment",
    }:
        return "actionable_productive", "revisar estruturalmente no aplicador produtivo"
    if kind == "python_string_reference":
        call = str(record.get("call") or "")
        if call in RUNTIME_PATH_CALLS or call.endswith((
            ".open", ".read_text", ".write_text", ".exists", ".is_file",
            ".resolve", ".with_name", ".joinpath",
        )):
            return "actionable_productive", "caminho usado por chamada de runtime"
        return "contextual_non_actionable", "string em chamada sem consumo operacional comprovado"
    return "contextual_non_actionable", "evidência textual sem consumo operacional comprovado"


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("candidate_key"),
        record.get("consumer_path"),
        record.get("line"),
        tuple(record.get("matched_tokens", [])),
    )


def deduplicate_reference_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = _record_identity(record)
        current = best.get(key)
        if current is None or KIND_PRIORITY.get(record["kind"], 0) > KIND_PRIORITY.get(current["kind"], 0):
            best[key] = record
    result: list[dict[str, Any]] = []
    for record in best.values():
        category, policy = classify_reference(record)
        enriched = dict(record)
        enriched["semantic_category"] = category
        enriched["action_policy"] = policy
        result.append(enriched)
    return sorted(
        result,
        key=lambda item: (
            item["candidate_key"], item["consumer_path"],
            item.get("line") or 0, item["kind"], item.get("excerpt", ""),
        ),
    )


def scan_python_consumers(
    *, path: Path, relative: str, tree: ast.Module, specs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for spec in specs:
                    if module_matches(alias.name, spec):
                        records.append({
                            "candidate_key": spec["key"],
                            "consumer_path": relative,
                            "line": node.lineno,
                            "kind": "static_import",
                            "form": "import",
                            "module": alias.name,
                            "name": None,
                            "asname": alias.asname,
                            "matched_tokens": [alias.name],
                            "excerpt": f"import {alias.name}",
                        })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for spec in specs:
                current_parent = spec["current_module"].rsplit(".", 1)[0] if "." in spec["current_module"] else ""
                stem = Path(spec["current_path"]).stem
                for alias in node.names:
                    direct_module_match = module_matches(module, spec)
                    parent_import_match = (
                        alias.name in {stem, spec["bare_module"]}
                        and (
                            module == current_parent
                            or (
                                node.level > 0
                                and module in {"", current_parent.rsplit(".", 1)[-1]}
                            )
                        )
                    )
                    if not (direct_module_match or parent_import_match):
                        continue
                    matched = module if direct_module_match else alias.name
                    records.append({
                        "candidate_key": spec["key"],
                        "consumer_path": relative,
                        "line": node.lineno,
                        "kind": "static_import",
                        "form": "from",
                        "module": module,
                        "name": alias.name,
                        "asname": alias.asname,
                        "matched_tokens": [matched],
                        "excerpt": f"from {'.' * node.level}{module} import {alias.name}",
                    })
        elif isinstance(node, ast.Call):
            name = call_name(node.func) or ""
            strings = literal_strings(node)
            if not strings:
                continue
            if name in DYNAMIC_IMPORT_CALLS:
                kind = "dynamic_import"
            elif name in SUBPROCESS_CALLS:
                kind = "subprocess_or_exec"
            else:
                kind = "python_string_reference"
            joined = "\n".join(strings)
            for spec in specs:
                mentions = text_mentions(joined, candidate_tokens(spec))
                if mentions:
                    records.append({
                        "candidate_key": spec["key"],
                        "consumer_path": relative,
                        "line": getattr(node, "lineno", None),
                        "kind": kind,
                        "call": name,
                        "matched_tokens": [token for token, _ in mentions],
                        "excerpt": joined[:300],
                    })
    return records


def scan_directory_references(relative: str, text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if SOFTWARE_DIRNAME not in line:
            continue
        records.append({
            "candidate_key": None,
            "consumer_path": relative,
            "line": line_number,
            "kind": "physical_directory_reference",
            "semantic_category": "physical_directory_reference",
            "action_policy": "reservar exclusivamente para AP-006",
            "matched_tokens": [SOFTWARE_DIRNAME],
            "excerpt": line.strip()[:300],
        })
    return records


def detect_pyproject_entrypoints(
    *, software_root: Path, specs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = software_root / PYPROJECT_REL
    if not path.is_file() or tomllib is None:
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    records: list[dict[str, Any]] = []
    scripts = data.get("project", {}).get("scripts", {})
    if isinstance(scripts, dict):
        for name, target in scripts.items():
            value = str(target)
            for spec in specs:
                if any(token in value for token in candidate_tokens(spec)):
                    records.append({
                        "candidate_key": spec["key"],
                        "type": "project-script",
                        "name": str(name),
                        "target": value,
                        "path": PYPROJECT_REL.as_posix(),
                    })
    return records


def file_record(path: Path, *, software_root: Path) -> dict[str, Any]:
    relative = path.relative_to(software_root).as_posix()
    raw = path.read_bytes()
    mode = path.stat().st_mode
    record: dict[str, Any] = {
        "path": relative,
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "executable": bool(mode & stat.S_IXUSR),
        "git_last_commit": None,
    }
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        record["text"] = False
        return record
    record["text"] = True
    record["line_count"] = len(source.splitlines())
    if path.suffix == ".py":
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            fail(f"Erro AST em {relative}:{exc.lineno}: {exc.msg}")
        record["python"] = ast_signature(tree)
    return record


def assignment_target_names(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    names: list[str] = []
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.append(child.id)
            elif isinstance(child, ast.Attribute):
                names.append(child.attr)
    return sorted(set(names))


def nearest_assignment(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.Assign | ast.AnnAssign | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.Assign, ast.AnnAssign)):
            return current
        if isinstance(current, (ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        current = parents.get(current)
    return None


def git_metadata_for_file(
    *, repository_root: Path, software_root: Path, path: Path
) -> dict[str, Any]:
    repo_path = repository_relative(path, repository_root=repository_root)
    result = git(
        repository_root,
        "log", "-1", "--format=%H%x00%cI%x00%s", "--", repo_path,
        check=False,
    )
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return {"commit": None, "committed_at": None, "subject": None}
    parts = raw.split("\x00", 2)
    while len(parts) < 3:
        parts.append("")
    return {"commit": parts[0], "committed_at": parts[1], "subject": parts[2]}


def build_candidate_inventory(
    *, software_root: Path, repository_root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    int,
]:
    specs = [dict(item) for item in CANDIDATE_SPECS]
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        path = software_root / spec["current_path"]
        if not path.is_file():
            fail(f"Arquivo candidato ausente: {spec['current_path']}")
        record = file_record(path, software_root=software_root)
        record["git_last_commit"] = git_metadata_for_file(
            repository_root=repository_root,
            software_root=software_root,
            path=path,
        )
        item = {
            **spec,
            "id": stable_id(PHASE, spec["key"], spec["current_path"]),
            "source": record,
            "references": [],
            "consumers": [],
            "actionable_consumers": [],
            "compatibility_contracts": [],
            "related_entrypoints": [],
            "decision": "inventory-only-no-change",
        }
        candidates.append(item)

    tracked = tracked_files(
        software_root=software_root, repository_root=repository_root
    )
    raw_records: list[dict[str, Any]] = []
    directory_records: list[dict[str, Any]] = []
    protected_consumer_paths: list[str] = []
    output_paths = {item.as_posix() for item in OUTPUT_RELS}
    for path in tracked:
        relative = path.relative_to(software_root).as_posix()
        if relative in output_paths:
            continue
        if protected_consumer_path(relative):
            protected_consumer_paths.append(relative)
        suffix = path.suffix.lower()
        known_text = suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES
        if not known_text:
            continue
        text = read_text(path)
        if text is None:
            continue
        directory_records.extend(scan_directory_references(relative, text))
        if suffix == ".py":
            try:
                tree = ast.parse(text, filename=relative)
            except SyntaxError as exc:
                fail(f"Erro AST em arquivo rastreado {relative}:{exc.lineno}: {exc.msg}")
            raw_records.extend(
                scan_python_consumers(
                    path=path, relative=relative, tree=tree, specs=specs
                )
            )
            parents = {
                child: parent
                for parent in ast.walk(tree)
                for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                assignment = nearest_assignment(node, parents)
                assignment_names = assignment_target_names(assignment) if assignment else []
                for spec in specs:
                    mentions = text_mentions(node.value, candidate_tokens(spec))
                    if not mentions:
                        continue
                    raw_records.append({
                        "candidate_key": spec["key"],
                        "consumer_path": relative,
                        "line": getattr(node, "lineno", None),
                        "kind": (
                            "python_path_assignment"
                            if assignment is not None and any(
                                PATH_TARGET_PATTERN.search(name)
                                for name in assignment_names
                            )
                            else "python_string_literal"
                        ),
                        "assignment_targets": assignment_names,
                        "matched_tokens": [token for token, _ in mentions],
                        "excerpt": node.value[:300],
                    })
        else:
            for spec in specs:
                for line_number, line in enumerate(text.splitlines(), start=1):
                    line_mentions = text_mentions(line, candidate_tokens(spec))
                    if not line_mentions:
                        continue
                    raw_records.append({
                        "candidate_key": spec["key"],
                        "consumer_path": relative,
                        "line": line_number,
                        "kind": (
                            "documentation_reference"
                            if documentation_path(relative)
                            else "test_reference"
                            if compatibility_contract_path(relative)
                            else "shell_or_config_reference"
                        ),
                        "matched_tokens": [token for token, _ in line_mentions],
                        "excerpt": line.strip()[:300],
                    })

    reference_records = deduplicate_reference_records(raw_records)
    # Diretório físico é uma preocupação independente da renomeação do módulo.
    directory_records = sorted(
        {
            (
                item["consumer_path"], item["line"], item["excerpt"]
            ): item
            for item in directory_records
        }.values(),
        key=lambda item: (item["consumer_path"], item["line"], item["excerpt"]),
    )

    entrypoints = detect_pyproject_entrypoints(
        software_root=software_root, specs=specs
    )
    for item in candidates:
        python_info = item["source"].get("python", {})
        if python_info.get("main_guard_calls"):
            item["related_entrypoints"].append({
                "type": "python-main-guard",
                "name": item["current_module"],
                "calls": python_info["main_guard_calls"],
                "path": item["current_path"],
            })
        item["related_entrypoints"].extend(
            ep for ep in entrypoints if ep["candidate_key"] == item["key"]
        )
        references = [
            record for record in reference_records
            if record["candidate_key"] == item["key"]
        ]
        actionable = [
            record for record in references
            if record["semantic_category"] == "actionable_productive"
        ]
        compatibility = [
            record for record in references
            if record["semantic_category"] == "compatibility_contract"
        ]
        effective = actionable + compatibility
        item["references"] = references
        item["actionable_consumers"] = actionable
        item["compatibility_contracts"] = compatibility
        item["consumers"] = sorted(
            effective,
            key=lambda record: (
                record["consumer_path"], record.get("line") or 0,
                record["kind"], record.get("excerpt", ""),
            ),
        )
        category_counts = Counter(
            record["semantic_category"] for record in references
        )
        item["reference_summary"] = {
            "total": len(references),
            "by_semantic_category": dict(sorted(category_counts.items())),
            "files_by_semantic_category": {
                category: sorted({
                    record["consumer_path"] for record in references
                    if record["semantic_category"] == category
                })
                for category in REFERENCE_CATEGORIES
                if any(
                    record["semantic_category"] == category
                    for record in references
                )
            },
        }
        item["consumer_summary"] = {
            "total": len(item["consumers"]),
            "actionable_total": len(actionable),
            "compatibility_total": len(compatibility),
            "by_kind": dict(sorted(Counter(
                record["kind"] for record in item["consumers"]
            ).items())),
            "files": sorted({
                record["consumer_path"] for record in item["consumers"]
            }),
            "actionable_files": sorted({
                record["consumer_path"] for record in actionable
            }),
            "compatibility_files": sorted({
                record["consumer_path"] for record in compatibility
            }),
        }
        item["compatibility_required"] = (
            item["classification"] == "renomeação com compatibilidade"
        )
        item["dynamic_reference_detected"] = any(
            record["kind"] in {
                "dynamic_import", "subprocess_or_exec",
                "shell_or_config_reference",
            }
            for record in actionable
        )

    collision = compare_fulltext_versions(
        software_root=software_root, candidates=candidates
    )
    return (
        candidates,
        reference_records,
        collision,
        sorted(set(protected_consumer_paths)),
        directory_records,
        len(raw_records),
    )


def compare_fulltext_versions(
    *, software_root: Path, candidates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    group = [item for item in candidates if item.get("collision_group") == "fulltext_executor"]
    if len(group) != 2:
        fail("Grupo de colisão fulltext_executor não possui exatamente duas origens.")
    first, second = sorted(group, key=lambda item: item["current_path"])
    first_path = software_root / first["current_path"]
    second_path = software_root / second["current_path"]
    first_text = first_path.read_text(encoding="utf-8")
    second_text = second_path.read_text(encoding="utf-8")
    first_lines = first_text.splitlines()
    second_lines = second_text.splitlines()
    matcher = difflib.SequenceMatcher(a=first_lines, b=second_lines, autojunk=False)
    opcodes = matcher.get_opcodes()
    changed_blocks = sum(1 for tag, *_ in opcodes if tag != "equal")
    inserted = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in {"insert", "replace"})
    deleted = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in {"delete", "replace"})
    first_python = first["source"].get("python", {})
    second_python = second["source"].get("python", {})
    first_functions = {item["name"] for item in first_python.get("top_level_functions", [])}
    second_functions = {item["name"] for item in second_python.get("top_level_functions", [])}
    return {
        "collision_id": stable_id("collision", "fulltext_executor"),
        "group": "fulltext_executor",
        "suspended_target": "executar_artigo_longo_fulltext.py",
        "origins": [first["current_path"], second["current_path"]],
        "identical_bytes": first["source"]["sha256"] == second["source"]["sha256"],
        "identical_ast": first_python.get("ast_sha256") == second_python.get("ast_sha256"),
        "line_similarity_ratio": round(matcher.ratio(), 6),
        "changed_blocks": changed_blocks,
        "inserted_or_replaced_lines_in_second": inserted,
        "deleted_or_replaced_lines_from_first": deleted,
        "functions_common": sorted(first_functions & second_functions),
        "functions_only_v1_13": sorted(first_functions - second_functions),
        "functions_only_v1_14": sorted(second_functions - first_functions),
        "consumer_files_v1_13": first["consumer_summary"]["files"],
        "consumer_files_v1_14": second["consumer_summary"]["files"],
        "decision": "suspended-manual-review-required",
        "required_manual_evidence": [
            "comparar comportamento e argumentos CLI",
            "comparar efeitos colaterais e artefatos gerados",
            "identificar consumidores operacionais reais",
            "executar testes caracterizadores específicos das duas versões",
            "não escolher automaticamente a versão numericamente maior",
        ],
    }


def build_manifest(
    *, software_root: Path, candidates: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    productive_relevant: set[str] = {
        PYPROJECT_REL.as_posix(),
        PACKAGE_MAIN_REL.as_posix(),
        PACKAGE_CLI_REL.as_posix(),
        PACKAGE_LEGACY_REL.as_posix(),
        PRISMA_ORCHESTRATION_REL.as_posix(),
    }
    for item in candidates:
        productive_relevant.add(item["current_path"])
        productive_relevant.update(item["consumer_summary"]["files"])

    control_relevant = {
        AP004A_INVENTORY_REL.as_posix(),
        AP004A_CONVENTION_REL.as_posix(),
    }

    def records(paths: Iterable[str], *, role: str) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        for relative in sorted(set(paths)):
            if protected_consumer_path(relative) or historical_reference_path(relative):
                # AP-004A é permitido apenas no manifesto de controle.
                if not (role == "control_baseline" and relative in control_relevant):
                    continue
            path = software_root / relative
            if not path.is_file():
                continue
            record: dict[str, Any] = {
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
                "role": role,
            }
            if path.suffix == ".py":
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=relative)
                record["ast_sha256"] = ast_signature(tree)["ast_sha256"]
            manifest[relative] = record
        return manifest

    return (
        records(productive_relevant, role="productive_or_compatibility"),
        records(control_relevant, role="control_baseline"),
    )


def build_inventory_data(
    *,
    software_root: Path,
    repository_root: Path,
    git_state: dict[str, Any],
    ap004a: dict[str, Any],
    architecture: dict[str, Any],
    tool_source: str,
) -> dict[str, Any]:
    (
        candidates,
        references,
        collision,
        protected_consumer_paths,
        directory_references,
        raw_reference_count,
    ) = build_candidate_inventory(
        software_root=software_root, repository_root=repository_root
    )
    source_manifest, control_manifest = build_manifest(
        software_root=software_root, candidates=candidates
    )
    semantic_counts = Counter(
        item["semantic_category"] for item in references
    )
    effective_consumers = [
        item for item in references
        if item["semantic_category"] in EFFECTIVE_CONSUMER_CATEGORIES
    ]
    classifications = Counter(item["classification"] for item in candidates)
    return {
        "phase": PHASE,
        "mode": MODE,
        "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "tool_revision": TOOL_REVISION,
        "inventory_revision": INVENTORY_REVISION,
        "generated_at_utc": utc_now(),
        "git": git_state,
        "ap004a_closure": ap004a,
        "ap003_architecture": architecture,
        "scope": {
            "productive_changes": [],
            "candidate_count": len(candidates),
            "candidate_paths": [item["current_path"] for item in candidates],
            "allowed_outputs": [item.as_posix() for item in OUTPUT_RELS],
            "contract_maintenance": {
                "path": AP004A_TEST_REL.as_posix(),
                "reason": (
                    "substituir invariantes transitórias de HEAD/árvore por "
                    "validação histórica do commit isolado AP-004A"
                ),
                "productive_change": False,
            },
            "physical_directory_rename": "proibida até AP-006",
            "functional_change": "proibida",
            "cli_semantics_change": "proibida",
            "collision_resolution": "não realizada automaticamente",
            "consumer_semantics": {
                "effective": sorted(EFFECTIVE_CONSUMER_CATEGORIES),
                "excluded_from_productive_manifest": [
                    "historical_immutable",
                    "physical_directory_reference",
                    "protected_operational",
                    "contextual_non_actionable",
                ],
            },
        },
        "statistics": {
            "candidates": len(candidates),
            "by_classification": dict(sorted(classifications.items())),
            "raw_candidate_reference_records": raw_reference_count,
            "deduplicated_candidate_reference_records": len(references),
            "reference_records_by_semantic_category": dict(sorted(semantic_counts.items())),
            "effective_consumer_records": len(effective_consumers),
            "effective_consumer_files": len({
                item["consumer_path"] for item in effective_consumers
            }),
            "actionable_productive_records": semantic_counts.get("actionable_productive", 0),
            "compatibility_contract_records": semantic_counts.get("compatibility_contract", 0),
            "historical_immutable_records": semantic_counts.get("historical_immutable", 0),
            "protected_operational_records": semantic_counts.get("protected_operational", 0),
            "contextual_non_actionable_records": semantic_counts.get("contextual_non_actionable", 0),
            "physical_directory_reference_records": len(directory_references),
            "physical_directory_reference_files": len({
                item["consumer_path"] for item in directory_references
            }),
            "source_manifest_files": len(source_manifest),
            "control_manifest_files": len(control_manifest),
            "destination_collisions": 1,
            "protected_consumer_paths_detected": len(protected_consumer_paths),
        },
        "candidates": candidates,
        "reference_records": references,
        "consumer_records": effective_consumers,
        "directory_reference_records": directory_references,
        "protected_consumer_paths_detected": protected_consumer_paths,
        "destination_collisions": [collision],
        "source_manifest": source_manifest,
        "control_manifest": control_manifest,
        "proposed_execution_order": [
            {
                "batch": "AP-004B-1",
                "candidate_key": "pipeline_orchestrator",
                "action": "criar módulo canônico e wrapper histórico",
                "blocked": True,
                "reason": "aguarda aplicador produtivo e aprovação específica",
            },
            {
                "batch": "AP-004B-2",
                "candidate_key": "toml_generator",
                "action": "criar módulo canônico e wrapper histórico",
                "blocked": True,
                "reason": "aguarda aplicador produtivo e aprovação específica",
            },
            {
                "batch": "AP-004B-3",
                "candidate_key": "prisma_ai_prescreen_configurator",
                "action": "criar script canônico e wrapper histórico",
                "blocked": True,
                "reason": "aguarda aplicador produtivo e aprovação específica",
            },
            {
                "batch": "AP-004B-4",
                "candidate_key": "article_diagnostic_log",
                "action": "criar script canônico e wrapper histórico",
                "blocked": True,
                "reason": "aguarda aplicador produtivo e aprovação específica",
            },
            {
                "batch": "AP-004B-5",
                "candidate_key": "fulltext_executor",
                "action": "manter ambas as versões sem renomeação",
                "blocked": True,
                "reason": "colisão não resolvida; exige caracterização manual",
            },
        ],
        "compatibility_rules": {
            "historical_wrapper_required": True,
            "wrapper_must_forward_main_and_exit_code": True,
            "public_entrypoints_preserved": [
                "academic-pipeline", "python -m academic_pipeline"
            ],
            "dynamic_paths_must_be_updated_or_supported": True,
            "aliases_expire_in": "AP-004E, salvo justificativa documentada",
            "no_textual_global_replacement": True,
            "ast_required_for_python_import_updates": True,
            "historical_documents_are_immutable": True,
            "physical_directory_references_deferred_to": "AP-006",
            "protected_operational_artifacts_excluded": True,
        },
        "protected": {
            "physical_directory": SOFTWARE_DIRNAME,
            "known_xfails": [
                "_refs_v6_strip_org",
                "extract_org_abstracts",
                "WorkflowState._normalize",
            ],
            "integration_branch": "refactor/academic-pipeline",
            "integration_forbidden_before_ap004_closure": True,
        },
        "tool": {
            "path": TOOL_REL.as_posix(),
            "sha256": sha256_bytes(tool_source.encode("utf-8")),
            "version": TOOL_VERSION,
            "revision": TOOL_REVISION,
        },
        "validation": {
            "py_compile": "pending",
            "git_diff_check": "pending",
            "specific_suite": {"status": "pending"},
            "consolidated_suite": {"status": "pending"},
        },
        "next_gate": {
            "productive_applicator_allowed": False,
            "requires_user_review": True,
            "required_decisions": [
                "aprovar consumidores produtivos acionáveis",
                "aprovar contratos de compatibilidade",
                "aprovar exclusões históricas, contextuais e operacionais",
                "manter a colisão full-text fora do primeiro aplicador",
                "autorizar explicitamente a criação do aplicador AP-004B",
            ],
        },
    }


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_strategy_document(data: dict[str, Any]) -> str:
    return normalize_output(f"""
# AP-004B — estratégia de módulos e arquivos

> Documento preparatório. Nenhuma movimentação ou renomeação produtiva foi realizada.

## Objetivo técnico

Normalizar os quatro nomes aprováveis sem alterar semântica, entrypoints
públicos, conteúdo gerado ou caminhos operacionais. O inventário v1.6 separa
consumo produtivo comprovado de contratos de compatibilidade e de evidências
históricas/contextuais.

## Taxonomia obrigatória

- `actionable_productive`: import, loader, subprocesso ou caminho usado por
  código produtivo; integra o futuro aplicador.
- `compatibility_contract`: camada legada ou teste que deve provar o wrapper
  histórico; integra o manifesto, mas não autoriza remover a compatibilidade.
- `historical_immutable`: documentação, snapshots e ferramentas de fases
  encerradas; nunca atualizar na AP-004B.
- `physical_directory_reference`: menção a
  `academic_pipeline_rc10_7_conformidade`; pertence exclusivamente à AP-006.
- `protected_operational`: aplicadores, backups, outputs, assets, estados e
  relatórios operacionais; fora do aplicador e do manifesto produtivo.
- `contextual_non_actionable`: texto ou documentação sem consumo operacional
  comprovado; fora do primeiro aplicador.

## Estratégia canônica

- Criar o caminho canônico dos quatro módulos aprováveis e manter o caminho
  histórico como wrapper transitório.
- Preservar argumentos, código de saída, `main()` e superfícies públicas.
- Atualizar imports Python por AST ou edição estrutural dirigida.
- Tratar somente registros `actionable_productive`; usar registros
  `compatibility_contract` para caracterizar equivalência.
- Não alterar referências históricas, contextuais, operacionais ou do diretório
  físico.
- Manter `v1_13` e `v1_14` intactos; a colisão full-text permanece fora do
  primeiro aplicador.
- Preservar `academic-pipeline` e `python -m academic_pipeline`.

## Ordem proposta de aplicação futura

1. Orquestrador canônico + wrapper `academic_pipeline_rc10.py`.
2. Gerador TOML canônico + wrapper versionado.
3. Configurador de pré-triagem + wrapper versionado.
4. Gerador de log diagnóstico + wrapper versionado.
5. Caracterização separada da colisão full-text, sem renomeação automática.

## Barreiras obrigatórias do futuro aplicador

- branch e HEAD exatos;
- árvore limpa ou estado preparatório reconhecido;
- remoto sincronizado;
- hashes de `source_manifest` e `control_manifest`;
- contratos AST dos seis candidatos;
- conjunto exato de arquivos permitidos;
- backup externo, escrita atômica e rollback integral;
- `py_compile`, `git diff --check`, suíte específica e suíte consolidada;
- ausência de alterações nos três `xfail` históricos;
- nenhum commit sem aprovação da consolidação.

## Estado

A criação do aplicador produtivo permanece bloqueada até aprovação expressa do
inventário semântico v1.6.
""")


def build_report(data: dict[str, Any]) -> str:
    stats = data["statistics"]
    lines: list[str] = [
        "# AP-004B — inventário de módulos e arquivos (v1.6)",
        "",
        "> Levantamento somente preparatório. Nenhum arquivo produtivo foi modificado.",
        "",
        "## Estado Git e base canônica",
        "",
        f"- Branch: `{data['git']['branch']}`.",
        f"- HEAD local/remoto: `{data['git']['head']}`.",
        f"- Commit AP-004A: `{data['ap004a_closure']['commit']}`.",
        f"- Inventário AP-004A: schema `{data['ap004a_closure']['schema_version']}`, revisão `{data['ap004a_closure']['revision']}`.",
        f"- Estado inicial aceito: `{data['git']['initial_tree_state']}`.",
        "",
        "## Resumo semântico",
        "",
        f"- Candidatos: **{stats['candidates']}**.",
        f"- Referências brutas: **{stats['raw_candidate_reference_records']}**.",
        f"- Referências deduplicadas: **{stats['deduplicated_candidate_reference_records']}**.",
        f"- Consumidores efetivos: **{stats['effective_consumer_records']}** em **{stats['effective_consumer_files']}** arquivos.",
        f"- Produtivos acionáveis: **{stats['actionable_productive_records']}**.",
        f"- Contratos de compatibilidade: **{stats['compatibility_contract_records']}**.",
        f"- Históricos imutáveis: **{stats['historical_immutable_records']}**.",
        f"- Operacionais protegidos: **{stats['protected_operational_records']}**.",
        f"- Contextuais não acionáveis: **{stats['contextual_non_actionable_records']}**.",
        f"- Referências ao diretório físico: **{stats['physical_directory_reference_records']}** em **{stats['physical_directory_reference_files']}** arquivos; reservadas à AP-006.",
        f"- Manifesto produtivo/compatibilidade: **{stats['source_manifest_files']}** arquivos.",
        f"- Manifesto de controle: **{stats['control_manifest_files']}** arquivos.",
        f"- Colisões de destino: **{stats['destination_collisions']}**.",
        "- Código produtivo alterado: **não**.",
        "",
        "## Matriz de decisão",
        "",
        "| Chave | Caminho atual | Destino proposto/suspenso | Classificação | Acionáveis | Compatibilidade | Excluídos | Política |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in data["candidates"]:
        target = item.get("proposed_path") or (
            "suspenso: " + str(item.get("suspended_target"))
        )
        by_category = item["reference_summary"]["by_semantic_category"]
        excluded = sum(
            count for category, count in by_category.items()
            if category not in EFFECTIVE_CONSUMER_CATEGORIES
        )
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} | {} | {} | {} |".format(
                md(item["key"]), md(item["current_path"]), md(target),
                md(item["classification"]),
                item["consumer_summary"]["actionable_total"],
                item["consumer_summary"]["compatibility_total"],
                excluded,
                md(item["compatibility_policy"]),
            )
        )

    lines.extend(["", "## Consumidores efetivos por candidato", ""])
    for item in data["candidates"]:
        lines.append(f"### `{item['current_path']}`")
        lines.append("")
        lines.append(f"- SHA-256: `{item['source']['sha256']}`.")
        python_info = item["source"].get("python", {})
        if python_info:
            lines.append(f"- AST SHA-256: `{python_info.get('ast_sha256')}`.")
            lines.append(f"- Guarda `__main__`: `{python_info.get('main_guard_calls', [])}`.")
        lines.append(
            f"- Acionáveis: **{item['consumer_summary']['actionable_total']}**; "
            f"compatibilidade: **{item['consumer_summary']['compatibility_total']}**."
        )
        excluded_summary = {
            category: count
            for category, count in item["reference_summary"]["by_semantic_category"].items()
            if category not in EFFECTIVE_CONSUMER_CATEGORIES
        }
        lines.append(f"- Evidências excluídas do aplicador: `{excluded_summary}`.")
        if item["consumers"]:
            lines.extend([
                "",
                "| Categoria | Tipo | Arquivo:linha | Evidência |",
                "|---|---|---|---|",
            ])
            for record in item["consumers"]:
                location = f"{record['consumer_path']}:{record.get('line') or '-'}"
                lines.append(
                    f"| `{md(record['semantic_category'])}` | `{md(record['kind'])}` | "
                    f"`{md(location)}` | `{md(record.get('excerpt', ''))}` |"
                )
        else:
            lines.append("- Nenhum consumidor produtivo ou contrato de compatibilidade comprovado.")
        lines.append("")

    lines.extend([
        "## Exclusões obrigatórias",
        "",
        "- Documentação, manifestos, snapshots e ferramentas da AP-003 são históricos imutáveis.",
        "- Artefatos finais da AP-004A permanecem históricos e não serão reescritos.",
        "- Menções ao diretório `academic_pipeline_rc10_7_conformidade` pertencem à AP-006.",
        "- Aplicadores, atualizadores, backups, outputs, assets, estados e relatórios operacionais ficam fora do aplicador e do `source_manifest`.",
        "- Textos e docstrings sem consumo operacional comprovado permanecem contextuais.",
        "",
    ])

    collision = data["destination_collisions"][0]
    lines.extend([
        "## Colisão full-text",
        "",
        f"- Destino suspenso: `{collision['suspended_target']}`.",
        f"- Origens: {', '.join('`' + item + '`' for item in collision['origins'])}.",
        f"- Bytes idênticos: **{'sim' if collision['identical_bytes'] else 'não'}**.",
        f"- AST idêntica: **{'sim' if collision['identical_ast'] else 'não'}**.",
        f"- Similaridade por linhas: **{collision['line_similarity_ratio']:.6f}**.",
        f"- Blocos alterados: **{collision['changed_blocks']}**.",
        "- Decisão: **suspensa para revisão manual; fora do primeiro aplicador**.",
        "",
        "## Validação",
        "",
        f"- `py_compile`: `{data['validation'].get('py_compile', 'pending')}`.",
        f"- `git diff --check`: `{data['validation'].get('git_diff_check', 'pending')}`.",
        f"- Suíte específica: `{data['validation'].get('specific_suite', {}).get('summary', data['validation'].get('specific_suite', {}).get('status', 'pending'))}`.",
        f"- Suíte consolidada: `{data['validation'].get('consolidated_suite', {}).get('summary', data['validation'].get('consolidated_suite', {}).get('status', 'pending'))}`.",
        "",
        "## Decisão de fase",
        "",
        "O aplicador produtivo da AP-004B permanece bloqueado até aprovação expressa deste inventário semântico.",
    ])
    return normalize_output("\n".join(lines))



def build_durable_ap004a_contract(software_root: Path) -> str:
    # Converte o contrato AP-004A transitório em contrato durável.
    path = software_root / AP004A_TEST_REL
    source = path.read_text(encoding="utf-8")

    if "EXPECTED_AP004A_SUBJECT" in source and "_find_ap004a_commit" in source:
        ast.parse(source, filename=AP004A_TEST_REL.as_posix())
        return normalize_output(source)

    constant_pattern = re.compile(r"^(EXPECTED_AP003G_COMMIT = .+)$", re.MULTILINE)
    source, replacements = constant_pattern.subn(
        r"\1\nEXPECTED_AP004A_SUBJECT = "
        r"'chore(academic-pipeline): consolidar inventário de nomes da AP-004A'",
        source, count=1,
    )
    if replacements != 1:
        fail("Contrato AP-004A não contém bloco de constantes esperado.")

    helper_needle = '''def _ephemeral(path: str) -> bool:\n    parts = Path(path).parts\n    return "__pycache__" in parts or ".pytest_cache" in parts or path.endswith((".pyc", ".pyo"))\n\n\n'''
    helper_replacement = helper_needle + '''def _software_relative(path: str) -> str:\n    normalized = path.strip().strip('"').replace("\\\\", "/")\n    prefix = "software/academic_pipeline_rc10_7_conformidade/"\n    return normalized[len(prefix):] if normalized.startswith(prefix) else normalized\n\n\ndef _commit_paths(commit: str) -> set[str]:\n    output = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)\n    return {_software_relative(line) for line in output.splitlines() if line.strip()}\n\n\ndef _find_ap004a_commit() -> str:\n    output = _run("git", "log", "--format=%H%x09%s", f"{EXPECTED_HEAD}..HEAD")\n    matches = []\n    for line in output.splitlines():\n        if "\\t" not in line:\n            continue\n        commit, subject = line.split("\\t", 1)\n        if subject == EXPECTED_AP004A_SUBJECT:\n            matches.append(commit)\n    assert len(matches) == 1, matches\n    return matches[0]\n\n\n'''
    if source.count(helper_needle) != 1:
        fail("Contrato AP-004A não contém helper efêmero esperado.")
    source = source.replace(helper_needle, helper_replacement, 1)

    old_first = '''def test_ap004a_v4_2_is_bound_to_current_head_and_ap003g() -> None:\n    data = _data()\n    assert data["phase"] == "AP-004A"\n    assert data["mode"] == "inventory-and-convention-v4.2-read-only"\n    assert data["inventory_schema_version"] == 4\n    assert data["inventory_revision"] == "4.2"\n    assert data["tool"]["version"] == 4\n    assert data["tool"]["revision"] == "4.2"\n    assert data["git"]["head"] == EXPECTED_HEAD\n    assert _run("git", "rev-parse", "HEAD") == EXPECTED_HEAD\n    assert data["ap003g_closure"]["commit"] == EXPECTED_AP003G_COMMIT\n    assert data["ap003g_closure"]["published"] is True\n\n\n'''
    new_first = '''def test_ap004a_v4_2_is_bound_to_inventory_baseline_and_ap003g() -> None:\n    data = _data()\n    assert data["phase"] == "AP-004A"\n    assert data["mode"] == "inventory-and-convention-v4.2-read-only"\n    assert data["inventory_schema_version"] == 4\n    assert data["inventory_revision"] == "4.2"\n    assert data["tool"]["version"] == 4\n    assert data["tool"]["revision"] == "4.2"\n    assert data["git"]["head"] == EXPECTED_HEAD\n    current_head = _run("git", "rev-parse", "HEAD")\n    if current_head != EXPECTED_HEAD:\n        closure = _find_ap004a_commit()\n        subprocess.run(\n            ("git", "merge-base", "--is-ancestor", closure, current_head),\n            cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,\n        )\n        assert _run("git", "rev-parse", f"{closure}^") == EXPECTED_HEAD\n    assert data["ap003g_closure"]["commit"] == EXPECTED_AP003G_COMMIT\n    assert data["ap003g_closure"]["published"] is True\n\n\n'''
    if source.count(old_first) != 1:
        fail("Contrato AP-004A não contém teste de HEAD transitório esperado.")
    source = source.replace(old_first, new_first, 1)

    old_last = '''def test_ap004a_v4_2_changes_only_allowed_files_and_generated_python_compiles() -> None:\n    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")\n    actual = {\n        path for line in status.splitlines() if line.strip()\n        for path in [_status_path(line)] if not _ephemeral(path)\n    }\n    assert actual == set(EXPECTED_OUTPUTS)\n    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256\n    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))\n    assert CONVENTION.is_file()\n    with tempfile.TemporaryDirectory(prefix="ap004a-contract-pyc-") as tmp:\n        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)\n        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)\n'''
    new_last = '''def test_ap004a_v4_2_commit_scope_and_generated_python_are_durable() -> None:\n    current_head = _run("git", "rev-parse", "HEAD")\n    if current_head == EXPECTED_HEAD:\n        status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")\n        actual = {\n            path for line in status.splitlines() if line.strip()\n            for path in [_status_path(line)] if not _ephemeral(path)\n        }\n        assert actual == set(EXPECTED_OUTPUTS)\n    else:\n        closure = _find_ap004a_commit()\n        assert _commit_paths(closure) == set(EXPECTED_OUTPUTS)\n        for relative in EXPECTED_OUTPUTS:\n            _run("git", "ls-files", "--error-unmatch", relative)\n    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256\n    ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))\n    assert CONVENTION.is_file()\n    with tempfile.TemporaryDirectory(prefix="ap004a-contract-pyc-") as tmp:\n        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)\n        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)\n'''
    if source.count(old_last) != 1:
        fail("Contrato AP-004A não contém teste transitório de escopo esperado.")
    source = source.replace(old_last, new_last, 1)

    source = normalize_output(source)
    tree = ast.parse(source, filename=AP004A_TEST_REL.as_posix())
    tests = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if len(tests) != 10:
        fail(f"Contrato AP-004A durável deveria manter 10 testes; encontrados: {len(tests)}")
    if "_find_ap004a_commit" not in source or "EXPECTED_AP004A_SUBJECT" not in source:
        fail("Manutenção durável AP-004A incompleta.")
    return source


def build_contract_test(*, head: str, tool_sha256: str) -> str:
    candidate_paths = [item["current_path"] for item in CANDIDATE_SPECS]
    output_paths = [item.as_posix() for item in OUTPUT_RELS]
    source = f'''\
from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / {INVENTORY_REL.as_posix()!r}
TOOL = ROOT / {TOOL_REL.as_posix()!r}
EXPECTED_HEAD = {head!r}
EXPECTED_TOOL_SHA256 = {tool_sha256!r}
EXPECTED_CANDIDATES = {candidate_paths!r}
EXPECTED_OUTPUTS = {output_paths!r}
EXPECTED_AP004A_FILES = {[item.as_posix() for item in AP004A_COMMIT_FILES]!r}
EFFECTIVE = {{"actionable_productive", "compatibility_contract"}}
EXCLUDED = {{
    "historical_immutable", "physical_directory_reference",
    "protected_operational", "contextual_non_actionable",
}}
STRUCTURAL_PATHS = {{
    {ORCHESTRATOR_REL.as_posix()!r},
    {PACKAGE_MAIN_REL.as_posix()!r},
    {PACKAGE_CLI_REL.as_posix()!r},
    {PACKAGE_LEGACY_REL.as_posix()!r},
    {PRISMA_ORCHESTRATION_REL.as_posix()!r},
}}


def _run(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    )
    return result.stdout.strip()


def _data() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_path(line: str) -> str:
    raw = line[2:].lstrip() if len(line) >= 2 else line
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    raw = raw.strip().strip('"').replace("\\\\", "/")
    prefix = "software/academic_pipeline_rc10_7_conformidade/"
    return raw[len(prefix):] if raw.startswith(prefix) else raw


def _ephemeral(path: str) -> bool:
    parts = PurePosixPath(path).parts
    ignored = {{"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}}
    return any(part in ignored for part in parts) or path.endswith((".pyc", ".pyo"))


def test_ap004b_v1_6_is_bound_to_clean_ap004a_head() -> None:
    data = _data()
    assert data["phase"] == "AP-004B"
    assert data["mode"] == "module-file-inventory-v1.6-read-only"
    assert data["inventory_schema_version"] == 2
    assert data["tool_version"] == 1
    assert data["tool_revision"] == "1.6"
    assert data["inventory_revision"] == "1.6"
    assert data["git"]["head"] == EXPECTED_HEAD
    assert _run("git", "rev-parse", "HEAD") == EXPECTED_HEAD
    assert _run("git", "rev-parse", "origin/ap-refactor/03-orchestrator-decomposition") == EXPECTED_HEAD
    assert data["ap004a_closure"]["commit"] == EXPECTED_HEAD
    assert data["ap004a_closure"]["changed_files"] == sorted(EXPECTED_AP004A_FILES)


def test_ap004b_v1_6_has_exact_candidate_matrix() -> None:
    data = _data()
    candidates = data["candidates"]
    assert [item["current_path"] for item in candidates] == EXPECTED_CANDIDATES
    assert len({{item["key"] for item in candidates}}) == 6
    assert sum(item["classification"] == "renomeação com compatibilidade" for item in candidates) == 4
    assert sum(item["classification"] == "renomeação de alto risco" for item in candidates) == 2
    assert all(item["decision"] == "inventory-only-no-change" for item in candidates)
    serialized = json.dumps(data, ensure_ascii=False)
    assert "gerar_log_diagnostico_artigo_v1_28" not in serialized
    assert "executar_artigo_longo_fulltext_v1_23" not in serialized
    assert "executar_artigo_longo_fulltext_v1_24" not in serialized


def test_ap004b_v1_6_candidate_hashes_and_ast_are_current() -> None:
    data = _data()
    for item in data["candidates"]:
        path = ROOT / item["current_path"]
        assert path.is_file()
        assert item["source"]["sha256"] == _sha256(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        dump = ast.dump(tree, include_attributes=False, annotate_fields=True)
        assert item["source"]["python"]["ast_sha256"] == hashlib.sha256(dump.encode()).hexdigest()


def test_ap004b_v1_6_consolidates_entrypoints_without_renaming_public_cli() -> None:
    data = _data()
    assert data["compatibility_rules"]["public_entrypoints_preserved"] == [
        "academic-pipeline", "python -m academic_pipeline"
    ]
    assert data["ap003_architecture"]["public_entrypoints"] == [
        "academic-pipeline", "python -m academic_pipeline"
    ]
    assert all(item["related_entrypoints"] for item in data["candidates"][:4])


def test_ap004b_v1_6_semantic_consumers_are_deduplicated_and_partitioned() -> None:
    data = _data()
    references = data["reference_records"]
    identities = [
        (
            item["candidate_key"], item["consumer_path"], item.get("line"),
            tuple(item.get("matched_tokens", [])),
        )
        for item in references
    ]
    assert len(identities) == len(set(identities))
    assert all(item["semantic_category"] in EFFECTIVE | EXCLUDED for item in references)
    assert data["consumer_records"] == [
        item for item in references if item["semantic_category"] in EFFECTIVE
    ]
    for candidate in data["candidates"]:
        assert candidate["consumers"] == candidate["actionable_consumers"] + candidate["compatibility_contracts"] or sorted(
            candidate["actionable_consumers"] + candidate["compatibility_contracts"],
            key=lambda item: (item["consumer_path"], item.get("line") or 0, item["kind"], item.get("excerpt", "")),
        ) == candidate["consumers"]
        assert all(item["semantic_category"] in EFFECTIVE for item in candidate["consumers"])


def test_ap004b_v1_6_does_not_confuse_module_with_physical_directory() -> None:
    data = _data()
    for item in data["reference_records"]:
        matched = item.get("matched_tokens", [])
        excerpt = item.get("excerpt", "")
        if matched == ["academic_pipeline_rc10"]:
            assert "academic_pipeline_rc10_7_conformidade" not in excerpt
    assert all(
        item["semantic_category"] == "physical_directory_reference"
        and item["candidate_key"] is None
        for item in data["directory_reference_records"]
    )
    assert data["compatibility_rules"]["physical_directory_references_deferred_to"] == "AP-006"


def test_ap004b_v1_6_manifests_exclude_historical_and_operational_consumers() -> None:
    data = _data()
    source_manifest = data["source_manifest"]
    control_manifest = data["control_manifest"]
    assert STRUCTURAL_PATHS <= set(source_manifest)
    assert set(control_manifest) == {{
        "docs/refactor/academic-pipeline/AP-004/AP-004_NAMING_CONVENTION.md",
        "docs/refactor/academic-pipeline/AP-004/ap004a_naming_inventory.json",
    }}
    for path in source_manifest:
        parts = PurePosixPath(path).parts
        assert "backups" not in parts
        assert "assets" not in parts
        assert not path.startswith("docs/refactor/academic-pipeline/AP-003/")
        assert not PurePosixPath(path).name.startswith(("aplicar_", "atualizar_", "install_", "setup_"))
    for path, record in {{**source_manifest, **control_manifest}}.items():
        actual = ROOT / path
        assert actual.is_file()
        assert record["sha256"] == _sha256(actual)


def test_ap004b_v1_6_preserves_collision_without_automatic_choice() -> None:
    data = _data()
    collision = data["destination_collisions"][0]
    assert collision["group"] == "fulltext_executor"
    assert collision["suspended_target"] == "executar_artigo_longo_fulltext.py"
    assert collision["origins"] == [
        "executar_artigo_longo_fulltext_v1_13.py",
        "executar_artigo_longo_fulltext_v1_14.py",
    ]
    assert collision["decision"] == "suspended-manual-review-required"
    assert data["proposed_execution_order"][-1]["candidate_key"] == "fulltext_executor"
    assert data["proposed_execution_order"][-1]["blocked"] is True


def test_ap004b_v1_6_requires_wrappers_for_compatible_renames() -> None:
    data = _data()
    compatible = [
        item for item in data["candidates"]
        if item["classification"] == "renomeação com compatibilidade"
    ]
    assert len(compatible) == 4
    assert all(item["compatibility_required"] for item in compatible)
    assert all("wrapper" in item["compatibility_policy"] for item in compatible)
    assert data["compatibility_rules"]["wrapper_must_forward_main_and_exit_code"] is True
    assert data["compatibility_rules"]["aliases_expire_in"].startswith("AP-004E")


def test_ap004b_v1_6_keeps_protected_scope_out_of_application() -> None:
    data = _data()
    assert data["protected"]["physical_directory"] == "academic_pipeline_rc10_7_conformidade"
    assert data["protected"]["known_xfails"] == [
        "_refs_v6_strip_org", "extract_org_abstracts", "WorkflowState._normalize"
    ]
    assert data["scope"]["productive_changes"] == []
    assert data["scope"]["functional_change"] == "proibida"
    assert data["scope"]["cli_semantics_change"] == "proibida"
    assert data["compatibility_rules"]["historical_documents_are_immutable"] is True
    assert data["compatibility_rules"]["protected_operational_artifacts_excluded"] is True


def test_ap004b_v1_6_proposed_order_is_blocked_and_collision_last() -> None:
    data = _data()
    order = data["proposed_execution_order"]
    assert len(order) == 5
    assert all(item["blocked"] is True for item in order)
    assert order[-1]["candidate_key"] == "fulltext_executor"
    assert "colisão" in order[-1]["reason"]
    assert data["next_gate"]["productive_applicator_allowed"] is False


def test_ap004b_v1_6_preserves_ap003_architecture_contract() -> None:
    data = _data()
    architecture = data["ap003_architecture"]
    assert architecture["top_level_function_names"].count("main") == 1
    assert architecture["top_level_function_names"].count("_ap003f_pipeline_core") == 1
    assert architecture["historical_alias_absent"] is True
    assert architecture["prisma_calls_internal_core"] is True
    assert STRUCTURAL_PATHS <= set(data["source_manifest"])


def test_ap004b_v1_6_changes_only_allowed_files_and_generated_python_compiles() -> None:
    data = _data()
    assert data["scope"]["allowed_outputs"] == EXPECTED_OUTPUTS
    assert _status_path(" M tests/example.py") == "tests/example.py"
    assert _status_path("M  tests/example.py") == "tests/example.py"
    assert _status_path("?? tests/example.py") == "tests/example.py"
    assert _status_path(
        " M software/academic_pipeline_rc10_7_conformidade/tests/example.py"
    ) == "tests/example.py"
    assert _status_path("R  old.py -> tests/example.py") == "tests/example.py"
    assert _sha256(TOOL) == EXPECTED_TOOL_SHA256
    with tempfile.TemporaryDirectory(prefix="ap004b-contract-") as tmp:
        py_compile.compile(str(TOOL), cfile=str(Path(tmp) / "tool.pyc"), doraise=True)
        py_compile.compile(str(Path(__file__)), cfile=str(Path(tmp) / "test.pyc"), doraise=True)
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    actual = {{
        _status_path(line)
        for line in status.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }}
    assert actual == set(EXPECTED_OUTPUTS)
'''
    return normalize_output(source)


def count_test_functions(source: str) -> int:
    tree = ast.parse(source)
    return sum(
        1 for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def whitespace_check(paths: Iterable[Path]) -> None:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.endswith((" ", "\t")):
                errors.append(f"{path}:{number}: trailing whitespace")
        if text and not text.endswith("\n"):
            errors.append(f"{path}: missing final newline")
    if errors:
        fail("Falhas de whitespace:\n" + "\n".join(errors))


def validate_allowed_final_status(
    *, repository_root: Path, software_root: Path
) -> None:
    lines = [
        line for line in git(
            repository_root, "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout.splitlines() if line.strip()
    ]
    actual: set[str] = set()
    for line in lines:
        path = software_relative_status_path(
            line, software_root=software_root, repository_root=repository_root
        )
        if not is_ephemeral(path):
            actual.add(path)
    expected = {item.as_posix() for item in OUTPUT_RELS}
    if actual != expected:
        fail(
            "Conjunto final de alterações divergente.\n"
            f"Esperado: {sorted(expected)}\nAtual: {sorted(actual)}"
        )


def parse_pytest_summary(result: CommandResult, *, label: str) -> dict[str, Any]:
    combined = "\n".join(
        value for value in (result.stdout.strip(), result.stderr.strip()) if value
    )
    matches = list(SUMMARY_PATTERN.finditer(combined))
    if not matches:
        fail(f"Resumo pytest não reconhecido ({label}):\n{combined}")
    match = matches[-1]
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "passed": int(match.group("passed")),
        "xfailed": int(match.group("xfailed") or 0),
        "summary": match.group(0),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_specific_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        ("pipenv", "run", "pytest", "-q", "-ra", TEST_REL.as_posix()),
        cwd=software_root, check=False, timeout=600,
    )
    parsed = parse_pytest_summary(result, label="AP-004B específica")
    if result.returncode != 0:
        fail(f"Suíte específica AP-004B falhou:\n{result.stdout}{result.stderr}")
    if parsed["passed"] != EXPECTED_CONTRACT_TESTS or parsed["xfailed"] != 0:
        fail(
            "Contagem específica AP-004B divergente.\n"
            f"Esperado: {EXPECTED_CONTRACT_TESTS} passed, 0 xfailed\n"
            f"Atual: {parsed['summary']}"
        )
    return parsed


def run_consolidated_suite(software_root: Path) -> dict[str, Any]:
    result = run(
        (
            "pipenv", "run", "pytest", "-q", "-ra",
            "app_bundle/tests", "tests",
        ),
        cwd=software_root, check=False, timeout=1800,
    )
    parsed = parse_pytest_summary(result, label="AP-004B consolidada")
    expected_passed = BASELINE_PASSED + EXPECTED_CONTRACT_TESTS
    if result.returncode != 0:
        fail(f"Suíte consolidada AP-004B falhou:\n{result.stdout}{result.stderr}")
    if parsed["passed"] != expected_passed or parsed["xfailed"] != BASELINE_XFAILED:
        fail(
            "Contagem consolidada AP-004B divergente.\n"
            f"Esperado: {expected_passed} passed, {BASELINE_XFAILED} xfailed\n"
            f"Atual: {parsed['summary']}"
        )
    return parsed


def write_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventário preparatório AP-004B de módulos e arquivos."
    )
    parser.add_argument(
        "--skip-remote-check", action="store_true",
        help="Uso excepcional offline; não comprova publicação no GitHub.",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Uso diagnóstico; gera artefatos sem validar pytest.",
    )
    return parser.parse_args()


def main() -> int:
    validate_static_candidate_configuration()
    validate_semantic_classifier_configuration()
    args = parse_args()
    software_root = Path.cwd().resolve()
    tool_source = Path(__file__).read_text(encoding="utf-8")
    ast.parse(tool_source, filename=str(Path(__file__)))

    repository_root, git_state = validate_git_state(
        software_root, skip_remote_check=args.skip_remote_check
    )
    if args.skip_remote_check:
        print(
            "[AVISO] Verificação remota ignorada; a execução não confirma "
            "publicação no GitHub.", file=sys.stderr,
        )
    ap004a = validate_ap004a_closure(
        software_root=software_root, repository_root=repository_root
    )
    architecture = validate_ap003_architecture(software_root)
    inventory = build_inventory_data(
        software_root=software_root,
        repository_root=repository_root,
        git_state=git_state,
        ap004a=ap004a,
        architecture=architecture,
        tool_source=tool_source,
    )
    ap004a_test_source = build_durable_ap004a_contract(software_root)
    test_source = build_contract_test(
        head=git_state["head"], tool_sha256=inventory["tool"]["sha256"]
    )
    if count_test_functions(test_source) != EXPECTED_CONTRACT_TESTS:
        fail(
            f"Esperados {EXPECTED_CONTRACT_TESTS} testes; "
            f"gerados {count_test_functions(test_source)}."
        )
    ast.parse(test_source, filename=TEST_REL.as_posix())
    strategy = build_strategy_document(inventory)
    report = build_report(inventory)
    outputs = {
        software_root / TOOL_REL: normalize_output(tool_source),
        software_root / STRATEGY_REL: strategy,
        software_root / INVENTORY_REL: write_json(inventory),
        software_root / REPORT_REL: report,
        software_root / TEST_REL: test_source,
        software_root / AP004A_TEST_REL: ap004a_test_source,
    }
    backup_root, backup_records = create_backups(
        outputs, software_root=software_root
    )
    try:
        for path, content in outputs.items():
            atomic_write(path, content)
        compile_python(software_root / TOOL_REL)
        compile_python(software_root / TEST_REL)
        compile_python(software_root / AP004A_TEST_REL)
        whitespace_check(outputs)
        diff_check = git(repository_root, "diff", "--check", check=False)
        if diff_check.returncode != 0:
            fail(f"git diff --check falhou:\n{diff_check.stdout}{diff_check.stderr}")
        validate_allowed_final_status(
            repository_root=repository_root, software_root=software_root
        )
        validation: dict[str, Any] = {
            "py_compile": "passed",
            "git_diff_check": "passed",
            "specific_suite": {"status": "skipped"},
            "consolidated_suite": {"status": "skipped"},
        }
        if not args.skip_tests:
            validation["specific_suite"] = run_specific_suite(software_root)
            validation["consolidated_suite"] = run_consolidated_suite(software_root)
        inventory["validation"] = validation
        # Atualiza JSON e relatório com o resultado efetivo, sem alterar o contrato.
        atomic_write(software_root / INVENTORY_REL, write_json(inventory))
        atomic_write(software_root / REPORT_REL, build_report(inventory))
        whitespace_check(outputs)
        validate_allowed_final_status(
            repository_root=repository_root, software_root=software_root
        )
    except Exception:
        rollback(backup_records)
        raise

    print("[OK] AP-004B inventariada sem alteração produtiva.")
    print(f"[OK] Branch: {git_state['branch']}")
    print(f"[OK] HEAD local/remoto: {git_state['head']}")
    print(
        "[OK] Commit AP-004A confirmado: "
        f"{ap004a['commit']} — {ap004a['subject']}"
    )
    print(f"[OK] Candidatos: {inventory['statistics']['candidates']}")
    print(
        "     renomeação com compatibilidade: "
        f"{inventory['statistics']['by_classification'].get('renomeação com compatibilidade', 0)}"
    )
    print(
        "     renomeação de alto risco: "
        f"{inventory['statistics']['by_classification'].get('renomeação de alto risco', 0)}"
    )
    print(f"[OK] Referências brutas/deduplicadas: {inventory['statistics']['raw_candidate_reference_records']}/{inventory['statistics']['deduplicated_candidate_reference_records']}")
    print(f"[OK] Consumidores efetivos: {inventory['statistics']['effective_consumer_records']} em {inventory['statistics']['effective_consumer_files']} arquivos")
    print(f"     produtivos acionáveis: {inventory['statistics']['actionable_productive_records']}")
    print(f"     contratos de compatibilidade: {inventory['statistics']['compatibility_contract_records']}")
    print(f"     históricos/contextuais/operacionais: {inventory['statistics']['historical_immutable_records']}/{inventory['statistics']['contextual_non_actionable_records']}/{inventory['statistics']['protected_operational_records']}")
    print(f"[OK] Referências ao diretório físico (AP-006): {inventory['statistics']['physical_directory_reference_records']}")
    print(f"[OK] Colisões de destino: {inventory['statistics']['destination_collisions']}")
    print(f"[OK] Relatório: {REPORT_REL}")
    print(f"[OK] Estratégia: {STRATEGY_REL}")
    print(f"[OK] JSON: {INVENTORY_REL}")
    print(f"[OK] Teste: {TEST_REL}")
    print(f"[OK] Ferramenta reexecutável: {TOOL_REL}")
    print(f"[OK] Backup externo: {backup_root}")
    if not args.skip_tests:
        print(
            "[OK] Suíte específica: "
            f"{inventory['validation']['specific_suite']['summary']}"
        )
        print(
            "[OK] Suíte consolidada: "
            f"{inventory['validation']['consolidated_suite']['summary']}"
        )
    else:
        print("[AVISO] Testes ignorados; inventário não validado para consolidação.")
    print("[OK] Nenhum commit foi criado.")
    print("[BLOQUEIO] Não criar aplicador produtivo sem aprovação do inventário AP-004B.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InventoryError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        raise SystemExit(1)
