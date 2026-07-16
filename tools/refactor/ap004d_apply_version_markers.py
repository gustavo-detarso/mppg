#!/usr/bin/env python3
"""Aplicador transacional da AP-004D — consolidação de marcadores de versão.

Este aplicador (revisão v7) foi autorizado para o inventário lógico
2059d15dceb68a105e6b03b4fa15e900730ab398e1dc1eb03dd13143578571b1.

Características de segurança:
- não executa alterações sem ``--apply``;
- valida a baseline Git isolada da AP-004C e a igualdade local/remoto;
- valida integralmente o inventário e as 16 transformações antes da primeira escrita;
- usa AST, tokenização Python e classificação estrutural de strings operacionais;
- altera somente os cinco arquivos produtivos aprovados e sincroniza seis contratos de caracterização duráveis das fases anteriores;
- preserva contratos protegidos, a estrutura AP-003 e os fulltext congelados;
- cria backup externo, escreve atomicamente e restaura tudo em qualquer falha;
- executa py_compile, git diff --check, testes específicos e a suíte canônica;
- nunca cria commit e nunca publica no remoto.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tokenize
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_BRANCH = "ap-refactor/03-orchestrator-decomposition"
REMOTE_NAME = "origin"
REMOTE_REF = f"{REMOTE_NAME}/{EXPECTED_BRANCH}"
EXPECTED_HEAD = "81293d79e86da8b4d0407b483fc3dedaf27768cb"
EXPECTED_HEAD_SUBJECT = (
    "refactor(academic-pipeline): consolidar símbolos internos da AP-004C"
)
EXPECTED_INVENTORY_SCHEMA = "ap004d-version-marker-inventory/2"
EXPECTED_INVENTORY_DIGEST = (
    "2059d15dceb68a105e6b03b4fa15e900730ab398e1dc1eb03dd13143578571b1"
)
EXPECTED_INVENTORY_RECORDS = 14742
EXPECTED_CANDIDATE_RECORDS = 20
EXPECTED_UNIQUE_TRANSFORMATIONS = 16

SOFTWARE_RELATIVE = PurePosixPath(
    "software/academic_pipeline_rc10_7_conformidade"
)
INVENTORY_RELATIVE = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/ap004d_version_marker_inventory.json"
)
INVENTORY_MD_RELATIVE = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_INVENTORY.md"
)
STRATEGY_MD_RELATIVE = PurePosixPath(
    "docs/refactor/academic-pipeline/AP-004/AP-004D_VERSION_MARKER_STRATEGY.md"
)
INVENTORY_TEST_RELATIVE = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap004d_version_marker_inventory_contract.py"
)
INVENTORY_TOOL_RELATIVE = PurePosixPath(
    "tools/refactor/ap004d_inventory_version_markers.py"
)
APPLICATOR_RELATIVE = PurePosixPath(
    "tools/refactor/ap004d_apply_version_markers.py"
)

PREPARATORY_PATHS = frozenset(
    {
        str(INVENTORY_RELATIVE),
        str(INVENTORY_MD_RELATIVE),
        str(STRATEGY_MD_RELATIVE),
        str(INVENTORY_TEST_RELATIVE),
        str(INVENTORY_TOOL_RELATIVE),
    }
)

DOCUMENT_ORCHESTRATION = SOFTWARE_RELATIVE / PurePosixPath(
    "academic_pipeline/document_orchestration.py"
)
HISTORICAL_ORCHESTRATOR = SOFTWARE_RELATIVE / PurePosixPath(
    "app_bundle/scripts/pipeline/academic_pipeline_rc10.py"
)
TOML_GENERATOR = SOFTWARE_RELATIVE / PurePosixPath(
    "app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py"
)
CONFIGURATION_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "app_bundle/tests/test_rc10_configuration_characterization.py"
)
SMOKE_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "app_bundle/tests/test_rc10_smoke.py"
)
AP003D_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap003d_document_contract.py"
)
AP003E_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap003e_prisma_generic_contract.py"
)
AP003F_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap003f_main_unification_contract.py"
)
AP003G_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap003g_stabilization_contract.py"
)
AP004C_APPLICATION_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap004c_internal_symbol_application_contract.py"
)
AP004C_INVENTORY_CONTRACT_TEST = SOFTWARE_RELATIVE / PurePosixPath(
    "tests/characterization/test_ap004c_internal_symbol_inventory_contract.py"
)

# Cinco superfícies produtivas aprovadas no inventário + seis contratos de
# caracterização que precisam acompanhar nomes e hashes canônicos ou validar
# fases anteriores pelo commit publicado, em vez de congelar indefinidamente a
# árvore de trabalho atual. Isso torna os contratos duráveis sem reescrever os
# manifestos históricos nem reabrir a arquitetura da AP-003/AP-004C.
SYMBOL_TRANSFORMATION_PATHS = (
    DOCUMENT_ORCHESTRATION,
    HISTORICAL_ORCHESTRATOR,
    TOML_GENERATOR,
    CONFIGURATION_TEST,
    SMOKE_TEST,
    AP003D_CONTRACT_TEST,
)
DURABLE_CONTRACT_PATHS = (
    AP003E_CONTRACT_TEST,
    AP003F_CONTRACT_TEST,
    AP003G_CONTRACT_TEST,
    AP004C_APPLICATION_CONTRACT_TEST,
    AP004C_INVENTORY_CONTRACT_TEST,
)
PRODUCTIVE_PATHS = SYMBOL_TRANSFORMATION_PATHS + DURABLE_CONTRACT_PATHS
PRODUCTIVE_PATH_SET = frozenset(str(path) for path in PRODUCTIVE_PATHS)
ALLOWED_STATUS_PATHS = PREPARATORY_PATHS | PRODUCTIVE_PATH_SET | {
    str(APPLICATOR_RELATIVE)
}

# A validação AST precisa cobrir somente as superfícies canônicas que podem
# definir ou consumir os símbolos aprovados. Diretórios de backup, outputs e
# ambientes não são código-fonte da AP-004D e podem conter árvores recursivas
# ou caminhos que excedem os limites do sistema de arquivos.
PYTHON_SCAN_ROOTS = (
    SOFTWARE_RELATIVE / PurePosixPath("academic_pipeline"),
    SOFTWARE_RELATIVE / PurePosixPath("app_bundle/scripts/pipeline"),
    SOFTWARE_RELATIVE / PurePosixPath("app_bundle/tests"),
    SOFTWARE_RELATIVE / PurePosixPath("tests"),
)
EXCLUDED_SCAN_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "archive",
        "archives",
        "backup",
        "backups",
        "build",
        "cache",
        "coverage",
        "dist",
        "export",
        "exports",
        "generated",
        "htmlcov",
        "log",
        "logs",
        "node_modules",
        "output",
        "outputs",
        "site-packages",
        "temp",
        "tmp",
        "venv",
    }
)

FROZEN_FULLTEXT_NAMES = frozenset(
    {
        "executar_artigo_longo_fulltext_v1_13.py",
        "executar_artigo_longo_fulltext_v1_14.py",
    }
)
PROTECTED_IDENTIFIERS = frozenset(
    {
        "_refs_v6_strip_org",
        "_ap003d_impl__refs_v6_strip_org",
        "extract_org_abstracts",
        "_ap003f_pipeline_core",
    }
)
PROTECTED_QUALIFIED = frozenset({"WorkflowState._normalize"})

# Os dois destinos abaixo incorporam as correções aprovadas após a revisão:
#   _wiz_disable_references_pre_v5_2 -> _wiz_disable_references_original
#   _rc10_4_imports -> _load_pipeline_imports
RENAMES: dict[str, str] = {
    "_refs_v6_disabled_impl": "_refs_disabled_impl",
    "_refs_v6_apply_runtime_policy_impl": "_refs_apply_runtime_policy_impl",
    "_refs_v6_clear_document_bibliography_impl": "_refs_clear_document_bibliography_impl",
    "_refs_v6_strip_org_impl": "_refs_strip_org_impl",
    "_refs_v6_disabled": "_refs_disabled",
    "_refs_v6_apply_runtime_policy": "_refs_apply_runtime_policy",
    "_refs_v6_original_load_config": "_refs_original_load_config",
    "_refs_v6_original_build_bibliography": "_refs_original_build_bibliography",
    "_refs_v6_clear_document_bibliography": "_refs_clear_document_bibliography",
    "_refs_v6_original_render_org_latex": "_refs_original_render_org_latex",
    "_WIZ_V5_REFERENCE_POLICY": "_WIZ_REFERENCE_POLICY",
    "_v5_collect_outputs_and_options_original": "_collect_outputs_and_options_original",
    "_v5_render_toml_original": "_render_toml_original",
    "_v5_original_ensure_reference_policy": "_original_ensure_reference_policy",
    "_wiz_disable_references_pre_v5_2": "_wiz_disable_references_original",
    "_rc10_4_imports": "_load_pipeline_imports",
}

# Propostas originais registradas pelo inventário. Os dois overrides acima são
# validados de maneira explícita, sem adulterar o inventário aprovado.
INVENTORY_PROPOSALS: dict[str, str] = {
    **RENAMES,
    "_wiz_disable_references_pre_v5_2": "_wiz_disable_references_pre",
    "_rc10_4_imports": "_imports",
}
DESTINATION_OVERRIDES: dict[str, tuple[str, str]] = {
    "_wiz_disable_references_pre_v5_2": (
        "_wiz_disable_references_pre",
        "_wiz_disable_references_original",
    ),
    "_rc10_4_imports": ("_imports", "_load_pipeline_imports"),
}

# Contagens AST reproduzem exatamente o inventário aprovado. Importações com
# ``as`` registram no AST o nome vinculado, não o nome originalmente importado;
# por isso as contagens tokenizadas são mantidas separadamente.
EXPECTED_AST_OCCURRENCE_COUNTS: dict[str, int] = {
    "_refs_v6_disabled_impl": 1,
    "_refs_v6_apply_runtime_policy_impl": 1,
    "_refs_v6_clear_document_bibliography_impl": 1,
    "_refs_v6_strip_org_impl": 1,
    "_refs_v6_disabled": 5,
    "_refs_v6_apply_runtime_policy": 4,
    "_refs_v6_original_load_config": 1,
    "_refs_v6_original_build_bibliography": 1,
    "_refs_v6_clear_document_bibliography": 1,
    "_refs_v6_original_render_org_latex": 1,
    "_WIZ_V5_REFERENCE_POLICY": 4,
    "_v5_collect_outputs_and_options_original": 4,
    "_v5_render_toml_original": 2,
    "_v5_original_ensure_reference_policy": 2,
    "_wiz_disable_references_pre_v5_2": 2,
    "_rc10_4_imports": 2,
}

EXPECTED_TOKEN_EDIT_COUNTS: dict[str, int] = {
    **EXPECTED_AST_OCCURRENCE_COUNTS,
    # definição no módulo extraído + nome original em ``from ... import ... as``
    "_refs_v6_disabled_impl": 2,
    "_refs_v6_apply_runtime_policy_impl": 2,
    "_refs_v6_clear_document_bibliography_impl": 2,
    "_refs_v6_strip_org_impl": 2,
}

# Metadados de exportação executáveis no ``__all__`` do módulo extraído.
EXPECTED_EXPORT_LITERAL_COUNTS: dict[str, int] = {
    "_refs_v6_disabled_impl": 1,
    "_refs_v6_apply_runtime_policy_impl": 1,
    "_refs_v6_clear_document_bibliography_impl": 1,
    "_refs_v6_strip_org_impl": 1,
}

# Chaves operacionais do dicionário ``runtime`` que precisam acompanhar os
# símbolos efetivamente renomeados. O wrapper protegido ``_refs_v6_strip_org``
# não está nesta tabela e permanece intacto.
EXPECTED_RUNTIME_LITERAL_COUNTS: dict[str, int] = {
    "_refs_v6_disabled": 3,
    "_refs_v6_apply_runtime_policy": 1,
    "_refs_v6_original_load_config": 1,
    "_refs_v6_original_build_bibliography": 1,
    "_refs_v6_clear_document_bibliography": 1,
    "_refs_v6_original_render_org_latex": 2,
}

# Consultas operacionais ao namespace global. A política do wizard é lida por
# ``globals().get("...")`` e precisa acompanhar a renomeação do identificador.
EXPECTED_GLOBALS_LITERAL_COUNTS: dict[str, int] = {
    "_WIZ_V5_REFERENCE_POLICY": 1,
}

# Contrato de caracterização AP-003D: nomes declarados em EXPECTED_IMPLS,
# EXPECTED_HELPERS e nas chaves de implementation_aliases. Essas strings não
# são documentação histórica; são metadados executáveis usados pelos testes.
EXPECTED_AP003D_CONTRACT_LITERAL_COUNTS: dict[str, int] = {
    "_refs_v6_disabled_impl": 1,
    "_refs_v6_apply_runtime_policy_impl": 1,
    "_refs_v6_clear_document_bibliography_impl": 1,
    "_refs_v6_strip_org_impl": 1,
    "_refs_v6_disabled": 2,
    "_refs_v6_apply_runtime_policy": 2,
    "_refs_v6_clear_document_bibliography": 2,
}


# Rebaselines estritamente necessárias após as renomeações AP-004D. Os hashes
# novos foram obtidos da transformação pré-validada e confirmados pela primeira
# execução transacional (posteriormente revertida por contratos ainda antigos).
AP004C_DOCUMENT_SHA256 = "3f2a3c95e08ccc3c19e3019a225c36fcf532cf4468f75b13c56b7c43bbc88a8e"
AP004D_DOCUMENT_SHA256 = "c28a6201fcbd40339240fc3eac897c6924b9989810d8107038f445dde78e2c06"
AP004C_ORCHESTRATOR_SHA256 = "c871997af007fcc465e36299985e27ddaed0e56b2223869f9bc81774c6fdc5ec"
AP004D_ORCHESTRATOR_SHA256 = "b7d2e0c8039e0a35ef1ffde343fa315dd15670728fe099fb1dd2c5c7b3fe517d"
EXPECTED_FULL_SUITE_PASSED = 486
EXPECTED_FULL_SUITE_XFAILED = 3

DURABLE_HASH_ASSIGNMENTS: dict[PurePosixPath, tuple[tuple[str, str | None, str, str], ...]] = {
    AP003E_CONTRACT_TEST: (
        ("EXPECTED_DOCUMENT_SHA256", None, AP004C_DOCUMENT_SHA256, AP004D_DOCUMENT_SHA256),
    ),
    AP003F_CONTRACT_TEST: (
        ("EXPECTED_DOCUMENT_SHA256", None, AP004C_DOCUMENT_SHA256, AP004D_DOCUMENT_SHA256),
    ),
    AP003G_CONTRACT_TEST: (
        ("EXPECTED_HASHES", "orchestrator", AP004C_ORCHESTRATOR_SHA256, AP004D_ORCHESTRATOR_SHA256),
        ("EXPECTED_HASHES", "document", AP004C_DOCUMENT_SHA256, AP004D_DOCUMENT_SHA256),
    ),
}

INITIAL_AP004C_APPLICATION_HASH_TEST = r'''def test_ap004c_orchestrator_hash_is_rebaselined_in_ap003g_contract() -> None:
    data = _data()
    expected = data["waves"]["wave_2"]["source_sha256_after"]
    source = (ROOT / "tests/characterization/test_ap003g_stabilization_contract.py").read_text(encoding="utf-8")
    assert expected in source
    assert _sha256(ROOT / "app_bundle/scripts/pipeline/academic_pipeline_rc10.py") == expected
'''

DURABLE_AP004C_APPLICATION_HASH_TEST = r'''def test_ap004c_orchestrator_hash_is_rebaselined_in_ap003g_contract() -> None:
    data = _data()
    expected = data["waves"]["wave_2"]["source_sha256_after"]
    commit = _find_commit()
    assert commit is not None
    contract = _run(
        "git",
        "show",
        f"{commit}:{SOFTWARE_PREFIX}tests/characterization/test_ap003g_stabilization_contract.py",
    )
    assert contract.returncode == 0, contract.stderr
    assert expected in contract.stdout
    orchestrator = _run(
        "git",
        "show",
        f"{commit}:{SOFTWARE_PREFIX}app_bundle/scripts/pipeline/academic_pipeline_rc10.py",
    )
    assert orchestrator.returncode == 0, orchestrator.stderr
    assert hashlib.sha256(orchestrator.stdout.encode("utf-8")).hexdigest() == expected
'''

INITIAL_AP004C_APPLICATION_SCOPE_TEST = r'''def test_ap004c_git_scope_is_exact_or_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_DIRTY_PATHS)
    if actual:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
    else:
        commit = _find_commit()
        assert commit is not None
        changed = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        assert changed.returncode == 0, changed.stderr
        normalized = {
            path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
            for path in changed.stdout.splitlines() if path
        }
        assert normalized == expected
        assert _run("git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD").returncode == 0
'''

DURABLE_AP004C_APPLICATION_SCOPE_TEST = r'''def test_ap004c_git_scope_is_exact_or_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_DIRTY_PATHS)
    commit = _find_commit()
    if commit is None:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
        return
    changed = _run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    assert changed.returncode == 0, changed.stderr
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.stdout.splitlines() if path
    }
    assert normalized == expected
    assert _run("git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD").returncode == 0
'''

INITIAL_AP004C_INVENTORY_SCOPE_TEST = r'''def test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_AP004C_APPLICATION_OUTPUTS)
    if actual:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
    else:
        result = _run(
            "git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD"
        )
        assert result.returncode == 0, result.stderr
        matches = []
        for line in result.stdout.splitlines():
            if "\x00" not in line:
                continue
            commit, subject = line.split("\x00", 1)
            if subject == EXPECTED_AP004C_APPLICATION_SUBJECT:
                matches.append(commit)
        assert len(matches) == 1
        changed = _run(
            "git", "diff-tree", "--no-commit-id", "--name-only", "-r", matches[0]
        )
        assert changed.returncode == 0, changed.stderr
        normalized = {
            path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
            for path in changed.stdout.splitlines() if path
        }
        assert normalized == expected
        assert _run(
            "git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"
        ).returncode == 0
'''

DURABLE_AP004C_INVENTORY_SCOPE_TEST = r'''def test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable() -> None:
    result = _run("git", "status", "--porcelain=v1", "--untracked-files=all")
    assert result.returncode == 0, result.stderr
    actual = {
        _status_path(line) for line in result.stdout.splitlines()
        if line.strip() and not _ephemeral(_status_path(line))
    }
    expected = set(EXPECTED_AP004C_APPLICATION_OUTPUTS)
    result = _run(
        "git", "log", "--format=%H%x00%s", f"{BASELINE_HEAD}..HEAD"
    )
    assert result.returncode == 0, result.stderr
    matches = []
    for line in result.stdout.splitlines():
        if "\x00" not in line:
            continue
        commit, subject = line.split("\x00", 1)
        if subject == EXPECTED_AP004C_APPLICATION_SUBJECT:
            matches.append(commit)
    if not matches:
        assert actual == expected
        assert _run("git", "rev-parse", "HEAD").stdout.strip() == BASELINE_HEAD
        return
    assert len(matches) == 1
    changed = _run(
        "git", "diff-tree", "--no-commit-id", "--name-only", "-r", matches[0]
    )
    assert changed.returncode == 0, changed.stderr
    normalized = {
        path[len(SOFTWARE_PREFIX):] if path.startswith(SOFTWARE_PREFIX) else path
        for path in changed.stdout.splitlines() if path
    }
    assert normalized == expected
    assert _run(
        "git", "merge-base", "--is-ancestor", BASELINE_HEAD, "HEAD"
    ).returncode == 0
'''

DURABLE_FUNCTION_REPLACEMENTS: dict[
    PurePosixPath, tuple[tuple[str, str, str], ...]
] = {
    AP004C_APPLICATION_CONTRACT_TEST: (
        (
            "test_ap004c_orchestrator_hash_is_rebaselined_in_ap003g_contract",
            INITIAL_AP004C_APPLICATION_HASH_TEST,
            DURABLE_AP004C_APPLICATION_HASH_TEST,
        ),
        (
            "test_ap004c_git_scope_is_exact_or_commit_is_durable",
            INITIAL_AP004C_APPLICATION_SCOPE_TEST,
            DURABLE_AP004C_APPLICATION_SCOPE_TEST,
        ),
    ),
    AP004C_INVENTORY_CONTRACT_TEST: (
        (
            "test_ap004c_current_status_is_output_scope_or_clean_and_commit_is_durable",
            INITIAL_AP004C_INVENTORY_SCOPE_TEST,
            DURABLE_AP004C_INVENTORY_SCOPE_TEST,
        ),
    ),
}

EXPECTED_PATHS_BY_SOURCE: dict[str, frozenset[str]] = {
    "_refs_v6_disabled_impl": frozenset({str(DOCUMENT_ORCHESTRATION)}),
    "_refs_v6_apply_runtime_policy_impl": frozenset({str(DOCUMENT_ORCHESTRATION)}),
    "_refs_v6_clear_document_bibliography_impl": frozenset({str(DOCUMENT_ORCHESTRATION)}),
    "_refs_v6_strip_org_impl": frozenset({str(DOCUMENT_ORCHESTRATION)}),
    "_refs_v6_disabled": frozenset(
        {str(HISTORICAL_ORCHESTRATOR), str(CONFIGURATION_TEST)}
    ),
    "_refs_v6_apply_runtime_policy": frozenset(
        {str(HISTORICAL_ORCHESTRATOR), str(CONFIGURATION_TEST)}
    ),
    "_refs_v6_original_load_config": frozenset({str(HISTORICAL_ORCHESTRATOR)}),
    "_refs_v6_original_build_bibliography": frozenset(
        {str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_refs_v6_clear_document_bibliography": frozenset(
        {str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_refs_v6_original_render_org_latex": frozenset(
        {str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_WIZ_V5_REFERENCE_POLICY": frozenset({str(TOML_GENERATOR)}),
    "_v5_collect_outputs_and_options_original": frozenset({str(TOML_GENERATOR)}),
    "_v5_render_toml_original": frozenset({str(TOML_GENERATOR)}),
    "_v5_original_ensure_reference_policy": frozenset({str(TOML_GENERATOR)}),
    "_wiz_disable_references_pre_v5_2": frozenset({str(TOML_GENERATOR)}),
    "_rc10_4_imports": frozenset({str(SMOKE_TEST)}),
}

EXPECTED_TOKEN_PATHS_BY_SOURCE: dict[str, frozenset[str]] = {
    **EXPECTED_PATHS_BY_SOURCE,
    "_refs_v6_disabled_impl": frozenset(
        {str(DOCUMENT_ORCHESTRATION), str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_refs_v6_apply_runtime_policy_impl": frozenset(
        {str(DOCUMENT_ORCHESTRATION), str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_refs_v6_clear_document_bibliography_impl": frozenset(
        {str(DOCUMENT_ORCHESTRATION), str(HISTORICAL_ORCHESTRATOR)}
    ),
    "_refs_v6_strip_org_impl": frozenset(
        {str(DOCUMENT_ORCHESTRATION), str(HISTORICAL_ORCHESTRATOR)}
    ),
}


SPECIFIC_TESTS = (
    "tests/characterization/test_ap004d_version_marker_inventory_contract.py",
    "tests/characterization/test_ap003d_document_contract.py",
    "tests/characterization/test_ap003e_prisma_generic_contract.py",
    "tests/characterization/test_ap003f_main_unification_contract.py",
    "tests/characterization/test_ap003g_stabilization_contract.py",
    "tests/characterization/test_ap004c_internal_symbol_application_contract.py",
    "tests/characterization/test_ap004c_internal_symbol_inventory_contract.py",
    "app_bundle/tests/test_rc10_configuration_characterization.py",
    "app_bundle/tests/test_rc10_smoke.py",
)


class ApplicatorError(RuntimeError):
    """Falha controlada que deve bloquear ou reverter a AP-004D."""


@dataclasses.dataclass(frozen=True)
class GitBaseline:
    repo_root: Path
    branch: str
    head: str
    subject: str
    remote_head: str
    status_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Occurrence:
    path: str
    line: int
    column: int
    name: str
    kind: str
    scope: str
    qualified_name: str


@dataclasses.dataclass(frozen=True)
class BackupEntry:
    relative: str
    existed: bool
    mode: int | None
    sha256: str | None


@dataclasses.dataclass(frozen=True)
class PreparedApplication:
    state: str
    baseline: GitBaseline
    inventory: Mapping[str, Any]
    tracked_python: tuple[PurePosixPath, ...]
    original_bytes: Mapping[PurePosixPath, bytes]
    transformed_bytes: Mapping[PurePosixPath, bytes]
    occurrences_before: tuple[Occurrence, ...]
    protected_snapshot: tuple[tuple[Any, ...], ...]
    frozen_hashes: Mapping[str, str]
    edit_counts: Mapping[str, int]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env) if env is not None else None,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(args)
        raise ApplicatorError(
            f"Comando falhou ({completed.returncode}): {rendered}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _discover_repo_root(explicit: str | None) -> Path:
    start = Path(explicit).expanduser().resolve() if explicit else Path.cwd().resolve()
    completed = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ApplicatorError(
            "Raiz Git não encontrada. Execute dentro do repositório canônico "
            "ou informe --repo-root."
        )
    return Path(completed.stdout.strip()).resolve()


def _parse_status(repo_root: Path) -> tuple[str, ...]:
    result = _run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=repo_root,
    )
    paths: set[str] = set()
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        path_part = raw_line[3:]
        if " -> " in path_part:
            old_path, new_path = path_part.split(" -> ", 1)
            paths.add(old_path.strip('"'))
            paths.add(new_path.strip('"'))
        else:
            paths.add(path_part.strip('"'))
    return tuple(sorted(paths))


def _validate_git_operation_state(repo_root: Path) -> None:
    git_dir_text = _run(["git", "rev-parse", "--git-dir"], cwd=repo_root).stdout.strip()
    git_dir = Path(git_dir_text)
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    markers = (
        git_dir / "MERGE_HEAD",
        git_dir / "CHERRY_PICK_HEAD",
        git_dir / "REVERT_HEAD",
        git_dir / "BISECT_LOG",
        git_dir / "rebase-merge",
        git_dir / "rebase-apply",
    )
    active = [str(path) for path in markers if path.exists()]
    if active:
        raise ApplicatorError(
            "Operação Git em andamento; finalize-a antes da AP-004D:\n- "
            + "\n- ".join(active)
        )


def _validate_git_baseline(repo_root: Path, *, fetch: bool) -> GitBaseline:
    if not (repo_root / SOFTWARE_RELATIVE).is_dir():
        raise ApplicatorError(
            f"Raiz canônica do software ausente: {repo_root / SOFTWARE_RELATIVE}"
        )
    if fetch:
        _run(["git", "fetch", REMOTE_NAME], cwd=repo_root)

    branch = _run(["git", "branch", "--show-current"], cwd=repo_root).stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise ApplicatorError(
            f"Branch incorreta: {branch!r}; esperada: {EXPECTED_BRANCH!r}."
        )

    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    subject = _run(
        ["git", "show", "-s", "--format=%s", "HEAD"], cwd=repo_root
    ).stdout.strip()
    remote_head = _run(["git", "rev-parse", REMOTE_REF], cwd=repo_root).stdout.strip()
    divergence = _run(
        ["git", "rev-list", "--left-right", "--count", f"HEAD...{REMOTE_REF}"],
        cwd=repo_root,
    ).stdout.strip()

    if head != EXPECTED_HEAD:
        raise ApplicatorError(
            f"HEAD inesperado: {head}. Esperado para a AP-004C: {EXPECTED_HEAD}."
        )
    if subject != EXPECTED_HEAD_SUBJECT:
        raise ApplicatorError(
            f"Assunto do HEAD inesperado: {subject!r}. Esperado: {EXPECTED_HEAD_SUBJECT!r}."
        )
    if remote_head != head or divergence != "0\t0" and divergence != "0 0":
        raise ApplicatorError(
            "A baseline local/remota divergiu: "
            f"local={head}, remoto={remote_head}, divergência={divergence!r}."
        )

    _validate_git_operation_state(repo_root)
    status_paths = _parse_status(repo_root)
    unexpected = sorted(set(status_paths) - ALLOWED_STATUS_PATHS)
    if unexpected:
        raise ApplicatorError(
            "Árvore contém caminhos fora do escopo autorizado da AP-004D:\n- "
            + "\n- ".join(unexpected)
        )

    missing_preparatory = sorted(
        path for path in PREPARATORY_PATHS if not (repo_root / path).is_file()
    )
    if missing_preparatory:
        raise ApplicatorError(
            "Artefatos preparatórios AP-004D ausentes:\n- "
            + "\n- ".join(missing_preparatory)
        )

    return GitBaseline(
        repo_root=repo_root,
        branch=branch,
        head=head,
        subject=subject,
        remote_head=remote_head,
        status_paths=status_paths,
    )


def _logical_inventory_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("inventory_sha256", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _load_and_validate_inventory(repo_root: Path) -> Mapping[str, Any]:
    path = repo_root / INVENTORY_RELATIVE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApplicatorError(f"Inventário AP-004D ilegível: {exc}") from exc

    if payload.get("schema_version") != EXPECTED_INVENTORY_SCHEMA:
        raise ApplicatorError(
            f"Schema inesperado: {payload.get('schema_version')!r}."
        )
    if payload.get("inventory_sha256") != EXPECTED_INVENTORY_DIGEST:
        raise ApplicatorError("Digest declarado do inventário não é o aprovado.")
    if _logical_inventory_digest(payload) != EXPECTED_INVENTORY_DIGEST:
        raise ApplicatorError("Conteúdo do inventário não corresponde ao digest aprovado.")
    if payload.get("git", {}).get("head") != EXPECTED_HEAD:
        raise ApplicatorError("Inventário não foi produzido a partir da baseline AP-004C aprovada.")
    if payload.get("git", {}).get("branch") != EXPECTED_BRANCH:
        raise ApplicatorError("Inventário registra branch diferente da autorizada.")

    summary = payload.get("summary", {})
    if summary.get("record_count") != EXPECTED_INVENTORY_RECORDS:
        raise ApplicatorError("Contagem total do inventário mudou.")
    if summary.get("candidate_count") != EXPECTED_CANDIDATE_RECORDS:
        raise ApplicatorError("Contagem de candidatos do inventário mudou.")
    if summary.get("collision_count") != 0:
        raise ApplicatorError("Inventário contém colisão de destino.")
    if summary.get("parse_error_count") != 0:
        raise ApplicatorError("Inventário contém erro de análise.")
    if payload.get("application_gate", {}).get("productive_applicator_allowed") is not False:
        raise ApplicatorError("Gate original do inventário foi adulterado.")

    candidates = [
        record for record in payload.get("records", []) if record.get("decision") == "candidato"
    ]
    if len(candidates) != EXPECTED_CANDIDATE_RECORDS:
        raise ApplicatorError("Lista material de candidatos diverge do resumo.")
    if any(record.get("collision") for record in candidates):
        raise ApplicatorError("Candidato marcado com colisão indevida.")
    if any(record.get("classification") != "marcador_privado_renomeavel_ast" for record in candidates):
        raise ApplicatorError("Há candidato fora da classificação AST aprovada.")
    if any(record.get("occurrence_type") != "python_identifier" for record in candidates):
        raise ApplicatorError("Há candidato que não é identificador Python.")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record.get("current"))].append(record)
    if set(grouped) != set(RENAMES):
        missing = sorted(set(RENAMES) - set(grouped))
        extra = sorted(set(grouped) - set(RENAMES))
        raise ApplicatorError(
            f"Conjunto lógico de candidatos mudou. Ausentes={missing}; extras={extra}."
        )
    if len(grouped) != EXPECTED_UNIQUE_TRANSFORMATIONS:
        raise ApplicatorError("Número de transformações lógicas não é 16.")

    for source, records in grouped.items():
        proposals = {str(record.get("proposed")) for record in records}
        if proposals != {INVENTORY_PROPOSALS[source]}:
            raise ApplicatorError(
                f"Proposta inventariada mudou para {source}: {sorted(proposals)}."
            )
        paths = {str(record.get("path")) for record in records}
        if not paths <= EXPECTED_PATHS_BY_SOURCE[source]:
            raise ApplicatorError(
                f"Candidato {source} apareceu em caminho não aprovado: {sorted(paths)}."
            )

    scope = payload.get("scope", {})
    if not PROTECTED_IDENTIFIERS - {"_ap003f_pipeline_core"} <= set(
        scope.get("protected_symbols", [])
    ):
        raise ApplicatorError("Inventário perdeu símbolos protegidos por xfail.")
    if not PROTECTED_QUALIFIED <= set(scope.get("protected_qualified_symbols", [])):
        raise ApplicatorError("Inventário perdeu WorkflowState._normalize.")
    if "_ap003f_pipeline_core" not in set(scope.get("structural_ap003_symbols", [])):
        raise ApplicatorError("Inventário perdeu o contrato estrutural da AP-003.")
    if not FROZEN_FULLTEXT_NAMES <= set(scope.get("frozen_fulltext_files", [])):
        raise ApplicatorError("Inventário perdeu os fulltext congelados.")

    return payload


class IdentifierVisitor(ast.NodeVisitor):
    """Coleta ocorrências de identificadores com a mesma semântica do inventário."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.scope_stack: list[str] = ["<module>"]
        self.occurrences: list[Occurrence] = []

    @property
    def scope(self) -> str:
        return ".".join(self.scope_stack)

    def _qualified(self, name: str) -> str:
        visible = [item for item in self.scope_stack if item != "<module>"]
        return ".".join([*visible, name]) if visible else name

    def _add(
        self,
        *,
        name: str,
        node: ast.AST,
        kind: str,
        scope_override: str | None = None,
    ) -> None:
        self.occurrences.append(
            Occurrence(
                path=self.path,
                line=int(getattr(node, "lineno", 1)),
                column=int(getattr(node, "col_offset", 0)),
                name=name,
                kind=kind,
                scope=scope_override or self.scope,
                qualified_name=self._qualified(name),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        parent_scope = self.scope
        self._add(
            name=node.name,
            node=node,
            kind="function_definition",
            scope_override=parent_scope,
        )
        self.scope_stack.append(node.name)
        self._visit_arguments(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            self.visit(node.returns)
        for statement in node.body:
            self.visit(statement)
        self.scope_stack.pop()
        return None

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        parent_scope = self.scope
        self._add(
            name=node.name,
            node=node,
            kind="class_definition",
            scope_override=parent_scope,
        )
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.scope_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope_stack.pop()
        return None

    def _visit_arguments(self, args: ast.arguments) -> None:
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg:
            all_args.append(args.vararg)
        if args.kwarg:
            all_args.append(args.kwarg)
        for argument in all_args:
            self._add(name=argument.arg, node=argument, kind="argument_definition")
            if argument.annotation:
                self.visit(argument.annotation)
        for default in [*args.defaults, *args.kw_defaults]:
            if default is not None:
                self.visit(default)

    def visit_Name(self, node: ast.Name) -> Any:
        self._add(
            name=node.id,
            node=node,
            kind=f"name_{node.ctx.__class__.__name__.lower()}",
        )
        return None

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self.visit(node.value)
        self._add(
            name=node.attr,
            node=node,
            kind=f"attribute_{node.ctx.__class__.__name__.lower()}",
        )
        return None

    def visit_alias(self, node: ast.alias) -> Any:
        bound_name = node.asname or node.name.split(".")[0]
        self._add(name=bound_name, node=node, kind="import_binding")
        return None

    def visit_Global(self, node: ast.Global) -> Any:
        for name in node.names:
            self._add(name=name, node=node, kind="global_declaration")
        return None

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        for name in node.names:
            self._add(name=name, node=node, kind="nonlocal_declaration")
        return None


def _parse_python(data: bytes, relative: PurePosixPath) -> ast.AST:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        source = data.decode(encoding)
        return ast.parse(source, filename=str(relative))
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise ApplicatorError(f"Python inválido em {relative}: {exc}") from exc


def _collect_occurrences_for_bytes(
    relative: PurePosixPath, data: bytes
) -> tuple[Occurrence, ...]:
    tree = _parse_python(data, relative)
    visitor = IdentifierVisitor(str(relative))
    visitor.visit(tree)
    return tuple(visitor.occurrences)


def _is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


def _is_scannable_python_path(relative: PurePosixPath) -> bool:
    if relative.suffix.lower() != ".py":
        return False
    if not any(_is_under(relative, root) for root in PYTHON_SCAN_ROOTS):
        return False
    lowered_parts = {part.lower() for part in relative.parts}
    if lowered_parts & EXCLUDED_SCAN_DIRECTORY_NAMES:
        return False
    return True


def _git_tracked_python_files(repo_root: Path) -> tuple[PurePosixPath, ...]:
    # O filtro é aplicado sobre os nomes retornados pelo índice antes de qualquer
    # stat/open/read. Isso impede que backups recursivos com caminhos enormes
    # alcancem pathlib ou o kernel.
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(SOFTWARE_RELATIVE)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ApplicatorError(
            "git ls-files falhou: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    paths: list[PurePosixPath] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        decoded = raw.decode("utf-8", errors="surrogateescape")
        relative = PurePosixPath(decoded)
        if _is_scannable_python_path(relative):
            paths.append(relative)
    selected = tuple(sorted(set(paths), key=str))
    missing_roots = [
        str(root)
        for root in PYTHON_SCAN_ROOTS
        if not any(_is_under(path, root) for path in selected)
    ]
    if missing_roots:
        raise ApplicatorError(
            "Escopo Python canônico incompleto; nenhuma fonte rastreada encontrada em:\n- "
            + "\n- ".join(missing_roots)
        )
    return selected


def _read_python_corpus(
    repo_root: Path, paths: Iterable[PurePosixPath]
) -> tuple[dict[PurePosixPath, bytes], tuple[Occurrence, ...]]:
    data_by_path: dict[PurePosixPath, bytes] = {}
    occurrences: list[Occurrence] = []
    for relative in paths:
        absolute = repo_root / relative
        try:
            data = absolute.read_bytes()
        except OSError as exc:
            raise ApplicatorError(
                f"Não foi possível ler fonte Python canônica {relative}: {exc}"
            ) from exc
        data_by_path[relative] = data
        occurrences.extend(_collect_occurrences_for_bytes(relative, data))
    return data_by_path, tuple(occurrences)


def _occurrence_index(
    occurrences: Iterable[Occurrence], names: Iterable[str]
) -> dict[str, tuple[Occurrence, ...]]:
    wanted = set(names)
    grouped: dict[str, list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence.name in wanted:
            grouped[occurrence.name].append(occurrence)
    return {
        name: tuple(
            sorted(
                grouped.get(name, []),
                key=lambda item: (
                    item.path,
                    item.line,
                    item.column,
                    item.kind,
                    item.scope,
                ),
            )
        )
        for name in wanted
    }


def _protected_snapshot(occurrences: Iterable[Occurrence]) -> tuple[tuple[Any, ...], ...]:
    snapshot: list[tuple[Any, ...]] = []
    for occurrence in occurrences:
        is_protected_identifier = occurrence.name in PROTECTED_IDENTIFIERS
        is_protected_qualified = occurrence.qualified_name in PROTECTED_QUALIFIED
        if is_protected_identifier or is_protected_qualified:
            snapshot.append(
                (
                    occurrence.path,
                    occurrence.line,
                    occurrence.column,
                    occurrence.name,
                    occurrence.kind,
                    occurrence.scope,
                    occurrence.qualified_name,
                )
            )
    return tuple(sorted(snapshot))


def _find_frozen_files(repo_root: Path) -> dict[str, str]:
    """Localiza e fotografa os dois fulltext congelados sem ampliar o corpus AST.

    Os fulltext são contratos históricos, não candidatos da onda AST. Portanto,
    sua localização não deve depender de ``PYTHON_SCAN_ROOTS``. A busca ocorre
    pelo nome exato no índice Git e descarta diretórios residuais antes de
    qualquer acesso ao sistema de arquivos.
    """

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", str(SOFTWARE_RELATIVE)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ApplicatorError(
            "Não foi possível localizar arquivos fulltext congelados: "
            + result.stderr.decode("utf-8", errors="replace")
        )

    candidates_by_name: dict[str, list[PurePosixPath]] = {
        name: [] for name in FROZEN_FULLTEXT_NAMES
    }
    excluded_matches: dict[str, list[str]] = {
        name: [] for name in FROZEN_FULLTEXT_NAMES
    }

    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        decoded = raw.decode("utf-8", errors="surrogateescape")
        relative = PurePosixPath(decoded)
        if relative.name not in FROZEN_FULLTEXT_NAMES:
            continue

        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & EXCLUDED_SCAN_DIRECTORY_NAMES:
            excluded_matches[relative.name].append(str(relative))
            continue

        candidates_by_name[relative.name].append(relative)

    missing = sorted(
        name for name, paths in candidates_by_name.items() if not paths
    )
    ambiguous = {
        name: sorted(map(str, paths))
        for name, paths in candidates_by_name.items()
        if len(paths) > 1
    }
    if missing or ambiguous:
        details: list[str] = []
        if missing:
            details.append("ausentes=" + repr(missing))
        if ambiguous:
            details.append("duplicados=" + repr(ambiguous))
        excluded_summary = {
            name: sorted(paths)
            for name, paths in excluded_matches.items()
            if paths
        }
        if excluded_summary:
            details.append("matches_residuais_excluídos=" + repr(excluded_summary))
        raise ApplicatorError(
            "Contrato dos fulltext congelados não pôde ser resolvido de forma "
            "unívoca: " + "; ".join(details)
        )

    found: dict[str, str] = {}
    for name in sorted(FROZEN_FULLTEXT_NAMES):
        relative = candidates_by_name[name][0]
        try:
            data = (repo_root / relative).read_bytes()
        except OSError as exc:
            raise ApplicatorError(
                f"Não foi possível ler fulltext congelado {relative}: {exc}"
            ) from exc
        found[str(relative)] = _sha256_bytes(data)

    if {PurePosixPath(path).name for path in found} != FROZEN_FULLTEXT_NAMES:
        raise ApplicatorError(
            "Validação interna inconsistente dos fulltext congelados: "
            f"{sorted(found)}"
        )
    return dict(sorted(found.items()))


def _validate_initial_or_applied_state(
    occurrences: tuple[Occurrence, ...]
) -> str:
    old_index = _occurrence_index(occurrences, RENAMES)
    new_index = _occurrence_index(occurrences, RENAMES.values())

    old_counts = {name: len(items) for name, items in old_index.items()}
    new_counts = {name: len(items) for name, items in new_index.items()}
    initial = all(
        old_counts[source] == EXPECTED_AST_OCCURRENCE_COUNTS[source]
        for source in RENAMES
    ) and all(new_counts[destination] == 0 for destination in RENAMES.values())
    applied = all(old_counts[source] == 0 for source in RENAMES) and all(
        new_counts[destination] == EXPECTED_AST_OCCURRENCE_COUNTS[source]
        for source, destination in RENAMES.items()
    )

    if initial:
        for source, expected_paths in EXPECTED_PATHS_BY_SOURCE.items():
            actual_paths = {item.path for item in old_index[source]}
            if actual_paths != set(expected_paths):
                raise ApplicatorError(
                    f"Escopo AST inesperado para {source}: {sorted(actual_paths)}; "
                    f"esperado={sorted(expected_paths)}."
                )
        return "initial"

    if applied:
        for source, destination in RENAMES.items():
            actual_paths = {item.path for item in new_index[destination]}
            expected_paths = set(EXPECTED_PATHS_BY_SOURCE[source])
            if actual_paths != expected_paths:
                raise ApplicatorError(
                    f"Escopo aplicado inesperado para {destination}: "
                    f"{sorted(actual_paths)}; esperado={sorted(expected_paths)}."
                )
        return "applied"

    diagnostics = []
    for source, destination in RENAMES.items():
        diagnostics.append(
            f"{source}={old_counts[source]}/{EXPECTED_AST_OCCURRENCE_COUNTS[source]}, "
            f"{destination}={new_counts[destination]}/"
            f"{EXPECTED_AST_OCCURRENCE_COUNTS[source]}"
        )
    raise ApplicatorError(
        "Árvore em estado misto ou divergente; aplicação recusada:\n- "
        + "\n- ".join(diagnostics)
    )


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, character in enumerate(text):
        if character == "\n":
            offsets.append(index + 1)
    return offsets


def _absolute_offset(offsets: Sequence[int], position: tuple[int, int]) -> int:
    line, column = position
    if line < 1 or line > len(offsets):
        raise ApplicatorError(f"Posição de token inválida: {position}.")
    return offsets[line - 1] + column


def _absolute_ast_offset(
    text: str,
    offsets: Sequence[int],
    position: tuple[int, int],
) -> int:
    """Converte coluna AST (bytes UTF-8) em índice de caracteres do texto."""

    line, byte_column = position
    if line < 1 or line > len(offsets):
        raise ApplicatorError(f"Posição AST inválida: {position}.")
    line_start = offsets[line - 1]
    line_end = text.find("\n", line_start)
    if line_end < 0:
        line_end = len(text)
    line_text = text[line_start:line_end]
    encoded = line_text.encode("utf-8")
    if byte_column < 0 or byte_column > len(encoded):
        raise ApplicatorError(f"Coluna AST inválida: {position}.")
    try:
        prefix = encoded[:byte_column].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApplicatorError(
            f"Coluna AST não coincide com fronteira UTF-8: {position}."
        ) from exc
    return line_start + len(prefix)


def _module_all_string_nodes(tree: ast.AST) -> set[int]:
    result: set[int] = set()
    if not isinstance(tree, ast.Module):
        return result
    for statement in tree.body:
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
        ):
            value = statement.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    result.add(id(element))
    return result


def _is_globals_namespace_call(node: ast.AST) -> bool:
    """Retorna ``True`` apenas para uma chamada direta e vazia a ``globals()``."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "globals"
        and not node.args
        and not node.keywords
    )


def _ap003d_contract_string_nodes(
    relative: PurePosixPath,
    tree: ast.AST,
) -> set[int]:
    """Localiza somente os literais executáveis do contrato AP-003D.

    A autorização é deliberadamente estreita:
    - elementos das listas de módulo ``EXPECTED_HELPERS`` e ``EXPECTED_IMPLS``;
    - chaves do dicionário local ``implementation_aliases`` dentro de
      ``test_historical_helpers_are_thin_wrappers``.

    Nenhuma outra string do teste, comentário, snapshot ou manifesto é elegível.
    """

    if relative != AP003D_CONTRACT_TEST or not isinstance(tree, ast.Module):
        return set()

    result: set[int] = set()
    allowed_lists = {"EXPECTED_HELPERS", "EXPECTED_IMPLS"}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            target_names = {
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            if target_names & allowed_lists and isinstance(
                statement.value, (ast.List, ast.Tuple, ast.Set)
            ):
                for element in statement.value.elts:
                    if isinstance(element, ast.Constant) and isinstance(
                        element.value, str
                    ):
                        result.add(id(element))

        if not (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "test_historical_helpers_are_thin_wrappers"
        ):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name)
                and target.id == "implementation_aliases"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    result.add(id(key))

    return result


def _classified_literal_replacements(
    relative: PurePosixPath,
    text: str,
) -> tuple[
    list[tuple[int, int, str, str]],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    """Classifica e prepara apenas strings executáveis semanticamente acopladas.

    São autorizadas quatro superfícies:
    - entradas exatas de ``__all__``;
    - chaves exatas de ``runtime["nome"]``;
    - consultas exatas ``globals().get("nome")`` ou ``globals()["nome"]``;
    - literais executáveis estritamente delimitados do contrato AP-003D.

    A superfície ``globals`` é ainda limitada aos símbolos declarados em
    ``EXPECTED_GLOBALS_LITERAL_COUNTS``. Qualquer outra string exata igual a um
    nome antigo ou novo bloqueia a aplicação, evitando substituição textual cega.
    """

    tree = ast.parse(text, filename=str(relative))
    offsets = _line_offsets(text)
    parent_by_id: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent

    module_all_ids = _module_all_string_nodes(tree)
    ap003d_contract_ids = _ap003d_contract_string_nodes(relative, tree)
    replacements: list[tuple[int, int, str, str]] = []
    export_counts: Counter[str] = Counter()
    runtime_counts: Counter[str] = Counter()
    globals_counts: Counter[str] = Counter()
    ap003d_contract_counts: Counter[str] = Counter()
    unexpected: list[str] = []
    names = set(RENAMES) | set(RENAMES.values())
    allowed_globals = set(EXPECTED_GLOBALS_LITERAL_COUNTS) | {
        RENAMES[source] for source in EXPECTED_GLOBALS_LITERAL_COUNTS
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if value not in names:
            continue

        context: str | None = None
        if id(node) in module_all_ids:
            context = "__all__"
        elif id(node) in ap003d_contract_ids:
            context = "ap003d_contract"
        else:
            parent = parent_by_id.get(id(node))
            if (
                isinstance(parent, ast.Subscript)
                and parent.slice is node
                and isinstance(parent.value, ast.Name)
                and parent.value.id == "runtime"
            ):
                context = "runtime_subscript"
            elif (
                isinstance(parent, ast.Subscript)
                and parent.slice is node
                and _is_globals_namespace_call(parent.value)
                and value in allowed_globals
            ):
                context = "globals_lookup"
            elif (
                isinstance(parent, ast.Call)
                and parent.args
                and parent.args[0] is node
                and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "get"
                and _is_globals_namespace_call(parent.func.value)
                and value in allowed_globals
            ):
                context = "globals_lookup"

        if context is None:
            unexpected.append(
                f"{relative}:{getattr(node, 'lineno', 0)}: string exata não "
                f"classificada {value!r}"
            )
            continue

        if value not in RENAMES:
            # Destinos são aceitos somente na verificação de estado aplicado;
            # não há substituição a preparar.
            continue

        start = _absolute_ast_offset(
            text,
            offsets,
            (int(node.lineno), int(node.col_offset)),
        )
        end = _absolute_ast_offset(
            text,
            offsets,
            (int(node.end_lineno or node.lineno), int(node.end_col_offset or node.col_offset)),
        )
        segment = text[start:end]
        try:
            literal_value = ast.literal_eval(segment)
        except (SyntaxError, ValueError) as exc:
            raise ApplicatorError(
                f"Literal estrutural ilegível em {relative}:{node.lineno}: {segment!r}"
            ) from exc
        if literal_value != value or segment.count(value) != 1:
            raise ApplicatorError(
                f"Literal estrutural não pode ser reescrito de modo exato em "
                f"{relative}:{node.lineno}: {segment!r}."
            )
        new_segment = segment.replace(value, RENAMES[value], 1)
        if ast.literal_eval(new_segment) != RENAMES[value]:
            raise ApplicatorError(
                f"Reescrita literal inválida em {relative}:{node.lineno}."
            )
        replacements.append((start, end, segment, new_segment))
        if context == "__all__":
            export_counts[value] += 1
        elif context == "runtime_subscript":
            runtime_counts[value] += 1
        elif context == "globals_lookup":
            globals_counts[value] += 1
        else:
            ap003d_contract_counts[value] += 1

    if unexpected:
        raise ApplicatorError(
            "Strings exatas de símbolos fora das superfícies autorizadas:\n- "
            + "\n- ".join(sorted(unexpected))
        )
    return (
        replacements,
        export_counts,
        runtime_counts,
        globals_counts,
        ap003d_contract_counts,
    )



def _normalized_function_ast(source: str, expected_name: str) -> str:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == expected_name
    ]
    if len(functions) != 1:
        raise ApplicatorError(
            f"Fonte de contrato interna inválida para {expected_name}: {len(functions)} funções."
        )
    return ast.dump(functions[0], include_attributes=False, annotate_fields=True)


def _top_level_function_node(tree: ast.Module, name: str) -> ast.AST:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ApplicatorError(
            f"Contrato esperava exatamente uma função {name!r}; encontrou {len(matches)}."
        )
    return matches[0]


def _assignment_literal_node(
    tree: ast.Module,
    assignment_name: str,
    dictionary_key: str | None,
) -> ast.Constant:
    values: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment_name
            for target in statement.targets
        ):
            values.append(statement.value)
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == assignment_name
            and statement.value is not None
        ):
            values.append(statement.value)
    if len(values) != 1:
        raise ApplicatorError(
            f"Contrato esperava uma atribuição {assignment_name!r}; encontrou {len(values)}."
        )
    value = values[0]
    if dictionary_key is None:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ApplicatorError(
                f"Atribuição {assignment_name!r} não contém string literal."
            )
        return value
    if not isinstance(value, ast.Dict):
        raise ApplicatorError(
            f"Atribuição {assignment_name!r} não contém dicionário literal."
        )
    matches: list[ast.Constant] = []
    for key, item in zip(value.keys, value.values):
        if (
            isinstance(key, ast.Constant)
            and key.value == dictionary_key
            and isinstance(item, ast.Constant)
            and isinstance(item.value, str)
        ):
            matches.append(item)
    if len(matches) != 1:
        raise ApplicatorError(
            f"Dicionário {assignment_name!r} esperava uma chave {dictionary_key!r}; "
            f"encontrou {len(matches)}."
        )
    return matches[0]


def _replace_literal_node(
    text: str,
    node: ast.Constant,
    old: str,
    new: str,
) -> tuple[int, int, str, str]:
    offsets = _line_offsets(text)
    start = _absolute_ast_offset(
        text, offsets, (int(node.lineno), int(node.col_offset))
    )
    end = _absolute_ast_offset(
        text,
        offsets,
        (int(node.end_lineno or node.lineno), int(node.end_col_offset or node.col_offset)),
    )
    segment = text[start:end]
    try:
        current = ast.literal_eval(segment)
    except (SyntaxError, ValueError) as exc:
        raise ApplicatorError(f"Literal de contrato ilegível: {segment!r}.") from exc
    if current != old:
        raise ApplicatorError(
            f"Literal de contrato divergente: atual={current!r}, esperado={old!r}."
        )
    replacement = repr(new)
    if ast.literal_eval(replacement) != new:
        raise ApplicatorError("Falha interna ao serializar hash de contrato.")
    return start, end, segment, replacement


def _durable_contract_state(
    relative: PurePosixPath,
    data: bytes,
) -> str | None:
    if relative not in DURABLE_HASH_ASSIGNMENTS and relative not in DURABLE_FUNCTION_REPLACEMENTS:
        return None
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        text = data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise ApplicatorError(f"Codificação inválida em contrato {relative}: {exc}") from exc
    tree = ast.parse(text, filename=str(relative))
    observed: set[str] = set()
    for assignment_name, dictionary_key, old, new in DURABLE_HASH_ASSIGNMENTS.get(relative, ()):
        node = _assignment_literal_node(tree, assignment_name, dictionary_key)
        if node.value == old:
            observed.add("initial")
        elif node.value == new:
            observed.add("applied")
        else:
            raise ApplicatorError(
                f"Hash inesperado em {relative}:{assignment_name}[{dictionary_key!r}]: "
                f"{node.value!r}."
            )
    for name, initial_source, durable_source in DURABLE_FUNCTION_REPLACEMENTS.get(relative, ()):
        node = _top_level_function_node(tree, name)
        actual = ast.dump(node, include_attributes=False, annotate_fields=True)
        initial = _normalized_function_ast(initial_source, name)
        durable = _normalized_function_ast(durable_source, name)
        if actual == initial:
            observed.add("initial")
        elif actual == durable:
            observed.add("applied")
        else:
            raise ApplicatorError(
                f"Função de contrato {relative}:{name} divergiu das formas inicial e durável."
            )
    if not observed:
        raise ApplicatorError(f"Nenhuma superfície durável foi validada em {relative}.")
    if len(observed) != 1:
        raise ApplicatorError(
            f"Contrato parcialmente rebaselined em {relative}: estados={sorted(observed)}."
        )
    return next(iter(observed))


def _validate_durable_contract_state(
    corpus: Mapping[PurePosixPath, bytes],
    expected_state: str,
) -> None:
    paths = set(DURABLE_HASH_ASSIGNMENTS) | set(DURABLE_FUNCTION_REPLACEMENTS)
    diagnostics: list[str] = []
    for relative in sorted(paths, key=str):
        if relative not in corpus:
            diagnostics.append(f"ausente={relative}")
            continue
        state = _durable_contract_state(relative, corpus[relative])
        if state != expected_state:
            diagnostics.append(
                f"{relative}: estado={state!r}, esperado={expected_state!r}"
            )
    if diagnostics:
        raise ApplicatorError(
            "Contratos duráveis divergentes:\n- " + "\n- ".join(diagnostics)
        )


def _apply_durable_contract_updates(
    relative: PurePosixPath,
    data: bytes,
) -> bytes:
    if relative not in DURABLE_HASH_ASSIGNMENTS and relative not in DURABLE_FUNCTION_REPLACEMENTS:
        return data
    state = _durable_contract_state(relative, data)
    if state != "initial":
        raise ApplicatorError(
            f"Atualização durável recusou contrato não inicial: {relative} ({state})."
        )
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    text = data.decode(encoding)
    tree = ast.parse(text, filename=str(relative))
    replacements: list[tuple[int, int, str, str]] = []
    for assignment_name, dictionary_key, old, new in DURABLE_HASH_ASSIGNMENTS.get(relative, ()):
        node = _assignment_literal_node(tree, assignment_name, dictionary_key)
        replacements.append(_replace_literal_node(text, node, old, new))
    offsets = _line_offsets(text)
    for name, initial_source, durable_source in DURABLE_FUNCTION_REPLACEMENTS.get(relative, ()):
        node = _top_level_function_node(tree, name)
        actual = ast.dump(node, include_attributes=False, annotate_fields=True)
        if actual != _normalized_function_ast(initial_source, name):
            raise ApplicatorError(
                f"Função inicial não corresponde ao contrato aprovado: {relative}:{name}."
            )
        start = _absolute_ast_offset(
            text, offsets, (int(node.lineno), int(node.col_offset))
        )
        end = _absolute_ast_offset(
            text,
            offsets,
            (int(node.end_lineno or node.lineno), int(node.end_col_offset or node.col_offset)),
        )
        replacements.append((start, end, text[start:end], durable_source.rstrip("\n")))
    cursor = 0
    chunks: list[str] = []
    for start, end, old, new in sorted(replacements):
        if start < cursor or text[start:end] != old:
            raise ApplicatorError(
                f"Edição durável inconsistente em {relative}: {start}:{end}."
            )
        chunks.append(text[cursor:start])
        chunks.append(new)
        cursor = end
    chunks.append(text[cursor:])
    result = "".join(chunks).encode(encoding)
    if _durable_contract_state(relative, result) != "applied":
        raise ApplicatorError(
            f"Contrato {relative} não atingiu estado durável após transformação."
        )
    return result

def _token_name_index(
    corpus: Mapping[PurePosixPath, bytes],
    names: Iterable[str],
) -> dict[str, tuple[tuple[str, int, int], ...]]:
    wanted = set(names)
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for relative, data in corpus.items():
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
            text = data.decode(encoding)
            tokens = tokenize.generate_tokens(io.StringIO(text).readline)
            for token in tokens:
                if token.type == tokenize.NAME and token.string in wanted:
                    grouped[token.string].append(
                        (str(relative), token.start[0], token.start[1])
                    )
        except (SyntaxError, UnicodeDecodeError, LookupError, tokenize.TokenError, IndentationError) as exc:
            raise ApplicatorError(
                f"Tokenização falhou em {relative}: {exc}"
            ) from exc
    return {
        name: tuple(sorted(grouped.get(name, [])))
        for name in wanted
    }


def _validate_tokenized_state(
    corpus: Mapping[PurePosixPath, bytes],
    state: str,
) -> None:
    index = _token_name_index(corpus, [*RENAMES, *RENAMES.values()])
    diagnostics: list[str] = []
    for source, destination in RENAMES.items():
        old_items = index[source]
        new_items = index[destination]
        expected = EXPECTED_TOKEN_EDIT_COUNTS[source]
        if state == "initial":
            valid = len(old_items) == expected and not new_items
            items = old_items
        else:
            valid = not old_items and len(new_items) == expected
            items = new_items
        actual_paths = {path for path, _, _ in items}
        expected_paths = set(EXPECTED_TOKEN_PATHS_BY_SOURCE[source])
        if actual_paths != expected_paths:
            valid = False
        if not valid:
            diagnostics.append(
                f"{source}->{destination}: antigos={old_items}, novos={new_items}, "
                f"esperado={expected}, caminhos_esperados={sorted(expected_paths)}"
            )
    if diagnostics:
        raise ApplicatorError(
            f"Estado tokenizado {state!r} divergente:\n- "
            + "\n- ".join(diagnostics)
        )


def _literal_counts_for_state(
    relative: PurePosixPath,
    data: bytes,
) -> tuple[
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        text = data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise ApplicatorError(f"Codificação inválida em {relative}: {exc}") from exc

    tree = ast.parse(text, filename=str(relative))
    parent_by_id: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent
    module_all_ids = _module_all_string_nodes(tree)
    ap003d_contract_ids = _ap003d_contract_string_nodes(relative, tree)

    old_export: Counter[str] = Counter()
    new_export: Counter[str] = Counter()
    old_runtime: Counter[str] = Counter()
    new_runtime: Counter[str] = Counter()
    old_globals: Counter[str] = Counter()
    new_globals: Counter[str] = Counter()
    old_ap003d_contract: Counter[str] = Counter()
    new_ap003d_contract: Counter[str] = Counter()
    names = set(RENAMES) | set(RENAMES.values())
    allowed_globals = set(EXPECTED_GLOBALS_LITERAL_COUNTS) | {
        RENAMES[source] for source in EXPECTED_GLOBALS_LITERAL_COUNTS
    }
    unexpected: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if value not in names:
            continue
        context: str | None = None
        if id(node) in module_all_ids:
            context = "export"
        elif id(node) in ap003d_contract_ids:
            context = "ap003d_contract"
        else:
            parent = parent_by_id.get(id(node))
            if (
                isinstance(parent, ast.Subscript)
                and parent.slice is node
                and isinstance(parent.value, ast.Name)
                and parent.value.id == "runtime"
            ):
                context = "runtime"
            elif (
                isinstance(parent, ast.Subscript)
                and parent.slice is node
                and _is_globals_namespace_call(parent.value)
                and value in allowed_globals
            ):
                context = "globals"
            elif (
                isinstance(parent, ast.Call)
                and parent.args
                and parent.args[0] is node
                and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "get"
                and _is_globals_namespace_call(parent.func.value)
                and value in allowed_globals
            ):
                context = "globals"
        if context is None:
            unexpected.append(
                f"{relative}:{getattr(node, 'lineno', 0)}:{value}"
            )
            continue

        if value in RENAMES:
            if context == "export":
                old_export[value] += 1
            elif context == "runtime":
                old_runtime[value] += 1
            elif context == "globals":
                old_globals[value] += 1
            else:
                old_ap003d_contract[value] += 1
        else:
            sources = [source for source, destination in RENAMES.items() if destination == value]
            if len(sources) != 1:
                raise ApplicatorError(f"Destino literal ambíguo: {value}.")
            source = sources[0]
            if context == "export":
                new_export[source] += 1
            elif context == "runtime":
                new_runtime[source] += 1
            elif context == "globals":
                new_globals[source] += 1
            else:
                new_ap003d_contract[source] += 1

    if unexpected:
        raise ApplicatorError(
            "Strings exatas não classificadas na validação literal:\n- "
            + "\n- ".join(sorted(unexpected))
        )
    return (
        old_export,
        new_export,
        old_runtime,
        new_runtime,
        old_globals,
        new_globals,
        old_ap003d_contract,
        new_ap003d_contract,
    )


def _validate_literal_state(
    corpus: Mapping[PurePosixPath, bytes],
    state: str,
) -> None:
    totals = [Counter() for _ in range(8)]
    for relative in SYMBOL_TRANSFORMATION_PATHS:
        if relative not in corpus:
            raise ApplicatorError(
                f"Arquivo de transformação simbólica ausente da validação literal: {relative}."
            )
        observed = _literal_counts_for_state(relative, corpus[relative])
        for total, current in zip(totals, observed):
            total.update(current)

    (
        old_export,
        new_export,
        old_runtime,
        new_runtime,
        old_globals,
        new_globals,
        old_ap003d_contract,
        new_ap003d_contract,
    ) = totals
    diagnostics: list[str] = []
    all_sources = (
        set(EXPECTED_EXPORT_LITERAL_COUNTS)
        | set(EXPECTED_RUNTIME_LITERAL_COUNTS)
        | set(EXPECTED_GLOBALS_LITERAL_COUNTS)
        | set(EXPECTED_AP003D_CONTRACT_LITERAL_COUNTS)
    )
    for source in sorted(all_sources):
        expected_export = EXPECTED_EXPORT_LITERAL_COUNTS.get(source, 0)
        expected_runtime = EXPECTED_RUNTIME_LITERAL_COUNTS.get(source, 0)
        expected_globals = EXPECTED_GLOBALS_LITERAL_COUNTS.get(source, 0)
        expected_ap003d_contract = EXPECTED_AP003D_CONTRACT_LITERAL_COUNTS.get(
            source, 0
        )
        values = (
            old_export[source],
            new_export[source],
            old_runtime[source],
            new_runtime[source],
            old_globals[source],
            new_globals[source],
            old_ap003d_contract[source],
            new_ap003d_contract[source],
        )
        if state == "initial":
            expected_values = (
                expected_export,
                0,
                expected_runtime,
                0,
                expected_globals,
                0,
                expected_ap003d_contract,
                0,
            )
        else:
            expected_values = (
                0,
                expected_export,
                0,
                expected_runtime,
                0,
                expected_globals,
                0,
                expected_ap003d_contract,
            )
        if values != expected_values:
            diagnostics.append(
                f"{source}: observado={values}, esperado={expected_values} "
                "(old_export,new_export,old_runtime,new_runtime,"
                "old_globals,new_globals,old_ap003d_contract,"
                "new_ap003d_contract)"
            )
    extra_sources = (
        set(old_export)
        | set(new_export)
        | set(old_runtime)
        | set(new_runtime)
        | set(old_globals)
        | set(new_globals)
        | set(old_ap003d_contract)
        | set(new_ap003d_contract)
    ) - all_sources
    if extra_sources:
        diagnostics.append(f"fontes literais extras={sorted(extra_sources)}")
    if diagnostics:
        raise ApplicatorError(
            f"Estado de strings estruturais {state!r} divergente:\n- "
            + "\n- ".join(diagnostics)
        )


def _token_aware_rename(
    relative: PurePosixPath,
    data: bytes,
) -> tuple[
    bytes,
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        text = data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise ApplicatorError(f"Codificação inválida em {relative}: {exc}") from exc

    offsets = _line_offsets(text)
    replacements: list[tuple[int, int, str, str]] = []
    token_counts: Counter[str] = Counter()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            if token.type != tokenize.NAME or token.string not in RENAMES:
                continue
            start = _absolute_offset(offsets, token.start)
            end = _absolute_offset(offsets, token.end)
            replacements.append((start, end, token.string, RENAMES[token.string]))
            token_counts[token.string] += 1
    except (tokenize.TokenError, IndentationError) as exc:
        raise ApplicatorError(f"Tokenização falhou em {relative}: {exc}") from exc

    (
        literal_replacements,
        export_counts,
        runtime_counts,
        globals_counts,
        ap003d_contract_counts,
    ) = _classified_literal_replacements(relative, text)
    replacements.extend(literal_replacements)

    if not replacements:
        return (
            data,
            token_counts,
            export_counts,
            runtime_counts,
            globals_counts,
            ap003d_contract_counts,
        )

    chunks: list[str] = []
    cursor = 0
    for start, end, old, new in sorted(replacements):
        if start < cursor or text[start:end] != old:
            raise ApplicatorError(
                f"Edição estrutural inconsistente em {relative}: {old!r} @{start}:{end}."
            )
        chunks.append(text[cursor:start])
        chunks.append(new)
        cursor = end
    chunks.append(text[cursor:])
    transformed = "".join(chunks).encode(encoding)
    return (
        transformed,
        token_counts,
        export_counts,
        runtime_counts,
        globals_counts,
        ap003d_contract_counts,
    )


def _prepare_transformed_files(
    repo_root: Path,
    state: str,
    corpus: Mapping[PurePosixPath, bytes],
) -> tuple[dict[PurePosixPath, bytes], dict[str, int]]:
    del repo_root  # reservado para futuras validações sem ampliar o escopo
    originals = {path: corpus[path] for path in PRODUCTIVE_PATHS}
    if state == "applied":
        return dict(originals), {
            source: (
                EXPECTED_TOKEN_EDIT_COUNTS[source]
                + EXPECTED_EXPORT_LITERAL_COUNTS.get(source, 0)
                + EXPECTED_RUNTIME_LITERAL_COUNTS.get(source, 0)
                + EXPECTED_GLOBALS_LITERAL_COUNTS.get(source, 0)
                + EXPECTED_AP003D_CONTRACT_LITERAL_COUNTS.get(source, 0)
            )
            for source in RENAMES
        }

    transformed: dict[PurePosixPath, bytes] = {}
    total_tokens: Counter[str] = Counter()
    total_exports: Counter[str] = Counter()
    total_runtime: Counter[str] = Counter()
    total_globals: Counter[str] = Counter()
    total_ap003d_contract: Counter[str] = Counter()
    for relative in PRODUCTIVE_PATHS:
        if relative in SYMBOL_TRANSFORMATION_PATHS:
            (
                updated,
                token_counts,
                export_counts,
                runtime_counts,
                globals_counts,
                ap003d_contract_counts,
            ) = _token_aware_rename(relative, originals[relative])
        else:
            updated = originals[relative]
            token_counts = Counter()
            export_counts = Counter()
            runtime_counts = Counter()
            globals_counts = Counter()
            ap003d_contract_counts = Counter()
        updated = _apply_durable_contract_updates(relative, updated)
        transformed[relative] = updated
        total_tokens.update(token_counts)
        total_exports.update(export_counts)
        total_runtime.update(runtime_counts)
        total_globals.update(globals_counts)
        total_ap003d_contract.update(ap003d_contract_counts)

    diagnostics: list[str] = []
    for source in RENAMES:
        observed = (
            total_tokens[source],
            total_exports[source],
            total_runtime[source],
            total_globals[source],
            total_ap003d_contract[source],
        )
        expected = (
            EXPECTED_TOKEN_EDIT_COUNTS[source],
            EXPECTED_EXPORT_LITERAL_COUNTS.get(source, 0),
            EXPECTED_RUNTIME_LITERAL_COUNTS.get(source, 0),
            EXPECTED_GLOBALS_LITERAL_COUNTS.get(source, 0),
            EXPECTED_AP003D_CONTRACT_LITERAL_COUNTS.get(source, 0),
        )
        if observed != expected:
            diagnostics.append(
                f"{source}: observado={observed}, esperado={expected} "
                "(tokens,__all__,runtime,globals,ap003d_contract)"
            )
    extras = (
        set(total_tokens)
        | set(total_exports)
        | set(total_runtime)
        | set(total_globals)
        | set(total_ap003d_contract)
    ) - set(RENAMES)
    if extras:
        diagnostics.append(f"mapeamentos extras={sorted(extras)}")
    if diagnostics:
        raise ApplicatorError(
            "Pré-validação encontrou divergências em todas as ondas estruturais:\n- "
            + "\n- ".join(diagnostics)
        )

    unchanged = [
        str(path) for path in PRODUCTIVE_PATHS if originals[path] == transformed[path]
    ]
    if unchanged:
        raise ApplicatorError(
            "Arquivos produtivos aprovados sem alteração, contrariando o inventário:\n- "
            + "\n- ".join(unchanged)
        )
    edit_counts = {
        source: (
            total_tokens[source]
            + total_exports[source]
            + total_runtime[source]
            + total_globals[source]
            + total_ap003d_contract[source]
        )
        for source in RENAMES
    }
    return transformed, dict(sorted(edit_counts.items()))


def _validate_transformed_corpus(
    *,
    tracked_python: tuple[PurePosixPath, ...],
    corpus: Mapping[PurePosixPath, bytes],
    transformed: Mapping[PurePosixPath, bytes],
    protected_before: tuple[tuple[Any, ...], ...],
) -> None:
    all_occurrences: list[Occurrence] = []
    mixed_corpus: dict[PurePosixPath, bytes] = {}
    for relative in tracked_python:
        data = transformed.get(relative, corpus[relative])
        mixed_corpus[relative] = data
        all_occurrences.extend(_collect_occurrences_for_bytes(relative, data))

    old_index = _occurrence_index(all_occurrences, RENAMES)
    new_index = _occurrence_index(all_occurrences, RENAMES.values())
    for source, destination in RENAMES.items():
        if old_index[source]:
            raise ApplicatorError(
                f"Identificador antigo permaneceu após transformação: {source}."
            )
        expected = EXPECTED_AST_OCCURRENCE_COUNTS[source]
        if len(new_index[destination]) != expected:
            raise ApplicatorError(
                f"Contagem AST pós-transformação inválida para {destination}: "
                f"{len(new_index[destination])}; esperada={expected}."
            )
        actual_paths = {item.path for item in new_index[destination]}
        if actual_paths != set(EXPECTED_PATHS_BY_SOURCE[source]):
            raise ApplicatorError(
                f"Destino {destination} apareceu fora do escopo AST aprovado: "
                f"{sorted(actual_paths)}."
            )

    _validate_tokenized_state(mixed_corpus, "applied")
    _validate_literal_state(mixed_corpus, "applied")
    _validate_durable_contract_state(mixed_corpus, "applied")

    protected_after = _protected_snapshot(all_occurrences)
    if protected_after != protected_before:
        raise ApplicatorError(
            "Snapshot AST de símbolos protegidos mudou; transformação recusada."
        )

def _prevalidate(
    repo_root: Path,
    *,
    fetch: bool,
) -> PreparedApplication:
    baseline = _validate_git_baseline(repo_root, fetch=fetch)
    inventory = _load_and_validate_inventory(repo_root)
    tracked_python = _git_tracked_python_files(repo_root)
    corpus, occurrences = _read_python_corpus(repo_root, tracked_python)
    state = _validate_initial_or_applied_state(occurrences)
    _validate_tokenized_state(corpus, state)
    _validate_literal_state(corpus, state)
    _validate_durable_contract_state(corpus, state)
    productive_status = set(baseline.status_paths) & PRODUCTIVE_PATH_SET
    if state == "initial" and productive_status:
        raise ApplicatorError(
            "A árvore ainda contém identificadores antigos, mas já possui alterações "
            "produtivas. A AP-004D recusou misturar mudanças prévias:\n- "
            + "\n- ".join(sorted(productive_status))
        )
    if state == "applied":
        changed = {
            line.strip()
            for line in _run(
                ["git", "diff", "--name-only"], cwd=repo_root
            ).stdout.splitlines()
            if line.strip()
        }
        if changed != PRODUCTIVE_PATH_SET:
            raise ApplicatorError(
                "Estado aplicado incompleto: o diff produtivo não contém exatamente "
                f"os onze arquivos autorizados. encontrado={sorted(changed)}"
            )
        canonical = repo_root / APPLICATOR_RELATIVE
        self_data = Path(__file__).resolve().read_bytes()
        if not canonical.is_file() or canonical.read_bytes() != self_data:
            raise ApplicatorError(
                "Estado aplicado sem a cópia canônica idêntica do aplicador."
            )
    protected_before = _protected_snapshot(occurrences)
    if not protected_before:
        raise ApplicatorError("Nenhum contrato protegido foi localizado na árvore.")
    frozen_hashes = _find_frozen_files(repo_root)

    missing_productive = [
        str(path) for path in PRODUCTIVE_PATHS if path not in corpus
    ]
    if missing_productive:
        raise ApplicatorError(
            "Arquivos produtivos aprovados não rastreados ou ausentes:\n- "
            + "\n- ".join(missing_productive)
        )

    transformed, edit_counts = _prepare_transformed_files(repo_root, state, corpus)
    _validate_transformed_corpus(
        tracked_python=tracked_python,
        corpus=corpus,
        transformed=transformed,
        protected_before=protected_before,
    )

    # Compilação em memória de todos os arquivos transformados antes da primeira escrita.
    for relative, data in transformed.items():
        tree = _parse_python(data, relative)
        compile(tree, str(relative), "exec")

    # A cópia canônica do próprio aplicador também é validada antes da transação.
    self_data = Path(__file__).resolve().read_bytes()
    try:
        compile(self_data.decode("utf-8"), str(APPLICATOR_RELATIVE), "exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ApplicatorError(f"Aplicador fornecido é inválido: {exc}") from exc

    return PreparedApplication(
        state=state,
        baseline=baseline,
        inventory=inventory,
        tracked_python=tracked_python,
        original_bytes={path: corpus[path] for path in PRODUCTIVE_PATHS},
        transformed_bytes=transformed,
        occurrences_before=occurrences,
        protected_snapshot=protected_before,
        frozen_hashes=frozen_hashes,
        edit_counts=edit_counts,
    )


def _backup_root(head: str) -> Path:
    base = Path.home() / ".cache" / "mppg-refactor" / "ap004d" / "backups"
    backup = base / f"{head[:12]}-{uuid.uuid4().hex}"
    backup.mkdir(parents=True, exist_ok=False)
    return backup


def _create_backup(
    repo_root: Path,
    backup_dir: Path,
    paths: Iterable[PurePosixPath],
) -> tuple[BackupEntry, ...]:
    entries: list[BackupEntry] = []
    files_dir = backup_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=False)
    for relative in paths:
        source = repo_root / relative
        if source.exists() and not source.is_file():
            raise ApplicatorError(f"Caminho de saída não é arquivo: {relative}")
        if source.is_file():
            data = source.read_bytes()
            mode = stat.S_IMODE(source.stat().st_mode)
            destination = files_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            os.chmod(destination, mode)
            entries.append(
                BackupEntry(
                    relative=str(relative),
                    existed=True,
                    mode=mode,
                    sha256=_sha256_bytes(data),
                )
            )
        else:
            entries.append(
                BackupEntry(
                    relative=str(relative),
                    existed=False,
                    mode=None,
                    sha256=None,
                )
            )
    manifest = {
        "phase": "AP-004D",
        "head": EXPECTED_HEAD,
        "inventory_sha256": EXPECTED_INVENTORY_DIGEST,
        "entries": [dataclasses.asdict(entry) for entry in entries],
    }
    (backup_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tuple(entries)


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _rollback(
    repo_root: Path,
    backup_dir: Path,
    entries: Iterable[BackupEntry],
) -> None:
    files_dir = backup_dir / "files"
    errors: list[str] = []
    for entry in entries:
        relative = PurePosixPath(entry.relative)
        destination = repo_root / relative
        try:
            if entry.existed:
                backup_file = files_dir / relative
                data = backup_file.read_bytes()
                if _sha256_bytes(data) != entry.sha256:
                    raise ApplicatorError(f"Backup corrompido para {relative}.")
                _atomic_write(destination, data, entry.mode or 0o644)
            elif destination.exists():
                if not destination.is_file():
                    raise ApplicatorError(
                        f"Rollback recusou remover caminho não arquivo: {relative}."
                    )
                destination.unlink()
        except Exception as exc:  # pragma: no cover - caminho de emergência
            errors.append(f"{relative}: {exc}")
    if errors:
        raise ApplicatorError(
            "Rollback incompleto; intervenção manual necessária:\n- "
            + "\n- ".join(errors)
        )


def _validate_frozen_hashes(repo_root: Path, expected: Mapping[str, str]) -> None:
    current = {
        path: _sha256_bytes((repo_root / path).read_bytes()) for path in expected
    }
    if current != dict(expected):
        raise ApplicatorError("Arquivo fulltext congelado foi alterado.")


def _post_write_structural_validation(
    repo_root: Path,
    prepared: PreparedApplication,
) -> None:
    corpus, occurrences = _read_python_corpus(repo_root, prepared.tracked_python)
    state = _validate_initial_or_applied_state(occurrences)
    if state != "applied":
        raise ApplicatorError("A árvore não atingiu o estado aplicado integral.")
    if _protected_snapshot(occurrences) != prepared.protected_snapshot:
        raise ApplicatorError("Contrato protegido mudou após a escrita.")
    _validate_frozen_hashes(repo_root, prepared.frozen_hashes)

    for relative in PRODUCTIVE_PATHS:
        expected = prepared.transformed_bytes[relative]
        actual = corpus[relative]
        if actual != expected:
            raise ApplicatorError(
                f"Conteúdo persistido diverge da pré-validação: {relative}."
            )


def _run_py_compile(repo_root: Path, backup_dir: Path) -> None:
    software_root = repo_root / SOFTWARE_RELATIVE
    pycache = backup_dir / "pycache"
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = str(pycache)
    relative_to_software = [
        str(path.relative_to(SOFTWARE_RELATIVE)) for path in PRODUCTIVE_PATHS
    ]
    _run(
        ["pipenv", "run", "python", "-m", "py_compile", *relative_to_software],
        cwd=software_root,
        env=env,
    )


def _run_diff_checks(repo_root: Path) -> None:
    _run(["git", "diff", "--check"], cwd=repo_root)
    changed = {
        line.strip()
        for line in _run(["git", "diff", "--name-only"], cwd=repo_root).stdout.splitlines()
        if line.strip()
    }
    if changed != PRODUCTIVE_PATH_SET:
        raise ApplicatorError(
            "Conjunto de arquivos rastreados alterados diverge do aprovado: "
            f"encontrado={sorted(changed)}, esperado={sorted(PRODUCTIVE_PATH_SET)}."
        )
    status_paths = set(_parse_status(repo_root))
    unexpected = sorted(status_paths - ALLOWED_STATUS_PATHS)
    if unexpected:
        raise ApplicatorError(
            "Testes ou aplicação criaram caminhos inesperados:\n- "
            + "\n- ".join(unexpected)
        )


def _run_specific_tests(repo_root: Path) -> None:
    software_root = repo_root / SOFTWARE_RELATIVE
    _run(
        ["pipenv", "run", "pytest", "-q", "-ra", *SPECIFIC_TESTS],
        cwd=software_root,
    )


def _run_full_suite(repo_root: Path) -> None:
    software_root = repo_root / SOFTWARE_RELATIVE
    completed = _run(
        [
            "pipenv",
            "run",
            "pytest",
            "-q",
            "-ra",
            "app_bundle/tests",
            "tests",
        ],
        cwd=software_root,
    )
    combined = completed.stdout + "\n" + completed.stderr
    summaries = re.findall(
        r"\b(?P<passed>\d+) passed\b[^\n]*?\b(?P<xfailed>\d+) xfailed\b",
        combined,
    )
    if not summaries:
        raise ApplicatorError(
            "Resumo consolidado da suíte não foi reconhecido. Saída:\n"
            + combined[-4000:]
        )
    passed, xfailed = map(int, summaries[-1])
    if passed != EXPECTED_FULL_SUITE_PASSED or xfailed != EXPECTED_FULL_SUITE_XFAILED:
        raise ApplicatorError(
            "Totais consolidados inesperados: "
            f"passed={passed}, xfailed={xfailed}; esperados="
            f"{EXPECTED_FULL_SUITE_PASSED} passed, "
            f"{EXPECTED_FULL_SUITE_XFAILED} xfailed. Saída:\n"
            + combined[-4000:]
        )
    if re.search(r"\b\d+ xpassed\b", combined):
        raise ApplicatorError(
            "A suíte apresentou xpass; um defeito histórico pode ter sido alterado. "
            "Saída:\n" + combined[-4000:]
        )


def _install_and_apply(prepared: PreparedApplication) -> Path:
    repo_root = prepared.baseline.repo_root
    if prepared.state == "applied":
        validation_dir = (
            Path.home()
            / ".cache"
            / "mppg-refactor"
            / "ap004d"
            / "validations"
            / f"{prepared.baseline.head[:12]}-{uuid.uuid4().hex}"
        )
        validation_dir.mkdir(parents=True, exist_ok=False)
        _post_write_structural_validation(repo_root, prepared)
        _run_py_compile(repo_root, validation_dir)
        _run_diff_checks(repo_root)
        _run_specific_tests(repo_root)
        _run_diff_checks(repo_root)
        _run_full_suite(repo_root)
        _run_diff_checks(repo_root)
        _validate_frozen_hashes(repo_root, prepared.frozen_hashes)
        print("[OK] A AP-004D já estava aplicada; revalidação integral concluída.")
        return validation_dir

    self_data = Path(__file__).resolve().read_bytes()
    transaction_paths = (*PRODUCTIVE_PATHS, APPLICATOR_RELATIVE)
    backup_dir = _backup_root(prepared.baseline.head)
    entries = _create_backup(repo_root, backup_dir, transaction_paths)
    try:
        for relative in PRODUCTIVE_PATHS:
            current_mode = stat.S_IMODE((repo_root / relative).stat().st_mode)
            _atomic_write(
                repo_root / relative,
                prepared.transformed_bytes[relative],
                current_mode,
            )
        _atomic_write(repo_root / APPLICATOR_RELATIVE, self_data, 0o755)

        _post_write_structural_validation(repo_root, prepared)
        _run_py_compile(repo_root, backup_dir)
        _run_diff_checks(repo_root)
        _run_specific_tests(repo_root)
        _run_diff_checks(repo_root)
        _run_full_suite(repo_root)
        _run_diff_checks(repo_root)
        _validate_frozen_hashes(repo_root, prepared.frozen_hashes)
    except Exception as original_exc:
        try:
            _rollback(repo_root, backup_dir, entries)
        except Exception as rollback_exc:
            raise ApplicatorError(
                f"Falha na AP-004D: {original_exc}\n"
                f"Falha adicional no rollback: {rollback_exc}\n"
                f"Backup externo: {backup_dir}"
            ) from original_exc
        raise ApplicatorError(
            f"Falha na AP-004D; rollback integral aplicado. Motivo: {original_exc}\n"
            f"Backup externo preservado: {backup_dir}"
        ) from original_exc
    return backup_dir


def _print_plan(prepared: PreparedApplication) -> None:
    print("[OK] Pré-validação integral concluída sem escrita.")
    print(f"[OK] Estado detectado: {prepared.state}")
    print(f"[OK] Branch: {prepared.baseline.branch}")
    print(f"[OK] HEAD local/remoto: {prepared.baseline.head}")
    print(f"[OK] Inventário: {EXPECTED_INVENTORY_DIGEST}")
    print(f"[OK] Transformações lógicas: {len(RENAMES)}")
    print("[OK] Importações aliased, __all__, chaves runtime, consultas globals e seis contratos duráveis validados estruturalmente.")
    print(f"[OK] Registros candidatos validados: {EXPECTED_CANDIDATE_RECORDS}")
    print(f"[OK] Fontes Python canônicas analisadas: {len(prepared.tracked_python)}")
    print("[OK] Diretórios de backup, output, cache e ambientes foram excluídos antes da leitura.")
    print("[OK] Colisões: 0")
    print("[OK] Arquivos autorizados (cinco produtivos + seis contratos duráveis):")
    for relative in PRODUCTIVE_PATHS:
        print(f"  - {relative}")
    print("[OK] Overrides aprovados:")
    for source, (inventoried, approved) in DESTINATION_OVERRIDES.items():
        print(f"  - {source}: {inventoried} -> {approved}")
    print("[BLOQUEIO] Nenhuma escrita ocorreu. Use --apply somente após autorização operacional.")


def _print_success(prepared: PreparedApplication, backup_dir: Path) -> None:
    print("[OK] Aplicação produtiva AP-004D concluída atomicamente.")
    print(f"[OK] Branch: {prepared.baseline.branch}")
    print(f"[OK] Baseline preservada no HEAD: {prepared.baseline.head}")
    print(f"[OK] Inventário aprovado: {EXPECTED_INVENTORY_DIGEST}")
    print(f"[OK] Transformações lógicas aplicadas: {len(RENAMES)}")
    print(f"[OK] Backup externo: {backup_dir}")
    print("[OK] Arquivos alterados (cinco produtivos + seis contratos duráveis):")
    for relative in PRODUCTIVE_PATHS:
        print(f"  - {relative}")
    print(f"[OK] Aplicador canônico instalado: {APPLICATOR_RELATIVE}")
    print("[OK] py_compile, git diff --check e testes específicos aprovados.")
    print(f"[OK] Suíte canônica: {EXPECTED_FULL_SUITE_PASSED} passed, {EXPECTED_FULL_SUITE_XFAILED} xfailed.")
    print("[BLOQUEIO] Nenhum commit foi criado e nenhum push foi realizado.")
    print("[BLOQUEIO] Revise o diff e aguarde aprovação expressa para consolidação Git.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pré-valida ou aplica atomicamente as 16 renomeações AST, tokenizadas "
            "e seus metadados executáveis aprovados para a AP-004D."
        )
    )
    parser.add_argument(
        "--repo-root",
        help="Raiz do repositório Git; por padrão, detectada a partir do diretório atual.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Autoriza escrita produtiva, testes e instalação da cópia canônica do aplicador.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help=(
            "Não executar git fetch origin; ainda exige a referência remota local idêntica. "
            "Uso excepcional/offline."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        repo_root = _discover_repo_root(args.repo_root)
        prepared = _prevalidate(repo_root, fetch=not args.no_fetch)
        if not args.apply:
            _print_plan(prepared)
            return 0
        backup_dir = _install_and_apply(prepared)
        _print_success(prepared, backup_dir)
        return 0
    except ApplicatorError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        print(
            "[BLOQUEIO] Nenhum commit ou push deve ser realizado para a AP-004D.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("[ERRO] Execução interrompida pelo usuário.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
