from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
PRISMA = ROOT / "academic_pipeline/prisma_generic_orchestration.py"
ARTICLE = ROOT / "app_bundle/scripts/pipeline/artigo_prisma_workflow.py"
TUI = ROOT / "app_bundle/scripts/pipeline/academic_pipeline_tui.py"
FREEZE = ROOT / "app_bundle/scripts/pipeline/prisma_congelar_artigo.py"
EXAMPLES = ROOT / "app_bundle/config/examples"

EXPECTED_DEPENDENCIES = [
    "openai>=1.0.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
    "pypdf>=4.0",
    "python-docx>=1.1",
    "openpyxl>=3.1",
]


def test_pep621_dependencies_and_package_data_are_explicit() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["name"] == "academic-pipeline-mppg"
    assert data["project"]["version"] == "0.1.0"
    assert data["project"]["dependencies"] == EXPECTED_DEPENDENCIES
    package_data = data["tool"]["setuptools"]["package-data"]["app_bundle"]
    assert len(package_data) == 21
    assert all("projetos" not in item for item in package_data)
    assert all("output" not in item for item in package_data)


def test_examples_exist_and_are_valid_toml() -> None:
    expected = {
        "atividade_rc10_exemplo.toml",
        "paper_rc10_exemplo.toml",
        "relatorio_prisma_rc10_exemplo.toml",
    }
    assert {path.name for path in EXAMPLES.glob("*.toml")} == expected
    for name in expected:
        payload = tomllib.loads((EXAMPLES / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload


def test_article_workflow_consumers_use_canonical_package_import() -> None:
    article = ARTICLE.read_text(encoding="utf-8")
    tui = TUI.read_text(encoding="utf-8")
    expected = "from app_bundle.scripts.pipeline.article_workflow import ArticleWorkflow"
    assert expected in article
    assert expected in tui


def test_prisma_subprocesses_use_installed_modules() -> None:
    source = PRISMA.read_text(encoding="utf-8")
    assert "'-m', 'academic_pipeline'" in source
    assert "'app_bundle.scripts.pipeline.prisma_exportar_bib'" in source
    assert "'app_bundle.scripts.pipeline.prisma_congelar_artigo'" in source
    assert "Path(__file__).with_name('prisma_exportar_bib.py')" not in source
    assert "Path(__file__).with_name('prisma_congelar_artigo.py')" not in source
    assert "[sys.executable, __file__" not in source


def test_productive_code_has_no_personal_home_path() -> None:
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "academic_pipeline",
            "app_bundle/scripts/pipeline",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
    )
    offenders = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if relative.suffix != ".py":
            continue
        path = ROOT / relative
        if "/home/gustavodetarso" in path.read_text(encoding="utf-8"):
            offenders.append(relative.as_posix())
    assert offenders == []


def test_freeze_helper_defaults_to_public_module_entrypoint() -> None:
    source = FREEZE.read_text(encoding="utf-8")
    assert "PACKAGE_ROOT = Path(__file__).resolve().parents[3]" in source
    assert '"academic_pipeline"' in source
    assert 'DEFAULT_ARTIGO_DIR = Path.cwd() / "artigo"' in source


def test_init_project_supports_external_destination(tmp_path: Path) -> None:
    pipeline_dir = ROOT / "app_bundle/scripts/pipeline"
    sys.path.insert(0, str(pipeline_dir))
    try:
        module = importlib.import_module("project_tools")
        result = module.init_project(
            "ap005e3_contract",
            project_type="atividade",
            base_dir=tmp_path,
            institution="fgv",
        )
    finally:
        sys.path.remove(str(pipeline_dir))
        sys.modules.pop("project_tools", None)

    expected = tmp_path / "app_bundle" / "projetos" / "ap005e3_contract"
    assert result.project_dir == expected
    assert result.config_path.is_file()
    assert result.documentos_zip_path.is_file()
    assert result.orientacoes_zip_path.is_file()
    assert result.doi_manifest_path.is_file()
    assert result.readme_path.is_file()
    payload = tomllib.loads(result.config_path.read_text(encoding="utf-8"))
    assert payload["instituicao"]["perfil"] == "fgv"
    readme = result.readme_path.read_text(encoding="utf-8")
    assert "/home/gustavodetarso" not in readme
    assert "academic-pipeline" in readme
