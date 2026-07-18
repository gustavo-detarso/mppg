from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT_ENV = "ACADEMIC_PIPELINE_PROJECT_ROOT"


class RepositoryRootError(RuntimeError):
    pass


def _is_project_root(candidate: Path) -> bool:
    return (
        candidate.is_dir()
        and (candidate / "pyproject.toml").is_file()
        and (candidate / "academic_pipeline").is_dir()
        and (candidate / "app_bundle").is_dir()
    )


def _candidate_chain(start: Path) -> Iterable[Path]:
    current = start if start.is_dir() else start.parent
    yield current
    yield from current.parents


def repository_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    override = os.environ.get(PROJECT_ROOT_ENV)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not _is_project_root(candidate):
            raise RepositoryRootError(
                f"{PROJECT_ROOT_ENV} does not point to a valid project root: "
                f"{candidate}"
            )
        return candidate

    anchor = Path(start) if start is not None else Path(__file__)
    anchor = anchor.expanduser()
    if not anchor.is_absolute():
        anchor = Path.cwd() / anchor
    anchor = anchor.resolve()

    for candidate in _candidate_chain(anchor):
        if _is_project_root(candidate):
            return candidate

    raise RepositoryRootError(
        "Could not resolve an Academic Pipeline repository project root. "
        f"Set {PROJECT_ROOT_ENV} explicitly for repository-backed resources."
    )


def repository_resource(
    *parts: str,
    start: str | os.PathLike[str] | None = None,
    must_exist: bool = True,
) -> Path:
    root = repository_project_root(start=start)
    candidate = root.joinpath(*parts).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RepositoryRootError(
            f"Repository resource escapes the project root: {candidate}"
        ) from exc

    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)

    return candidate
