#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def classify(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    writes_export_el = 0
    export_sites = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node.func)
        expression = ast.unparse(node)
        if (
            name.endswith("write_text")
            or name.endswith("write_bytes")
            or name in {"open", "Path.open"}
        ) and "_export_pdf.el" in expression:
            writes_export_el += 1
        if (
            "_export_pdf.el" in expression
            or "org-latex-export-to-pdf" in expression
        ):
            export_sites += 1

    if writes_export_el:
        return "direct_export_el_writer"
    if export_sites:
        return "pdf_export_executor_not_el_writer"
    return "not_export_el_related"


def validate(repo: Path, manifest: Path) -> dict[str, object]:
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["phase"] == "AP-006D.4B"
    assert data["decision"] == (
        "preserve_current_generated_artifacts_with_documented_provenance"
    )
    assert data["summary"]["preserved_artifact_count"] == 7
    assert data["summary"]["current_old_reference_line_count"] == 9
    assert data["summary"]["historical_old_reference_line_count"] == 2

    records = [
        *data["artifacts"]["current"],
        data["artifacts"]["historical"],
    ]
    for record in records:
        path = repo / record["path"]
        assert path.is_file(), record["path"]
        assert sha256(path) == record["sha256"], record["path"]

    bridge = repo / data["bridge"]["path"]
    assert bridge.is_symlink()
    assert bridge.readlink().as_posix() == data["bridge"]["expected_symlink_target"]

    generator = repo / data["generator"]["path"]
    observed = classify(generator)
    assert observed == "pdf_export_executor_not_el_writer"
    assert observed == data["generator"]["classification"]

    return {
        "status": "ok",
        "artifact_count": len(records),
        "generator_classification": observed,
        "fingerprint_sha256": data["fingerprint_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        validate(args.repo.resolve(), args.manifest.resolve()),
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
