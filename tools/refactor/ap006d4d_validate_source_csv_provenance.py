#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_KEY = "source_csv"
READ_METHODS = {"open", "read_text", "read_bytes", "read_csv"}
EXISTENCE_METHODS = {"exists", "is_file", "is_dir"}
CONTAINER_METHODS = {"append", "extend", "insert", "update", "add"}
SERIALIZER_LEAVES = {"write_csv", "writerow", "writerows", "json.dumps"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def segment(text: str, node: ast.AST) -> str:
    value = ast.get_source_segment(text, node)
    if value is not None:
        return value
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def call_name(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:
        return ""


def loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
    }


def assigned_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    if isinstance(node, ast.Name):
        result.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for child in node.elts:
            result.update(assigned_names(child))
    return result


def contains_key(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant) and child.value == SOURCE_KEY
        for child in ast.walk(node)
    )


def method_receiver(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return call.func.value.id
    return None


def analyze_target(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    main = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )

    tainted_scalars: set[str] = set()
    tainted_containers: set[str] = set()

    changed = True
    while changed:
        changed = False
        for node in ast.walk(main):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = set()
                value = None
                if isinstance(node, ast.Assign):
                    for target_node in node.targets:
                        targets.update(assigned_names(target_node))
                    value = node.value
                else:
                    targets.update(assigned_names(node.target))
                    value = node.value
                if not targets or value is None:
                    continue
                dependencies = loaded_names(value)
                if contains_key(value) or bool(
                    dependencies & (tainted_scalars | tainted_containers)
                ):
                    for name in targets:
                        if name not in tainted_scalars:
                            tainted_scalars.add(name)
                            changed = True

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = method_receiver(node)
                if receiver is None:
                    continue
                leaf = node.func.attr
                arguments = set()
                for argument in node.args:
                    arguments.update(loaded_names(argument))
                for keyword in node.keywords:
                    arguments.update(loaded_names(keyword.value))
                if leaf in CONTAINER_METHODS and (
                    arguments & (tainted_scalars | tainted_containers)
                ):
                    if receiver not in tainted_containers:
                        tainted_containers.add(receiver)
                        changed = True

    read_sink_count = 0
    existence_sink_count = 0
    serialization_sink_count = 0
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        leaf = name.rsplit(".", 1)[-1]
        receiver = method_receiver(node)
        receiver_tainted = (
            receiver in tainted_scalars or receiver in tainted_containers
            if receiver is not None
            else False
        )
        arguments = set()
        for argument in node.args:
            arguments.update(loaded_names(argument))
        for keyword in node.keywords:
            arguments.update(loaded_names(keyword.value))
        tainted_arguments = arguments & (tainted_scalars | tainted_containers)
        if not (receiver_tainted or tainted_arguments or contains_key(node)):
            continue
        if leaf in READ_METHODS:
            read_sink_count += 1
        elif leaf in EXISTENCE_METHODS:
            existence_sink_count += 1
        elif leaf in SERIALIZER_LEAVES or name in SERIALIZER_LEAVES:
            serialization_sink_count += 1

    return {
        "module_sha256": sha256_path(path),
        "read_sink_count": read_sink_count,
        "existence_sink_count": existence_sink_count,
        "serialization_sink_count": serialization_sink_count,
        "tainted_scalars": sorted(tainted_scalars),
        "tainted_containers": sorted(tainted_containers),
    }


def csv_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    try:
        delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    values = [(row.get(SOURCE_KEY) or "") for row in rows]
    return {
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "delimiter": delimiter,
        "row_count": len(rows),
        "nonempty_source_csv_count": sum(bool(value) for value in values),
        "unique_source_csv_values": sorted(set(values)),
    }


def validate(repo: Path, manifest: Path) -> dict[str, object]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4D"
    assert data["status"] == "source_csv_provenance_preservation_materialized"
    assert data["decision"] == (
        "preserve_source_csv_verbatim_as_historical_provenance"
    )
    assert data["summary"]["pair_count"] == 4
    assert data["summary"]["total_row_count"] == 308
    assert data["summary"]["unique_source_csv_value_count"] == 1
    assert data["constraints"]["rewrite_existing_csv_forbidden"] is True
    assert data["constraints"][
        "runtime_path_resolver_forbidden_without_new_evidence"
    ] is True

    for record in data["pairs"]:
        source = repo / record["source_path"]
        cache = repo / record["cache_path"]
        assert source.is_file() and cache.is_file()
        assert source.read_bytes() == cache.read_bytes()
        observed = csv_record(source)
        for key in (
            "sha256",
            "size_bytes",
            "delimiter",
            "row_count",
            "nonempty_source_csv_count",
            "unique_source_csv_values",
        ):
            assert observed[key] == record[key], (record["source_path"], key)

    target_record = data["productive_contract"]
    target = repo / target_record["module_path"]
    observed_target = analyze_target(target)
    assert observed_target["module_sha256"] == target_record["module_sha256"]
    assert observed_target["read_sink_count"] == 0
    assert observed_target["existence_sink_count"] == 0
    assert observed_target["serialization_sink_count"] >= 1
    assert observed_target["tainted_scalars"] == target_record["tainted_scalars"]
    assert observed_target["tainted_containers"] == target_record["tainted_containers"]

    bridge = repo / data["bridge"]["path"]
    assert bridge.is_symlink()
    assert bridge.readlink().as_posix() == data["bridge"]["expected_symlink_target"]

    fingerprint_payload = {
        "decision": data["decision"],
        "source_csv_value": data["source_csv_value"],
        "pairs": data["pairs"],
        "productive_contract": data["productive_contract"],
        "validated_audit": data["validated_audit"],
        "constraints": data["constraints"],
    }
    observed_fingerprint = sha256_bytes(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert observed_fingerprint == data["fingerprint_sha256"]

    return {
        "status": "ok",
        "pair_count": len(data["pairs"]),
        "total_row_count": data["summary"]["total_row_count"],
        "read_sink_count": observed_target["read_sink_count"],
        "existence_sink_count": observed_target["existence_sink_count"],
        "serialization_sink_count": observed_target["serialization_sink_count"],
        "fingerprint_sha256": observed_fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.repo.resolve(), args.manifest.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
