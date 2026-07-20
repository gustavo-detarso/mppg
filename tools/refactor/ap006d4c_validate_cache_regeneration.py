#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, csv, hashlib, json
from pathlib import Path
from typing import Any
OLD_ROOT = "academic_pipeline_rc10_7_conformidade"
def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def csv_observation(path: Path) -> dict[str, Any]:
    raw = path.read_bytes(); text = raw.decode("utf-8-sig")
    try: delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error: delimiter = ","
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "encoding": "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8", "delimiter": delimiter, "header": rows[0] if rows else [], "data_row_count": max(0, len(rows)-1), "old_reference_line_count": sum(OLD_ROOT in line for line in text.splitlines())}
def function_contract(path: Path, name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8"); tree = ast.parse(text)
    function = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    segment = ast.get_source_segment(text, function) or ast.unparse(function)
    calls, metadata_assignment, path_attribute_read = [], False, False
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            try: calls.append(ast.unparse(node.func))
            except Exception: pass
        if isinstance(node, ast.Attribute) and node.attr == "path": path_attribute_read = True
        if isinstance(node, ast.Subscript):
            try: rendered = ast.unparse(node)
            except Exception: rendered = ""
            if "metadata" in rendered and "fulltext_cache_path" in rendered: metadata_assignment = True
    return {"module_sha256": sha256(path), "function_sha256": hashlib.sha256(segment.encode()).hexdigest(), "arguments": [arg.arg for arg in function.args.args], "required_calls": {"shutil.rmtree": any(v.endswith("shutil.rmtree") for v in calls), "cache.mkdir": any(v.endswith("cache.mkdir") for v in calls), "shutil.copy2": any(v.endswith("shutil.copy2") for v in calls)}, "reads_doc_path": path_attribute_read, "updates_fulltext_cache_metadata": metadata_assignment}
def validate(repo: Path, manifest: Path) -> dict[str, object]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4C" and data["status"] == "cache_regeneration_contract_materialized"
    assert data["decision"] == "preserve_authoritative_sources_and_reproduce_caches_through_productive_copy_primitive"
    assert data["summary"]["pair_count"] == 4
    assert data["validated_dry_run"]["gate"] == "ready_for_ap006d4c_materialization_design"
    assert data["validated_dry_run"]["pytest"] == {"passed": 35, "returncode": 0, "xfailed": 1}
    for record in data["pairs"]:
        source, cache = repo / record["source_path"], repo / record["cache_path"]
        assert source.is_file() and cache.is_file() and source.read_bytes() == cache.read_bytes()
        observed = csv_observation(source)
        for key in ("sha256", "size_bytes", "encoding", "delimiter", "header", "data_row_count", "old_reference_line_count"): assert observed[key] == record[key], (record["source_path"], key)
        assert record["source_introduction_commit"] == data["introduction_commit"] == record["cache_introduction_commit"]
    target_record = data["productive_contract"]["target"]
    observed_contract = function_contract(repo / target_record["module_path"], target_record["function"])
    for key in ("module_sha256", "function_sha256", "arguments", "required_calls", "reads_doc_path", "updates_fulltext_cache_metadata"): assert observed_contract[key] == target_record[key], key
    caller_record = data["productive_contract"]["caller"]; assert sha256(repo / caller_record["module_path"]) == caller_record["module_sha256"]
    toml_record = data["productive_contract"]["toml"]; assert sha256(repo / toml_record["path"]) == toml_record["sha256"]
    bridge = repo / data["bridge"]["path"]; assert bridge.is_symlink() and bridge.readlink().as_posix() == data["bridge"]["expected_symlink_target"]
    fp = {"decision": data["decision"], "pairs": data["pairs"], "productive_contract": data["productive_contract"], "validated_dry_run": data["validated_dry_run"]}
    observed_fingerprint = hashlib.sha256(json.dumps(fp, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert observed_fingerprint == data["fingerprint_sha256"]
    return {"status": "ok", "pair_count": len(data["pairs"]), "exact_match_count": sum((repo / item["source_path"]).read_bytes() == (repo / item["cache_path"]).read_bytes() for item in data["pairs"]), "copy_primitive": target_record["copy_primitive"], "fingerprint_sha256": observed_fingerprint}
def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(validate(args.repo.resolve(), args.manifest.resolve()), ensure_ascii=False, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
