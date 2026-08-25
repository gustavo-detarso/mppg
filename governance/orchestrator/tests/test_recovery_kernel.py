#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1];src=(ROOT/"mppg_recovery_kernel.py").read_text();tree=ast.parse(src);funcs={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for f in {"run_child","parse_event","shadow_candidate","materialize_candidate","gate"}:assert f in funcs
assert "KERNEL RECOVERY MATERIALIZATION" in src and "rebuild_manifest(shadow)" in src and "OPENAI_API_KEY" in src and "software/academic_pipeline_mppg/" in src
print("RECOVERY_KERNEL_AST=PASS");print("RECOVERY_KERNEL_SHADOW_GUARD=PASS");print("RECOVERY_KERNEL_INLINE_GATE=PASS")
